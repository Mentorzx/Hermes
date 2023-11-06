from logging.handlers import RotatingFileHandler
from twitterwebcrawler import TwitterScraper
from svm_algorithm import SvmThinker
from translate_dataset import STranslator
from flask import Flask, abort, jsonify, request
from sklearn.metrics import confusion_matrix
from threading import Thread
from functools import wraps
from typing import Union
import snscrape.modules.twitter as sntwitter
import pandas as pd
import logging
import zipfile
import math
import yaml
import re
import io
import os


app = Flask(__name__)


def configure_logging(log_filename: str) -> logging.Logger:
    """
    Configure logging settings and return the logger object.

    Args:
        log_filename (str): The name of the log file.
    Returns:
        logging.Logger: The logger object.
    """
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    log_handler = RotatingFileHandler(
        log_filename,
        mode="a",
        maxBytes=10 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    log_handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(log_handler)
    return logger


def load_config() -> Union[dict[str, str], dict[str, list[str]]]:
    """
    Load the configuration from the config.yml file.

    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open("config.yml", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error occurred while loading configuration: {str(e)}")
        exit()


def get_twitter_trends() -> list[str]:
    """
    Get the current Twitter trends.

    Returns:
        list: List of trending topics on Twitter.
    """
    try:
        logging.info("Collecting Trends...")
        trends = [trend.name for trend in sntwitter.TwitterTrendsScraper().get_items()]
        trends_translated = STranslator().translate_list("pt", "en", trends)
        logging.info(f"Trends Collected: {trends_translated}")
        return trends_translated
    except Exception as e:
        logging.error(f"Error occurred while getting Twitter trends: {str(e)}")
        exit()


def train_thinker() -> tuple[SvmThinker, list[dict[str, int]]]:
    """
    Train the SvmThinker model and return the trained model and metrics.

    Returns:
        tuple[SvmThinker, dict[str, int]]: A tuple containing the trained SvmThinker model
        and a dictionary of metrics.
    """
    datasets = [config.get("datasets_path")]
    datasets += config.get("datasets") or []
    datasets = list(filter(None, datasets))  # Remove None from the list
    datasets = [item for sublist in datasets for item in (sublist if isinstance(sublist, list) else [sublist])] if datasets else []
    thinker = SvmThinker(logger)
    thinker.load_data(datasets, is_zip=True)
    thinker.preprocess_data(spell_check=False)
    thinker.train_model()
    metrics = thinker.evaluate_model()
    return thinker, metrics


def separate_tweets_by_keywords(keyword_list: list[str]) -> pd.DataFrame:
    """
    Reads a CSV file located in the 'datasets' folder within 'datasets.zip' and separates the tweets based on the provided keyword list.

    Args:
        keyword_list (list[str]): A list of keywords to search for in the tweets.

    Returns:
        pandas.DataFrame: A DataFrame containing the tweets that match the provided keywords.

    The function reads the CSV file located in the 'datasets' folder within 'datasets.zip' and searches for each keyword in the 'text' column of the DataFrame.
    It creates a new DataFrame with the tweets that contain any of the provided keywords.
    The 'word' column in the DataFrame represents the keyword associated with each tweet.

    Note:
        The CSV file must be located within the 'datasets' folder in 'datasets.zip'.
        The function assumes that the CSV file has the following columns: 'target', 'ids', 'date', 'flag', 'user', 'text'.
    """
    try:
        logger.info("Loading trained data...")
        datasets_path = config.get("datasets_path")
        databank = config.get("databank")
        if isinstance(datasets_path, str) and isinstance(databank, str):
            with open(datasets_path, "rb") as zip_file:
                content_zip = zip_file.read()
            file_zip = io.BytesIO(content_zip)
            with zipfile.ZipFile(file_zip, "r") as zip_ref:
                with zip_ref.open(databank) as file:
                    df = pd.read_csv(
                        file,
                        encoding="latin-1",
                        names=["target", "ids", "date", "flag", "user", "text"],
                        dtype={
                            "target": int,
                            "ids": int,
                            "date": str,
                            "flag": str,
                            "user": str,
                            "text": str,
                        },
                    )
            logger.info("Trained data read.")
        else:
            raise TypeError("The datasets path and databank must be strings")
    except BaseException as e:
        logger.error(f"Error occurred while loading data: {str(e)}")
        exit()
    logger.info("Starting trained data handling for dataframe...")
    keyword_pattern = f"(?:{'|'.join(keyword_list)})"
    keyword_mask = df["text"].str.contains(keyword_pattern, case=False, regex=True)
    filtered_df = df[keyword_mask].copy()
    filtered_df["word"] = (
        filtered_df["text"]
        .str.extractall(f"({keyword_pattern})", flags=re.IGNORECASE)
        .groupby(level=0)
        .apply(lambda x: ",".join(x[0]))
    )
    logger.info("Trained data handling for dataframe finished!")
    return filtered_df


def process_tweets_by_word(
    word: str, tweets: list[str], positive: int, negative: int
) -> list[Union[dict[str, int], dict[str, list[str]]]]:
    sentiment_tweets, figure_tweets = thinker.classify(word, tweets)
    sentiment_count = [len(tweets) for tweets in sentiment_tweets.values()]
    for analysis in (sentiment_tweets, figure_tweets):
        if analysis:
            log_metrics_results(
                word, analysis, {"positive": positive, "negative": negative}
            )
    return [{key: len(value) for key, value in sentiment_tweets.items()}, figure_tweets]


def process_twitter_data(
    words: list[str],
) -> dict[str, list[Union[int, dict[str, str]]]]:
    """
    Process Twitter data for the given words using the SvmThinker model.

    Args:
        words (list[str]): A list of words to process.
        thinker (SvmThinker): The SvmThinker model used for processing.

    Returns:
        dict[str, list[Union[int, dict[str, str]]]]: A dictionary containing the results
        of the Twitter data processing for each word.
    """
    try:
        result = {}
        # scrapper = TwitterScraper(config, logger=logger)
        # tweets_dict = scrapper.get_tweets(words)
        # max_len = max(len(v) for v in tweets_dict.values())
        # filled_tweets_dict = {}
        # for k, v in tweets_dict.items():
        #     if len(v) < max_len:
        #         v = list(v) + [None] * (max_len - len(v))
        #     else:
        #         v = list(v)
        #     filled_tweets_dict[k] = v
        df = separate_tweets_by_keywords(words)
        logger.info("Starting process twitter data...")
        for word in words:
            tweets = df.loc[df["word"] == word, "text"].tolist()
            if not len(tweets):
                logger.warning(f"No tweets found for '{word}'")
                targets = []
            else:
                targets = df.loc[df["word"] == word, "target"].tolist()
            positive = len([x for x in targets if x > 2])
            negative = len([x for x in targets if x <= 2])
            result[word] = process_tweets_by_word(word, tweets, positive, negative)
        logger.info("Twitter data processed!")
        return result
    except Exception as e:
        logger.error(f"Error occurred during Twitter data processing: {str(e)}")
        exit()


def log_metrics_results(
    word: str, analysis: dict[str, list[str]], real_values: dict[str, int]
) -> None:
    """
    Log the metrics results based on the provided analysis and real values.

    Args:
        word (str): The word or subject of the analysis.
        analysis (dict[str, list[str]]): A dictionary containing the analysis results with different keys.
            Each key represents a specific type of analysis, and its value is a list of tweets related to that type.
        real_values (dict[str, int]): A dictionary containing the real values or counts for each tweet type.
            Each key represents a specific tweet type, and its value is the count of tweets for that type.

    Returns:
        None

    Logs the following information:
    - The total number of tweets for each analysis type: 'Total <key>-trained tweets about '<word>': <count>'
    - If 'analysis' is one of the tweet types ['positive', 'negative', 'neutral', 'irrelevant'],
      the total number of real tweets for each tweet type: 'Total <key>-real tweets about '<word>': <count>'
    - The precision value based on the minimum count between the analysis and real values for 'positive' and 'negative' tweet types.
    - The total number of tweets for all types of analysis: 'Total tweets types about '<word>': <total_size>'
    """
    for key in analysis:
        logger.info(f"Total {key}-trained tweets about '{word}': {len(analysis[key])}")
    logger.info("----------------------------------------------")
    if any(
        key in ["positive", "negative", "neutral", "irrelevant"] for key in analysis
    ):
        for key in real_values:
            logger.info(f"Total {key}-real tweets about '{word}': {real_values[key]}")
        positive_ac = min(
            len(analysis_pos := analysis.get("positive", [])),
            real_values_pos := real_values.get("positive", 0),
        )
        negative_ac = min(
            len(analysis_neg := analysis.get("negative", [])),
            real_values_neg := real_values.get("negative", 0),
        )
        logger.warning(
            f"{(((positive_ac + negative_ac) / (real_values_pos + real_values_neg)) * 100):.2f}% precision"
        )
    else:
        total_size = sum(len(value) for value in analysis.values())
        logger.info(f"Total tweets types about '{word}': {total_size}")


def process_trends_search(trends_search: list[str]) -> list[dict]:
    """
    Process the trends search by calling the 'process_twitter_data' function and return the results.

    Args:
        trends_search (list[str]): A list of trends to search.

    Returns:
        list: A list of dictionaries containing the results for each trend, including the trend name,
            positive count, negative count, and tweet types.
    """
    words = process_twitter_data(trends_search)
    results = []
    for trend in trends_search:
        trend_data = words.get(trend, [0, 0, 0, 0, {}])
        trend_results = []
        for dic in trend_data[-2:]:
            result = {
                "trend": trend,
                "negative_count": trend_data[0],
                "neutral_count": trend_data[1],
                "irrelevant_count": trend_data[2],
                "positive_count": trend_data[3],
                "tweet_types": trend_data[-1],
            }
            if isinstance(dic, dict):
                for key, value in dic.items():
                    result[key] = value
            trend_results.append(result)
        results.extend(trend_results)
    return results


def create_response(page: int = 1, total_pages: int = 1, results: list = []) -> dict:
    # sourcery skip: default-mutable-arg
    """
    Create a response dictionary containing pagination information and the results.

    Args:
        page (int): The current page number (default is 1).
        total_pages (int): The total number of pages (default is 1).
        results (list): The list of results (default is an empty list).

    Returns:
        dict: A dictionary containing the response information, including the total number of pages,
            the current page number, the results, and the next page URL if applicable.

    """
    response = {
        "total_pages": total_pages,
        "current_page": page,
        "results": results,
    }
    if page < total_pages:
        next_page = f"http://hermesproject.pythonanywhere.com/trends?page={page + 1}"
        response["next_page"] = next_page
    return response


def validate_api_key(config: dict):
    """
    Validate the API key from the request headers against the configured API key.

    Args:
        config (dict[str, str]): A dictionary containing the configuration parameters.

    Raises:
        HTTPException: If the API key is missing or doesn't match the configured API key.
    """
    api_key = config.get("api_key")
    if "X-API-Key" not in request.headers or request.headers["X-API-Key"] != api_key:
        logger.warning("Unauthorized API Key!")
        abort(401, "Unauthorized")


@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    """
    Clear the cache by removing all stored trends from the cache.

    Returns:
        dict: A JSON response indicating that the cache has been cleared.
    """
    # validate_api_key(config)
    trends_cache.clear()
    logging.info("Cache Cleared.")
    return jsonify({"message": "Cache cleared."})


@app.route("/search", methods=["POST"])
def search_tweets():
    """
    Search tweets based on a keyword.

    Request JSON:
    {
        "keyword": "Keyword"
    }

    Response JSON:
    {
        "results": [
            {
                "trend": "keyword",
                "positive_count": 10,
                "negative_count": 5,
                "tweet_types": {"ironic": ["tweet 1", ...], "sarcasm": ["tweet 2", ...], "regular": ["tweet 3", ...], "figurative": ["tweet 4", ...]}
            }
        ]
    }
    """
    # validate_api_key(config)
    request_data = request.get_json()
    if request_data and "keyword" in request_data:
        keyword = request_data["keyword"]
        result = process_trends_search([keyword])
        return jsonify({"results": result})
    else:
        abort(400, "Missing 'keyword' parameter in the request")


@app.route("/", methods=["GET"])
def get_trends():
    """
    Get the content of the README file.

    Returns:
        dict: A JSON response containing the content of the README file.
    """
    readme_path = os.path.join(app.root_path, "README.md")
    with open(readme_path, "r") as file:
        readme_content = file.read()
    response = {"content": readme_content}
    return jsonify(response)


@app.route("/metrics", methods=["GET"])
def get_twitter_metrics():
    """
    Get the algorithm metrics.

    Returns:
        dict: A JSON response containing the Twitter metrics.
    """
    try:
        logging.info(f"Catching metrics: {metrics}")
        return jsonify(metrics)
    except Exception as e:
        logging.error(f"Error while getting metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/trends", methods=["GET"])
def get_trends_sentiment():
    """
    Get the sentiment of Twitter trends.

    Returns:
        dict: A JSON response containing the sentiment of Twitter trends.
    """
    page = int(request.args.get("page", 1))
    trends_per_page = 5
    total_results = 20
    total_pages = math.ceil(total_results / trends_per_page)
    if page in trends_cache:
        json = jsonify(trends_cache[page])
        logging.info(f"Cache hit for page: {page}: {trends_cache[page]}")
        return json
    # trends = get_twitter_trends()
    trends = [
        "Book",
        "Discrimination",
        "Exploitation",
        "Addiction",
        "Mental health",
        "Anita",
        "Digital nomad",
        "Propaganda",
        "Wellness",
        "Disinformation",
        "Fake news",
        "Influencer",
        "Self-care",
        "Upcycling",
        "Ruby",
        "Mindfulness",
        "Slow living",
        "Fight",
        "Sustainability",
        "Cancel culture",
    ]

    def validate_page(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if page < 1 or page > total_pages:
                logging.error(f"Invalid page number: {page}")
                return jsonify({"error": "Invalid page number"}), 400
            return func(*args, **kwargs)

        return wrapper

    def get_page_index(page):
        start_index = (page - 1) * trends_per_page
        end_index = page * trends_per_page
        return start_index, end_index

    @validate_page
    def get_page_response(page):
        """
        Get the response for a specific page of Twitter trends.

        Args:
            page (int): The page number.

        Returns:
            dict: A JSON response containing the sentiment of Twitter trends for the specified page.
        """
        start_index, end_index = get_page_index(page)
        trends_search = trends[start_index:end_index]
        results = process_trends_search(trends_search)
        response = create_response(page, total_pages, results)
        trends_cache[page] = response
        json = jsonify(response)
        logging.info(f"Processed page {page}: {response}")
        return json

    def process_remaining_pages(page):
        """
        Process the remaining pages of Twitter trends.

        This function is executed in a separate thread to process the remaining pages while the current page
        response is being returned.

        Returns:
            None
        """
        remaining_results = []
        pages = set(range(1, total_pages + 1))
        pages.discard(page)
        for remaining_page in pages:
            start_index, end_index = get_page_index(remaining_page)
            remaining_trends_search = trends[start_index:end_index]
            remaining_results += process_trends_search(remaining_trends_search)
            trends_cache[remaining_page] = create_response(
                remaining_page, total_pages, remaining_results
            )
            logging.info(f"Processed remaining trends for page {remaining_page}")

        # # Combine the tweet classifications for all trends
        # all_predictions = []
        # all_targets = []
        # for page in trends_cache:
        #     for trend_data in trends_cache[page]["results"]:
        #         for prediction, tweets in trend_data["tweet_types"].items():
        #             all_predictions += [prediction] * len(tweets)
        #             all_targets += [prediction] * len(tweets)

        # # Generate the confusion matrix
        # confusion = confusion_matrix(all_targets, all_predictions)
        # logging.info("Confusion Matrix:")
        # logging.info(confusion)

    try:
        return get_page_response(page)
    finally:
        thread = Thread(target=process_remaining_pages, args=(page,))
        thread.start()


config = load_config()
logger = configure_logging("logging.log")
thinker, metrics = train_thinker()
trends_cache = {}


if __name__ == "__main__":
    try:
        app.run()
    except BaseException as e:
        logger.error(f"Error occurred in the main process: {str(e)}")
        exit()
