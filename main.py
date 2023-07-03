from logging.handlers import RotatingFileHandler
from twitterwebcrawler import TwitterScraper
from svm_algorithm import SvmThinker
from translate_dataset import STranslator
from flask import Flask, abort, jsonify, request
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


app = Flask(__name__)


def configure_logging(log_filename: str) -> logging.Logger:
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
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)
    return logger


def load_config() -> dict[str, str]:
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


def train_thinker() -> SvmThinker:
    thinker = SvmThinker(logger)
    thinker.load_data()
    thinker.preprocess_data()
    thinker.load_sentiment_words()
    thinker.train_model()
    return thinker


def separate_tweets_by_keywords(keyword_list: list[str]) -> pd.DataFrame:
    """
    Reads a CSV file named 'tweets_trained.csv' located in the 'datasets' folder within 'datasets.zip' and separates the tweets based on the provided keyword list.

    Args:
        keyword_list (list[str]): A list of keywords to search for in the tweets.

    Returns:
        pandas.DataFrame: A DataFrame containing the tweets that match the provided keywords.

    Example:
        keyword_list = ['bummer', 'outside']
        filtered_df = separate_tweets_by_keywords(keyword_list)

    The function reads the CSV file located in the 'datasets' folder within 'datasets.zip' and searches for each keyword in the 'text' column of the DataFrame.
    It creates a new DataFrame with the tweets that contain any of the provided keywords.
    The 'word' column in the DataFrame represents the keyword associated with each tweet.

    Note:
        The CSV file must be located within the 'datasets' folder in 'datasets.zip'.
        The function assumes that the CSV file has the following columns: 'target', 'ids', 'date', 'flag', 'user', 'text'.
    """
    try:
        logger.info("Loading trained data...")
        with open("datasets/datasets.zip", "rb") as zip_file:
            conteudo_zip = zip_file.read()
        arquivo_zip = io.BytesIO(conteudo_zip)
        with zipfile.ZipFile(arquivo_zip, "r") as zip_ref:
            with zip_ref.open("tweets_trained.csv") as file:
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
        logger.info("Trained data readed.")
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


def process_twitter_data(
    words: list[str], thinker: SvmThinker
) -> dict[str, list[Union[int, dict[str, str]]]]:
    """
    Process Twitter data: collect tweets for each trend, store them in a pandas DataFrame,
    and use the SvmThinker class to process the data.
    """
    try:
        result = {}
        result_compare = {}
        positive, negative = 0, 0
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
            targets = df.loc[df["word"] == word, "target"].tolist()
            positive = len([x for x in targets if x > 2])
            negative = len([x for x in targets if x <= 2])
            positive_count, negative_count, tweets_types = thinker.classify_tweets(
                word, tweets
            )
            if len(tweets_types) > 0:
                logger.info(
                    f"Total trained-positive tweets about '{word}': {positive_count}"
                )
                logger.info(f"Total real-positive tweets about '{word}': {positive}")
                logger.info(
                    f"Total trained-negative tweets about '{word}': {negative_count}"
                )
                logger.info(f"Total real-negative tweets about '{word}': {negative}")
                if positive_count <= positive:
                    positive_ac = positive_count
                else:
                    positive_ac = positive
                if negative_count <= negative:
                    negative_ac = negative_count
                else:
                    negative_ac = negative
                logger.warning(
                    f"{(((positive_ac+negative_ac)/(positive+negative))*100):.2f}% precision"
                )
                logger.info(f"Total tweets types about '{word}': {tweets_types}")
            result[f"{word}"] = [positive_count, negative_count, tweets_types]
            result_compare[f"{word}"] = [positive, negative]
        logger.info("Twitter data processed!")
        return result
    except Exception as e:
        logger.error(f"Error occurred during Twitter data processing: {str(e)}")
        exit()


def process_trends_search(trends_search: list[str]) -> list:
    words = process_twitter_data(trends_search, thinker)
    results = []
    for trend in trends_search:
        trend_data = words.get(trend, [0, 0, {}])
        results.append(
            {
                "trend": trend,
                "positive_count": trend_data[0],
                "negative_count": trend_data[1],
                "tweet_types": trend_data[2],
            }
        )
    return results


def create_response(page: int = 1, total_pages: int = 1, results: list = []) -> dict:
    response = {
        "total_pages": total_pages,
        "current_page": page,
        "results": results,
    }
    if page < total_pages:
        next_page = f"http://hermesproject.pythonanywhere.com/trends?page={page + 1}"
        response["next_page"] = next_page
    return response


def validate_api_key(config: dict[str, str]):
    api_key = config.get("api_key")
    if "X-API-Key" not in request.headers or request.headers["X-API-Key"] != api_key:
        logger.warning("Unauthorized API Key!")
        abort(401, "Unauthorized")


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
    Get 20 of the Twitter trends.

    Response JSON:
    {
        "description": "ReadMe of the project",
        "results": [
            "trend 1", "trend 2", "trend 3", "trend 4", ...
        ]
    }
    """
    # validate_api_key(config)
    trends = get_twitter_trends()
    response = {
        "description": """
        Project Hermes

        Description

        Project Hermes is an algorithm that uses machine learning techniques to classify figures of speech in tweets. Currently, the algorithm uses Support Vector Machine (SVM) for this classification. Another SVM will be added soon to classify the words present in the figures of speech.

        The algorithm no longer uses web scraping to collect tweets. Initially, web scraping was used, but due to conflicts with API hosting, a database with 1,600,000 tweets was used instead. The link to the database is: https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download. However, the other datasets used to train the algorithm were obtained from sources that were not properly credited. We apologize for this lack of credits, but the datasets will be available in the GitHub link: https://github.com/Mentorzx/Hermes.

        The algorithm analyzes the collected tweets and classifies the figures of speech found in them. The goal is to identify and evaluate the opinions of Twitter users about certain subjects, using natural language processing and machine learning techniques.

        Features

        Project Hermes has the following features:

        - Tweet collection: The algorithm uses a database with 1,600,000 tweets to perform the analysis of figures of speech. The tweets are pre-classified into figures of speech, such as figurative, ironic, regular, and sarcastic.

        - Classification of figures of speech: The algorithm uses Support Vector Machine (SVM) to classify the figures of speech present in the tweets.

        - Sentiment classification: The algorithm uses Support Vector Machine (SVM) to classify whether the searched keywords have a negative or positive sentiment according to the found tweets.

        - Keyword insertion: It is possible to search for your own keyword, analyze its sentiment, and retrieve the tweets with their respective figures of speech.

        - Retrieving current Brazilian trends.

        Usage

        To use Project Hermes, follow the steps below:

        1. Set up the environment: Make sure you have the necessary dependencies installed in your development environment. Refer to the requirements.txt file (still under development) for a complete list of dependencies and their corresponding versions.

        2. Configuration of the config.yml file: The config.yml file contains the algorithm's configurations, such as Twitter access credentials and other relevant settings. Fill in all the necessary information correctly before running the algorithm.

        3. Running the algorithm: Run the main.py file to start the API for the analysis of figures of speech in tweets. The algorithm will use the database of 1,600,000 tweets to perform the classification.

        Using the API

        The Project Hermes API allows you to make queries and obtain results through HTTP requests. Below are examples of how to use the API with cURL:

        1. Search for a keyword:
        ("Potato" as an exemple)
        curl -X POST -H "Content-Type: application/json" -H "{insert key here without "}"}" -d '{"keyword": "potato"}' http://hermesproject.pythonanywhere.com/search

        2. Get the sentiment of the trends:]
        curl -X GET -H "{insert key here without "}"}" http://hermesproject.pythonanywhere.com/trends

        3. Get only the trends (without sentiment):
        curl -X GET -H "{insert key here without "}"}" http://hermesproject.pythonanywhere.com/

        The above changes add the option to specify the page on which you want to get trend sentiment. Simply replace `{insert key here without "}"}` with the correct key from the `config.yml` file and insert the desired page number in `?page=1` (for example, `?page=1` for the first page, `?page=2` for the second page, and so on).

        Please remember to follow the applicable policies and terms of use when using the API.

        Legal Disclaimer

        Project Hermes was developed exclusively for academic purposes and to enhance the resume of the developer Alex Lira. Web scraping initially used to collect tweets was replaced by a database of 1,600,000 tweets. The link to the database is: https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download. The other datasets used in training the algorithm were obtained from sources that were not properly credited. The datasets will be available in the GitHub link: https://github.com/Mentorzx/Hermes.

        Developer Alex Lira is not responsible for any misuse or rights violations resulting from the use of Project Hermes. The user is solely responsible for ensuring that their use complies with the applicable policies and terms of use.

        Contact

        For more information about Project Hermes or to contact developer Alex Lira, you can access his profile on GitHub.

        Website developed by Jéssica Andrade and João Paulo Rios (Front-End Team) and Alex Lira (Back-end).
        """,
        "results": trends,
    }
    return jsonify(response)


@app.route("/trends", methods=["GET"])
def get_trends_sentiment():
    page = int(request.args.get("page", 1))
    trends_per_page = 5
    total_results = 20
    total_pages = math.ceil(total_results / trends_per_page)
    trends = get_twitter_trends()

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
        if page in trends_cache:
            json = jsonify(trends_cache[page])
            logging.info(f"Cache hit for page: {page}: {trends_cache[page]}")
            return json
        start_index, end_index = get_page_index(page)
        trends_search = trends[start_index:end_index]
        results = process_trends_search(trends_search)
        response = create_response(page, total_pages, results)
        trends_cache[page] = response
        json = jsonify(response)
        logging.info(f"Processed page {page}: {response}")
        return json

    def process_remaining_pages():
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

    try:
        return get_page_response(page)
    finally:
        thread = Thread(target=process_remaining_pages)
        thread.start()


config = load_config()
logger = configure_logging("logging.log")
thinker = train_thinker()
trends_cache = {}


if __name__ == "__main__":
    try:
        app.run()
    except BaseException as e:
        logger.error(f"Error occurred in the main process: {str(e)}")
        exit()
