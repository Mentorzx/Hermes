from logging.handlers import RotatingFileHandler
from twitterwebcrawler import TwitterScraper
from svm_algorithm import SvmThinker
from translate_dataset import STranslator
from flask import Flask, abort, jsonify, request
from typing import Union
import snscrape.modules.twitter as sntwitter
import pandas as pd
import logging
import zipfile
import yaml
import io

app = Flask(__name__)


def get_twitter_trends() -> list[str]:
    """
    Get the current Twitter trends.

    Returns:
        list: List of trending topics on Twitter.
    """
    try:
        trends = [trend.name for trend in sntwitter.TwitterTrendsScraper().get_items()]
        trends_translated = STranslator().translate_list("pt", "en", trends)
        logging.info(f"Trends Collected: {trends_translated}")
        return trends_translated
    except Exception as e:
        logging.error(f"Error occurred while getting Twitter trends: {str(e)}")
        exit()


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


def separate_tweets_by_keywords(keyword_list) -> pd.DataFrame:
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
                )
    except BaseException as e:
        logger.error(f"Error occurred while loading data: {str(e)}")
        exit()
    filtered_df = pd.DataFrame(columns=["word"] + df.columns.tolist())
    for keyword in keyword_list:
        filtered_tweets = df[df["text"].str.contains(keyword, case=False)]
        filtered_tweets["word"] = keyword
        filtered_df = pd.concat([filtered_df, filtered_tweets])
    return filtered_df


def process_twitter_data(
    words: list[str], config: dict[str, str], logger
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
        thinker = SvmThinker(logger=logger)
        thinker.load_data()
        thinker.preprocess_data()
        thinker.load_sentiment_words()
        thinker.train_model()
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
                logger.info(f"Total tweets types about '{word}': {tweets_types}")
            result[f"{word}"] = [positive_count, negative_count, tweets_types]
            result_compare[f"{word}"] = [positive, negative]
        return result
    except Exception as e:
        logger.error(f"Error occurred during Twitter data processing: {str(e)}")
        exit()


def validate_api_key(config: dict[str, str]):
    api_key = config.get("api_key")
    if "X-API-Key" not in request.headers or request.headers["X-API-Key"] != api_key:
        abort(401, "Unauthorized")


@app.route("/irony", methods=["POST"])
def check_irony():
    """
    Check if a tweet is ironic.

    Request JSON:
    {
        "tweet": "Tweet text"
    }

    Response JSON:
    {
        "is_ironic": true or false
    }
    """
    # tweet = request.json["tweet"]
    # # Implement irony checking logic here
    # is_ironic = False  # Placeholder

    # return jsonify({"is_ironic": is_ironic})
    abort(400, "Function not implemented yet")


@app.route("/sentiment", methods=["POST"])
def analyze_sentiment():
    """
    Analyze the sentiment of a list of tweets.

    Request JSON:
    {
        "tweets": ["Tweet 1", "Tweet 2", ...]
    }

    Response JSON:
    {
        "results": [
            {
                "positive_count": 10,
                "negative_count": 5
            },
            ...
        ]
    }
    """
    # tweets = request.json["tweets"]
    # results = []

    # for trend in trends:
    #     positive_count, negative_count = process_tweet_data(trend, tweets)
    #     results.append({"positive_count": positive_count, "negative_count": negative_count})

    # return jsonify({"results": results})
    abort(400, "Function not implemented yet")


@app.route("/trends", methods=["GET"])
def get_trends_sentiment():
    """
    Get the sentiment analysis of Twitter trends.

    Response JSON:
    {
        "results": [
            {
                "trend": "Trend 1",
                "positive_count": 10,
                "negative_count": 5,
                "tweet_types": {"ironic": ["tweet 1", ...], "sarcasm": ["tweet 2", ...], "regular": ["tweet 3", ...], "figurative": ["tweet 4", ...]}
            },
            ...
        ]
    }
    """
    results = []
    config = load_config()
    validate_api_key(config)
    logger = logging.getLogger()
    trends = get_twitter_trends()
    words = process_twitter_data(trends, config, logger)
    for trend in words:
        trend_data = words.get(
            trend,
            [0, 0, {}],
        )
        results.append(
            {
                "trend": trend,
                "positive_count": trend_data[0],
                "negative_count": trend_data[1],
                "tweet_types": trend_data[2],
            }
        )
    return jsonify({"results": results})


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


if __name__ == "__main__":
    log_filename = "logging.log"
    logger = configure_logging(log_filename)
    try:
        config = load_config()
        app.run()  # Start the Flask application
    except BaseException as e:
        logger.error(f"Error occurred in the main process: {str(e)}")
        exit()
