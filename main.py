from logging.handlers import RotatingFileHandler
from twitterwebcrawler import TwitterScraper
from svm_algorithm import SvmThinker
from flask import Flask, abort, jsonify, request
import snscrape.modules.twitter as sntwitter
import pandas as pd
import logging
import yaml


app = Flask(__name__)


def get_twitter_trends() -> list[str]:
    """
    Get the current Twitter trends.

    Returns:
        list: List of trending topics on Twitter.
    """
    try:
        trends = [
            trend.name for trend in sntwitter.TwitterTrendsScraper().get_items()]
        logging.info(f"Trends Collected: {trends}")
        return trends
    except Exception as e:
        logging.error(f"Error occurred while getting Twitter trends: {str(e)}")
        raise


def load_config() -> dict[str, str]:
    """
    Load the configuration from the config.yml file.

    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open("config.yml", encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error occurred while loading configuration: {str(e)}")
        raise


def process_twitter_data(words: list[str], config: dict[str, str], logger) -> dict[str, list[int]]:
    """
    Process Twitter data: collect tweets for each trend, store them in a pandas DataFrame,
    and use the SvmThinker class to process the data.
    """
    try:
        result = {}
        scrapper = TwitterScraper(config, logger=logger)
        tweets_dict = scrapper.get_tweets(words)
        max_len = max(len(v) for v in tweets_dict.values())
        filled_tweets_dict = {}
        for k, v in tweets_dict.items():
            if len(v) < max_len:
                v = list(v) + [None] * (max_len - len(v))
            else:
                v = list(v)
            filled_tweets_dict[k] = v
        df = pd.DataFrame(filled_tweets_dict)
        thinker = SvmThinker(logger=logger)
        thinker.load_data()
        thinker.preprocess_data()
        thinker.load_sentiment_words()
        thinker.train_model()
        for word in words:
            tweets = [tweet for tweet in df[word].tolist()
                      if tweet is not None]
            positive_count, negative_count = thinker.classify_tweets(
                word, tweets)
            logger.info(
                f"Total positive tweets about '{word}': {positive_count}")
            logger.info(
                f"Total negative tweets about '{word}': {negative_count}")
            result[f"{word}"] = [positive_count, negative_count]
        return result
    except Exception as e:
        logger.error(
            f"Error occurred during Twitter data processing: {str(e)}")
        raise


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
                "negative_count": 5
            },
            ...
        ]
    }
    """
    results = []
    config = load_config()
    validate_api_key(config)
    trends = get_twitter_trends()
    words = process_twitter_data(trends, config, logger)
    for trend in words:
        results.append({
            "trend": trend,
            "positive_count": words[trend][0],
            "negative_count": words[trend][1]
        })

    return jsonify({"results": results})


def configure_logging(log_filename: str) -> logging.Logger:
    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    log_handler = RotatingFileHandler(
        log_filename, mode='a', maxBytes=10*1024*1024, backupCount=2, encoding='utf-8')
    log_handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)
    return logger


if __name__ == "__main__":
    log_filename = 'logging.log'
    logger = configure_logging(log_filename)
    try:
        config = load_config()
        app.run()  # Start the Flask application
    except BaseException as e:
        logger.error(f"Error occurred in the main process: {str(e)}")
        raise
