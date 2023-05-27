from logging.handlers import RotatingFileHandler
from twitterwebcrawler import TwitterScraper
from svm_algorithm import SvmThinker
import snscrape.modules.twitter as sntwitter
import pandas as pd
import logging
import yaml


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


def process_twitter_data(logger):
    """
    Process Twitter data: collect tweets for each trend, store them in a pandas DataFrame,
    and use the SvmThinker class to process the data.
    """
    try:
        trends = get_twitter_trends()
        config = load_config()
        scrapper = TwitterScraper(config, logger=logger)
        tweets_dict = scrapper.get_tweets(trends)
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
        for trend in trends:
            tweets = [tweet for tweet in df[trend].tolist()
                      if tweet is not None]
            positive_count, negative_count = thinker.classify_tweets(
                trend, tweets)
            logger.info(
                f"Total positive tweets about '{trend}': {positive_count}")
            logger.info(
                f"Total negative tweets about '{trend}': {negative_count}")
    except Exception as e:
        logger.error(
            f"Error occurred during Twitter data processing: {str(e)}")
        raise


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
        process_twitter_data(logger)
    except BaseException as e:
        logger.error(f"Error occurred in the main process: {str(e)}")
        raise
