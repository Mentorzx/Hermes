from twitterwebcrawler import TwitterScrapper
from svm_algorithm import SvmThinker
import snscrape.modules.twitter as sntwitter
import pandas as pd
import yaml


def get_twitter_trends():
    """
    Get the current Twitter trends.

    Returns:
        list: List of trending topics on Twitter.
    """
    trends = [trend.name for trend in sntwitter.TwitterTrendsScraper().get_items()]
    return trends


def load_config():
    """
    Load the configuration from the config.yml file.

    Returns:
        dict: Configuration dictionary.
    """
    with open("config.yml", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def process_twitter_data():
    """
    Process Twitter data: collect tweets for each trend, store them in a pandas DataFrame,
    and use the SvmThinker class to process the data.
    """
    trends = get_twitter_trends()
    config = load_config()
    scrapper = TwitterScrapper(config)
    tweets_dict = {}
    for trend in trends:
        tweets = scrapper.get_tweets(trend)
        tweets_dict[trend] = tweets
    df = pd.DataFrame.from_dict(tweets_dict)
    thinker = SvmThinker()
    thinker.process_data(df)


if __name__ == "__main__":
    process_twitter_data()
