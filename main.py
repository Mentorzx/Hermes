import snscrape.modules.twitter as sntwitter

trends = [trend.name for trend in sntwitter.TwitterTrendsScraper().get_items()]
print(trends)
