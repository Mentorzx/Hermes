from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from translate_dataset import STranslator
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from langdetect import detect
from sklearn.svm import SVC
import pandas as pd
import logging
import nltk
import re


class SvmThinker:
    """
    A class that uses Support Vector Machines (SVM) for tweet classification.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """
        Initialize the SvmThinker object.

        Args:
            logger (logging.Logger): Logger object for logging messages.
        """
        self.logger = logger
        self.vectorizer = TfidfVectorizer()
        self.svm = SVC(kernel="linear")
        self.positive_words = []
        self.negative_words = []

    def load_data(self) -> None:
        """
        Load the dataset and sentiment words.
        """
        try:
            self.logger.info("Loading data...")
            # dataset that contains one column with one tweet and another column that contains the respective speech figure (irony, regular, figurative or sarcasm)
            self.data = pd.read_csv("datasets/speech_figures.csv", encoding="utf-8")
            # dataset that contains one column with positive words and another column that contains the negative words
            self.sentiment_data = pd.read_csv(
                "datasets/sentiment_words.csv", encoding="utf-8"
            )
        except BaseException as e:
            self.logger.error(f"Error occurred while loading data: {str(e)}")
            exit()

    def preprocess_data(self):
        """
        Preprocess the data if necessary.
        """
        try:
            self.logger.info("Preprocessing data...")
            pattern = r"[^\w\s]|(\d+)|http\S+|#\S+|@\S+|[^\x00-\x7F]+"
            nltk.download("stopwords")
            nltk.download("punkt")
            nltk.download("wordnet")
            stop_words = set(stopwords.words("english"))
            lemmatizer = WordNetLemmatizer()
            spell = SpellChecker()
            # Replace np.nan with empty string (this is not necessary if you preprocess the data before filling the NaNs)
            self.data["tweet_text"] = self.data["tweet_text"].fillna("")
            # Combine multiple regular expressions into a single pattern
            self.data["tweet_text"] = self.data["tweet_text"].apply(
                lambda x: re.sub(pattern, "", x)
            )
            # Convert to lowercase
            self.data["tweet_text"] = self.data["tweet_text"].str.lower()
            # Fix spelling errors
            # fix = lambda x: spell.correction(x) or x
            # words = self.data["tweet_text"].str.split()
            # mask = words.apply(lambda x: any(spell.unknown(x)))
            # words_exploded = words[mask].explode()
            # words_exploded = words_exploded.apply(fix)
            # words[mask] = words_exploded.groupby(level=0).agg(list)
            # self.data["tweet_text"] = words.str.join(" ")
            # Preprocess text using NLTK: remove stopwords and lemmatize
            # self.data["tweet_text"] = self.data["tweet_text"].apply(
            #     lambda x: " ".join(
            #         lemmatizer.lemmatize(word) or word
            #         for word in word_tokenize(x)
            #         if word not in stop_words
            #     )
            # )
        except BaseException as e:
            self.logger.error(f"Error occurred while preprocessing data: {str(e)}")
            exit()

    def train_model(self) -> None:
        """
        Train the SVM model.
        """
        try:
            self.logger.info("Training model...")
            X = self.data["tweet_text"]
            y = self.data["class_label"].fillna("")
            X = self.vectorizer.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            self.svm.fit(X_train, y_train)
            score = self.svm.score(X_test, y_test)
            self.logger.info(f"Successfully trained model! Model score: {score}")
        except BaseException as e:
            self.logger.error(f"Error occurred while training model: {str(e)}")
            exit()

    def load_sentiment_words(self) -> None:
        """
        Load the positive and negative sentiment words.
        """
        try:
            self.logger.info("Loading sentiment words...")
            self.positive_words = self.sentiment_data["positive_words"].tolist()
            self.negative_words = self.sentiment_data["negative_words"].tolist()
        except BaseException as e:
            self.logger.error(f"Error occurred while loading sentiment words: {str(e)}")
            exit()

    def classify_tweets(
        self, subject: str, tweets: list[str]
    ) -> tuple[int, int, dict[str, str]]:
        """
        Classify the given tweets based on the trained model and sentiment words.

        Args:
            subject (str): The subject to analyze.
            tweets (list[str]): The list of tweets to classify.
        """
        try:
            self.logger.info(f"Classifying tweets for '{subject}'...")
            positive_tweets = 0
            negative_tweets = 0
            tweet_types = {}
            if len(tweets) == 0:
                self.logger.warning(f"No tweets found for '{subject}'")
                return positive_tweets, negative_tweets, tweet_types
            self.logger.debug(f"Received {len(tweets)} tweets")
            X_new = self.vectorizer.transform(tweets)
            predictions = self.svm.predict(X_new)
            self.logger.debug(f"Distribution of predictions: {Counter(predictions)}")
            for tweet, prediction in zip(tweets, predictions):
                tweet_words = tweet.split()
                # tweet_translated = []mas é isso que falei
                # if detect(tweet) not in ["en", "nl", "so", "hr"]:
                #     self.logger.warning(
                #         f"Tweet '{tweet}' not in english, trying to translate..."
                #     )
                #     try:
                #         translated_words = STranslator().translate_list(
                #             detect(tweet), "en", tweet_words
                #         )
                #         tweet_translated.append(translated_words)
                #         self.logger.warning(f"Sucessful!")
                #     except Exception:
                #         self.logger.error(f"Tweet {tweet} can't be translated.")
                #         tweet_translated.append(tweet_words)
                # else:
                #     tweet_translated.append(tweet)
                positive_count = sum(
                    word in self.positive_words for word in tweet_words
                )
                negative_count = sum(
                    word in self.negative_words for word in tweet_words
                )
                if prediction == "ironic":
                    positive_count, negative_count = negative_count, positive_count
                if positive_count > negative_count:
                    positive_tweets += 1
                else:
                    negative_tweets += 1
                tweet_types.setdefault(prediction, []).append(tweet)
            return (
                positive_tweets,
                negative_tweets,
                tweet_types,
            )
        except BaseException as e:
            self.logger.error(f"Error occurred while classifying tweets: {str(e)}")
            exit()
