from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd
import logging


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
        self.svm = SVC(kernel='linear')
        self.positive_words = []
        self.negative_words = []

    def load_data(self) -> None:
        """
        Load the dataset and sentiment words.
        """
        try:
            self.logger.info("Loading data...")
            self.data = pd.read_csv(
                'datasets/translated_speech_figures.csv', encoding='utf-8', delimiter=';')
            self.sentiment_data = pd.read_csv(
                'datasets/translated_sentiment_words.csv', encoding='utf-8')
        except Exception as e:
            self.logger.error(f"Error occurred while loading data: {str(e)}")
            raise

    def preprocess_data(self):
        """
        Preprocess the data if necessary.
        """
        pass

    def train_model(self) -> None:
        """
        Train the SVM model.
        """
        try:
            self.logger.info("Training model...")
            X = self.data['tweet_text']
            y = self.data['class_label']
            X = self.vectorizer.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            self.svm.fit(X_train, y_train)
            self.logger.info("Successfully trained model!")
        except Exception as e:
            self.logger.error(f"Error occurred while training model: {str(e)}")
            raise

    def load_sentiment_words(self) -> None:
        """
        Load the positive and negative sentiment words.
        """
        try:
            self.logger.info("Loading sentiment words...")
            self.positive_words = self.sentiment_data['positive_words'].tolist(
            )
            self.negative_words = self.sentiment_data['negative_words'].tolist(
            )
        except Exception as e:
            self.logger.error(
                f"Error occurred while loading sentiment words: {str(e)}")
            raise

    def classify_tweets(self, subject: str, tweets: list[str]) -> tuple[int, int]:
        """
        Classify the given tweets based on the trained model and sentiment words.

        Args:
            subject (str): The subject to analyze.
            tweets (list[str]): The list of tweets to classify.
        """
        try:
            self.logger.info(f"Classifying tweets for '{subject}'...")
            X_new = self.vectorizer.transform(tweets)
            predictions = self.svm.predict(X_new)
            positive_tweets = 0
            negative_tweets = 0
            for i in range(len(predictions)):
                if predictions[i] == 'positivo':
                    positive_tweets += 1
                elif predictions[i] == 'negativo':
                    negative_tweets += 1
                else:  # Handling for ironic tweets
                    tweet_words = tweets[i].lower().split()
                    positive_count = sum(
                        word in self.positive_words for word in tweet_words)
                    negative_count = sum(
                        word in self.negative_words for word in tweet_words)
                    if positive_count > negative_count:
                        negative_tweets += 1
                    else:
                        positive_tweets += 1
            return positive_tweets, negative_tweets
        except Exception as e:
            self.logger.error(
                f"Error occurred while classifying tweets: {str(e)}")
            raise
