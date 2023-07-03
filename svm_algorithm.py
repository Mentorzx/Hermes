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
from sklearn import metrics
import pandas as pd
import zipfile
import logging
import nltk
import re
import io


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
            with open("datasets/datasets.zip", "rb") as zip_file:
                conteudo_zip = zip_file.read()
            arquivo_zip = io.BytesIO(conteudo_zip)
            with zipfile.ZipFile(arquivo_zip, "r") as zip_ref:
                with zip_ref.open("speech_figures.csv") as file:
                    self.data = pd.read_csv(file, encoding="utf-8")
                with zip_ref.open("sentiment_words.csv") as file:
                    self.sentiment_data = pd.read_csv(file, encoding="utf-8")
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
            y_pred = self.svm.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred) * 100
            precision = (
                metrics.precision_score(y_test, y_pred, average="weighted") * 100
            )
            recall = metrics.recall_score(y_test, y_pred, average="weighted") * 100
            f1_score = metrics.f1_score(y_test, y_pred, average="weighted") * 100
            self.logger.info("Successfully trained model!")
            self.logger.info(f"Accuracy: {accuracy:.2f}%")
            self.logger.info(f"Precision: {precision:.2f}%")
            self.logger.info(f"Recall: {recall:.2f}%")
            self.logger.info(f"F1-Score: {f1_score:.2f}%")
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
                # tweet_translated = []
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
                elif prediction == "sarcasm":
                    negative_tweets += 1
                elif positive_count > negative_count:
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
