import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report


class SvmThinker:
    """
    A class that uses Support Vector Machines (SVM) for tweet classification.
    """

    def __init__(self):
        """
        Initialize the SvmThinker object.
        """
        self.vectorizer = TfidfVectorizer()
        self.svm = SVC(kernel='linear')
        self.positive_words = []
        self.negative_words = []

    def load_data(self):
        """
        Load the dataset and sentiment words.
        """
        self.data = pd.read_csv('datasets/speech_figures.csv')
        self.sentiment_data = pd.read_csv('datasets/sentiment_words.csv')

    def preprocess_data(self):
        """
        Preprocess the data if necessary.
        """
        pass

    def train_model(self):
        """
        Train the SVM model.
        """
        X = self.data['tweet_text']
        y = self.data['irony_label']
        X = self.vectorizer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        self.svm.fit(X_train, y_train)

    def load_sentiment_words(self):
        """
        Load the positive and negative sentiment words.
        """
        self.positive_words = self.sentiment_data['positive_words'].tolist()
        self.negative_words = self.sentiment_data['negative_words'].tolist()

    def classify_tweets(self, subject, tweets):
        """
        Classify the given tweets based on the trained model and sentiment words.

        Args:
            subject (str): The subject to analyze.
            tweets (list[str]): The list of tweets to classify.
        """
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

        print(f"Total positive tweets about '{subject}': {positive_tweets}")
        print(f"Total negative tweets about '{subject}': {negative_tweets}")
