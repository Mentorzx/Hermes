from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from translate_dataset import STranslator
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from langdetect import detect
from tqdm import tqdm
import numpy as np
import pandas as pd
import zipfile
import logging
import nltk
import re
import io
import os


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
        self.models = []
        self.vectorizers = []
        self.positive_words = []
        self.negative_words = []

    def load_data(self, file_names: list[str], is_zip: bool = False) -> None:
        """
        Load data from a zip file or a list of paths.

        Args:
            file_names (list[str]): List of file names or paths to load the data from.
            is_zip (bool, optional): Indicates if the data is in a zip file or not. Defaults to False.

        Raises:
            BaseException: If an error occurs while loading the data.

        Notes:
            If is_zip is True, the first element of file_names should be the path to the zip file.
            The data is expected to be in CSV format with UTF-8 encoding.
        """
        try:
            self.logger.info("Loading datas...")
            self.logger.debug(
                f"Received parameters: file_names={file_names}, is_zip={is_zip}"
            )
            self.data = {}
            if is_zip:
                zip_path = file_names[0]
                with open(zip_path, "rb") as zip_file:
                    conteudo_zip = zip_file.read()
                arquivo_zip = io.BytesIO(conteudo_zip)
                with zipfile.ZipFile(arquivo_zip, "r") as zip_ref:
                    for file_name in file_names[1:]:
                        with zip_ref.open(file_name) as file:
                            self.data[file_name] = pd.read_csv(file, encoding="utf-8")
            else:
                for file_path in file_names:
                    file_name = os.path.basename(file_path)
                    self.data[file_name] = pd.read_csv(file_path, encoding="utf-8")
            self.logger.info(f"Successfully loaded {len(self.data)} files.")
            for file_name, df in self.data.items():
                self.logger.info(
                    f"Dataframe {file_name} has {df.shape[0]} rows and {df.shape[1]} columns."
                )
                self.logger.debug(
                    f"Dataframe {file_name} has columns: {df.columns.tolist()}"
                )
            self.logger.info("Finished load_data function.")
        except BaseException as e:
            self.logger.error(f"Error occurred while loading data: {str(e)}")
            exit()

    def preprocess_data(self, clean=True, spell_check=True, lemmatize=True) -> None:
        """
        Preprocesses the data by performing several steps including cleaning, spell-checking, and lemmatization.

        Args:
            clean (bool): If True, removes special characters, numbers, URLs, hashtags, and usernames. Defaults to True.
            spell_check (bool): If True, fixes spelling errors using a spell checker. Defaults to True.
            lemmatize (bool): If True, lemmatizes words using the WordNet lemmatizer and removes stopwords. Defaults to True.

        Raises:
            BaseException: If an error occurs during the preprocessing steps.

        Notes:
            - This function assumes that the data is stored in a dictionary of pandas DataFrames, accessible via the attribute 'data'.
            - The preprocessing steps include:
                1. Removing special characters, numbers, URLs, hashtags, and usernames.
                2. Downloading necessary NLTK resources (stopwords, punkt, and wordnet).
                3. Removing rows where the second column has the value 'figurative'.
                4. Removing rows with empty values in the first and second columns.
                5. Converting the columns to lowercase.
                6. Fixing spelling errors using a spell checker.
                7. Lemmatizing words using the WordNet lemmatizer and removing stopwords.

            Please note that the original data DataFrames are modified in-place during the preprocessing steps.
        """
        try:
            self.logger.info("Preprocessing data...")
            pattern = r"[^\w\s]|(\d+)|http\S+|#[^\s]+|@\S+|[^\x00-\x7F]+"
            self.logger.debug("Downloading nltk resources...")
            try:
                nltk.download("stopwords")
                nltk.download("punkt")
                nltk.download("wordnet")
            except BaseException as e:
                self.logger.error(
                    f"Error occurred while downloading nltk resources: {str(e)}"
                )
                exit()
            self.logger.debug("Download completed.")
            stop_words = set(stopwords.words("english"))
            lemmatizer = WordNetLemmatizer()
            spell = SpellChecker()
            for key, df in self.data.items():
                self.logger.debug(f"Processing dataframe {df}...")
                df = df.drop(df[df.iloc[:, 1] == "figurative"].index)
                df.iloc[:, 0] = df.iloc[:, 0].replace("", np.nan)
                df.iloc[:, 1] = df.iloc[:, 1].replace("", np.nan)
                df = df.dropna(subset=[df.columns[0], df.columns[1]], how="any")
                if clean:
                    df.iloc[:, 0] = df.iloc[:, 0].apply(
                        lambda x: re.sub(pattern, "", x)
                    )
                    df.iloc[:, 0] = df.iloc[:, 0].str.lower()
                    df.iloc[:, 1] = df.iloc[:, 1].str.lower()
                if spell_check:
                    tqdm.pandas(desc="Fixing spelling errors")
                    fix = lambda x: spell.correction(x) or x
                    words = df.iloc[:, 0].str.split()
                    mask = words.apply(lambda x: any(spell.unknown(x)))
                    words_exploded = words[mask].explode()
                    words_exploded = words_exploded.progress_apply(fix)
                    words[mask] = words_exploded.groupby(level=0).agg(list)
                    df.iloc[:, 0] = words.str.join(" ")
                if lemmatize:
                    tqdm.pandas(desc="Lemmatizing words")
                    df.iloc[:, 0] = df.iloc[:, 0].progress_apply(
                        lambda x: " ".join(
                            lemmatizer.lemmatize(word) or word
                            for word in word_tokenize(x)
                            if word not in stop_words
                        )
                    )
                self.data.update({key: df})
        except BaseException as e:
            self.logger.error(f"Error occurred while preprocessing data: {str(e)}")
            exit()

    def split_data(
        self, dataset_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the data into train and test sets.

        Args:
            dataset_index (int): The index of the dataset to use in the self.data dictionary.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The train and test sets for X and y.
        """
        df = self.data[dataset_index]
        X = df.iloc[:, 0]
        y = df.iloc[:, 1]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(X)
        self.vectorizers.append(vectorizer)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def train_model(self) -> None:
        """
        Train an SVM model for each dataset in the self.data dictionary and store them in self.models.

        Raises:
            BaseException: If an error occurs during the model training.
        """
        try:
            self.logger.info("Training models...")
            for dataset_index in self.data.keys():
                svm = SVC(verbose=True)
                X_train, _, y_train, _ = self.split_data(dataset_index)
                svm.fit(X_train, y_train)
                self.models.append(svm)
                self.logger.info(
                    f"Successfully trained model for dataset {dataset_index}!"
                )
        except BaseException as e:
            self.logger.error(f"Error occurred while training models: {str(e)}")
            exit()

    def evaluate_model(self) -> list[dict[str, int]]:
        """
        Evaluate the performance of each trained model on a test set.

        Returns:
            list[dict[str, int]]: A list of dictionaries containing evaluation metrics for each model.

        Raises:
            BaseException: If an error occurs during the model evaluation.
        """
        try:
            self.logger.info("Evaluating models...")
            metrics_list = []
            for dataset_index, model in zip(self.data.keys(), self.models):
                metrics_dict = {}
                _, X_test, _, y_test = self.split_data(dataset_index)
                y_pred = model.predict(X_test)
                metrics_dict["accuracy"] = metrics.accuracy_score(y_test, y_pred) * 100
                metrics_dict["precision"] = (
                    metrics.precision_score(y_test, y_pred, average="weighted") * 100
                )
                metrics_dict["recall"] = (
                    metrics.recall_score(y_test, y_pred, average="weighted") * 100
                )
                metrics_dict["f1_score"] = (
                    metrics.f1_score(y_test, y_pred, average="weighted") * 100
                )
                confusion = metrics.confusion_matrix(y_test, y_pred)
                self.logger.info(
                    f"Successfully evaluated model for dataset {dataset_index}!"
                )
                self.logger.info(f"Accuracy: {metrics_dict['accuracy']:.2f}%")
                self.logger.info(f"Precision: {metrics_dict['precision']:.2f}%")
                self.logger.info(f"Recall: {metrics_dict['recall']:.2f}%")
                self.logger.info(f"F1-Score: {metrics_dict['f1_score']:.2f}%")
                self.logger.info("Confusion Matrix:")
                self.logger.info(confusion)
                metrics_list.append(metrics_dict)
            return metrics_list
        except BaseException as e:
            self.logger.error(f"Error occurred while evaluating models: {str(e)}")
            exit()

    def classify(self, subject: str, phrases: list[str]) -> list[dict[str, list[str]]]:
        """
        Classify the given phrases based on the trained models in self.model.

        Args:
            subject (str): The subject to analyze.
            phrases (list[str]): The list of phrases to classify.

        Returns:
            list[dict[str, list[str]]]: A list of dictionaries containing phrase types categorized by each model.

        Raises:
            BaseException: If an error occurs during phrase classification.
        """
        try:
            phrases_types_list = []
            if len(phrases) == 0:
                return phrases_types_list
            self.logger.info(f"Classifying phrases for '{subject}'...")
            self.logger.debug(f"Received {len(phrases)} phrases")
            for index, model in enumerate(self.models):
                vectorizer = self.vectorizers[index]
                X_new = vectorizer.transform(phrases)
                phrases_types = {}
                predictions = model.predict(X_new)
                self.logger.debug(
                    f"Distribution of predictions: {Counter(predictions)}"
                )
                for phrase, prediction in zip(phrases, predictions):
                    phrases_types.setdefault(prediction, []).append(phrase)
                phrases_types_list.append(phrases_types)
            return phrases_types_list
        except BaseException as e:
            self.logger.error(f"Error occurred while classifying phrases: {str(e)}")
            exit()
