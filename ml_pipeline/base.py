import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

class SentimentDataLoader:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def load(self):
        try:
            logging.info("Loading training data from %s", self.train_path)
            df_train = pd.read_csv(self.train_path)
            logging.info("Loading test data from %s", self.test_path)
            df_test = pd.read_csv(self.test_path)
            for df in [df_train, df_test]:
                df['sentiment'] = (df['rating'] > 5).astype(int)
                df['review'] = df['review'].fillna('')
            logging.info("Data loaded successfully.")
            return df_train, df_test
        except Exception as e:
            logging.error("Error loading data: %s", e, exc_info=True)
            raise

class TextPreprocessor:
    def __init__(self, max_features=10000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')

    def fit_transform(self, train_texts, test_texts):
        try:
            logging.info("Starting TF-IDF vectorization...")
            all_texts = pd.concat([train_texts, test_texts])
            self.vectorizer.fit(all_texts)
            X_train = self.vectorizer.transform(train_texts)
            X_test = self.vectorizer.transform(test_texts)
            logging.info("TF-IDF vectorization complete.")
            return X_train, X_test
        except Exception as e:
            logging.error("Error during TF-IDF vectorization: %s", e, exc_info=True)
            raise