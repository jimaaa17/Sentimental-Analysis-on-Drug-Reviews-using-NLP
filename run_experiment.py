import argparse
import config
import numpy as np
from ml_pipeline.base import SentimentDataLoader, TextPreprocessor
from ml_pipeline.models import BaseSentimentModel
from ml_pipeline.utils import get_model
from ml_pipeline.utils import setup_logging

setup_logging() 
def main(args):
    # Load data
    loader = SentimentDataLoader(args.train, args.test)
    df_train, df_test = loader.load()

    # Preprocess text
    preprocessor = TextPreprocessor()
    X_train, X_test = preprocessor.fit_transform(df_train['review'], df_test['review'])
    y_train = df_train['sentiment'].values
    y_test = df_test['sentiment'].values

    # Get model
    model = get_model(args.model)
    sentiment_model = BaseSentimentModel(model)

    # Train and evaluate
    if args.model == "gbt":
        # Compute sample weights inversely proportional to class frequency
        class_counts = np.bincount(y_train)
        class_weights = {i: sum(class_counts) / (2 * c) for i, c in enumerate(class_counts)}
        sample_weight = np.array([class_weights[y] for y in y_train])
        sentiment_model.model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        sentiment_model.train(X_train, y_train)
    sentiment_model.evaluate(X_test, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=config.DATA_TRAIN_PATH)
    parser.add_argument('--test', type=str, default=config.DATA_TEST_PATH)
    parser.add_argument('--model', type=str, choices=['gbt', 'logistic', 'naive_bayes', 'random_forest', 'svm'], default='gbt')
    parser.add_argument('--max_features', type=int, default=config.TFIDF_MAX_FEATURES)
    parser.add_argument('--save_model', action='store_true')
    args = parser.parse_args()
    main(args)