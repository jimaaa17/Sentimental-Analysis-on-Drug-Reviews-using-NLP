from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


class LogisticClassifier:
    """Wrapper around Spark MLlib LogisticRegression."""
    def __init__(self):
        self.classifier = LogisticRegression(featuresCol='features', labelCol='label')
        self.model = None

    def fit(self, training_df):
        self.model = self.classifier.fit(training_df)
        return self.model

    def evaluate(self, test_df):
        if self.model is None:
            raise ValueError('Model has not been trained')
        evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')
        return evaluator.evaluate(self.model.transform(test_df))
