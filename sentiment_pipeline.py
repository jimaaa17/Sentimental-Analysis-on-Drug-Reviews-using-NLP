class DataLoader:
    """Load and preprocess drug review data using Spark."""
    def __init__(self, spark_session):
        self.spark = spark_session

    def _clean_columns(self, df):
        import pyspark.sql.functions as F
        columnmap = {}
        for column in df.columns:
            if column.endswith("\r"):
                columnmap[column] = column.rstrip()
        for old in columnmap:
            df = df.withColumn(columnmap[old], F.col(old))
            df = df.drop(old)
        return df

    def load(self, train_path, test_path):
        df_train = self.spark.read.csv(train_path, inferSchema=True, header=True, quote='"', escape='\\', multiLine=True)
        df_test = self.spark.read.csv(test_path, inferSchema=True, header=True, quote='"', escape='\\', multiLine=True)
        df_train = self._clean_columns(df_train)
        df_test = self._clean_columns(df_test)
        from pyspark.sql.functions import round, when, col, length
        df_train = df_train.withColumn("usefulCount", round(df_train["usefulCount"]).cast('integer'))
        df = df_train.join(df_test, on=['uniqueID', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'], how='left_outer')
        sentiment = when(col("rating") <= 5, 0).otherwise(1)
        df = df.withColumn("sentiment", sentiment)
        df = df.withColumn('length', length(df['review']))
        self.data = df
        return df

from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

class SentimentModel:
    """Pipeline and model training for sentiment classification."""
    def __init__(self, spark_session):
        self.loader = DataLoader(spark_session)
        self.pipeline = None
        self.model = None
        self.training = None
        self.testing = None

    def prepare_data(self, train_path, test_path):
        df = self.loader.load(train_path, test_path)
        tokenizer = Tokenizer(inputCol="review", outputCol="token_text")
        stopremove = StopWordsRemover(inputCol='token_text', outputCol='stop_tokens')
        count_vec = CountVectorizer(inputCol='stop_tokens', outputCol='c_vec')
        idf = IDF(inputCol="c_vec", outputCol="tf_idf")
        label_indexer = StringIndexer(inputCol='sentiment', outputCol='label')
        clean_up = VectorAssembler(inputCols=['tf_idf', 'length'], outputCol='features')
        self.pipeline = Pipeline(stages=[label_indexer, tokenizer, stopremove, count_vec, idf, clean_up])
        cleaner = self.pipeline.fit(df)
        clean_data = cleaner.transform(df).select(['label', 'features'])
        self.training, self.testing = clean_data.randomSplit([0.7, 0.3])
        return self.training, self.testing

    def train(self):
        if self.training is None:
            raise ValueError("Data not prepared. Run prepare_data first.")
        lr = LogisticRegression(featuresCol='features', labelCol='label')
        self.model = lr.fit(self.training)
        return self.model

    def evaluate(self):
        if self.model is None or self.testing is None:
            raise ValueError("Model not trained or data not prepared.")
        evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
        predictions = self.model.transform(self.testing)
        return evaluator.setMetricName('areaUnderROC').evaluate(predictions)

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName('drug_dataset').getOrCreate()
    model = SentimentModel(spark)
    model.prepare_data(
        's3://capstone-drug-dataset/captsone-drug-dataset/train_raw.csv',
        's3://capstone-drug-dataset/captsone-drug-dataset/test_raw.csv'
    )
    model.train()
    acc = model.evaluate()
    print(f"Testing accuracy: {acc}")
