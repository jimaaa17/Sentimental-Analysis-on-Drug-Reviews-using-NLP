from pyspark.sql import SparkSession
import pyspark.sql.functions as F

class DataLoader:
    """Load and preprocess drug review data using Spark."""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def _clean_columns(self, df):
        columnmap = {c: c.rstrip() for c in df.columns if c.endswith("\r")}
        for old, new in columnmap.items():
            df = df.withColumn(new, F.col(old)).drop(old)
        return df

    def load(self, train_path: str, test_path: str):
        df_train = self.spark.read.csv(train_path, inferSchema=True, header=True, quote='"', escape='\\', multiLine=True)
        df_test = self.spark.read.csv(test_path, inferSchema=True, header=True, quote='"', escape='\\', multiLine=True)
        df_train = self._clean_columns(df_train)
        df_test = self._clean_columns(df_test)
        df_train = df_train.withColumn("usefulCount", F.round(F.col("usefulCount")).cast('integer'))
        df = df_train.join(df_test, on=['uniqueID','drugName','condition','review','rating','date','usefulCount'], how='left_outer')
        df = df.withColumn("sentiment", F.when(F.col("rating") <= 5, 0).otherwise(1))
        df = df.withColumn('length', F.length('review'))
        return df
