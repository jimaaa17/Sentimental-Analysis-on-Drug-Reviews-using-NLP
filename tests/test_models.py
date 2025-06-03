import pytest
pyspark = pytest.importorskip('pyspark')
from pyspark.sql import SparkSession

from src.dataloader import DataLoader
from src.preprocessing import build_preprocessing_pipeline
from src.models.logistic import LogisticClassifier


def test_logistic_training(tmp_path):
    spark = SparkSession.builder.master('local[1]').appName('test').getOrCreate()
    train = str(tmp_path/'train.csv')
    test = str(tmp_path/'test.csv')
    data = 'uniqueID,drugName,condition,review,rating,date,usefulCount\n1,A,C,"good",8,2020-01-01,1\n2,B,D,"bad",3,2020-01-02,1\n'
    with open(train, 'w') as f:
        f.write(data)
    with open(test, 'w') as f:
        f.write(data)
    loader = DataLoader(spark)
    df = loader.load(train, test)
    pipeline = build_preprocessing_pipeline()
    clean = pipeline.fit(df).transform(df).select('label','features')
    train_df, test_df = clean.randomSplit([0.7,0.3], seed=42)
    model = LogisticClassifier()
    model.fit(train_df)
    auc = model.evaluate(test_df)
    assert isinstance(auc, float)
    spark.stop()
