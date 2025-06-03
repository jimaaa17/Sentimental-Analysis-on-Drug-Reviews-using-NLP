import pytest
pyspark = pytest.importorskip('pyspark')
from pyspark.sql import SparkSession

from src.dataloader import DataLoader


def test_load(tmp_path):
    spark = SparkSession.builder.master('local[1]').appName('test').getOrCreate()
    train_path = str(tmp_path/'train.csv')
    test_path = str(tmp_path/'test.csv')
    with open(train_path, 'w') as f:
        f.write('uniqueID,drugName,condition,review,rating,date,usefulCount\n1,A,C,"good",8,2020-01-01,1\n')
    with open(test_path, 'w') as f:
        f.write('uniqueID,drugName,condition,review,rating,date,usefulCount\n1,A,C,"good",8,2020-01-01,1\n')
    loader = DataLoader(spark)
    df = loader.load(train_path, test_path)
    assert df.count() == 1
    assert 'sentiment' in df.columns
    spark.stop()
