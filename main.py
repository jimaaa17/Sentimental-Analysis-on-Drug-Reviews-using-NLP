import argparse
from pyspark.sql import SparkSession

from src.dataloader import DataLoader
from src.preprocessing import build_preprocessing_pipeline
from src.models.logistic import LogisticClassifier
from src.models.random_forest import RandomForestModel


def run_pipeline(train_path, test_path, model_type='logistic'):
    spark = SparkSession.builder.appName('drug_dataset').getOrCreate()
    loader = DataLoader(spark)
    df = loader.load(train_path, test_path)
    pipeline = build_preprocessing_pipeline()
    clean_df = pipeline.fit(df).transform(df).select('label', 'features')
    train_df, test_df = clean_df.randomSplit([0.7, 0.3], seed=42)

    if model_type == 'random_forest':
        model = RandomForestModel()
    else:
        model = LogisticClassifier()

    model.fit(train_df)
    acc = model.evaluate(test_df)
    print(f"Test ROC-AUC ({model_type}): {acc}")
    spark.stop()


def parse_args():
    parser = argparse.ArgumentParser(description="Drug review sentiment analysis")
    parser.add_argument("--train", required=True, help="Path to training CSV")
    parser.add_argument("--test", required=True, help="Path to testing CSV")
    parser.add_argument("--model", choices=["logistic", "random_forest"], default="logistic", help="Model type")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.train, args.test, args.model)
