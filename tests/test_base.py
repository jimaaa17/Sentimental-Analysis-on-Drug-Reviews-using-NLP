
import unittest
from ml_pipeline.base import SentimentDataLoader

class TestSentimentDataLoader(unittest.TestCase):
    def test_load(self):
        loader = SentimentDataLoader("data/train/train_raw.csv", "data/test/drugsComTest_raw.csv")
        df_train, df_test = loader.load()
        self.assertFalse(df_train.empty)
        self.assertFalse(df_test.empty)

if __name__ == "__main__":
    unittest.main()