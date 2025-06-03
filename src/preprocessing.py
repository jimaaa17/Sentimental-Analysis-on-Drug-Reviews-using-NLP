from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline


def build_preprocessing_pipeline():
    tokenizer = Tokenizer(inputCol="review", outputCol="token_text")
    stopremove = StopWordsRemover(inputCol="token_text", outputCol="stop_tokens")
    count_vec = CountVectorizer(inputCol="stop_tokens", outputCol="c_vec")
    idf = IDF(inputCol="c_vec", outputCol="tf_idf")
    label_indexer = StringIndexer(inputCol="sentiment", outputCol="label")
    assembler = VectorAssembler(inputCols=['tf_idf', 'length'], outputCol='features')
    return Pipeline(stages=[label_indexer, tokenizer, stopremove, count_vec, idf, assembler])
