# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#Importing pyspark session
import pyspark


# %%
#Importing pyspark package
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col,trim,round, length
from pyspark import SparkContext

import pyspark.sql.functions as F
from pyspark.sql.types import *
spark=SparkSession.builder.appName('drug_dataset').getOrCreate()

# %% [markdown]
# ## Clean and Prepare the Data

# %%
#Importing train data from S3
df_train = spark.read.csv('s3://capstone-drug-dataset/captsone-drug-dataset/train_raw.csv',inferSchema=True, header=True,quote='"',escape= "\"",multiLine=True)
columnmap = {}
for column in df_train.columns:
  if column.endswith("\r"):
    columnmap[column] = column.rstrip()
for c in columnmap.keys():
  df_train = df_train.withColumn(columnmap[c], F.col(c))
  df_train = df_train.drop(c)


# %%
#Importing test data from S3
df_test = spark.read.csv('s3://capstone-drug-dataset/captsone-drug-dataset/test_raw.csv',inferSchema=True, header=True,quote='"',escape= "\"",multiLine=True)
for column in df_test.columns:
  if column.endswith("\r"):
    columnmap[column] = column.rstrip()
for c in columnmap.keys():
  df_test = df_test.withColumn(columnmap[c], F.col(c))
  df_test = df_test.drop(c)


# %%
#Test data samples
df_test.show(10)
df_train.printSchema
df_train = df_train.withColumn("usefulCount",round(df_train["usefulCount"]).cast('integer'))


# %%
#Joining train and test data set
df = df_train.join(df_test, on=['uniqueID', 'drugName', 'condition','review','rating','date','usefulCount'], how='left_outer')


# %%
#Computing setniment column based on rating
sentiment = when(col("rating")<=5, 0).otherwise(1)

df = df.withColumn("sentiment",sentiment)
df = df.withColumn('length',length(df['review']))

# %% [markdown]
# ## Feature Transformation

# %%
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer

tokenizer = Tokenizer(inputCol="review", outputCol="token_text")
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
idf = IDF(inputCol="c_vec", outputCol="tf_idf")
pos_neg = StringIndexer(inputCol='sentiment',outputCol='label')


# %%
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector


# %%
clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')


# %%
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import OneVsRest

# %% [markdown]
# ## Building pipeline and fit model

# %%
from pyspark.ml import Pipeline
data_prep_pipe = Pipeline(stages=[pos_neg,tokenizer,stopremove,count_vec,idf,clean_up])
cleaner = data_prep_pipe.fit(df)
cleaner


# %%
clean_data = cleaner.transform(df)
clean_data = clean_data.select(['label','features'])
clean_data.show()

# %% [markdown]
# ## Logistic Regression Model Estimator and Training the data

# %%
(training,testing) = clean_data.randomSplit([0.7,0.3])
logr = LogisticRegression(featuresCol='features', labelCol='label')
logr_model = logr.fit(training)

# %% [markdown]
# ## Prediction on training data

# %%
pred_training_logr = logr_model.transform(training)
show_columns = ['features', 'label', 'prediction', 'rawPrediction', 'probability']
pred_training_logr.select(show_columns).show(5, truncate=True)

# %% [markdown]
# ## Evaluator

# %%
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
print('Accuracy on training data (areaUnderROC): ', evaluator.setMetricName('areaUnderROC').evaluate(pred_training_logr))

# %% [markdown]
# ## Prediction on test data

# %%
pred_testing_logr = logr_model.transform(testing)
pred_testing_logr.select(show_columns).show(5, truncate=True)


# %%
print('Accuracy on testing data (areaUnderROC): ', evaluator.setMetricName('areaUnderROC').evaluate(pred_testing_logr))

# %% [markdown]
# ## Confusion Matrix

# %%
label_pred_train = pred_training_logr.select('label', 'prediction')
label_pred_train.rdd.zipWithIndex().countByKey()


label_pred_test = pred_testing_logr.select('label', 'prediction')
label_pred_test.rdd.zipWithIndex().countByKey()

# %% [markdown]
# ## Accuracy of the model

# %%
from pyspark.ml.evaluation import BinaryClassificationEvaluator
acc_eval = BinaryClassificationEvaluator()
acc = acc_eval.evaluate(pred_testing_logr)
print("Accuracy of model at predicting sentiment was: {}".format(acc))


# %%


