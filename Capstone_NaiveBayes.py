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
#Computing sentiment column based on rating
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
# ## Naive Bayes classification
# %% [markdown]
# ### Split data into training and test sets
# %% [markdown]
# ## Build cross-validation model
# ### Estimator

# %%
(training,testing) = clean_data.randomSplit([0.7,0.3])
from pyspark.ml.classification import NaiveBayes
naivebayes = NaiveBayes(featuresCol="features", labelCol="label")

# %% [markdown]
# ### Parameter grid

# %%
from pyspark.ml.tuning import ParamGridBuilder
param_grid = ParamGridBuilder().    addGrid(naivebayes.smoothing, [0, 1, 2, 4, 8]).    build()

# %% [markdown]
# ### Evaluator

# %%
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator()

# %% [markdown]
# ## Build cross-validation model

# %%
from pyspark.ml.tuning import CrossValidator
crossvalidator = CrossValidator(estimator=naivebayes, estimatorParamMaps=param_grid, evaluator=evaluator)

# %% [markdown]
# ## Fit cross-validation model

# %%
crossvalidation_mode = crossvalidator.fit(training)

# %% [markdown]
# ## Prediction on training and test sets

# %%
pred_train = crossvalidation_mode.transform(training)
pred_train.show(5)


# %%
pred_test = crossvalidation_mode.transform(testing)
pred_test.show(5)

# %% [markdown]
# ## Best model from cross validation

# %%
print("The parameter smoothing has best value:",
      crossvalidation_mode.bestModel._java_obj.getSmoothing())

# %% [markdown]
# ### Prediction accuracy on train data

# %%
print('training data (f1):', evaluator.setMetricName('f1').evaluate(pred_train), "\n",
     'training data (weightedPrecision): ', evaluator.setMetricName('weightedPrecision').evaluate(pred_train),"\n",
     'training data (weightedRecall): ', evaluator.setMetricName('weightedRecall').evaluate(pred_train),"\n",
     'training data (accuracy): ', evaluator.setMetricName('accuracy').evaluate(pred_train))

# %% [markdown]
# ### Prediction accuracy on test data

# %%
print('test data (f1):', evaluator.setMetricName('f1').evaluate(pred_test), "\n",
     'test data (weightedPrecision): ', evaluator.setMetricName('weightedPrecision').evaluate(pred_test),"\n",
     'test data (weightedRecall): ', evaluator.setMetricName('weightedRecall').evaluate(pred_test),"\n",
     'test data (accuracy): ', evaluator.setMetricName('accuracy').evaluate(pred_test))

# %% [markdown]
# ## Confusion matrix
# %% [markdown]
# ### Confusion matrix on training data

# %%
train_conf_mat = pred_train.select('label', 'prediction')
train_conf_mat.rdd.zipWithIndex().countByKey()

# %% [markdown]
# ### Confusion matrix on testing data

# %%
test_conf_mat = pred_test.select('label', 'prediction')
test_conf_mat.rdd.zipWithIndex().countByKey()


# %%


