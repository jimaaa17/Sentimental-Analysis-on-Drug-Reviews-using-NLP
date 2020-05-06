# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np 
import pandas as pd 
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
import random
from sklearn.metrics import classification_report
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_chisquare
from scipy.stats import chisquare
import pickle
from bs4 import BeautifulSoup
from collections import defaultdict
import requests
from statsmodels.stats.multitest import fdrcorrection_twostage
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
#Reading the csv file
df_train = pd.read_csv("C:/MITA Spring 19/Turkoz/Capstone/Drug_Dataset/train_raw/train_csv.csv")
df_test = pd.read_csv("C:/MITA Spring 19/Turkoz/Capstone/Drug_Dataset/test_raw/test_raw.csv")

#Shape of data
print("Shape of train data: ",df_train.shape)
print("Shape of test data: ",df_test.shape)
df_train.head()


# %%
#Computing the new column sentinment on basis of rating 
def sentiment(x):
    if x>5:
        return 1
    else:
        return 0
df_train['sentiment'] = df_train['rating'].apply(lambda x: sentiment(x))
df_test['sentiment'] = df_test['rating'].apply(lambda x: sentiment(x))

df_train['sentiment'].dtypes


# %%
#Checking the train dataframe
df_train.head()


# %%
#Checking the test dataframe
df_test.head()

# %% [markdown]
# # Exploratory Data Analysis

# %%
#Analyze the count of sentiment
df_train.groupby('sentiment').size()


# %%
#Visualizing sentiment for each features of Drug dataset
df_train.groupby('sentiment').count().plot.bar()


# %%
#converting the date feature to appropriate date time
df_train['date'] = pd.to_datetime(df_train['date'])
df_train['month'] = df_train['date'].apply(lambda x: x.month)
df_train.head()


# %%
#Visualize the rating feature monthly
fig, ax = plt.subplots(1,1, figsize=(5,5))

plt.style.use('seaborn')

# Score by Moth


ax.plot(df_train.groupby('month').rating.mean())
ax.set_ylabel('Average Rating')
ax.set_xlabel('Month')
ax.set_xticks(range(1,13))
ax.set_title('Average Rating vs. Month')



plt.show()


# %%
#Visualizing number of comments/review by year
df_train.groupby('date')['review'].size().plot(figsize=(12,4))
plt.ylabel('Number of comments')
plt.title("Number of comments posting against date",fontsize=15)


# %%
# Analyze the counts of comments against each condition of disease
df_condition_for_month = pd.DataFrame(df_train.groupby(['condition','month']).size()).reset_index()

df_comments = df_condition_for_month.pivot_table(index='condition',columns='month',values=0)

fig = plt.figure(figsize=(15,5))
df_comments.sum(1).sort_values(ascending=False).iloc[:50].plot(kind='bar')
plt.title("Number of comments against each conditions/diseases",fontsize=15)
plt.tight_layout()
fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
plt.xticks(fontsize=10)

# plt.savefig("Plot/num_comments_each_condition.png")


# %%
#Percentage of comments in each month for each disease
n = 50
top_n_index = df_comments.sum(1).sort_values(ascending=False).iloc[:n].index
top_comments = df_comments.loc[top_n_index]
top_comments_percentage = top_comments.div(top_comments.sum(1), axis=0)
top_comments_percentage.plot.bar(stacked=True, figsize=(15,7))
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title("Percentage of comments in each month")
plt.tight_layout()


# %%
# a pie chart to represent the sentiments of the patients

size = [161491, 53572]
colors = ['yellow', 'skyblue']
labels = "Positive Sentiment","Negative Sentiment"
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (10, 10)
plt.pie(size, colors = colors, labels = labels, explode = explode, autopct = '%.2f%%')
plt.axis('off')
plt.title('A Pie Chart Representing the Sentiments of Patients', fontsize = 30)
plt.legend()
plt.show()


# %%
## avg of comments for each condition in each month
df_comments_sorted = df_comments.loc[df_comments.sum(1).sort_values(ascending=False).index]
df_comments_sorted.sum() / len(df_comments_sorted)

# %% [markdown]
# # Chi-Squared Test

# %%
sum_months = df_comments_sorted.sum()
all_sum = sum_months.sum()

i = 0 
#df_con_mon_pivot_sorted.iloc[i,:]
expected = df_comments_sorted.sum(0) * (df_comments_sorted.iloc[i,:].sum() / all_sum)
observed = df_comments_sorted.iloc[i,:]
chisquare(observed,f_exp=expected)


# %%
sum_months = df_comments_sorted.sum()
all_sum = sum_months.sum()
chi_result =[]

## only take T > 5 (the cell need 5 records at least)
df_comments_sorted_T5 = df_comments_sorted[(df_comments_sorted > 5).all(1)]
df_comments_sorted_T5.head()


# %%
# compare expected and observed using chi-square test
for idx in df_comments_sorted_T5.index:
    ## use 
    expected = df_comments_sorted.sum(0) * (df_comments_sorted.loc[idx,:].sum() / all_sum)
    observed = df_comments_sorted_T5.loc[idx,:]
    chi_result.append([idx,chisquare(observed,f_exp=expected)])


# %%
df_chi = pd.DataFrame([[a[0],a[1].statistic,a[1].pvalue] for a in chi_result])
df_chi.columns = ['condition','statistic','pvalue']
df_chi['adj_pvalue'] = fdrcorrection_twostage(df_chi['pvalue'])[1]
df_chi.head()


# %%

df_chi_sorted = df_chi[df_chi['adj_pvalue'] < 0.05].sort_values('pvalue')

df_chi_sorted.head(10)


# %%
plt.figure(figsize=(12,12))
i =1
for idx in df_chi_sorted['condition'][:10]:
    
    each = df_comments_sorted_T5.loc[idx,:]/df_comments_sorted_T5.loc[idx,:].sum()
    
    plt.subplot(5, 2, i)
    plt.title(idx)
    plt.plot(each)
    plt.xticks(np.arange(1,13,1))
    plt.xlabel('months')
    plt.ylabel('% of comments')
    i = i +1
    
plt.tight_layout()


# %%
df_train.describe(include="all")


# %%


