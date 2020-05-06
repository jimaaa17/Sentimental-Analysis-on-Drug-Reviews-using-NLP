# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Statistical Packages
import numpy as np 
import pandas as pd 

# Text Blob and nltk
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re

# Packages required for Neural Networks 
from keras.utils import to_categorical
import random
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense,Dropout,Embedding,LSTM
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Sequential
from tqdm import tqdm
import warnings
from keras.preprocessing.sequence import pad_sequences
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
lemmatizer = WordNetLemmatizer()

# Packages Required for Logistic Regression 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score , f1_score
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import string
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

#set random seed for the session and also for tensorflow that runs in background for keras
# set_random_seed(123)
# random.seed(123)


# %%
df_train = pd.read_csv("C:/MITA Spring 19/Turkoz/Capstone/Drug_Dataset/train_raw/train_csv.csv")
df_test = pd.read_csv("C:/MITA Spring 19/Turkoz/Capstone/Drug_Dataset/test_raw/test_raw.csv")


# %%
print("Shape of the training data- ",df_train.shape)
print("Shape of the testing data- ",df_test.shape)


# %%
##### Columns present in the testing dataset
df_train.columns


# %%

######### Displaying how the reviews look like
df_train['review'].head(5)


# %%
df_train['sentiment'] = df_train['rating'].apply(lambda x: 1 if x>5 else 0)
df_test['sentiment'] = df_test['rating'].apply(lambda x: 1 if x>5 else 0)


# %%
reviews = df_train['review']
sentiments = []
for review in tqdm(reviews):
    blob = TextBlob(review)
    sentiments += [blob.sentiment.polarity]


# %%
df_train["sentiment"] = sentiments
df_train.head(5)


# %%
row_indexes1=df_train[df_train['rating']>=5.0].index
df_train.loc[row_indexes1,'rating_new']="1"
row_indexes0=df_train[df_train['rating']<5.0].index
df_train.loc[row_indexes0,'rating_new']="0"


# %%
row_indexes1=df_test[df_test['rating']>=5.0].index
df_test.loc[row_indexes1,'rating_new']="1"
row_indexes0=df_test[df_test['rating']<5.0].index
df_test.loc[row_indexes0,'rating_new']="0"


# %%
df_train['sentiment'] = df_train['rating'].apply(lambda i:1 if i>=5 else 0)
df_test['sentiment'] = df_test['rating'].apply(lambda i:1 if i>=5 else 0)


# %%
df_train.head(5)


# %%
df_test.head(5)


# %%
from tqdm import tqdm

def clean_sentences(df):
    reviews = []

    for sent in tqdm(df['review']):
        
        #remove html content
        review_text = BeautifulSoup(sent).get_text()
            
        #remove non-alphabetic characters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
    
        #tokenize the sentences
        words = word_tokenize(review_text.lower())
    
        #lemmatize each word to its lemma
        lemma_words = [lemmatizer.lemmatize(i) for i in words]
    
        reviews.append(lemma_words)

    return(reviews)


# %%

df_train['tokens'] = clean_sentences(df_train)
df_test['tokens'] = clean_sentences(df_test)


# %%

from nltk.corpus import stopwords
stopword = stopwords.words('english')
negated_words = ['don',"don't",'ain','aren',"aren't",'couldn',"couldn't",
                 'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',"isn't",
                'against','no','not','no','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'shouldn',
                 "shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn',"wouldn't"]
stop = []
for w in stopword:
    if w not in negated_words:
        stop.append(w)


# %%
def clean(sen):
    return [word for word in sen if word not in stop]


# %%

df_train['cleaned']= df_train['tokens'].apply(clean)
df_test['cleaned'] = df_test['tokens'].apply(clean)


# %%

df_train.head(5)

# %% [markdown]
# ## LSTM With CNN

# %%
############### FOR TRAINING #################
all_training_words = [word for tokens in df_train["cleaned"] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in df_train["cleaned"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
print("Max sentence length is %s" % max(training_sentence_lengths))


# %%
############### FOR TESTING ##################
all_testing_words = [word for tokens in df_test["cleaned"] for word in tokens]
testing_sentence_lengths = [len(tokens) for tokens in df_test["cleaned"]]
TESTING_VOCAB = sorted(list(set(all_testing_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_testing_words), len(TESTING_VOCAB)))
print("Max sentence length is %s" % max(testing_sentence_lengths))


# %%
############### FOR TRAINING #################
cleaned_2 = []
for i in df_train["cleaned"]:
    text = " ".join(i)
    cleaned_2.append(text)
df_train["cleaned_new"] = cleaned_2


# %%

############## FOR TESTING ##################
cleaned_2_test = []
for i in df_test["cleaned"]:
    text = " ".join(i)
    cleaned_2_test.append(text)
df_test["cleaned_new"] = cleaned_2_test
print(len(cleaned_2_test))


# %%
df_test.head(5)


# %%
tokenizer_train = Tokenizer(num_words= len(TRAINING_VOCAB), lower=True, char_level=False)
tokenizer_train.fit_on_texts(df_train['cleaned_new'].tolist())

sequences_train = tokenizer_train.texts_to_sequences(df_train['cleaned_new'].tolist())
data_train = pad_sequences(sequences_train, maxlen=max(training_sentence_lengths))

tokenizer_test = Tokenizer(num_words= len(TESTING_VOCAB), lower=True, char_level=False)
tokenizer_test.fit_on_texts(df_test['cleaned_new'].tolist())

sequences_test = tokenizer_test.texts_to_sequences(df_test['cleaned_new'].tolist())
data_test = pad_sequences(sequences_test, maxlen=max(training_sentence_lengths))


# %%
train_word_index = tokenizer_train.word_index
test_word_index = tokenizer_test.word_index
print('Found %s unique tokens in the traing data set.' % len(train_word_index))
print('Found %s unique tokens in the testing data set.' % len(test_word_index))
total_words = len(train_word_index)+1


# %%
print("Shape of training data",data_train.shape)
print("Shape of testing data",data_test.shape)


# %%
# LSTM Model Architecture
model_lstm = Sequential()
model_lstm.add(Embedding(41914, 100, input_length=973))
model_lstm.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model_lstm.summary())


# %%
train_labels = df_train["sentiment"]


# %%
def create_conv_model():
    model_conv = Sequential()
    model_conv.add(Embedding(41914, 100, input_length=973))
    model_conv.add(Dropout(0.2))
    model_conv.add(Conv1D(64, 5, activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(LSTM(100))
    model_conv.add(Dense(1, activation='sigmoid'))
    model_conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_conv


# %%
model_conv = create_conv_model()


# %%
# Training the model 
#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)
model_conv = create_conv_model()
history1 = model_conv.fit(data_train, np.array(train_labels), batch_size=100, validation_split=0.3, epochs = 3,callbacks=[early_stopping_monitor],verbose=1)


# %%
plt.plot(history1.history['loss'],label='train')
plt.plot(history1.history['val_loss'],label='validation')
plt.legend()
plt.title('model loss')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.savefig('LSTM_model_loss.png')


# %%

plt.plot(history1.history['acc'],label='train')
plt.plot(history1.history['val_acc'],label='validation')
plt.title('model accuracy')
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('LSTM_model_accuracy.png')


# %%


