# Importing Libraries

import pandas as pd
import numpy as np
import itertools
import re
import tensorflow as tf

import nltk
# nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from google.cloud import bigquery
from sklearn import metrics




class TextClassification:
    def __init__(self,data,target_column):
        self.data = data
        self.target_column = target_column
    
    def display_column(self):
        print('*'*30+'COLUMN NAMES'+'*'*30+'\n\t\t' ,self.data.columns)
    def data_preprocessing(self,vAR_test_data):
        print('*'*30+'DATA PRE-PROCESSING'+'*'*30+'\n\t\t1.Remove Stop Words\n\t\t2.Stemming/Lemmatization')
        vAR_ps = PorterStemmer()
        vAR_corpus = []
        if vAR_test_data is None:
            data = self.data
        else:
            data = vAR_test_data
        for i in range(0, len(data)):
            vAR_review = re.sub('[^a-zA-Z]', ' ', data['comment_text'][i])
            vAR_review = vAR_review.lower()
            vAR_review = vAR_review.split()

            vAR_review = [vAR_ps.stem(word) for word in vAR_review if not word in stopwords.words('english')]
            vAR_review = ' '.join(vAR_review)
            vAR_corpus.append(vAR_review)
        return vAR_corpus
    def bagofwords_vectorization(self,vAR_corpus,vAR_test_data):
        vAR_cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
        vAR_X = vAR_cv.fit_transform(vAR_corpus).toarray()
        if vAR_test_data is None:
            vAR_y = self.data[self.target_column]
        else: 
            vAR_y = vAR_test_data[self.target_column]
        return vAR_X,vAR_y
    def tfidf_vectorization(self):
        pass
    def word_embedding_vectorization(self,vAR_corpus,vAR_test_data):
        vAR_voc_size=10000
        vAR_sent_length=8
        vAR_onehot_repr=[one_hot(words,vAR_voc_size)for words in vAR_corpus]
        vAR_embedded_docs=pad_sequences(vAR_onehot_repr,padding='pre',maxlen=vAR_sent_length)
        vAR_model=Sequential()
        vAR_model.add(Embedding(vAR_voc_size,10,input_length=vAR_sent_length))
        vAR_model.compile('adam','mse')
        vAR_X = vAR_model.predict(vAR_embedded_docs)
        if vAR_test_data is None:
            vAR_y = self.data[self.target_column]
        else: 
            vAR_y = vAR_test_data[self.target_column]
        return vAR_X,vAR_y
        
    def train_test_split(self,vAR_X,vAR_y):
        vAR_X_train, vAR_X_test, vAR_y_train, vAR_y_test = train_test_split(vAR_X, vAR_y, test_size=0.3, random_state=0)
        return vAR_X_train,vAR_y_train,vAR_X_test,vAR_y_test
    def test_model(self,vAR_model,vAR_X_test):
        vAR_prediction = vAR_model.predict(vAR_X_test)
        return vAR_prediction
    def accuracy_score(self,vAR_prediction,vAR_y_test):
        score = metrics.accuracy_score(vAR_y_test, vAR_prediction)
        return score


class ClassificationModels(TextClassification):
    def __init__(self,data,target_column):
        TextClassification.__init__(self,data,target_column)
    def train_model_naivebayes(self,vAR_X_train,vAR_y_train):
        vAR_model=MultinomialNB()
        vAR_model = MultiOutputClassifier(vAR_model)
        vAR_model.fit(vAR_X_train, vAR_y_train)
        return vAR_model
    def train_model_random_forest(self,vAR_X_train,vAR_y_train):
        vAR_model=RandomForestClassifier()
        # vAR_model = MultiOutputClassifier(vAR_model)
        vAR_model.fit(vAR_X_train, vAR_y_train)
        return vAR_model
    def train_model_lstm(self,vAR_X_train,vAR_y_train,vAR_X_test,vAR_y_test):
        # vAR_embedding_vector_features=40
        vAR_model=Sequential()
        # vAR_model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
        vAR_model.add(LSTM(100))
        vAR_model.add(Dense(units=20, activation="relu"))
        vAR_model.add(Dense(units=20, activation="relu"))
        vAR_model.add(Dense(6,activation='relu'))
        vAR_model.compile(loss='binary_crossentropy',optimizer='SGD',metrics=['accuracy'])
        vAR_model.fit(vAR_X_train,vAR_y_train,validation_data=(vAR_X_test,vAR_y_test),epochs=50,batch_size=32)
        vAR_model.save('/home/jupyter/LSTM_MODEL_50epoch/lstm_model.h5')
        return vAR_model


def read_bq_data():
    vAR_bqclient = bigquery.Client()
    vAR_query_string = """
        select * from `flydubai-338806.DSAI_DMV_DATASET.DSAI_DMV_TOXIC_COMMENTS`
        """
    vAR_dataframe = (
            vAR_bqclient.query(vAR_query_string)
            .result()
            .to_dataframe(
                create_bqstorage_client=True,
            )
        )
    return vAR_dataframe


if __name__ == "__main__":
    vAR_data = read_bq_data()
    vAR_data = read_bq_data().head(200)
    # vAR_data = pd.read_csv('/home/jupyter/DSAI_DMV_Text_Analyzer/DSAI_Dataset/train.csv')
    # vAR_data = read_bq_data().head(500)
    vAR_target_columns = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    vAR_model_obj = ClassificationModels(vAR_data,vAR_target_columns)
    vAR_model_obj.display_column()
    vAR_test_data = None
    vAR_corpus = vAR_model_obj.data_preprocessing(vAR_test_data)
    print('Data Preprocessing Completed')

# ############# *****Execute Below Code For NaiveBayes Classification***** ##############

#     vAR_X,vAR_y = vAR_model_obj.bagofwords_vectorization(vAR_corpus,vAR_test_data)
#     print('Vectorization Completed Using Bag of Words')
#     vAR_X_train,vAR_y_train,vAR_X_test,vAR_y_test = vAR_model_obj.train_test_split(vAR_X,vAR_y)
#     print('Train & Test Data Splitted Successfully')

#     vAR_model = vAR_model_obj.train_model_naivebayes(vAR_X_train,vAR_y_train)
#     print('Naive Bayes Model Trained successfully')
#     vAR_prediction = vAR_model_obj.test_model(vAR_model,vAR_X_test)
#     print('Naive Bayes Model Tested successfully')
#     accuracy = vAR_model_obj.accuracy_score(vAR_prediction,vAR_y_test)
#     print('Naive Bayes Model Accuracy - ',accuracy)

# ############# ******************************************************** ###############

# ############# *****Execute Below Code For Random Forest Classification***** ##############

    # vAR_X,vAR_y = vAR_model_obj.bagofwords_vectorization(vAR_corpus,vAR_test_data)
    # print('Vectorization Completed Using Bag of Words')
    # vAR_X_train,vAR_y_train,vAR_X_test,vAR_y_test = vAR_model_obj.train_test_split(vAR_X,vAR_y)
    # print('Train & Test Data Splitted Successfully')
    # vAR_model = vAR_model_obj.train_model_random_forest(vAR_X_train,vAR_y_train)
    # print('Random Forest Model Trained successfully')
    # vAR_prediction = vAR_model_obj.test_model(vAR_model,vAR_X_test)
    # print('Random Forest Model Tested successfully')
    # accuracy = vAR_model_obj.accuracy_score(vAR_prediction,vAR_y_test)
    # print('Random Forest Model Accuracy - ',accuracy)

# ############# ******************************************************** ###############

# ############# *****Execute Below Code For LSTM RNN Deep Learning Model***** ############## 

    vAR_X,vAR_y = vAR_model_obj.word_embedding_vectorization(vAR_corpus,vAR_test_data)
    print('Vectorization Completed Using Word Embedding')
    vAR_X_train,vAR_y_train,vAR_X_test,vAR_y_test = vAR_model_obj.train_test_split(vAR_X,vAR_y)
    print('Train & Test Data Splitted Successfully')
    
    vAR_model = vAR_model_obj.train_model_lstm(vAR_X_train,vAR_y_train,vAR_X_test,vAR_y_test)
    print('LSTM Model Trained successfully')
    vAR_prediction = vAR_model_obj.test_model(vAR_model,vAR_X_test)
    print('LSTM Model Tested successfully')
