# -*- coding:utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
df = pd.read_csv('labeledTrainData.tsv', sep='\t', escapechar='\\', nrows=None)
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words('english'))

def clean_text(text, remove_stopwords=False):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in eng_stopwords]
    return words
model_name = '300features_40minwords_10context.model'
#导入已经训练好的模型
model = Word2Vec.load('models/300features_40minwords_10context.model')
print(df.head())
def to_review_vector(review):
    words = clean_text(review, remove_stopwords=True)
    array = np.array([model[w] for w in words if w in model])
    return pd.Series(array.mean(axis=0))
#对训练集做处理
train_data_features = df.review.apply(to_review_vector)
#print(train_data_features.head())
#使用随机森林作为分类器
forest = RandomForestClassifier(n_estimators = 100, random_state=42)
forest = forest.fit(train_data_features, df.sentiment)
confusion_matrix(df.sentiment, forest.predict(train_data_features))
#删除多余变量，节省内存
del df
del train_data_features
#导入测试集
df = pd.read_csv('testData.tsv', sep='\t', escapechar='\\', nrows=None)
#print(df.shape)
#对测试集进行处理
test_data_features = df.review.apply(to_review_vector)
#print(test_data_features.head())
#最后输出预测结果
result = forest.predict(test_data_features)
output = pd.DataFrame({'id': df.id, 'sentiment': result})
output.to_csv('Word2Vec_model.csv', index=False)
output.head()

