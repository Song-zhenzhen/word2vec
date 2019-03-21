# -*- coding:utf-8 -*-
import pandas as pd
import os
import re
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import nltk
from nltk.corpus import stopwords
datafile = 'G:\\study\\NLP\\NLP\\课程\\word2vec\\课件资料\\第3课\\kaggle-word2vec-ipynb\\data\\'
df = pd.read_csv(datafile+'labeledTrainData.tsv', sep='\t', escapechar='\\')
def display(text, title):
    print(title)
    print("\n--------split-------\n")
    print(text)
raw_example = df['review'][1]
#display(raw_example, 'original data')
example = BeautifulSoup(raw_example, 'html.parser').get_text() #去网页解析
example_letters = re.sub(r'[^a-zA-Z]', ' ', example) #去掉标点
words = example_letters.lower().split() #小写化,按空格解开
words_nostop = [w for w in words if w not in stopwords.words('english')] #去掉停用词
eng_stopwords = set(stopwords.words('english'))
#集成写成函数的形式
def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [w for w in words if w not in eng_stopwords]
    return ' '.join(words)
df['clean_review'] = df.review.apply(clean_text)
vectorizer = CountVectorizer(max_features=5000)
train_data_features = vectorizer.fit_transform(df.clean_review).toarray()
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, df.sentiment)
print(confusion_matrix(df.sentiment, forest.predict(train_data_features)))
test_data_features = vectorizer.transform(df.clean_review).toarray() #对测试集同样进行清洗
result = forest.predict(test_data_features) #输出预测结果
output = pd.DataFrame({'id':df.id, 'sentiment':result})











