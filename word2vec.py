# -*- coding:utf-8 -*-
import os
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import nltk.data
from bs4 import BeautifulSoup
from gensim.models.word2vec import Word2Vec

def load_dataset(name, nrows=None):
	datasets = {
		'unlabeled_train': 'unlabeledTrainData.tsv',
		'labeled_train': 'labeledTrainData.tsv',
		'test': 'testData.tsv'
	}
	if name not in datasets:
		raise ValueError(name)
	data_file = os.path.join('G:\\study\\NLP\\NLP\\课程\\word2vec\\课件资料\\第3课\\kaggle-word2vec-ipynb', 'data', datasets[name])
	df = pd.read_csv(data_file, sep='\t', escapechar='\\')
	print('number of reviews: {}'.format(len(df)))
	return df


df = load_dataset('unlabeled_train')
eng_stopwords = set(stopwords.words('english'))

def clean_text(text):
	text = BeautifulSoup(text, 'html.parser').get_text()
	text = re.sub(r'[^a-zA-Z]', ' ', text)
	words = text.lower().split()
	words = [w for w in words if w not in eng_stopwords]
	return ' '.join(words)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')#可以下载nltk_data包
def print_call_counts(f):
	n = 0
	def wrapped(*args, **kwargs):
		nonlocal n
		n += 1
		if n % 1000 == 1:
			print('method{} called {} times'.format(f.__name__, n))
		return wrapped
def split_sentences(review):
	raw_sentences = tokenizer.tokenize(review.strip())
	sentences = [clean_text(s) for s in raw_sentences if s]
	return sentences
sentences = sum(df.review.apply(split_sentences), [])
print(sentences)
print('{} reviews -> {} sentences'.format(len(df), len(sentences)))
num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)
model.init_sims(replace=True)
model_name = '{}features_{}minwords_{}context.model'.format(num_features, min_word_count, context)
model.save(os.path.join('G:\\project\\word2vec', 'models', model_name))
print(model.doesnt_match('man woman child kitchen'.split()))
print(model.doesnt_match('Watching Time Chasers'.split()))
print(model.most_similar('man'))




