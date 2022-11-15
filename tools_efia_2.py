# -*- coding: utf-8 -*-
#%%
import os
from bs4 import BeautifulSoup as bs
import numpy as np

import nltk

from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import pandas as pd
from collections import Counter 

#%%
dir1  = '/Users/EfiaA/OneDrive - Imperial College London/Imperial academic work/University life/Y4/MSci project/Project_Coding/anonymised_reports/anonymised_reports/year_1_2017/cycle_1/xml'
filename = 'GS_BKZ271_Redacted'


def openAllFiles(direc):
    out = []
    file_names = []
    count = 0
    for file in os.listdir(direc):
        file_names.append(file)
        with open(os.path.join(direc, file), 'r', encoding = 'utf-8') as f:
            contents = f.read()
            count +=1  # there are 86 files  (unit test checks)
            soup = bs(contents, 'xml')
            page = [head.get_text() for head in soup.find_all('p')]
            out.append(page[5:]) # manage this more 
    #text_array = np.concatenate(out, axis = 0)
    return  out

y =["BAL", "BAL", "THE", "THE", "THE",
"BAL", "BAL", "EXP", "THE", "THE", "BAL", "EXP",
"BAL", "BAL ", "BAL", "THE","EXP","THE","EXP",
"BAL", "BAL","THE", "BAL", "BAL", "BAL", "BAL", 
"BAL", "BAL","BAL", "BAL", "BAL", "NONE", "BAL",
"BAL","BAL", "BAL","THE", "BAL", "THE", "BAL", 
"BAL", "BAL", "BAL","THE", "BAL","THE", "THE",
"BAL"]

print(len(y))


#%%


def together(arr):
    text_array = np.concatenate(arr, axis = 0)
    return text_array

def tokeniser(arr):
    # does tokenisation and punctuation removal
    n_a = []
    for i in arr:
        #text = " ".join(str(i)
        text = ' '.join(str(x) for x in i)
        Tokenizer = RegexpTokenizer(r'\w+')
        new_text  = Tokenizer.tokenize(text)
        n_a.append(new_text)
    #new_text2 =  " ".join(str(x) for x in new_text)
    return n_a

def stem(arr):
    stem_sent = []
    porter = PorterStemmer()
    for array in arr:
        words = [porter.stem(i) for i in array]
        stem_sent.append(words)
    return stem_sent

def lemmatizer(arr):
    lem_sent = []
    wordnet_lemmatizer = WordNetLemmatizer()
    for word in arr:
        w = [wordnet_lemmatizer.lemmatize(i) for i in word]
        lem_sent.append(w)
    return lem_sent

def bow(arr):
    bow_per_doc = []
    for i in arr:
        x = Counter(i)
        results = x.items()
        data    = list(results)
        bow_per_doc.append(data)        
    return  bow_per_doc

def tfidf(arr):

    tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
    x = tfidf.fit(arr)
    tfidf_matrix = tfidf.transform(arr)
    dense = tfidf_matrix.toarray()
    
    
    '''
    Interpreting the output of the tf-idf
    you get a list of values each in the form of:
        (A, B) C
    A - Document index
    B - Specific word-vector index
    c - TFIDF score for word B in document A
    
    This is a sparse matrix. It indicates the tfidf score for all non-zero
    values in the word vector for each document. 
    '''
    return dense

def tfidf2(arr):
    tf_transformer = TfidfTransformer(use_idf=True).fit(arr)
    X_train_tf = tf_transformer.transform(arr)
    size = X_train_tf.shape
    return X_train_tf, size

def train_validate_train_data_split(x):
    #train, validation and test datasets
    t_data_60 = int(0.6*len(x))
    v_data_60_80 = int(0.8*len(x))
    train, valid, test = np.split(x,np.array([t_data_60, v_data_60_80], dtype = "object"))
    return train, valid, test

def pre_process_text(directory):
    array_of_text = openAllFiles(directory)
    tokenised_text = tokeniser(array_of_text)
    stemmed_text = stem(tokenised_text)
    lemmatised_text = lemmatizer(stemmed_text)
    necessary_files = lemmatised_text[22:70]
    return necessary_files

def dframe(arr1, arr2):
    df = pd.DataFrame({'1': arr1, '2': arr2})
    return df
#%%
#pre-processed text
pptxt = pre_process_text(dir1)


#%%
# feature extraction
tfidf_text = tfidf(pptxt)
print(tfidf_text)

#%%
#dataset = pd.DataFrame({'labels': y, 'Documents': tfidf_text})
#print(dataset)
#%%
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split( tfidf_text, y, test_size= 0.3, random_state=0)

clf       = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)

print('MultinomialNB Accuracy:', metrics.accuracy_score(y_test, predicted))
print(classification_report(y_test, predicted))



