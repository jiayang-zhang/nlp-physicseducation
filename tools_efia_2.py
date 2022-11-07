# -*- coding: utf-8 -*-
#%%
import os
from bs4 import BeautifulSoup as bs
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
            out.append(page[8:]) # manage this more 
    text_array = np.concatenate(out, axis = 0)
    return  out

y = ["BAL", "BAL", "THE", "THE", "THE", "BAL", "BAL", "EXP", "THE", "THE", "BAL", "EXP",
"BAL", "BAL ", "BAL", "THE","EXP","THE","EXP","BAL", "BAL","THE", "BAL", "BAL", "BAL", "BAL", 
"BAL", "BAL","BAL", "BAL", "BAL", "NONE", "BAL"
"BAL","BAL", "BAL","THE", "BAL", "THE", "BAL", 
"BAL", "BAL", "BAL","THE", "BAL","THE", "THE", "BAL", "THE"]

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
    tfidf_matrix = tfidf.fit_transform(arr)
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


def train_validate_train_data_split(x):
    #train, validation and test datasets
    t_data_60 = int(0.6*len(x))
    v_data_60_80 = int(0.8*len(x))
    train, valid, test = np.split(x, [t_data_60, v_data_60_80])
    return train, valid, test

def pre_process_text(directory):
    array_of_text = openAllFiles(directory)
    tokenised_text = tokeniser(array_of_text)
    stemmed_text = stem(tokenised_text)
    lemmatised_text = lemmatizer(stemmed_text)
    necessary_files = lemmatised_text[22:69]
    return necessary_files


#%%
#pre-processed text
pptxt = pre_process_text(dir1)
#%%
x_train_data = train_validate_train_data_split(pptxt)[0]
x_valid_data = train_validate_train_data_split(pptxt)[1]
x_test_data  = train_validate_train_data_split(pptxt)[2]

#%%
#y data - labels
y_train_data = train_validate_train_data_split(y)[0]
y_valid_data = train_validate_train_data_split(y)[1]
y_test_data  = train_validate_train_data_split(y)[2]
print(len(y_train_data))
#%%
# feature extraction
tfidf_text_train = tfidf(x_train_data)
tfidf_text_test = tfidf(x_test_data)
bow_text_train   = bow(x_train_data)
bow_text_test   = bow(x_test_data)

print(len(tfidf_text_train))
print(len(tfidf_text_test))

#%%
'''
FOOD FOR THOUGHT:
    - import csv file of the high level rating s
    - Uses the pandas data frame to associate the high level reasoning to the individual 
    array elements
'''

#%%
# we want to choose features 
#import Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

#classification report offered by sklear
from sklearn.metrics  import classification_report

'''
Naive Bayes Classifier
'''
nb_model = MultinomialNB()
nb_model.fit(tfidf_text_train, y_train_data)
y_pred = nb_model.predict(tfidf_text_test)

#classification report 
print(classification_report(y_test_data, y_pred))
#%%
x = [1,22]
print(type(x))


    