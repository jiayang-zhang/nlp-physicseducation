'''
performs:
tf-ifd
naive_bayes
'''

import os
import pandas as pd
import numpy as np


# feature extraction imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#machine learning method imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# metrics imports
from sklearn import metrics
from sklearn.metrics import classification_report
# ================================================================================================
#path = '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/nlp-physicseducation/testfiles'
files = ['test01.txt', 'test02.txt']
path= '/Users/EfiaA/OneDrive - Imperial College London/Imperial academic work/University life/Y4\MSci project/Project_Coding/nlp-physicseducation/testfiles'
# ================================================================================================


# -- Get files ---
df_files = build_files_dataframe(dir_txtfldr, 'GS_', '.txt')

# -- Get labels ---
df_labels = build_labels_dataframe('data/labels.xlsx')

# -- Merge dataframes --
df = pd.merge(df_files, df_labels, left_on='StudentID', right_on='StudentID')      # merged dataframe: StudentID, Content, ArgumentLevel, ReasoningLevel


"""
Examples
"""
# To call 'ArgumentLevel' labels
ArgumentLevel_list = print(df['ArgumentLevel'].tolist())
# To see the corresponding document names
print(df['StudentID'].tolist())
# To call of string lists of reports for feature extraction
print(df['Content'].tolist())
# To see everything
print(df.drop(['Content'], axis = 1)) #  the 'Content' column is too overwhelming



# -- Feature extraction: TF-IDF ---
# word vectors
corpus_wordvec_names = None                    # word vector name list
corpus_wordvec_counts = None                   # word vector frequncy list    # len(corpus_wordvec_counts) = index(dataframe)



# -- Feature: TF-IDF ---
def tf_idf(dataframe):
    # performs tf_idf on the dataframe
    v    = TfidfVectorizer()
    x    = v.fit_transform(dataframe)
    #s_m  = x.toarray()
    return x

print(tf_idf(df['Content']))



# --- Classification: Naive Bayes classifier ----
X = dtm.toarray()
y = df['Reasoning level']
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf       = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)

print('MultinomialNB Accuracy:', metrics.accuracy_score(y_test, predicted))
print(classification_report(y_test, predicted))
