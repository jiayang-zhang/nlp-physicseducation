#%%
import pandas as pd
import numpy as np

# feature extraction imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#machine learning method imports
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold

# metrics imports
from sklearn import metrics
from sklearn.metrics import classification_report
from tools import utils, ml_tools


# ================================================================================================
#path = '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/nlp-physicseducation/testfiles'
path= '/Users/EfiaA/OneDrive - Imperial College London/Imperial academic work/University life/Y4\MSci project/Project_Coding/nlp-physicseducation/testfiles'
dir_txtfldr = '/Users/EfiaA/OneDrive - Imperial College London/Imperial academic work/University life/Y4/MSci project/Project_Coding/anonymised_reports/anonymised_reports/year_1_2017/cycle_1/txt'
# ================================================================================================


# -- Get files ---
df_files = utils.build_files_dataframe(dir_txtfldr, 'GS_', '.txt')

# -- Get labels ---
df_labels = utils.build_labels_dataframe('data/labels.xlsx')

# -- Merge dataframes --
df = pd.merge(df_files, df_labels, left_on='StudentID', right_on='StudentID')      # merged dataframe: StudentID, Content, ArgumentLevel, ReasoningLevel

from sklearn.feature_extraction.text import TfidfVectorizer
def tf_idf(corpus):
    # performs tf_idf on the dataframe
    v    = TfidfVectorizer()
    x    = v.fit_transform(corpus)
    s_m  = x.toarray()
    return s_m
    
# -- Bag of Words ---
wordvec_names, wordvec_counts= ml_tools.BoW(df['Content'].tolist())
# -- Feature extraction: TF-IDF ---
X_t= tf_idf(df['Content'].tolist())

# --- Classification: Naive Bayes classifier ----

#%%
# --- classify using bow feature extraction ---
y = df['ReasoningLevel']
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(wordvec_counts, df['ReasoningLevel'].tolist(), train_size = 0.8)
clf_b       = MultinomialNB().fit(X_train_b, y_train_b)
predicted_b = clf_b.predict(X_test_b)


# --- classify using tf-idf feature extraction---
#%%
#from sklearn.model_selection import StratifiedKFold
kf = KFold(n_splits=5)
#skf = StratifiedKFold(n_splits=10) --> the results are worse for stratified sampling 



# -- NB, TF-IDF, KFOLD ----
array = []
for train_index, test_index in kf.split(X_t):
    #print('TRAIN:', train_index, 'TEST:', test_index)
    X_train_t, X_test_t = X_t[train_index], X_t[test_index]
    y_train_t, y_test_t = y[train_index], y[test_index]
    clf_t      = MultinomialNB().fit(X_train_t, y_train_t)
    predicted_t = clf_b.predict(X_test_t)
    #print(predicted_t)
    acc_score = metrics.accuracy_score(y_test_t, predicted_t)
    #print('MultinomialNB Accuracy, using TF-IDF feature extraction:', acc_score)
    array.append(acc_score)

average_accuracy = sum(array)/len(array)]
print('Array of accuracy values:', array)
print('Average accuracy:', average_accuracy)

'''
----- KEY INFORMATION! --------
For the train_test_split the documentation shows
 1. random_state = 42 | controls the shuffling applied in data before applying the split#
link: https://scikit-learn.org/stable/glossary.html#term-random_state
 2. shuffle = True | you can shuffle before splitting 
 3. stratify = 

More on the different functions to use to split shuffle and stratify data
 https://scikit-learn.org/stable/modules/cross_validation.html#stratification

 Stratified k fold
 stratified shuffle split

 best source, used it for tfidf kfold training
 https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right
'''
#%%
print('MultinomialNB Accuracy, using BOW feature extraction:', metrics.accuracy_score(y_test_b, predicted_b))
print('MultinomialNB Accuracy, using TF-IDF feature extraction:', metrics.accuracy_score(y_test_t, predicted_t))
#print(classification_report(y_test_, predicted))

