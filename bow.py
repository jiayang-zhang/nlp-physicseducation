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



# create file&token dataframe
df = pd.DataFrame(columns=['File','Content'])
# read files
for file in files:
    with open(os.path.join(path, file), 'r') as f:
        content = f.read()
    # add new row to DataFrame
    df.loc[len(df)] = [file, content]


# -- Feature: Bag of Words ---
countvec = CountVectorizer()
dtm = countvec.fit_transform( df['Content'].tolist() ) # document-term matrix
# create BoW dataframe
BoW = pd.DataFrame(dtm.toarray(), columns= countvec.get_feature_names())
# join two dataframes
df = pd.merge(df, BoW, left_index=True, right_index=True)
# print dataframe (drop content column)
print(df.drop('Content',axis=1))

#%%
#%%
# -- Feature: TF-IDF ---
def tf_idf(dataframe):
    # performs tf_idf on the dataframe
    v    = TfidfVectorizer()
    x    = v.fit_transform(dataframe)
    #s_m  = x.toarray()
    return x

print(tf_idf(df['Content']))

#%%


# -- Classification: logistic regression ---
'''
X = dtm.toarray()
y = np.array([0,1]) #dummy_label
# Create training and test split
X_train, X_test, y_train, y_test = train_test_split(X, y)
# Create an instance of LogisticRegression classifier
#

lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
#
# Fit the model
#
lr.fit(X_train, y_train)
#
# Create the predictions
#
y_predict = lr.predict(X_test)

# Use metrics.accuracy_score to measure the score
print("LogisticRegression Accuracy %.3f" %metrics.accuracy_score(y_test, y_predict))
'''

# --- Classification: Naive Bayes classifier ----
X = dtm.toarray()
y = df['Reasoning level']
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf       = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)

print('MultinomialNB Accuracy:', metrics.accuracy_score(y_test, predicted))
print(classification_report(y_test, predicted))