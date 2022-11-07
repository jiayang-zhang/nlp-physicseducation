from tools import *
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# ================================================================================================
path = '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/anonymised_reports/year_1_2017/cycle_1/txt'
# ================================================================================================

'''
dataframe structure
index       File        Content         Words...
    0       GS_x.txt    xyz             (frequency)10
'''
# Variables
corpus_text = []
corpus_wordvec_names = None
corpus_wordvec_counts = None



# create file&token dataframe
df = pd.DataFrame(columns=['File','Content'])

counter = 0
for file in os.listdir(path):
    if file.startswith('GS_') and file.endswith('.txt'):
        counter += 1
        with open(os.path.join(path, file), 'r') as f:
            content = f.read()
        content = preprocess(content)
        # add to corpus_text
        corpus_text.append(content)
        # add new row to DataFrame
        df.loc[len(df)] = [file, content]
print('Total number of files:', counter)

# print(df)


# -- Feature: Bag of Words ---
countvec = CountVectorizer()
dtm = countvec.fit_transform( df['Content'].tolist() ) # document-term matrix

# word vectors
corpus_wordvec_names = countvec.get_feature_names()
corpus_wordvec_counts = dtm.toarray()
# print(corpus_wordvec_names)

# create BoW dataframe
BoW = pd.DataFrame(dtm.toarray(), columns= countvec.get_feature_names())
df = pd.merge(df, BoW, left_index=True, right_index=True)
# # print dataframe (drop content column)
# print(df)


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
