'''
performs:
bag of words
logistic regression
'''

from tools.xlsxer import *
from tools.xmler import *
import os
import pandas as pd
pd.set_option('max_colu', 10)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# ================================================================================================
dir_txtfldr = '/Users/jiayangzhang/Library/CloudStorage/OneDrive-ImperialCollegeLondon/year4/anonymised_reports/year_1_2017/cycle_1/txt'
# ================================================================================================

# -- Get files ---
df_files = build_files_dataframe(dir_txtfldr, 'GS_', '.txt')

# -- Get labels ---
df_labels = build_labels_dataframe('data/labels.xlsx')

# -- Merge dataframes --
df = pd.merge(df_files, df_labels, left_on='StudentID', right_on='StudentID')      # merged dataframe: StudentID, Content, ArgumentLevel, ReasoningLevel
print(df.drop(['Content'], axis = 1))
df.to_csv('data/labels_cleaned.csv')    # save to a csv file



# -- Feature extraction: Bag of Words ---
countvec = CountVectorizer()
dtm = countvec.fit_transform( df['Content'].tolist() )  # document-term matrix

# word vectors
corpus_wordvec_names = countvec.get_feature_names()     # word vector name list
corpus_wordvec_counts = dtm.toarray()                   # word vector frequncy list    # len(corpus_wordvec_counts) = index(dataframe)



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
