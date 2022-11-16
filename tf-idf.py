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



# -- Feature extraction: TF-IDF ---
def tf_idf(corpus):
    # performs tf_idf on the dataframe
    v    = TfidfVectorizer()
    x    = v.fit_transform(corpus)
    #s_m  = x.toarray()
    return x


# --- Classification: Naive Bayes classifier ----
X = tf_idf(df['Content'].tolist())
y = df['ReasoningLevel']
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf       = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)

print('MultinomialNB Accuracy:', metrics.accuracy_score(y_test, predicted))
print(classification_report(y_test, predicted))
