#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# feature extraction imports
from sklearn.feature_extraction.text import TfidfVectorizer

#machine learning method imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tools import utils, ml_tools, formats
import time

#%%

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

#%%
def tf_idf(corpus):
    # performs tf_idf on the dataframe
    v    = TfidfVectorizer()
    x    = v.fit_transform(corpus)
    s_m  = x.toarray()
    return s_m

    
# -- Bag of Words ---
wordvec_names, wordvec_counts= ml_tools.BoW(df['Content'].tolist())
y_b = df['ReasoningLevel'].tolist()


# -- Feature extraction: TF-IDF ---
X_t = tf_idf(df['Content'].tolist())
y_t = df['ReasoningLevel']

# --- Classification: Naive Bayes classifier ----
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(wordvec_counts, y_b , train_size = 0.8)
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t, y_t , train_size = 0.8)

# ------- NB, BoW------
nb_bow = ml_tools.naive_bayes(X_train_b, y_train_b, X_test_b, y_test_b)

# ------ NB,  TF-IDF-----
nb_tfidf = ml_tools.naive_bayes(X_train_t, y_train_t, X_test_t, y_test_t)

# --- RF: USING BoW feature extraction ------
rf_bow = ml_tools.random_forest(X_train_t, y_train_t, X_test_t, y_test_t )

# --- RF: USING tf-idf feature extraction -----

rf_tfidf = ml_tools.random_forest(X_train_t, y_train_t, X_test_t, y_test_t )

#%%
labels = ['ArgumentLevel','ReasoningLevel'] # 'ArgumentLevel', 'ReasoningLevel'
features = ['ifidf','bow'] #'bow', 'ifidf'
num_epochs = 10
train_sizes = [0.5,0.6,0.7,0.8,0.9] # proportion of training data

# %%

# loop over labels, feature extractions
for label in labels:
    for feature in features:
        # -- Feature extraction: TF-IDF ---
        if feature ==  'ifidf':
            wordvec_names, wordvec_counts = ml_tools.tf_idf(df['Content'].tolist())
        # -- Feature extraction: Bag of Words ---
        elif feature == 'bow':
            wordvec_names, wordvec_counts = ml_tools.BoW(df['Content'].tolist())
        formats.lr_accuracy_trainsize_plot_general(ml_tools.naive_bayes, wordvec_counts, df[label].tolist(),label, feature, num_epochs, train_sizes)
# %%
