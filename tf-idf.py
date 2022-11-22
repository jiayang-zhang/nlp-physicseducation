#%%
import pandas as pd
import numpy as np

# feature extraction imports
from sklearn.feature_extraction.text import TfidfVectorizer

#machine learning method imports
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer

# metrics imports
from sklearn import metrics
from sklearn.metrics import classification_report
from tools import utils, ml_tools
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


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
no_of_iterations = 10
kf = KFold(n_splits= no_of_iterations)

#%%

# --- classify using bow feature extraction -----
# leave out method
leave_out_array = []
iteration_number = []
for i in range(no_of_iterations):
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(wordvec_counts, y_b , train_size = 0.8)
    clf_b       = MultinomialNB().fit(X_train_b, y_train_b)
    predicted_b = clf_b.predict(X_test_b)
    acc_score_l =  metrics.accuracy_score(y_test_b, predicted_b)
    leave_out_array.append(acc_score_l)
    iteration_number.append(i)

mean_lo_values = sum(leave_out_array)/ len(leave_out_array)
print('array of leave out accuracy values', leave_out_array)
print('Avg leave out accuracy values:', mean_lo_values)


# NB, BoW, KFold
model       = MultinomialNB()
result = cross_val_score(model, wordvec_counts, df['ReasoningLevel'].tolist(), cv = kf)
print(result)
print('Avg accuracy - BoW {}'.format(result.mean()))


import matplotlib.pyplot as plt
plt.plot(iteration_number, leave_out_array,'x')
plt.plot(iteration_number,leave_out_array,  '--')
plt.plot(iteration_number, result, 'o')
plt.plot(iteration_number, result, '--')
plt.title('NB, BoW  CV: Leave out vs Kfold')
plt.ylabel('Accuracy value')
plt.xlabel('Iteration number')
plt.legend()
plt.grid()

#%%

# --- classify using tf-idf feature extraction---


# NB,  TF-IDF, KFold
array = []
for train_index, test_index in kf.split(X_t):
    #print('TRAIN:', train_index, 'TEST:', test_index)
    X_train_t, X_test_t = X_t[train_index], X_t[test_index]
    y_train_t, y_test_t = y_t[train_index], y_t[test_index]
    clf_t      = MultinomialNB().fit(X_train_t, y_train_t)
    predicted_t = clf_b.predict(X_test_t)
    acc_score = metrics.accuracy_score(y_test_t, predicted_t)
    array.append(acc_score)

average_accuracy = sum(array)/len(array)
print('Array of accuracy values:', array)
print('Average accuracy - tfidf:', average_accuracy)


# graph of k fold for tidf vs BoW 
plt.plot(iteration_number, result,'x')
plt.plot(iteration_number,result,  '--', label = 'BoW')
plt.plot(iteration_number, array, 'o')
plt.plot(iteration_number, array, '--', label = 'tf-idf')
plt.title('NB, CV-Kfold: BoW vs TF-IDF')
plt.ylabel('Accuracy value')
plt.xlabel('Iteration number')
plt.legend()
plt.grid()

#%%

print('MultinomialNB Accuracy, using BOW feature extraction:', metrics.accuracy_score(y_test_b, predicted_b))
print('MultinomialNB Accuracy, using TF-IDF feature extraction:', metrics.accuracy_score(y_test_t, predicted_t))

#%%

#---- Classification: Random Forest ---------
from sklearn.ensemble import RandomForestClassifier
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(wordvec_counts, y_b , train_size = 0.8)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_t, y_t , train_size = 0.8)

# --- RF: USING BoW feature extraction ------
clf_rf_b            = RandomForestClassifier(max_depth= None, random_state = 0).fit(wordvec_counts, y_b)
predicted_rf_b      = clf_rf_b.predict(X_test_b)
accuracy_score_rf_b = metrics.accuracy_score(y_test_b, predicted_rf_b)
print(accuracy_score_rf_b)


# --- RF: USING tf-idf feature extraction -----
clf_rf_t            = RandomForestClassifier(max_depth= None, random_state = 0).fit(X_t, y_t)
predicted_rf_t      = clf_rf_t.predict(X_test_t)
accuracy_score_rf_t = metrics.accuracy_score(y_test_t, predicted_rf_t)
print(accuracy_score_rf_t)


# --- RF, BoW, KFold -----
model_rf            = RandomForestClassifier(max_depth= None, random_state = 0)
result_rf_b         = cross_val_score(model_rf, wordvec_counts, y_b, cv = kf)
print(result_rf_b)
print('Avg accuracy - Bo W, RF {}'.format(result_rf_b.mean()))


#%%
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