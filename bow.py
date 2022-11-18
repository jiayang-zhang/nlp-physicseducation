from tools import ml_tools, utils
import os
import pandas as pd
pd.set_option('max_colu', 10)


from sklearn.model_selection import train_test_split

# =======================================================================================================================================
dir_csv = 'outputs/labels_cleaned.csv'
# =======================================================================================================================================

df = pd.read_csv(dir_csv, encoding='utf-8')

# # -- Feature extraction: Bag of Words ---
wordvec_names, wordvec_counts = ml_tools.BoW(df['Content'].tolist())


# # -- classifion: logistic regression --
# iterations = 1
# sum = 0
# for i in range(iterations):
#     # split train, test data
#     X_train, X_test, y_train, y_test = train_test_split(wordvec_counts, df['ArgumentLevel'].tolist(), train_size = 0.8)
#     # create trained model
#     y_test_predict = ml_tools.logistic_regression(X_train, X_test, y_train)
#     # prediction
#     accuracy_score = ml_tools.sanity_check(X_test, y_test, y_test_predict)
#     sum += accuracy_score
# print('average accuracy score:', sum/iterations)
