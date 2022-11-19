import os
import pandas as pd
pd.set_option('max_colu', 10)
import time
from sklearn.model_selection import train_test_split

from tools import ml_tools, utils, formats

# =======================================================================================================================================
dir_csv = 'outputs/labels_cleaned.csv'
# =======================================================================================================================================

df = pd.read_csv(dir_csv, encoding='utf-8')

'''
# -- Feature extraction: Bag of Words ---
wordvec_names, wordvec_counts = ml_tools.BoW(df['Content'].tolist())
'''

# -- Feature extraction: TF-IDF ---
wordvec_names, wordvec_counts = ml_tools.tf_idf(df['Content'].tolist())


# # -- classifion: logistic regression --
sizes = [0.5,0.6,0.7,0.8,0.9]
num_epochs = 10

accuracies = []
for size in sizes:
    sum = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        # split train/test data
        X_train, X_test, y_train, y_test = train_test_split(wordvec_counts, df['ArgumentLevel'].tolist(), train_size = size)
        # train model + prediction
        y_test_predict = ml_tools.logistic_regression(X_train, X_test, y_train)

        # Accuracy
        accuracy_score = ml_tools.sanity_check(X_test, y_test, y_test_predict, printWrong=False)
        # print("Epoch {}/{}, Accuracy: {:.3f}".format(epoch+1,num_epochs, accuracy_score))
        sum += accuracy_score
    accuracies.append(sum/num_epochs)
    print('average accuracy score:', sum/num_epochs)
    print("--- %s seconds ---" % (time.time() - start_time))


formats.scatter_plot(xvalue = sizes, yvalue = accuracies, xlabel = 'Training Size', ylabel = 'Accuracy', filepath = 'outputs/ifidf-lr-{}epochs.png'.format(num_epochs))
