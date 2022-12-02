import pandas as pd
pd.set_option('max_colu', 10)

import time
import numpy as np
from sklearn.model_selection import train_test_split   # conda install scikit-learn
from tools import ml_tools, formats


# =======================================================================================================================================
# * set which dataset you would like to train *
dir_csv = 'outputs/labels_cleaned_y1c1c2.csv'
# =======================================================================================================================================
# * set how you would like it to be trained *
labels = ['ArgumentLevel','ReasoningLevel'] # 'ArgumentLevel', 'ReasoningLevel'
features = ['ifidf','bow'] #'bow', 'ifidf'
num_epochs = 3
train_sizes = [0.5,0.6,0.7,0.8,0.9] # proportion of training data
# =======================================================================================================================================

# unpack dataframe
df = pd.read_csv(dir_csv, encoding='utf-8')

# -- classifion: logistic regression --
def lr_accuracy_trainsize_plot(label, feature, num_epochs, train_sizes):
    accuracies = []
    accuracies_sd = []
    for size in train_sizes:
        sum = 0
        start_time = time.time()


        dummy = []
        for epoch in range(num_epochs):
            # split train/test data
            X_train, X_test, y_train, y_test = train_test_split(wordvec_counts, df[label].tolist(), train_size = size)
            # train model + prediction
            y_test_predict = ml_tools.logistic_regression(X_train, X_test, y_train)

            # Accuracy
            accuracy_score = ml_tools.sanity_check(X_test, y_test, y_test_predict, printWrong=False)
            # print("Epoch {}/{}, Accuracy: {:.3f}".format(epoch+1,num_epochs, accuracy_score))
            dummy.append(accuracy_score)

        accuracies.append(np.sum(dummy)/num_epochs)
        accuracies_sd.append(np.std(dummy))
        print('average accuracy score:', np.sum(dummy)/num_epochs)
        print("--- %s seconds ---" % (time.time() - start_time))


    # specify figure output path
    filepath = 'outputs/{}-lr-{}epochs-{}.png'.format(feature, num_epochs, label) # ** always change name **
    formats.scatter_plot(xvalue = train_sizes, yvalue = accuracies, yerr = accuracies_sd, xlabel = 'Training Size', ylabel = 'Accuracy', filepath = filepath)
    return

# loop over labels, feature extractions
for label in labels:
    for feature in features:

        # -- Feature extraction: TF-IDF ---
        if feature ==  'ifidf':
            wordvec_names, wordvec_counts = ml_tools.tf_idf(df['Content'].tolist())
        # -- Feature extraction: Bag of Words ---
        elif feature == 'bow':
            wordvec_names, wordvec_counts = ml_tools.BoW(df['Content'].tolist())
        lr_accuracy_trainsize_plot(label, feature, num_epochs, train_sizes)
