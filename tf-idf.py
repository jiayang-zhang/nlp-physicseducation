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
labels = ['ArgumentLevel','ReasoningLevel'] # 'ArgumentLevel', 'ReasoningLevel'
features = ['ifidf','bow'] #'bow', 'ifidf'
num_epochs = 40
train_sizes = [0.5,0.6,0.7,0.8,0.9] # proportion of training data
'''
What is the accuracies?
the accuracy arra contains 4 lists corresponding to each feature extraction technique used
each list contains 5 accuracy values corresponding to the training size used
each single value is the average accuracy score for 10 or more iterations (epoch is defined by the user)
'''

# loop over labels for training size, feature extractions: Naive bayes
accuracies = []
accuracies_sd = []
feature2 = []
for label in labels:
    for feature in features:
        # -- Feature extraction: TF-IDF ---
        if feature ==  'ifidf':
            wordvec_names, wordvec_counts = ml_tools.tf_idf(df['Content'].tolist())
            print('tfidf')
            t = ml_tools.lr_accuracy_trainsize_plot_general(ml_tools.naive_bayes, wordvec_counts, df[label].tolist(),label, feature, num_epochs, train_sizes)
            accuracies.append(t[0])
            accuracies_sd.append(t[1])
            feature2.append(feature)
        # -- Feature extraction: Bag of Words ---
        elif feature == 'bow':
            wordvec_names, wordvec_counts = ml_tools.BoW(df['Content'].tolist())
            print('bow')
            b = ml_tools.lr_accuracy_trainsize_plot_general(ml_tools.naive_bayes, wordvec_counts, df[label].tolist(),label, feature, num_epochs, train_sizes)
            accuracies.append(b[0])
            accuracies_sd.append(b[1])
            feature2.append(feature)


# %%
#------- Dataframe  --------
df1 = pd.DataFrame({'feature extraction':feature2,'accuracy':accuracies, 'standdev': accuracies_sd})

#----- pickled dataframe
utils.save_as_pickle_file(df1,'testnb')
#%%

unpickle_df = utils.load_pickle_file_to_df('testnb')


#%%
# --- inidividual graphs --------

for i in range(len(accuracies)):
    if i%2 == 0:
        filepath = 'outputs/{}-NB-{}epochs-{}.png'.format(feature[0], num_epochs, label[0]) # ** always change name **
        formats.scatter_plot(xvalue = train_sizes, yvalue = accuracies[i], yerr = accuracies_sd[i], xlabel = 'Training Size', ylabel = 'Accuracy', filepath = filepath)
    else: 
        filepath = 'outputs/{}-NB-{}epochs-{}.png'.format(feature[1], num_epochs, label[1]) # ** always change name **
        formats.scatter_plot(xvalue = train_sizes, yvalue = accuracies[i], yerr = accuracies_sd[i], xlabel = 'Training Size', ylabel = 'Accuracy', filepath = filepath)


#%%
#    #--- collective graphs ---
#    filepath = 'outputs/{}-NB-{}epochs-{}.png'.format(feature[1], num_epochs, label[1]) # ** always change name **
#    formats.scatter_plot_asone(xvalue = train_sizes, yvalue = accuracies, yerr = accuracies_sd, xlabel = 'Training Size', ylabel = 'Accuracy', filepath = filepath, feature = features, label = labels)
