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
labels2 = ['ArgumentLevel','ReasoningLevel','ArgumentLevel','ReasoningLevel'] # 'ArgumentLevel', 'ReasoningLevel'
features = ['ifidf','bow'] #'bow', 'ifidf'
num_epochs = 2500
train_sizes = [0.5,0.6,0.7,0.8,0.9] # proportion of training data
'''
What is the accuracies?
the accuracy arra contains 4 lists corresponding to each feature extraction technique used
each list contains 5 accuracy values corresponding to the training size used
each single value is the average accuracy score for 10 or more iterations (epoch is defined by the user)
'''
#%%
#------ trainsize NB-----------
# loop over labels for training size, feature extractions: Naive bayes
accuracies = []
accuracies_sd = []
feature2 = []
labels2  = []
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
            labels2.append(label)
        # -- Feature extraction: Bag of Words ---
        elif feature == 'bow':
            wordvec_names, wordvec_counts = ml_tools.BoW(df['Content'].tolist())
            print('bow')
            b = ml_tools.lr_accuracy_trainsize_plot_general(ml_tools.naive_bayes, wordvec_counts, df[label].tolist(),label, feature, num_epochs, train_sizes)
            accuracies.append(b[0])
            accuracies_sd.append(b[1])
            feature2.append(feature)
            labels2.append(label)

df1 = pd.DataFrame({'feature extraction':feature2,'accuracy':accuracies, 'standdev': accuracies_sd, 'Label': labels2})
#%%
#----- pickled dataframe
utils.save_as_pickle_file(df1,'NB_trainingsize_plot_2500epochs2')
print(df1['Label'])
#%%
unpickle_df = utils.load_pickle_file_to_df('NB_trainingsize_plot_2500epochs2')
print(unpickle_df['Label'])
#%%
for i in range(len(unpickle_df)):
    filepath = 'outputs/{}-NB2-{}epochs-{}-graph.png'.format(unpickle_df['feature extraction'][i], num_epochs, unpickle_df['Label'][i]) # ** always change name **
    formats.scatter_plot(xvalue = train_sizes, yvalue = unpickle_df['accuracy'][i], yerr = unpickle_df['standdev'][i], xlabel = 'Training Size', ylabel = 'Accuracy', filepath = filepath)



#%%
# --- inidividual graphs --------
'''
for i in range(len(unpickle_df)):
    if i%2 == 0:
        filepath = 'outputs/{}-RF-{}epochs-{}-graph-{}.png'.format(features[0], num_epochs, labels[0]) # ** always change name **
        formats.scatter_plot(xvalue = train_sizes, yvalue = unpickle_df['accuracy'][i], yerr = unpickle_df['standdev'][i], xlabel = 'Training Size', ylabel = 'Accuracy', filepath = filepath)
    else: 
        filepath = 'outputs/{}-RF-{}epochs-{}-graph-{}.png'.format(features[1], num_epochs, labels[1], i) # ** always change name **
        formats.scatter_plot(xvalue = train_sizes,  yvalue = unpickle_df['accuracy'][i],yerr = unpickle_df['standdev'][i], xlabel = 'Training Size', ylabel = 'Accuracy', filepath = filepath)
'''
#%%
#=============================================================================================================================
#                       ------------- RF ----------------
#=============================================================================================================================
#%%
#------ trainsize RF-----------
# loop over labels for training size, feature extractions: Naive bayes
accuracies_rf = []
accuracies_sd_rf = []
feature2_rf = []
labels2_rf  = []
for label in labels:
    for feature in features:
        # -- Feature extraction: TF-IDF ---
        if feature ==  'ifidf':
            wordvec_names, wordvec_counts = ml_tools.tf_idf(df['Content'].tolist())
            print('tfidf')
            t = ml_tools.lr_accuracy_trainsize_plot_general(ml_tools.random_forest, wordvec_counts, df[label].tolist(),label, feature, num_epochs, train_sizes)
            accuracies_rf.append(t[0])
            accuracies_sd_rf.append(t[1])
            feature2_rf.append(feature)
            labels2_rf.append(label)
        # -- Feature extraction: Bag of Words ---
        elif feature == 'bow':
            wordvec_names, wordvec_counts = ml_tools.BoW(df['Content'].tolist())
            print('bow')
            b = ml_tools.lr_accuracy_trainsize_plot_general(ml_tools.random_forest, wordvec_counts, df[label].tolist(),label, feature, num_epochs, train_sizes)
            accuracies_rf.append(b[0])
            accuracies_sd_rf.append(b[1])
            feature2_rf.append(feature)
            labels2_rf.append(label)

df1_rf = pd.DataFrame({'feature extraction':feature2_rf,'accuracy':accuracies_rf, 'standdev': accuracies_sd_rf, 'Label': labels2_rf})
#%%
#----- pickled dataframe
utils.save_as_pickle_file(df1_rf,'RF_trainingsize_plot_2500epochs1')

#%%
unpickle_df = utils.load_pickle_file_to_df('RF_trainingsize_plot_2500epochs')
print(unpickle_df['Label'])
#%%
for i in range(len(unpickle_df)):
    filepath = 'outputs/{}-RF2-{}epochs-{}-graph.png'.format(unpickle_df['feature extraction'][i], num_epochs, unpickle_df['Label'][i]) # ** always change name **
    formats.scatter_plot(xvalue = train_sizes, yvalue = unpickle_df['accuracy'][i], yerr = unpickle_df['standdev'][i], xlabel = 'Training Size', ylabel = 'Accuracy', filepath = filepath)

#%%
#=============================================================================================================
#                                   ---------------- LR -------------
#==============================================================================================================
#------ trainsize LR -----------
# loop over labels for training size, feature extractions: Naive bayes
accuracies_lr = []
accuracies_sd_lr = []
feature2_lr = []
labels2_lr = []
for label in labels:
    for feature in features:
        # -- Feature extraction: TF-IDF ---
        if feature ==  'ifidf':
            wordvec_names, wordvec_counts = ml_tools.tf_idf(df['Content'].tolist())
            print('tfidf')
            t = ml_tools.lr_accuracy_trainsize_plot_general(ml_tools.logistic_regression2, wordvec_counts, df[label].tolist(),label, feature, num_epochs, train_sizes)
            accuracies_lr.append(t[0])
            accuracies_sd_lr.append(t[1])
            feature2_lr.append(feature)
            labels2_lr.append(label)
        # -- Feature extraction: Bag of Words ---
        elif feature == 'bow':
            wordvec_names, wordvec_counts = ml_tools.BoW(df['Content'].tolist())
            print('bow')
            b = ml_tools.lr_accuracy_trainsize_plot_general(ml_tools.logistic_regression2, wordvec_counts, df[label].tolist(),label, feature, num_epochs, train_sizes)
            accuracies_lr.append(b[0])
            accuracies_sd_lr.append(b[1])
            feature2_lr.append(feature)
            labels2_lr.append(label)

df1_lr= pd.DataFrame({'feature extraction':feature2_lr,'accuracy':accuracies_lr, 'standdev': accuracies_sd_lr, 'Label': labels2_lr})
#%%
#----- pickled dataframe --------
utils.save_as_pickle_file(df1_lr,'LR_trainingsize_plot_2500epochs1')

#%%
unpickle_df_lr = utils.load_pickle_file_to_df('LR_trainingsize_plot_2500epochs')

#%%
for i in range(len(unpickle_df_lr)):
    filepath = 'outputs/{}-LR2-{}epochs-{}-graph.png'.format(unpickle_df_lr['feature extraction'][i], num_epochs, unpickle_df_lr['Label'][i]) # ** always change name **
    formats.scatter_plot(xvalue = train_sizes, yvalue = unpickle_df_lr['accuracy'][i], yerr = unpickle_df_lr['standdev'][i], xlabel = 'Training Size', ylabel = 'Accuracy', filepath = filepath)

# %%
