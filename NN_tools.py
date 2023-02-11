# importing tool functions
import numpy as np
import matplotlib.pyplot as plt
from tools import utils, ml_tools
import pandas as pd
import numpy as np

# machine learning imports
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from sklearn.metrics import cohen_kappa_score

# graphing tools
import seaborn as sns
from scipy.stats import sem
import matplotlib
import scienceplots

plt.style.use(['science', 'ieee','no-latex'])
matplotlib.rc('font', family='times new roman')

# time
import time

# Standard Neural Network

# Standard Neural Network
def NN_ck(model1, input1, ephs, X_t, y_t, X_test, y_test):
    maxlen = 100 
    model = model1
    model.add(layers.Dense(12, input_dim = input1, activation  = 'relu'))
    model.add(layers.Dense(8, input_dim = input1, activation  = 'relu'))
    model.add(layers.Dense(1,activation = 'sigmoid'))
    model.add(layers.Flatten())
    model.compile(loss= 'binary_crossentropy', optimizer= 'rmsprop', metrics= ['accuracy'])
    model.build(input1)
    model.summary()
    history = model.fit(X_t,y_t,epochs = ephs, verbose=True, validation_data=(X_test, y_test), batch_size=30 )
    predictions = model.predict(X_test[:11])
    cohen_score = cohen_kappa_score(y_test[:11], predictions)
    return cohen_score

def NN_optimised_parameters(model1, input1, ephs, X_t, y_t, X_test, y_test):
    maxlen = 100 
    model = model1
    model.add(layers.Dense(12, input_dim = input1, activation  = 'relu'))
    model.add(layers.Dense(8, input_dim = input1, activation  = 'relu'))
    model.add(layers.Dense(1,activation = 'sigmoid'))
    model.add(layers.Flatten())
    model.compile(loss= 'binary_crossentropy', optimizer= 'rmsprop', metrics= ['accuracy'])
    model.build(input1)
    model.summary()
    history = model.fit(X_t,y_t,epochs = ephs, verbose=True, validation_data=(X_test, y_test), batch_size=30 )
    predictions = model.predict(X_test[:11])
    cohen_score = cohen_kappa_score(y_test[:11], predictions)
    return history

def NN_default_parameters(model1, input1, ephs, X_t, y_t, X_test, y_test):
    maxlen = 100 
    model = model1
    model.add(layers.Dense(12, input_dim = input1, activation  = 'relu'))
    model.add(layers.Dense(8, input_dim = input1, activation  = 'relu'))
    model.add(layers.Dense(1,activation = 'sigmoid'))
    model.add(layers.Flatten())
    model.compile(loss= 'binary_crossentropy', optimizer= 'Adam', metrics= ['accuracy'])
    model.build(input1)
    model.summary()
    history = model.fit(X_t,y_t,epochs = ephs, verbose=True, validation_data=(X_test, y_test), batch_size=30 )
    predictions = model.predict(X_test[:11])
    cohen_score = cohen_kappa_score(y_test[:11], predictions)
    return history, cohen_score


def NN_data(Neural, X, y,t_size,epoch_no, str_dataname,str_featext, str_year, dir):
    accuracies = []
    accuracies_sem = []
    loss = []
    val_loss = []
    ck          = []

    dummy = []
    dummy_loss = []
    dummy_val_loss = []
    dummy_cohen  = []
    for i in t_size:
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y , train_size = i)
        input = X_train_b.shape[1]
        #input = len(y_train_b)
        nn1 = Neural(Sequential(), input, epoch_no, X_train_b, y_train_b, X_test_b, y_test_b)

        dummy.append(nn1.history['accuracy'])
        dummy_loss.append(nn1.history['loss'])
        dummy_val_loss.append(nn1.history['loss'])


    for array in dummy:
        accuracies.append(np.sum(array)/len(array))
        accuracies_sem.append(sem(array))
    
    for array in dummy_loss:
        loss.append(np.sum(array)/len(array))
    
    for array in dummy_val_loss:
        val_loss.append(np.sum(array)/len(array))
    
        ck.append(np.sum(dummy_cohen)/len(dummy_cohen))

    dict_rl = {'trainsize':t_size, 'accuracy':accuracies, 'sem': accuracies_sem, 'loss':loss, 'valloss': val_loss, 'ck':ck}
    acc_rl_bow = pd.DataFrame(dict_rl)
    name = 'W5_NN_{}_{}_{}_{}ephs_recent'. format(str_year, str_featext,str_dataname,epoch_no)
    utils.save_as_pickle_file(acc_rl_bow,name, dir)
    #name = 'NN_{}_{}_trainsize_accuracy_sem_{}ephs_{}'. format(str_dataname,str_featext,'1000', str_year)
    return acc_rl_bow


def NN_data_iteration(Neural, X, y,t_size,epoch_no, it_no, str_dataname,str_featext, str_year, dir):
    
    accuracies     = []
    accuracies_sem = []
    loss           = []
    val_loss       = []
    ck             = []

    accuracies1     = []
    accuracies_sem1 = []
    loss1          = []
    val_loss1       = []

    for i in t_size:

        dummy          = []
        dummy_loss     = []
        dummy_val_loss = []
        dummy_ck       = []

        for iteration in range(it_no):
            X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y , train_size = i)
            input = X_train_b.shape[1]
            nn1 = Neural(Sequential(), input, epoch_no, X_train_b, y_train_b, X_test_b, y_test_b)

            dummy.append(nn1.history['accuracy'])
            dummy_loss.append(nn1.history['loss'])
            dummy_val_loss.append(nn1.history['loss'])

        
        for no in range(len(dummy)):
            accuracies1.append(np.sum(dummy[no])/len(dummy[no]))
            accuracies_sem1.append(np.sum(sem(dummy[no]))/len(dummy[no]))
            loss1.append(np.sum(dummy_loss[no])/len(dummy[no]))
            val_loss1.append(np.sum(dummy_val_loss[no])/len(dummy[no]))

    

        accuracies.append(np.sum(accuracies1)/len(accuracies1))
        accuracies_sem.append(np.sum(accuracies_sem1)/len(accuracies_sem1))
        loss.append(np.sum(loss1)/len(loss1))
        val_loss.append(np.sum(val_loss1)/len(val_loss1))
        ck.append(np.sum(dummy_ck)/len(dummy_ck))

    dict_rl = {'trainsize':t_size, 'accuracy':accuracies, 'sem': accuracies_sem, 'loss':loss, 'valloss': val_loss, 'ck':ck}
    acc_rl_bow = pd.DataFrame(dict_rl)
    name = 'W5_NN_{}_iteration_{}_{}_{}_{}'. format(it_no, str_year, str_featext,str_dataname,epoch_no)
    utils.save_as_pickle_file(acc_rl_bow,name, dir)
    return acc_rl_bow

def NN_data_ck(Neural, X, y,t_size,epoch_no, str_year):
    ck    = []
    dummy = []
    for i in t_size:
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y , train_size = i)
        input = X_train_b.shape[1]
        nn1 = Neural(Sequential(), input, epoch_no, X_train_b, y_train_b, X_test_b, y_test_b)

        dummy.append(nn1)

    dict_rl = {'trainsize':t_size, 'ck':ck}
    acc_rl_bow = pd.DataFrame(dict_rl)
    name = 'W5_NN_cohenkappa_{}_{}ephs_recent'. format(str_year, epoch_no)
    utils.save_as_pickle_file(acc_rl_bow,name, dir)
    return acc_rl_bow

def plot(dframe_train, dframe_acc, dframe_sem):
    plt.plot(dframe_train, dframe_acc, 'o')
    plt.errorbar(dframe_train, dframe_acc, yerr = dframe_sem, markersize=0.5, capsize=3, elinewidth=1, color= 'black' )
    plt.xlabel(' Training size')
    plt.ylabel('Accuracy score')

    # save figure 
    # filepath = 'outputs/NN-RL-BOW-1000ephs_NEW'
    # plt.savefig(filepath)
    return


def one_hot_enc_labels_bow(whole_df, name_of_labels_str):
    df_y = whole_df[name_of_labels_str].tolist()
    lb   = LabelBinarizer()
    lb.fit(df_y)
    df_y1 = lb.transform(df_y)
    return df_y1


def one_hot_enc_labels_tf(whole_df, name_of_labels_str):
    df_y =np.array(whole_df[name_of_labels_str].tolist(), dtype=object)
    lb   = LabelBinarizer()
    lb.fit(df_y)
    df_y1 = lb.transform(df_y)
    return df_y1


def convert_to_preferred_format(sec):
   sec = sec % (24 * 3600)
   hour = sec // 3600
   sec %= 3600
   min = sec // 60
   sec %= 60
   print("seconds value in hours:",hour)
   print("seconds value in minutes:",min)
   return "%02d:%02d:%02d" % (hour, min, sec) 

def nn_graph(train_size, df, mlmodel):
    #train_sizes = [0.5,0.6,0.7,0.8,0.9]
    c = ['r','b','g', 'm'] # colour notation
    for i in range(4):
        # all of the arrays will be of size 4 for training size graphs anyway
        plt.plot(train_size,df['accuracy'][i], label = '{}-{}-{}'.format(mlmodel,df['feature extraction'][i],df['Label'][i]), color =c[i],marker='o', markersize =4 )
        #plt.scatter(train_size, df['accuracy'][i], 'o', color = c[i])
        plt.errorbar(train_size, df['accuracy'][i], yerr = df['sem'][i], linewidth=0.2, capsize=2, c = 'black', markeredgewidth = 0.5)
    plt.xlabel(' Training size')
    plt.ylabel('Accuracy score')
    plt.legend( prop={'size': 3})
    #filepath = 'outputs/comparison-{}-2500epochs-overallgraph_new.png'.format(mlmodel)
    #plt.savefig(filepath)
    return 

def nn_graph_loss(train_size, df, mlmodel):
    #train_sizes = [0.5,0.6,0.7,0.8,0.9]
    c = ['r','b','g', 'm'] # colour notation
    for i in range(4):
        # all of the arrays will be of size 4 for training size graphs anyway
        plt.plot(train_size,df['loss'][i], label = '{}-{}-{}-{}'.format(mlmodel,df['feature extraction'][i],df['Label'][i], 'loss'), color =c[i],marker='o', markersize =4 )
        #plt.plot(train_size,df['val_loss'][i], label = '{}-{}-{}-{}'.format(mlmodel,df['feature extraction'][i],df['Label'][i], 'val_loss'), color =c[i],marker='o', markersize =4 )
        #plt.scatter(train_size, df['accuracy'][i], 'o', color = c[i])
        #plt.errorbar(train_size, df['loss'][i], yerr = df['sem'][i], linewidth=0.2, capsize=2, c = 'black', markeredgewidth = 0.5)
    plt.xlabel(' Training size')
    plt.ylabel('Loss')
    plt.legend( prop={'size': 3})
    #filepath = 'outputs/comparison-{}-2500epochs-overallgraph_new.png'.format(mlmodel)
    #plt.savefig(filepath)
    return
