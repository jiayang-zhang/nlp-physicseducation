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
from tensorflow.keras.layers import Dropout

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
    return model

def NN_optimised_parameters(model1, input1, ephs, X_t, y_t, X_test, y_test):
    maxlen = 100 
    model = model1
    model.add(layers.Dense(12, input_dim = input1, activation  = 'relu'))
   # model.add(Dropout(0.7))
    model.add(layers.Dense(8, input_dim = input1, activation  = 'relu'))
    #model.add(Dropout(0.7))
    model.add(layers.Dense(1,activation = 'sigmoid'))
    #model.add(layers.Dense(1,activation = 'softmax'))
    model.add(layers.Flatten())
    model.compile(loss= 'binary_crossentropy', optimizer= 'rmsprop', metrics= ['accuracy'])
    #model.compile(loss= 'categorical_crossentropy', optimizer= 'rmsprop', metrics= ['accuracy'])
    model.build(input1)
    model.summary()
    #history = model.fit(X_t,y_t,epochs = ephs, verbose=True, validation_data=(X_test, y_test), batch_size=30 )
    return model

def NN_optimised_parameters_al(model1, input1, ephs, X_t, y_t, X_test, y_test):
    maxlen = 100 
    model = model1
    model.add(layers.Dense(12, input_dim = input1, activation  = 'relu'))
   # model.add(Dropout(0.7))
    model.add(layers.Dense(8, input_dim = input1, activation  = 'relu'))
    #model.add(Dropout(0.7))
    #model.add(layers.Dense(1,activation = 'sigmoid'))
    model.add(layers.Dense(5,activation = 'softmax'))
    model.add(layers.Flatten())
    #model.compile(loss= 'binary_crossentropy', optimizer= 'rmsprop', metrics= ['accuracy'])
    model.compile(loss= 'categorical_crossentropy', optimizer= 'rmsprop', metrics= ['accuracy'])
    model.build(input1)
    model.summary()
    #history = model.fit(X_t,y_t,epochs = ephs, verbose=True, validation_data=(X_test, y_test), batch_size=30 )
    return model
def NN_optimised_parameters_epist(model1, input1, ephs, X_t, y_t, X_test, y_test):
    maxlen = 100 
    model = model1
    model.add(layers.Dense(12, input_dim = input1, activation  = 'relu'))
   # model.add(Dropout(0.7))
    model.add(layers.Dense(8, input_dim = input1, activation  = 'relu'))
    #model.add(Dropout(0.7))
    #model.add(layers.Dense(1,activation = 'sigmoid'))
    model.add(layers.Dense(4,activation = 'softmax'))
    model.add(layers.Flatten())
    #model.compile(loss= 'binary_crossentropy', optimizer= 'rmsprop', metrics= ['accuracy'])
    model.compile(loss= 'categorical_crossentropy', optimizer= 'rmsprop', metrics= ['accuracy'])
    model.build(input1)
    model.summary()
    #history = model.fit(X_t,y_t,epochs = ephs, verbose=True, validation_data=(X_test, y_test), batch_size=30 )
    return model

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
    #history = model.fit(X_t,y_t,epochs = ephs, verbose=True, validation_data=(X_test, y_test), batch_size=30 )
    return model


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
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y , train_size = i, shuffle= True)
        input = X_train_b.shape[1]
        #input = len(y_train_b)
        model = Neural(Sequential(), input, epoch_no, X_train_b, y_train_b, X_test_b, y_test_b)
        history = model.fit(X_train_b,y_train_b,epochs = epoch_no, verbose=True, validation_data=(X_test_b, y_test_b), batch_size=30 )
        # predictions =  model.predict(X_test_b) 
        # predictions =  np.argmax(predictions, axis=1)
        # y_test_b    =  np.argmax(y_test_b, axis=1)
        # cohen_score = cohen_kappa_score(y_test_b, predictions)

        # dummy.append(cohen_score)
        dummy.append(history.history['accuracy'])
        dummy_loss.append(history.history['loss'])
        dummy_val_loss.append(history.history['loss'])
        dummy_cohen.append(cohen_kappa_score)


    for array in dummy:
        accuracies.append(np.sum(array)/len(array))
        accuracies_sem.append(sem(array))
    
    for array in dummy_loss:
        loss.append(np.sum(array)/len(array))
    
    for array in dummy_val_loss:
        val_loss.append(np.sum(array)/len(array))
    
    for array in dummy_cohen:
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
        dummy_predict  = []
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y , train_size = i, shuffle=True)
        input = X_train_b.shape[1]
        for iteration in range(it_no):
            model = Neural(Sequential(), input, epoch_no, X_train_b, y_train_b, X_test_b, y_test_b)
            history = model.fit(X_train_b,y_train_b,epochs = epoch_no, verbose=True, validation_data=(X_test_b, y_test_b), batch_size=30 )

            # dummy_predict.append(predictions)
            dummy.append(history.history['accuracy'])
            dummy_loss.append(history.history['loss'])
            dummy_val_loss.append(history.history['loss'])
            #dummy_ck.append(cohen_kappa_score)

        print(dummy_ck)
        for no in range(len(dummy)):
            accuracies1.append(np.sum(dummy[no])/len(dummy[no]))
            accuracies_sem1.append(np.sum(sem(dummy[no]))/len(dummy[no]))
            loss1.append(np.sum(dummy_loss[no])/len(dummy[no]))
            val_loss1.append(np.sum(dummy_val_loss[no])/len(dummy[no]))


                
    

        accuracies.append(np.sum(accuracies1)/len(accuracies1))
        accuracies_sem.append(np.sum(accuracies_sem1)/len(accuracies_sem1))
        loss.append(np.sum(loss1)/len(loss1))
        val_loss.append(np.sum(val_loss1)/len(val_loss1))

        # if dummy_ck == 0.0:
        #     ck.append(dummy_ck)
        # else:
        #     ck.append(np.sum(dummy_ck))
        
        #print(ck)


    #dict_rl = {'trainsize':t_size, 'accuracy':accuracies, 'sem': accuracies_sem, 'loss':loss, 'valloss': val_loss, 'ck':ck}
    dict_rl = {'trainsize':t_size, 'accuracy':accuracies, 'sem': accuracies_sem, 'loss':loss, 'valloss': val_loss }
    acc_rl_bow = pd.DataFrame(dict_rl)
    #name = 'W5_NN_{}_iteration_{}_{}_{}_{}'. format(it_no, str_year, str_featext,str_dataname,epoch_no)
    #utils.save_as_pickle_file(acc_rl_bow,name, dir)
    return acc_rl_bow

def NN_data_ck(Neural, X, y,t_size,epoch_no, str_year):
    ck    = []
    dummy = []

    for i in t_size:
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y , train_size = i)
        input = X_train_b.shape[1]
        nn1 = Neural(Sequential(), input, epoch_no, X_train_b, y_train_b, X_test_b, y_test_b)
        history = nn1.fit(X_train_b,y_train_b,epochs = epoch_no, verbose=True, validation_data=(X_test_b, y_test_b), batch_size=30 )
        predictions =  nn1.predict(X_test_b) 
        predictions =  np.argmax(predictions, axis=1)
        y_test_b    =  np.argmax(y_test_b, axis=1)
        cohen_score = cohen_kappa_score(y_test_b, predictions)
        dummy.append(cohen_score)
    
    for array in dummy:
        ck.append(np.sum(array)/epoch_no)

    dict_rl = {'trainsize':t_size, 'ck':ck}
    acc_rl_bow = pd.DataFrame(dict_rl)
    #name = 'W5_NN_cohenkappa_{}_{}ephs_recent'. format(str_year, epoch_no)
    #utils.save_as_pickle_file(acc_rl_bow,name, dir)
    return acc_rl_bow


###### Stratified k fold test train and split on 0.8 trainsize ########################################
from sklearn.model_selection import StratifiedKFold

def NN_data_iteration_strat(Neural, X, y,t_size,epoch_no, it_no, k_fold_number, str_dataname,str_featext, str_year, dir):
    
    accuracies     = []
    accuracies_sem = []
    loss           = []
    val_loss       = []
    ck             = []

    accuracies1     = []
    accuracies_sem1 = []
    loss1          = []
    val_loss1       = []


    dummy          = []
    dummy_loss     = []
    dummy_val_loss = []

    skf = StratifiedKFold(n_splits= k_fold_number, shuffle = True, random_state=999)
    
    for train, test in skf.split(X,y.argmax(1)):
        #X_train_b, X_test_b = X[train_idex], X[test_index]
        #y_train_b, y_test_b = y[train_idex], y[test_index]
        
        #X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y , train_size = t_size, shuffle=True)
    
        input = X[train].shape[1]

        for iteration in range(it_no):

            model = Neural(Sequential(), input, epoch_no, X[train], y[train], X[test], y[test])
            history = model.fit(X[train],y[train],epochs = epoch_no, verbose=True, validation_data=(X[test], y[test]), batch_size=30 )

            # dummy_predict.append(predictions)
            dummy.append(history.history['accuracy'])
            dummy_loss.append(history.history['loss'])
            dummy_val_loss.append(history.history['loss'])
            #dummy_ck.append(cohen_kappa_score)

    print(dummy)
    for no in range(len(dummy)):
        accuracies1.append(np.sum(dummy[no])/len(dummy[no]))
        accuracies_sem1.append(np.sum(sem(dummy[no]))/len(dummy[no]))
        loss1.append(np.sum(dummy_loss[no])/len(dummy[no]))
        val_loss1.append(np.sum(dummy_val_loss[no])/len(dummy[no]))


                
    

        accuracies.append(np.sum(accuracies1)/len(accuracies1))
        accuracies_sem.append(np.sum(accuracies_sem1)/len(accuracies_sem1))
        loss.append(np.sum(loss1)/len(loss1))
        val_loss.append(np.sum(val_loss1)/len(val_loss1))

        # if dummy_ck == 0.0:
        #     ck.append(dummy_ck)
        # else:
        #     ck.append(np.sum(dummy_ck))
        
        #print(ck)


    #dict_rl = {'trainsize':t_size, 'accuracy':accuracies, 'sem': accuracies_sem, 'loss':loss, 'valloss': val_loss, 'ck':ck}
    dict_rl = {'trainsize':t_size, 'accuracy':accuracies, 'sem': accuracies_sem, 'loss':loss, 'valloss': val_loss }
    acc_rl_bow = pd.DataFrame(dict_rl)
    #name = 'W5_NN_{}_iteration_{}_{}_{}_{}'. format(it_no, str_year, str_featext,str_dataname,epoch_no)
    #utils.save_as_pickle_file(acc_rl_bow,name, dir)
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
        plt.errorbar(train_size, df['accuracy'][i], yerr = df['sem'][i], linewidth=0.2, capsize=2, c = 'black', markeredgewidth = 0.5)
    plt.xlabel(' Training size')
    plt.ylabel('Average Accuracy score')
    plt.legend( prop={'size': 3})
    #filepath = 'outputs/comparison-{}-2500epochs-overallgraph_new.png'.format(mlmodel)
    #plt.savefig(filepath)
    return 

def nn_graph_loss(train_size, df, mlmodel):

    c = ['r','b','g', 'm'] # colour notation

    for i in range(4):
        # all of the arrays will be of size 4 for training size graphs anyway
        plt.plot(train_size,df['loss'][i], label = '{}-{}-{}-{}'.format(mlmodel,df['feature extraction'][i],df['Label'][i], 'loss'), color =c[i],marker='o', markersize =4 )


    plt.xlabel(' Training size')
    plt.ylabel('Average Loss')
    plt.legend( prop={'size': 3})
    #filepath = 'outputs/comparison-{}-2500epochs-overallgraph_new.png'.format(mlmodel)
    #plt.savefig(filepath)
    return

def nn_graph_general(train_size, df_col, df_ft, df_label, mlmodel, y_axis_string):

    c = ['r','b','g', 'm'] # colour notation

    for i in range(4):
        # all of the arrays will be of size 4 for training size graphs anyway
        plt.plot(train_size,df_col[i], label = '{}-{}-{}-{}'.format(mlmodel,df_ft[i],df_label[i], 'loss'), color =c[i],marker='o', markersize =4 )


    plt.xlabel(' Training size')
    plt.ylabel(y_axis_string)
    plt.legend( prop={'size': 3})
    #filepath = 'outputs/comparison-{}-2500epochs-overallgraph_new.png'.format(mlmodel)
    #plt.savefig(filepath)
    return


##################################### NN INVESTIGATIONS  ##########################################################################################################

#==================================== NUMBER OF NODES ==========================================================
def NN_invest(model1, input1, ephs, X_t, y_t, X_test, y_test, node1, node2, node3):
    maxlen = 100 
    model = model1
    model.add(layers.Dense(node1, input_dim = input, activation  = 'relu'))
    model.add(layers.Dense(node2, input_dim = input, activation  = 'relu'))
    model.add(layers.Dense(node3 ,activation = 'sigmoid'))
    model.add(layers.Flatten())
    model.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics= ['accuracy'])
    model.build(input1)
    model.summary()
    history = model.fit(X_t,y_t,epochs = ephs, verbose=True, validation_data=(X_test, y_test), batch_size=30 )
    return history



def NN_data_invest(X, y, tsize,  epoch_no, node1, node2, node3):
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y , train_size = tsize )
    input = X_train_b.shape[1] 
    nn1 = NN_invest(Sequential(), input, epoch_no, X_train_b, y_train_b, X_test_b, y_test_b, node1, node2, node3)
    return pd.DataFrame(nn1.history)


################################################################### EXTRA CODE ######################################################################
            # predictions =  model.predict(X_test_b) 
            # predictions =  np.argmax(predictions, axis=1)
            # y_test_b    =  np.argmax(y_test_b, axis=1)
            # print('prediction', predictions)
            # print('y_test_b', y_test_b)

            #cohen_score = cohen_kappa_score(y_test_b, predictions)

            # len(x) for x in lst
            #dummy_ck.append(cohen_score)

def NN_opt(model1, input1, ephs, X_t, y_t, X_test, y_test, node1, node2, node3, opt):
    maxlen = 100 
    model = model1
    model.add(layers.Dense(node1, input_dim = input, activation  = 'relu'))
    model.add(layers.Dense(node2, input_dim = input, activation  = 'relu'))
    #model.add(layers.Dense(node2, input_dim = input, activation  = 'relu'))
    model.add(layers.Dense(node3 ,activation = 'sigmoid'))
    model.add(layers.Flatten())
    model.compile(loss= 'binary_crossentropy', optimizer= opt, metrics= ['accuracy'])
    model.build(input1)
    model.summary()
    history = model.fit(X_t,y_t,epochs = ephs, verbose=True, validation_data=(X_test, y_test), batch_size=30 )
    return history


optimizers = ['Adadelta', 'Adagrad', 'Adam', 'RMSprop', 'SGD']
def NN_data_opt(X, y, tsize, epoch_no, node1, node2, node3, opt):
    accuracy = []
    loss     = []
    val_loss = []
    for i in opt:
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y , train_size = tsize)
        input = X_train_b.shape[1] 
        nn1 = NN_opt(Sequential(), input, epoch_no, X_train_b, y_train_b, X_test_b, y_test_b, node1, node2, node3, i)
        accuracy.append(nn1.history['accuracy'])
        loss.append(nn1.history['loss'])
        val_loss.append(nn1.history['val_loss'])
    dictionary = {'accuracy':accuracy, 'loss': loss, 'val_loss':val_loss}
    df = pd.DataFrame(dictionary)
    return df


def NN_batch(model1, input1, ephs, X_t, y_t, X_test, y_test, node1, node2, node3, bsize):
    maxlen = 100 
    model = model1
    model.add(layers.Dense(node1, input_dim = input, activation  = 'relu'))
    model.add(layers.Dense(node2, input_dim = input, activation  = 'relu'))
    model.add(layers.Dense(node3 ,activation = 'sigmoid'))
    model.add(layers.Flatten())
    model.compile(loss= 'binary_crossentropy', optimizer= 'Adam', metrics= ['accuracy'])
    model.build(input1)
    model.summary()
    history = model.fit(X_t,y_t,epochs = ephs, verbose=True, validation_data=(X_test, y_test), batch_size=bsize)
    return history


Batchsize = [16,32,64,128,256]
def NN_data_batch(X, y,tsize, epoch_no, node1, node2, node3, batch_sizes):
    accuracy = []
    loss     = []
    val_loss = []
    times    = []
    for i in batch_sizes:
        
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y , train_size = tsize)
        input = X_train_b.shape[1] 
        start_time = time()
        nn1 = NN_batch(Sequential(), input, epoch_no, X_train_b, y_train_b, X_test_b, y_test_b, node1, node2, node3, i)
        accuracy.append(nn1.history['accuracy'])
        loss.append(nn1.history['loss'])
        val_loss.append(nn1.history['val_loss'])
        times.append(time()- start_time)

    dictionary = {'accuracy':accuracy, 'loss': loss, 'val_loss':val_loss, 'batch': batch_sizes, 'times':times}
    df = pd.DataFrame(dictionary)
    return df

####################################################################### LEARNING RATE ##########################################
# have to build new NN as I do not want to mix keras and tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical

def NN_opt_lr(model1, input1, ephs, X_t, y_t, X_test, y_test, node1, node2, node3,lrate):
    maxlen = 100 
    model = model1
    model.add(Dense(node1, input_dim = input1, activation  = 'relu'))
    model.add(Dense(node2, input_dim = input1, activation  = 'relu'))
    model.add(Dense(node3,  activation = 'sigmoid'))
    model.add(layers.Flatten())
    opt = Adam(learning_rate= lrate)
    model.compile(loss= 'binary_crossentropy', optimizer= opt, metrics= ['accuracy'])
    model.build(input1)
    model.summary()
    history = model.fit(X_t,y_t,epochs = ephs, verbose=True, validation_data=(X_test, y_test), batch_size=30 )

    return history

########################################################### GRID SEARCH OPT ##########################################
def NN_tune(Optimizer_Trial, Neurons_Trial):
    maxlen = 100 
    model = Sequential()
    model.add(layers.Dense(Neurons_Trial, input_dim = input, activation  = 'relu'))
    model.add(layers.Dense(Neurons_Trial, input_dim = input, activation  = 'relu'))
    model.add(layers.Dense(1 ,activation = 'sigmoid'))
    model.add(layers.Flatten())
    model.compile(loss= 'binary_crossentropy', optimizer= Optimizer_Trial, metrics= ['accuracy'])
    #model.build(input1)
    #model.summary()
    #history = model.fit(X_t,y_t,epochs = ephs, verbose=True, validation_data=(X_test, y_test), batch_size=bsize)
    return model
    


Batchsize = [16,32,64,128,256]
def NN_data_batch2(X, y,tsize, epoch_no, node1, node2, node3, batch_sizes):
    accuracy = []
    loss     = []
    val_loss = []
    for i in batch_sizes:
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y , train_size = tsize)
        input = X_train_b.shape[1] 
        nn1 = NN_batch(Sequential(), input, epoch_no, X_train_b, y_train_b, X_test_b, y_test_b, node1, node2, node3, i)
        accuracy.append(nn1.history['accuracy'])
        loss.append(nn1.history['loss'])
        val_loss.append(nn1.history['val_loss'])
    dictionary = {'accuracy':accuracy, 'loss': loss, 'val_loss':val_loss, 'batch': batch_sizes}
    df = pd.DataFrame(dictionary)
    return df

def NN_data_iteration_investigation(X, y,tsize, epoch_no,it_no, ts):
    accuracy = []
    loss     = []
    val_loss = []
    accuracy_sem = []

    dummy= []
    dummy_loss     = []
    dummy_val_loss = []
    
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y , train_size = ts, shuffle=True)
    input = X_train_b.shape[1]
    for iteration in range(it_no):
        model = NN_default_parameters(Sequential(), input, epoch_no, X_train_b, y_train_b, X_test_b, y_test_b)
        history = model.fit(X_train_b,y_train_b,epochs = epoch_no, verbose=True, validation_data=(X_test_b, y_test_b), batch_size=30 )

        dummy.append(history.history['accuracy'])
        dummy_loss.append(history.history['loss'])
        dummy_val_loss.append(history.history['loss'])


    for no in range(len(dummy)):
        accuracy.append(np.sum(dummy[no])/len(dummy[no]))
        accuracy_sem.append(np.sum(sem(dummy[no]))/len(dummy[no]))
        loss.append(np.sum(dummy_loss[no])/len(dummy[no]))
        val_loss.append(np.sum(dummy_val_loss[no])/len(dummy[no]))

    dictionary = {'accuracy':accuracy, 'sem':accuracy_sem, 'loss': loss, 'val_loss':val_loss}
    df = pd.DataFrame(dictionary)
    return df