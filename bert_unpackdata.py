import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science','ieee','no-latex'])


# ****Path****
fldr = 'bert_argumentlevel/year2' 


# ****Classifier Parameters****
# set epochs
epochs = 4
# set batch size
batch_size = 16
# set random seed
seed_val = 42
# set learnning rate
lr = 1e-4   # args.learning rate - default is 5e-5

# ****Training Parameters****
# set the number of loops 
run_num = 3
# set training data ratio
ratios = [0.5,0.6,0.7,0.8,0.9]

avg_list, seom_list = [], []

for r in ratios:

    # unpack dataframes
    df = pd.read_pickle(r'./Pickledfiles/'+fldr+'/'+str(r)+'trainratio_'+str(run_num)+'runs_'+str(epochs)+'epochs_'+str(batch_size)+'batch_size_'+str(lr)+'lr_'+str(seed_val)+'seed_val_.pkl')
    # df = df.iloc[[3]] # the 3rd row only
    df = df[3::4] # start from the 3rd row, get subsequent 4th row
    # print(df)
    accy_list = (df['Valid. Accur.'].tolist())

    # append average and standard error of mean to lists
    avg_list.append(np.average(accy_list))
    seom_list.append( np.std(accy_list, ddof=1) / np.sqrt(np.size(accy_list)) ) 

print(avg_list)
print(seom_list)




plt.figure()
plt.scatter(ratios, avg_list, c=(0, 0, 0, 1))
plt.errorbar(ratios, avg_list, yerr= seom_list, color=(0, 0, 0, 1),fmt='none', capsize = 2)
plt.plot(ratios, avg_list, linestyle='dashed', color=(0.3, 0.3, 0.45 ,.4))
plt.xlabel('Training size')
plt.ylabel('Accuracy score')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.savefig('./figures/'+fldr+'/'+str(run_num)+'runs_'+str(epochs)+'epochs_'+str(batch_size)+'batch_size_'+str(lr)+'lr_'+str(seed_val)+'seed_val_.png')
plt.show()







"""

lr_list

fldr = 'bert_argumentlevel/year2/'
run_num_list = np.arange(1,21,1)    #[1,2,3,4,...,20]

avg_list = []
seom_list = []
for run_num in run_num_list:
    df = pd.read_pickle('./Pickledfiles/'+fldr+'0.9trainratio_'+str(run_num)+'runs_1epochs_16batch_size_0.0001lr_42seed_val_.pkl')
    accy_list = df['Valid. Accur.'].tolist() 
    

    # append average and standard error of mean to lists
    avg_list.append(np.average(accy_list))
    seom_list.append( np.std(accy_list, ddof=1) / np.sqrt(np.size(accy_list)) ) 
seom_list = [0 if math.isnan(x) else x for x in seom_list]
print(avg_list, seom_list)


plt.figure()
plt.scatter(run_num_list, avg_list, c=(0, 0, 0, 1))
plt.errorbar(run_num_list, avg_list, yerr= seom_list, color=(0, 0, 0, 1),fmt='none', capsize = 2)
plt.plot(run_num_list, avg_list, linestyle='dashed', color=(0.3, 0.3, 0.45 ,.4))
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.savefig('./figures/'+fldr+'vary_run_num_0.9trainratio_'+str(run_num)+'runs_1epochs_16batch_size_0.0001lr_42seed_val_.png')
plt.show()
"""


"""

# ============ vary lr ===========
fldr = 'bert_argumentlevel/year2/'

lr_list = [1e0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]

avg_list = []
seom_list = []
for lr in lr_list:
    df = pd.read_pickle('./Pickledfiles/'+fldr+'0.9trainratio_2runs_4epochs_16batch_size_'+str(lr)+'lr_42seed_val_.pkl')
    accy_list = (df.loc[4])['Valid. Accur.'].tolist() 
    
    # append average and standard error of mean to lists
    avg_list.append(np.average(accy_list))
    seom_list.append( np.std(accy_list, ddof=1) / np.sqrt(np.size(accy_list)) ) 
print(avg_list, seom_list)


plt.figure()
plt.scatter(np.log(lr_list), avg_list, c=(0, 0, 0, 1))
plt.errorbar(np.log(lr_list), avg_list, yerr= seom_list, color=(0, 0, 0, 1),fmt='none', capsize = 2)
plt.plot(np.log(lr_list), avg_list, linestyle='dashed', color=(0.3, 0.3, 0.45 ,.4))
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Accuracy')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.savefig('./figures/'+fldr+'vary_lr_0.9trainratio_2runs_4epochs_16batch_size_'+str(lr)+'lr_42seed_val_.png')
plt.show()

"""

"""
# ============ vary batch_size ===========

batch_size_list = [2,4,8,16]
fldr = 'bert_argumentlevel/year2/'

avg_list = []
seom_list = []
for batch_size in batch_size_list:
    df = pd.read_pickle('./Pickledfiles/'+fldr+'0.9trainratio_2runs_4epochs_'+str(batch_size)+'batch_size_1e-05lr_42seed_val_.pkl')
    accy_list = (df.loc[4])['Valid. Accur.'].tolist() 
    
    # append average and standard error of mean to lists
    avg_list.append(np.average(accy_list))
    seom_list.append( np.std(accy_list, ddof=1) / np.sqrt(np.size(accy_list)) ) 
print(avg_list, seom_list)


plt.figure()
plt.scatter(batch_size_list, avg_list, c=(0, 0, 0, 1))
plt.errorbar(batch_size_list, avg_list, yerr= seom_list, color=(0, 0, 0, 1),fmt='none', capsize = 2)
plt.plot(batch_size_list, avg_list, linestyle='dashed', color=(0.3, 0.3, 0.45 ,.4))
plt.xlabel('Batch size')
plt.ylabel('Accuracy')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.savefig('./figures/'+fldr+'vary_batch_size_0.9trainratio_2runs_4epochs_'+str(batch_size)+'batch_size_1e-05lr_42seed_val_.png')
plt.show()

"""