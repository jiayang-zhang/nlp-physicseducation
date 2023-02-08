#%%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science','ieee','no-latex'])

# ****Path****
fldr = 'bert_argumentlevel/year1year2' 


# ****Classifier Parameters****
# set epochs
epochs = 2
# set batch size
batch_size = 16
# set random seed
seed_val = 42
# set learnning rate
lr = 1e-5   # args.learning rate - default is 5e-5



# ****Training Parameters****
# set the number of loops 
run_num = 5
ratios = [0.5,0.6,0.7,0.8,0.9]
avg_list, seom_list = [], []

for r in ratios:

    # unpack dataframes
    df = pd.read_pickle(r'./Pickledfiles/'+fldr+'/'+str(r)+'trainratio_'+str(run_num)+'runs_'+str(epochs)+'epochs_'+str(batch_size)+'batch_size_'+str(lr)+'lr_'+str(seed_val)+'seed_val_.pkl')
    # df = df.iloc[[3]] # the 3rd row only
    df = df[3::4] # start from the 3rd row, get subsequent 4th row
    accy_list = (df['Valid. Accur.'].tolist())

    # append average and standard error of mean to lists
    avg_list.append(np.average(accy_list))
    seom_list.append( np.std(accy_list, ddof=1) / np.sqrt(np.size(accy_list)) ) 

print(avg_list)
print(seom_list)

#%%%
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
%%
