#%%
import matplotlib.pyplot as plt
import ml_tools
import time
plt.style.use('seaborn-notebook')
#plt.style.available

#%%
def bar_plot(xvalue = [0,0], yvalue = [0,0], xlabel = None, ylabel = None, filepath = 'outputs/example.png'):
    plt.figure()
    plt.bar(xvalue, yvalue, color=(0.3, 0.3, 0.45 ,.4), edgecolor=(0, 0, 0, 1))
    # plt.grid(ls= ':', color='#6e6e6e', lw=0.5);
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filepath)
    return


def scatter_plot(xvalue = [0,0], yvalue = [0,0], yerr = [0,0], xlabel = None, ylabel = None, filepath = 'outputs/example.png'):
    plt.figure()
    plt.scatter(xvalue, yvalue, c=(0, 0, 0, 1), s=25)
    plt.errorbar(xvalue, yvalue, yerr, color=(0, 0, 0, 1),fmt='none', capsize = 3)
    plt.plot(xvalue, yvalue, linestyle='dashed', color=(0.3, 0.3, 0.45 ,.4))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filepath)
    return


#=============================================================================================================================================
# GENERAL plotting function
#=============================================================================================================================================

def lr_accuracy_trainsize_plot_general(classifier, x, y, label, feature, num_epochs, train_sizes):
    accuracies = []
    accuracies_sd = []
    for size in train_sizes:
        sum = 0
        start_time = time.time()

        dummy = []
        for epoch in range(num_epochs):
            # split train/test data
            X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = size)
            # train model + prediction
            y_test_predict = classifier(X_train, X_test, y_train, y_test)

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
    scatter_plot(xvalue = train_sizes, yvalue = accuracies, yerr = accuracies_sd, xlabel = 'Training Size', ylabel = 'Accuracy', filepath = filepath)
    return
