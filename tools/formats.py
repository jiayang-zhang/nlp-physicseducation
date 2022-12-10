import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import pandas as pd
#plt.style.use(['science', 'ieee'])
plt.style.use('seaborn-notebook')
plt.style.available



def bar_plot(xvalue = [0,0], yvalue = [0,0], xlabel = None, ylabel = None, filepath = 'outputs/example.png'):
    ax = plt.figure().gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.bar(xvalue, yvalue, color=(0.3, 0.3, 0.45 ,.4), edgecolor=(0, 0, 0, 1))
    # plt.grid(ls= ':', color='#6e6e6e', lw=0.5);
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.figure.savefig(filepath)
    return


def scatter_plot(xvalue = [0,0], yvalue = [0,0], yerr = [0,0], xlabel = None, ylabel = None, filepath = 'outputs/example.png'):
    plt.figure()
    plt.scatter(xvalue, yvalue, c=(0, 0, 0, 1), s=25)
    plt.errorbar(xvalue, yvalue, yerr, color=(0, 0, 0, 1),fmt='none', capsize = 3)
    plt.plot(xvalue, yvalue, linestyle='dashed', color=(0.3, 0.3, 0.45 ,.4))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig(filepath)
    return


def histogram(xvalue, xlabel, ylabel, file_path,size):
    plt.figure()
    plt.hist(xvalue)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(file_path)
    return

