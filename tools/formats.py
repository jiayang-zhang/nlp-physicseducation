import matplotlib.pyplot as plt
#plt.style.use('ieee')
plt.style.available

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
