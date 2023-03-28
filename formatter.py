import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib import rc, rcParams
import pandas as pd


def three_figures(x_0, y_0, yerr_0, 
                  x_1, y_1, yerr_1,   
                  x_2, y_2, yerr_2, 
                  filepath):
    
    plt.style.use(['science', 'ieee', 'no-latex'])
    plt.rcParams['font.family'] = "Arial"
    rc('font', weight='normal')

    # Create a figure with three subplots in a row
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (9,2)) #, figsize=(12, 4)

    # Plot the first subplot on the left
    axes[0].scatter(x_0, y_0, color = '#A20346', marker = 's', s= 10, label = 'red')
    axes[0].errorbar(x_0, y_0, yerr_0, color='#A20346', fmt='none')
    axes[0].plot(x_0, y_0, linestyle='solid', color='#A20346', alpha=0.5)
    axes[0].set_xlabel('Training ratio', fontsize=10)
    axes[0].set_ylabel('Accuracy', fontsize=10)
    axes[0].xaxis.set_ticks(np.arange(0.3, 1.0, 0.1))
    axes[0].yaxis.set_ticks(np.arange(0.2, 1.1, 0.1))
    axes[0].minorticks_off()
    axes[0].set_title('(a)')


    # Plot the second subplot in the middle
    axes[1].scatter(x_1, y_1, color = '#004E7E', marker = 's', s= 10, label = 'blue')
    axes[1].errorbar(x_1, y_1, yerr_1, color='#004E7E', fmt='none')
    axes[1].plot(x_1, y_1, linestyle='solid', color='#004E7E', alpha=0.5)
    axes[1].set_xlabel('Training ratio', fontsize=10)
    axes[1].set_ylabel('Accuracy', fontsize=10)
    axes[1].xaxis.set_ticks(np.arange(0.3, 1.0, 0.1))
    axes[1].yaxis.set_ticks(np.arange(0.2, 1.1, 0.1))
    axes[1].minorticks_off()
    axes[1].set_title('(b)')


    # Plot the third subplot on the right
    axes[2].scatter(x_2, y_2, color = '#3D4E1D', marker = 's', s= 10, label = 'red')
    axes[2].errorbar(x_2, y_2, yerr_2, color='#3D4E1D', fmt='none')
    axes[2].plot(x_2, y_2, linestyle='solid', color='#3D4E1D', alpha=0.5)
    axes[2].set_xlabel('Training ratio', fontsize=10) # , fontweight = 'bold'
    axes[2].set_ylabel('Accuracy', fontsize=10) # fontweight = 'bold'
    axes[2].xaxis.set_ticks(np.arange(0.2, 1.0, 0.1))
    axes[2].yaxis.set_ticks(np.arange(0.2, 1.1, 0.1))
    axes[2].minorticks_off()
    axes[2].set_title('(c)')


    # Add a shared y-axis label to the leftmost plot
    # fig.text(0.06, 0.5, 'Amplitude', ha='center', va='center', rotation='vertical')
    # Add a shared x-axis label to the bottom plot
    # fig.text(0.5, 0.06, 'Time', ha='center', va='center')

    # Adjust the spacing between the subplots
    plt.subplots_adjust(wspace=0.4)

    plt.savefig(filepath, dpi = 1000)
    
    # Show the plot
    plt.show()
    
    return




def three_plots_nine_lines(

    # FIRST PLOT 
    x1_plot1, y1_plot1, y1err_plot1,
    x2_plot1, y2_plot1, y2err_plot1,
    x3_plot1, y3_plot1, y3err_plot1,
    
    # # SECOND PLOT
    x1_plot2, y1_plot2, y1err_plot2,
    x2_plot2, y2_plot2, y2err_plot2,
    x3_plot2, y3_plot2, y3err_plot2,
    
    # # THIRD PLOT
    x1_plot3, y1_plot3, y1err_plot3,
    x2_plot3, y2_plot3, y2err_plot3,
    x3_plot3, y3_plot3, y3err_plot3,
    
    filepath, 
    
):
    # colour
    # red = #A20346
    # blue = #004E7E
    # yellow = #F27C0A
    # green = #3D4E1D
    # purple = #2E206B
    # tur = #006D7D
    
    color1 = '#004E7E'
    color2 = '#A20346'
    color3 = '#F27C0A'
    
    markersize = 3


    plt.style.use(['science', 'ieee', 'no-latex'])
    plt.rcParams['font.family'] = "Arial"
    rc('font', weight='normal')
    

    # Create a figure with three subplots in a row
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (9,2)) #, figsize=(12, 4)


    # Plot the first subplot on the left
    axes[0].plot(x1_plot1, y1_plot1 , color= color1, marker = 's', label = 'Y1', markersize=markersize, linestyle = 'solid')
    axes[0].errorbar(x1_plot1, y1_plot1 , yerr = y1err_plot1, color=color1, fmt='none')

    axes[0].plot(x2_plot1, y2_plot1 , color= color2, marker = '^', label = 'Y2', markersize=markersize, linestyle = 'dotted')
    axes[0].errorbar(x2_plot1, y2_plot1 , yerr = y2err_plot1, color=color2, fmt='none')

    axes[0].plot(x3_plot1, y3_plot1 , color= color3, marker = 'o', label = 'Y1&Y2', markersize=markersize, linestyle = 'dashdot')
    axes[0].errorbar(x3_plot1, y3_plot1 , yerr = y3err_plot1, color=color3, fmt='none')

    axes[0].set_ylabel('Accuracy', fontsize=10)
    axes[0].xaxis.set_ticks(np.arange(0.3, 1.0, 0.1))
    axes[0].yaxis.set_ticks(np.arange(0.2, 1.1, 0.1))
    axes[0].minorticks_off()
    legend = axes[0].legend( loc='upper left', framealpha=1, frameon=True, edgecolor = 'black', prop = {'size' : 7})
    legend.get_frame().set_linewidth(0.2)


    # Plot the second subplot in the middle
    axes[1].plot(x1_plot2,y1_plot2 , color = '#004E7E', marker = 's', label = 'Y1', markersize=markersize, linestyle = 'solid')
    axes[1].errorbar(x1_plot2, y1_plot2, yerr= y1err_plot2, color= '#004E7E', fmt='none')

    axes[1].plot(x2_plot2, y2_plot2 , color= color2, marker = '^', label = 'Y2', markersize=markersize, linestyle = 'dotted')
    axes[1].errorbar(x2_plot2, y2_plot2, yerr = y2err_plot2, color=color2, fmt='none')

    axes[1].plot(x3_plot2, y3_plot2 , color= color3, marker = 'o', label = 'Y1&Y2', markersize=markersize, linestyle = 'dashdot')
    axes[1].errorbar(x3_plot2, y3_plot2 , yerr = y3err_plot2, color=color3, fmt='none')

    axes[1].set_xlabel('Training ratio', fontsize=10)
    axes[1].xaxis.set_ticks(np.arange(0.3, 1.0, 0.1))
    axes[1].yaxis.set_ticks(np.arange(0.2, 1.1, 0.1))
    axes[1].minorticks_off()

    legend = axes[1].legend( loc='upper left', framealpha=1, frameon=True, edgecolor = 'black', prop = {'size' : 7})
    legend.get_frame().set_linewidth(0.2)



    # # Plot the third subplot on the right
    # axes[2].scatter(x3value_y1, y3value_y1 , color = blu , marker = 's', s= 10)
    axes[2].plot(x1_plot3,y1_plot3 , color = '#004E7E', marker = 's', label = 'Y1', markersize=markersize, linestyle = 'solid')
    axes[2].errorbar(x1_plot3, y1_plot3, yerr= y1err_plot3, color= '#004E7E', fmt='none')

    axes[2].plot(x2_plot3, y2_plot3 , color= color2, marker = '^', label = 'Y2', markersize=markersize, linestyle = 'dotted')
    axes[2].errorbar(x2_plot3, y2_plot3, yerr = y2err_plot3, color=color2, fmt='none')

    axes[2].plot(x3_plot3, y3_plot3, color= color3, marker = 'o', label = 'Y1&Y2', markersize=markersize, linestyle = 'dashdot')
    axes[2].errorbar(x3_plot3, y3_plot3, yerr = y3err_plot3, color=color3, fmt='none')

    axes[2].xaxis.set_ticks(np.arange(0.2, 1.0, 0.1))
    axes[2].yaxis.set_ticks(np.arange(0.2, 1.1, 0.1))
    axes[2].minorticks_off()

    legend = axes[2].legend( loc='upper left', framealpha=1, frameon=True, edgecolor = 'black', prop = {'size' : 7})
    legend.get_frame().set_linewidth(0.2)


    # Adjust the spacing between the subplots
    # plt.subplots_adjust(wspace=0.4)
    plt.savefig(filepath, dpi = 1000)

    # Show the plotgit 
    plt.show()
    
    
    
def four_figures(x_0, y_0, yerr_0, 
                 x_1, y_1, yerr_1,   
                 x_2, y_2, yerr_2, 
                 x_3, y_3, yerr_3, 
                 filepath):
    

    plt.style.use(['science', 'ieee', 'no-latex'])
    plt.rcParams['font.family'] = "Arial"
    rc('font', weight='normal')


    # Create a figure with three subplots in a row
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize = (12.5,2.15)) 


    # Plot the first subplot on the left
    axes[0].scatter(x_0, y_0, color = '#A20346', marker = 's', s= 10, label = 'red')
    axes[0].errorbar(x_0, y_0, yerr_0, color='#A20346', fmt='none')
    axes[0].plot(x_0, y_0, linestyle='solid', color='#A20346', alpha=0.5)
    # axes[0].set_xlabel('Training ratio', fontsize=10)
    axes[0].set_ylabel('Accuracy', fontsize=10)
    axes[0].xaxis.set_ticks(np.arange(0.5, 1.0, 0.1))
    axes[0].yaxis.set_ticks(np.arange(0.2, 1.1, 0.1))
    axes[0].minorticks_off()


    # Plot the second subplot in the middle
    axes[1].scatter(x_1, y_1, color = '#004E7E', marker = 's', s= 10, label = 'blue')
    axes[1].errorbar(x_1, y_1, yerr_1, color='#004E7E', fmt='none')
    axes[1].plot(x_1, y_1, linestyle='solid', color='#004E7E', alpha=0.5)
    # axes[1].set_xlabel('Training ratio', fontsize=10)
    # axes[1].set_ylabel('Accuracy', fontsize=10)
    axes[1].xaxis.set_ticks(np.arange(0.5, 1.0, 0.1))
    axes[1].yaxis.set_ticks(np.arange(0.2, 1.1, 0.1))
    axes[1].minorticks_off()


    # Plot the third subplot on the right
    axes[2].scatter(x_2, y_2, color = '#3D4E1D', marker = 's', s= 10, label = 'red')
    axes[2].errorbar(x_2, y_2, yerr_2, color='#3D4E1D', fmt='none')
    axes[2].plot(x_2, y_2, linestyle='solid', color='#3D4E1D', alpha=0.5)
    # axes[2].set_xlabel('Training ratio', fontsize=10) # , fontweight = 'bold'
    # axes[2].set_ylabel('Accuracy', fontsize=10) # fontweight = 'bold'
    axes[2].xaxis.set_ticks(np.arange(0.5, 1.0, 0.1))
    axes[2].yaxis.set_ticks(np.arange(0.2, 1.1, 0.1))
    axes[2].minorticks_off()

    # Plot the fourth subplot on the right
    axes[3].scatter(x_3, y_3, color = '#3D4E1D', marker = 's', s= 10, label = 'red')
    axes[3].errorbar(x_3, y_3, yerr_3, color='#3D4E1D', fmt='none', )
    axes[3].plot(x_3, y_3, linestyle='solid', color='#3D4E1D', alpha=0.5)
    # axes[3].set_xlabel('Training ratio', fontsize=10) # , fontweight = 'bold'
    # axes[3].set_ylabel('Accuracy', fontsize=10) # fontweight = 'bold'
    axes[3].xaxis.set_ticks(np.arange(0.5, 1.0, 0.1))
    axes[3].yaxis.set_ticks(np.arange(0.2, 1.1, 0.1))
    axes[3].minorticks_off()



    # Add a shared y-axis label to the leftmost plot
    # fig.text(0.06, 0.5, 'Amplitude', ha='center', va='center', rotation='vertical')
    # Add a shared x-axis label to the bottom plot
    fig.text(0.5, 0.06, 'Training ratio', ha='center', va='center')

  
    # Adjust the spacing between the subplots
    plt.subplots_adjust(wspace=0.3)


    plt.savefig(filepath, dpi = 1000)
    
    # Show the plot
    plt.show()
    
    return