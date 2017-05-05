# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#!/usr/bin/env python
import os
import collections
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from itertools import cycle
import sys

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

#  =================
#  = Specify files =
#  =================

# Location of formatted experimental & FDS data file directories
data_dir = '../FDS_Output_Files/Sensitivity_Analysis/'
OG_data_dir = '../FDS_Output_Files/Official_Repo_Files/'

# Location of file with timing information
all_times_file = '../Experimental_Data/Plot_Times.csv'

# Load exp. timings and description file
all_times = pd.read_csv(all_times_file)
all_times = all_times.set_index('Time')

#  =================
#  = Specify plots =
#  =================

# Specify structure
specify_struct  = False
specific_struct = 'West'

# Specify plots to generate
plot_CO     = False
plot_CO2    = False 
plot_O2     = True   
plot_TC     = False 
plot_HF     = False 
plot_BDP    = False 
plot_HGL    = False     # [STILL NEED TO ADD CAPABILITY]
plot_cjetTC = True      # Ceiling jet temperature = upper most TC 
plot_OG     = False     # 55 cells per mesh in y direction vs. 10 cm -> 40 cells

# Duration of pre-test time
pre_test_time = 0

# Location to save/output figures
save_dir = '../Plots/Grid_Sensitivity/'

#  ==========================
#  = User defined functions =
#  ==========================

# # Prints an error message and stops code
def error_message(message):
    lineno = inspect.currentframe().f_back.f_lineno
    print('[ERROR, line '+str(lineno)+']:')
    print('  ' + message)
    sys.exit()

def setup_fig(color_list, y_label, x_max):
    fig = plt.figure()
    plt.rc('axes', color_cycle=color_list)
    ax1 = plt.gca()
    plt.ylabel(y_label, fontsize=20)
    ax1.set_xlim(0,x_max)
    plt.xlabel('Time (s)', fontsize=20)
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    return ax1

#  ==========================
#  = Loop through directory =
#  ==========================
if specify_struct:
    if specific_struct == 'East':
        test_ls = ['Test_04']
    elif specific_struct == 'West':
        test_ls = ['Test_25']
    else:
        error_message('Invalid name for specific_struct.')
else:
    test_ls = ['Test_04','Test_25']

for test in test_ls:
    grid_L = pd.read_csv(data_dir+test+'_14cm/'+test+'_14cm_devc.csv',
        skiprows=1, index_col='Time')
    grid_M = pd.read_csv(OG_data_dir+test+'_devc.csv',
        skiprows=1, index_col='Time')
    if test == 'Test_04':
        S_size = '_5cm'
    else:
        S_size = '_7cm'

    grid_S = pd.read_csv(data_dir+test+S_size+'/'+test+S_size+'_devc.csv',
        skiprows=1, index_col='Time')

    print ('--- Loaded Different Grid Size FDS Files for '+test+' ---')

    # Make list of column header prefixes corresponding to plot data
    plot_types = []
    if plot_CO:
        plot_types.append('CO_')
    if plot_CO2:
        plot_types.append('CO2_')
    if plot_O2:
        plot_types.append('O2_')
    if plot_TC:
        plot_types.append('TC_')
    if plot_HF:
        plot_types.append('HF_')
    if plot_BDP:
        plot_types.append('BDP_')

    # iterate through exp data column headers and generate desired plots
    plotted_groups = []
    x_max = grid_M.index.values.astype(float)[-1]

    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    for column in grid_M:
        # Check if column's data type should be plotted
        if any([substring in column for substring in plot_types]):     
            # Check if channel's sensor group has been plotted
            if any([substring in column for substring in plotted_groups]):
                continue

            legend = True

            # Add legend/timing information later
            if 'TC_' in column:
                group = column[:-2]
                channel_ls = []
                for i in range(1,9):
                    channel_name = group+'_'+str(i)
                    try: 
                        df = grid_M[channel_name]
                        channel_ls.append(channel_name)
                    except KeyError:
                        continue
                plotted_groups.append(group+'_') 
                data_type = 'Temperature'
                plot_colors = tableau20[:len(channel_ls)]
                y_label = r'Temperature ($^\circ$C)'
                legend = False

            elif 'BDP_' in column:
                group = column[:-2]
                channel_ls = []
                for i in range(1,9):
                    channel_name = group+'_'+str(i)
                    try: 
                        df = grid_M[channel_name]
                        channel_ls.append(channel_name)
                    except KeyError:
                        continue
                plotted_groups.append(group+'_') 
                data_type = 'Velocity' 
                plot_colors = tableau20[:len(channel_ls)]
                y_label = r'Velocity (m/s)'
                legend = False

            elif 'HF_' in column:
                group = column
                channel_ls = [column, 'RAD'+column[2:]]
                plotted_groups.append(column)
                plotted_groups.append('RAD'+column[2:])
                data_type = 'Heat_Flux' 
                plot_colors = tableau20[:len(channel_ls)]
                y_label = r'Heat Flux (kW/m$^2$)'
                leg_loc = 'upper right'

            elif 'RAD_' in column:
                group = 'HF'+column[3:]
                channel_ls = [group, column]
                plotted_groups.append(column)
                plotted_groups.append('HF'+column[3:])
                data_type = 'Heat_Flux' 
                plot_colors = tableau20[:len(channel_ls)]
                y_label = r'Heat Flux (kW/m$^2$)'
                leg_loc = 'upper right'

            elif 'CO_' in column:
                group = 'CO'
                channel_ls = ['CO_A', 'CO_B']
                plotted_groups.append('CO_')
                data_type = 'Gas_Concentration'
                plot_colors = ['0.3',tableau20[6]]
                y_label = 'Volume Fraction'
                leg_loc = 'upper right'

            elif 'CO2_' in column:
                group = 'CO2'
                channel_ls = ['CO2_A', 'CO2_B']
                plotted_groups.append('CO2_')
                data_type = 'Gas_Concentration'
                plot_colors = ['0.3',tableau20[6]]
                y_label = 'Volume Fraction'
                leg_loc = 'upper right'

            elif 'O2_' in column:
                group = 'O2'
                channel_ls = ['O2_A', 'O2_B']
                plotted_groups.append('O2_')
                data_type = 'Gas_Concentration'
                plot_colors = ['0.3',tableau20[6]]
                y_label = 'Volume Fraction'
                leg_loc = 'lower right'

            ax1 = setup_fig(plot_colors,y_label,x_max)

            if group == 'O2':
                ax1.set_ylim(0,0.23)
                legend_loc = 'lower right'
            else:
                legend_loc = 'upper right'

            for name in channel_ls:
                plt.plot(grid_L.index.values.astype(float), 
                    grid_L[name].values.astype(float),
                    marker='s', markevery=int(x_max/100), mew=2, mec='none', ms=10, 
                    ls='--', lw=2, label=name+'(14 cm Grid)')
                print ('    Plotting '+name+' (14 cm Grid) Data')

            for name in channel_ls:
                plt.plot(grid_M.index.values.astype(float), 
                    grid_M[name].values.astype(float),
                    marker='o', markevery=int(x_max/100), mew=2, mec='none', ms=10, 
                    ls='-', lw=2, label=name+'(10 cm Grid)')
                print ('    Plotting '+name+' (10 cm Grid) Data')               

            for name in channel_ls:
                plt.plot(grid_S.index.values.astype(float), 
                    grid_S[name].values.astype(float),
                    marker='^', markevery=int(x_max/100), mew=2, mec='none', ms=10, 
                    ls=':', lw=2, label=name+' ('+S_size[1:]+' Grid)')
                print ('    Plotting '+name+'  ('+S_size[1:]+' Grid) Data')
            
            plt.grid(color='0.75', linestyle='-.', linewidth=1)

            # # Add vertical lines and labels for timing information (if available)
            # ax3 = ax1.twiny()
            # ax1_xlims = ax1.axis()[0:2]
            # ax3.set_xlim(ax1_xlims)
            # # Remove NaN items from event timeline
            # events = all_times[test].dropna()
            # # Ignore events that are commented starting with a pound sign
            # events = events[~events.str.startswith('#')]
            # [plt.axvline(_x, color='0.4', lw=1) for _x in events.index.values]
            # ax3.set_xticks(events.index.values)
            # plt.setp(plt.xticks()[1], rotation=60)
            # ax3.set_xticklabels(events.values, fontsize=10, ha='left')
            # plt.xlim([0, x_max])

            handles1, labels1 = ax1.get_legend_handles_labels()
            plt.legend(handles1, labels1, loc=legend_loc, fontsize=10, handlelength=3)
            # Save plot to file
            print ('    Saving ' + group + ' figure')
            print
            plt.savefig(save_dir+data_type+'/'+test+ '_' +group+'.pdf')
            plt.close('all')

            if plot_OG:
                ax1 = setup_fig(plot_colors,y_label,x_max)

                if y_label == 'Volume Fraction':
                    ax1.set_ylim(0,0.25)

                for name in channel_ls:
                    plt.plot(grid_OG.index.values.astype(float), 
                        grid_OG[name].values.astype(float),
                        marker='s', markevery=int(x_max/100), mew=2, mec='none', ms=10, 
                        ls='--', lw=1.5, label=name+'(OG Grid)')
                    print ('    Plotting '+name+' (OG Grid) Data')

                for name in channel_ls:
                    plt.plot(grid_M.index.values.astype(float), 
                        grid_M[name].values.astype(float),
                        marker='o', markevery=int(x_max/100), mew=1.5, mec='none', ms=10, 
                        ls='-', lw=1.5, label=name+'(10 cm Grid)')
                    print ('    Plotting '+name+' (10 cm Grid) Data')
                
                if legend:              
                    handles1, labels1 = ax1.get_legend_handles_labels()
                    plt.legend(handles1, labels1, loc=leg_loc, fontsize=10, handlelength=3)

                plt.grid(True)
                # Save plot to file
                print ('    Saving ' + group + ' OG and M comparison Figure')
                print
                plt.savefig(save_dir+data_type+'/'+test+ '_' +group+ 'OG_vs_M.pdf')
                plt.close('all')
        else:
            continue

    if plot_cjetTC:
        # set plot colors 
        plot_colors = ['0.3','0.3','0.3',tableau20[6],tableau20[6],tableau20[6],tableau20[0],tableau20[0],tableau20[0]]

        # Set array of TCs to plot (each row corresponds to a figure)
        if test == 'Test_04':
            channel_array = np.array([['TC_A1_1', 'TC_A3_1', 'TC_A5_1'],['TC_A2_1', 'TC_A4_1', 'None']]) 
        else:
            channel_array = np.array([['TC_A1_1', 'TC_A2_1', 'TC_A3_1'],['TC_A7_1', 'TC_A8_1', 'TC_A9_1']])

        exp_error = 0.15
        FDS_error = 0.07

        for i in range(0,channel_array.shape[0]):
            # Setup figure to plot set of TCs
            ax1 = setup_fig(plot_colors,r'Temperature ($^\circ$C)',x_max)
            for j in range(0, channel_array.shape[1]):
                name = channel_array[i,j]
                if name != 'None':
                    # if i == 0 and test == 'Test_25':
                    #     L_name = name[:-1]+'2'                    
                    # else:
                    #     L_name = name

                    plt.plot(grid_L.index.values.astype(float), grid_L[name].values.astype(float),
                        marker='s', markevery=int(x_max/100), mew=2, mec='none', ms=10, 
                        ls='--', lw=2, label=name+'(14 cm Grid)')    
                    print ('    Plotting '+name+' (14 cm Grid) Data')

                    plt.plot(grid_M.index.values.astype(float), grid_M[name].values.astype(float),
                        marker='o', markevery=int(x_max/100), mew=2, mec='none', ms=10, 
                        ls='-', lw=2, label=name+'(10 cm Grid)')
                    print ('    Plotting '+name+' (10 cm Grid) Data')

                    plt.plot(grid_S.index.values.astype(float), grid_S[name].values.astype(float),
                        marker='^', markevery=int(x_max/100), mew=2, mec='none', ms=10, 
                        ls='-.', lw=2, label=name+'('+S_size[1:]+' Grid)')            
                    print ('    Plotting '+name+' ('+S_size[1:]+' Grid) Data')

                    # plt.errorbar(x[0::10], y[0::10], yerr=FDS_error*y[0::10], ecolor=next(err_colors), ls='none', fmt='.')
                else:
                    continue

            plt.grid(color='0.75', linestyle='-.', linewidth=1)

            # # Add vertical lines and labels for timing information (if available)
            # ax3 = ax1.twiny()
            # ax1_xlims = ax1.axis()[0:2]
            # ax3.set_xlim(ax1_xlims)
            # # Remove NaN items from event timeline
            # events = all_times[test].dropna()
            # # Ignore events that are commented starting with a pound sign
            # events = events[~events.str.startswith('#')]
            # [plt.axvline(_x, color='0.4', lw=1) for _x in events.index.values]
            # ax3.set_xticks(events.index.values)
            # plt.setp(plt.xticks()[1], rotation=60)
            # ax3.set_xticklabels(events.values, fontsize=10, ha='left')
            # plt.xlim([0, x_max])

            handles1, labels1 = ax1.get_legend_handles_labels()
            plt.legend(handles1, labels1, loc='upper right', fontsize=10, handlelength=3)

            # Save plot to file
            print ('    Saving '+test+'_cjet_'+str(i+1))
            plt.savefig(save_dir+'Temperature/'+test+'_cjet_'+str(i+1)+'.pdf')
            plt.close('all')
        print
