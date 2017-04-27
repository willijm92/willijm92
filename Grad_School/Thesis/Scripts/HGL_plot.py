# Generates HGL plots using .input files in FDS output directories 
#   (still need to make modified data .input files)

# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#!/usr/bin/env python

import os
import collections
import numpy as np
import pandas as pd
import math
import linecache
import matplotlib.pyplot as plt
from itertools import cycle
import sys

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

#  =================
#  = Specify files =
#  =================

# Specify name
specify_test  = False
specific_name = 'Test_5_East_062614'

# Specify structure
specify_struct  = False
specific_struct = 'West'

# Location of formatted experimental & FDS data file directories
exp_data_dir = '../Experimental_Data/Official_Repo_Data/'
orig_FDS_data_dir = '../FDS_Output_Files/Official_Repo_Files/'
mod_FDS_data_dir = '../FDS_Output_Files/Modified_Cases/'

# Location of file with timing information
all_times_file = '../Experimental_Data/All_Times.csv'

# Load exp. timings and description file
all_times = pd.read_csv(all_times_file)
all_times = all_times.set_index('Time')

# Specify data to plot (original FDS vs. exp and/or new FDS case results)
orig_FDS_cases  = False
mod_FDS_cases   = True
exp_data        = True

# # Location to save/output figures
# if mod_FDS_cases:
#     save_dir = '../Plots/Modified_Cases/Temperature/'
# else:
#     save_dir = '../Plots/Validation/Temperature/'

#  ==========================
#  = User defined functions =
#  ==========================

# # Prints an error message and stops code
def error_message(message):
    lineno = inspect.currentframe().f_back.f_lineno
    print('[ERROR, line '+str(lineno)+']:')
    print('  ' + message)
    sys.exit()

# Checks if file should be skipped, returns True if it should
def check_name(test_name):
    if test_name[:4] != 'Test':     # skips hidden files
        return(True)

    if specify_struct:
        if specific_struct == 'West':
            if specific_struct not in test_name:
                return(True)
        elif specific_struct == 'East': 
            if 'West' in test_name:
                return(True)
        else:
            error_message('Invalid name for specific_struct')

    return(False)

def setup_fig(color_list, y_label, x_max):
    fig = plt.figure()
    plt.rc('axes', color_cycle=color_list)
    ax1 = plt.gca()
    plt.ylabel(y_label, fontsize=20)
    ax1.set_xlim(0,x_max)
    plt.xlabel('Time (s)', fontsize=20)
    fig.set_size_inches(10, 6)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    return ax1

#  ===============================
#  = Loop through all data files =
#  ===============================
for f in os.listdir(mod_FDS_data_dir):
    if f.endswith('.input'):
        if f[:4] != 'Test':
            continue
        print(f)
        info_file = mod_FDS_data_dir+f
        tmp = np.zeros((20000,500))

        # read number of trees
        ntrees = int(linecache.getline(info_file, 1)[:-1])

        # read number of TCs per tree
        ntc = int(linecache.getline(info_file, 2)[:-1])

        # read each TC height and data file column number for each tree
        ztc = []    # list to store TC heights
        icol = np.zeros((ntc,ntrees),dtype=int)     # array of data file column numbers
        for n in range(0, ntc):
            line = linecache.getline(info_file, n+3)[:-1]
            S = line.split() # read line dividing into values using delimiter ' ', convert them to float)
            ztc.append(float(S[0]))
            for nn in range(0,ntrees):
                icol[n,nn]=int(S[nn+1])
        
        # weight of each tree
        wgt = []
        line = linecache.getline(info_file, ntc+3)[:-1]
        S = line.split()    # read weights as floats dividing line into values using delimiter ' '
        for nn in range(0, ntrees):
            wgt.append(float(S[nn]))

        # data file name
        infile = linecache.getline(info_file, ntc+4)[:-1]

        # read number of columns in data file
        nc = int(linecache.getline(info_file, ntc+5)[:-1])

        # read row number where data starts
        nr = int(linecache.getline(info_file, ntc+6)[:-1])

        # read ceiling height
        z_h = float(linecache.getline(info_file, ntc+7)[:-1])

        # read starting time
        t_start = int(linecache.getline(info_file, ntc+8)[:-1])

        # read name of output file
        outfile = linecache.getline(info_file, ntc+9)[:-1]

        # Read data from file
        M = pd.read_csv(mod_FDS_data_dir+infile, skiprows=1)
        t = M.iloc[:,0]
        d = M.iloc[:,1:nc]

        z_0 = 0
        z = []
        for n in range(0,ntc-2):
            z.append((ztc[n]+ztc[n+1])/2)
        z.append(z_h)

        # Create output files and lists to fill for the columns
        fout = pd.DataFrame(columns=['Time','Height','T_lower','T_upper'])
        tmpl = []
        tmph = []
        zint = []

        for i in range(0,len(t)):
            for nn in range(0, ntrees):
                for n in range(0,ntc):
                    tmp[i,n] = tmp[i,n] + (273+d.iloc[i,icol[n,nn]-2])*wgt[nn]
            i1 = 0
            i2 = 0
            for n in range(0,ntc-1):
                if n == 0:
                    i1 = i1 + tmp[i,n]*(z[n]-z_0)
                    i2 = i2 + (1./tmp[i,n])*(z[n]-z_0)
                else:
                    i1 = i1 + tmp[i,n]*(z[n]-z[n-1])
                    i2 = i2 + (1./tmp[i,n])*(z[n]-z[n-1])
        
            zint.append(tmp[i,0]*(i1*i2-z_h**2)/(i1+i2*tmp[i,1]**2-2.*tmp[i,0]*z_h))
            tmpl.append(tmp[i,0])
            i1 = 0
            for n in range(0,ntc-1):
                if n == 0:
                    if z[n]>zint[i]:
                        if z_0>=zint[i]:
                            i1 = i1 + tmp[i,n]*(z[n]-z_0)
                        if z_0<zint[i]:
                            i1 = i1 + tmp[i,n]*(z[n]-zint[i])
                else:
                    if z[n]>zint[i]:
                        if z[n-1]>=zint[i]:
                            i1 = i1 + tmp[i,n]*(z[n]-z[n-1])
                        if z[n-1]<zint[i]:
                            i1 = i1 + tmp[i,n]*(z[n]-zint[i])
            
            tmph.append(max(tmpl[i],(1./(z_h-zint[i]))*i1))

            newrow = [int(t.iloc[i]),round(zint[i],2),round(tmpl[i]-273,2),round(tmph[i]-273,2)]
            fout.loc[i] = newrow
        fout.to_csv(path_or_buf=mod_FDS_data_dir+f[:-6]+'.csv', index=False)

        # if check_name(test_name):     # check if file should be skipped
        #     continue
        # else:   # Load exp. and FDS data files
        #     exp_data = pd.read_csv(exp_data_dir + f)
        #     exp_data = exp_data.set_index('Time')
        #     exp_data = exp_data.drop('s')

        #     if len(test_name) < 7:
        #         FDS_data_file = test_name[:5]+'0'+test_name[5]+'_devc.csv'
        #     else:
        #         FDS_data_file = test_name+'_devc.csv'

        #     try:
        #         FDS_data = pd.read_csv(FDS_data_dir+test_name[:-12]+'/'+FDS_data_file,skiprows=1)
        #     except:
        #         continue

        #     FDS_data = FDS_data.set_index('Time')
        #     print ('--- Loaded FDS & Exp Data Files for ' + test_name + ' ---')
        
        # # Make list of column header prefixes corresponding to plot data
        # plot_types = []
        # if plot_CO:
        #     plot_types.append('CO_')
        # if plot_CO2:
        #     plot_types.append('CO2_')
        # if plot_O2:
        #     plot_types.append('O2_')
        # if plot_TC:
        #     plot_types.append('TC_')
        # if plot_HF:
        #     plot_types.append('HF_')
        # if plot_BDP:
        #     plot_types.append('BDP_')

        # # These are the "Tableau 20" colors as RGB.
        # tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
        #              (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
        #              (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
        #              (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
        #              (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

        # # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
        # for i in range(len(tableau20)):
        #     r, g, b = tableau20[i]
        #     tableau20[i] = (r / 255., g / 255., b / 255.)

        # # iterate through exp data column headers and generate desired plots
        # plotted_groups = []
        # x_max = exp_data.index.values.astype(float)[-1]

        # for column in FDS_data:
        #     # Check if column's data type should be plotted
        #     if any([substring in column for substring in plot_types]):     
        #         # Check if channel's sensor group has been plotted
        #         if any([substring in column for substring in plotted_groups]):
        #             continue
        #         legend = True

        #         # Add legend/timing information later
        #         if 'TC_' in column:
        #             group = column[:-2]
        #             channel_ls = []
        #             for i in range(1,9):
        #                 channel_name = group+'_'+str(i)
        #                 try: 
        #                     df = exp_data[channel_name]
        #                     channel_ls.append(channel_name)
        #                 except KeyError:
        #                     continue
        #             plotted_groups.append(group+'_') 
        #             data_type = 'Temperature'
        #             plot_colors = tableau20[:len(channel_ls)]
        #             y_label = r'Temperature ($^\circ$C)'
        #             legend = False

        #         elif 'BDP_' in column:
        #             group = column[:-2]
        #             channel_ls = []
        #             for i in range(1,9):
        #                 channel_name = group+'_'+str(i)
        #                 try: 
        #                     df = exp_data[channel_name]
        #                     channel_ls.append(channel_name)
        #                 except KeyError:
        #                     continue
        #             plotted_groups.append(group+'_') 
        #             data_type = 'Velocity' 
        #             plot_colors = tableau20[:len(channel_ls)]
        #             y_label = r'Velocity (m/s)'
        #             legend = False

        #         elif 'HF_' in column:
        #             group = column
        #             channel_ls = [column, 'RAD'+column[2:]]
        #             plotted_groups.append(column)
        #             plotted_groups.append('RAD'+column[2:])
        #             data_type = 'Heat_Flux' 
        #             plot_colors = tableau20[:len(channel_ls)]
        #             y_label = r'Heat Flux (kW/m$^2$)'

        #         elif 'RAD_' in column:
        #             group = 'HF'+column[3:]
        #             channel_ls = [group, column]
        #             plotted_groups.append(column)
        #             plotted_groups.append('HF'+column[3:])
        #             data_type = 'Heat_Flux' 
        #             plot_colors = tableau20[:len(channel_ls)]
        #             y_label = r'Heat Flux (kW/m$^2$)'

        #         elif 'CO_' in column:
        #             group = 'CO'
        #             channel_ls = ['CO_A', 'CO_B']
        #             plotted_groups.append('CO_')
        #             data_type = 'Gas_Concentration'
        #             plot_colors = ['k', 'r']
        #             y_label = 'Volume Fraction'

        #         elif 'CO2_' in column:
        #             group = 'CO2'
        #             channel_ls = ['CO2_A', 'CO2_B']
        #             plotted_groups.append('CO2_')
        #             data_type = 'Gas_Concentration'
        #             plot_colors = ['k', 'r']
        #             y_label = 'Volume Fraction'

        #         elif 'O2_' in column:
        #             group = 'O2'
        #             channel_ls = ['O2_A', 'O2_B']
        #             plotted_groups.append('O2_')
        #             data_type = 'Gas_Concentration'
        #             plot_colors = ['k', 'r']
        #             y_label = 'Volume Fraction'

        #         ax1 = setup_fig(plot_colors,y_label,x_max)
                
        #         if y_label == 'Volume Fraction':
        #             ax1.set_ylim(0,0.25)

        #         for name in channel_ls:
        #             plt.plot(exp_data.index.values.astype(float), 
        #                 exp_data[name].values.astype(float), 
        #                 ls='-', lw=1.5, label='Exp '+name)
        #             print ('    Plotting ' + name + ' Exp Data')

        #         for name in channel_ls:
        #             plt.plot(FDS_data.index.values.astype(float), 
        #                 FDS_data[name].values.astype(float), 
        #                 ls='--', lw=1.5, label='FDS '+name)
        #             print ('    Plotting ' + name + ' FDS Data')
                
        #         if legend:              
        #             handles1, labels1 = ax1.get_legend_handles_labels()
        #             plt.legend(handles1, labels1, loc='upper right', fontsize=8, handlelength=3)

        #         plt.grid(True)
        #         # Save plot to file
        #         print ('    Saving ' + group + ' figure')
        #         print
        #         plt.savefig(save_dir+data_type+'/'+test_name + '_' +group+ '.pdf')
        #         plt.close('all')

        #     else:
        #         continue

        # if plot_O2_CO2:
        #     plot_colors = ['k','k','r','r']
        #     exp_time = exp_data.index.values.astype(float)
        #     FDS_time = FDS_data.index.values.astype(float)
        #     x_max = max(exp_time)
        #     y_label = ''
        #     ax1 = setup_fig(plot_colors,y_label,x_max)     
        #     for species in ['O2','CO2']: 
        #         channel = species+'_A'
        #         exp_gas_data = exp_data[channel].dropna().values.astype(float)
        #         FDS_gas_data = FDS_data[channel].dropna().values.astype(float)
        #         if species == 'O2':
        #             O2_exp_max = np.max(exp_gas_data)
        #             O2_FDS_max = np.max(FDS_gas_data)

        #             exp_gas_data = (O2_exp_max-exp_gas_data)*(3./5.)
        #             FDS_gas_data = (O2_FDS_max-FDS_gas_data)*(3./5.)

        #         plt.plot(exp_time[:len(exp_gas_data)], exp_gas_data, lw=1.5, ls='-', label='Exp '+channel)
        #         print ('    Plotting ' + channel + ' Exp Data')
        #         plt.plot(FDS_time[:len(FDS_gas_data)], FDS_gas_data, lw=1.5, ls='--', label='FDS '+channel)
        #         print ('    Plotting ' + channel + ' FDS Data')               

        #     handles1, labels1 = ax1.get_legend_handles_labels()
        #     plt.legend(handles1, labels1, loc='upper left', fontsize=8, handlelength=3)

        #     plt.grid(True)
        #     # Save plot to file
        #     print ('    Saving O2_A on CO2_A plot')
        #     plt.savefig(save_dir + '/Gas_Concentration/'+test_name + '_O2_A_with_CO2_A.pdf')
        #     plt.close('all')
