# -*- coding: utf-8 -*-
from __future__ import unicode_literals, division
#!/usr/bin/env python

# Script generates data plots comparing FDS data to experimental data
#   and also has capability to include original repo data from FDS validation
#   guide data files 

# Still need to add:
#   - Splitting to TC and BDP plots
# 	- add error bars
# 	- combine HF plots

# Currently, script is set up to plot formatted experimental gas data with FDS gas data 
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
from matplotlib import colors as mcolors

#  =================
#  = Specify files =
#  =================

# Specify name to run a single test
specify_test  = False
specific_name = 'Test_4'

# Specify structure to plot data from specific structure 
specify_struct  = False
specific_struct = 'West'

# Location of formatted experimental & FDS data file directories
exp_data_dir = '../Experimental_Data/Official_Repo_Data/'
FDS_data_dir = '../FDS_Output_Files/Modified_Cases/'

# Location of file with timing information
all_times_file = '../Experimental_Data/Plot_Times.csv'
plot_settings_file = '../Experimental_Data/plot_settings.csv'

# Load exp. timings and description file
all_times = pd.read_csv(all_times_file)
all_times = all_times.set_index('Time')

plot_settings = pd.read_csv(plot_settings_file)

#  =================
#  = Specify plots =
#  =================

# Location to save/output figures
save_dir = '../Plots/Validation/'

# Specify plots to generate
plot_CO2     = False
plot_O2      = False
plot_TC      = False
plot_HF      = True
plot_BDP     = True
plot_HGL     = False
plot_cjetTC  = False

plot_CO      = False
plot_O2_CO2  = False

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

	if specify_test:
		if test_name != specific_name:
			return(True)

	if specify_struct:
		if specific_struct == 'West':
			if len(test_name)<7:
				return(True)
		elif specific_struct == 'East': 
			if len(test_name)>6:
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
	# presentation plots
	fig.set_size_inches(10, 7)
	# fig.set_size_inches(10, 8)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)

	return ax1

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
if plot_HGL:
	HGL_test_plots = []

# font size for presentation plots
leg_font = 12

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

#  ===============================
#  = Loop through all data files =
#  ===============================
for f in os.listdir(exp_data_dir):
	if f[-7:-4] != 'HGL':
		if f.endswith('.csv'):
			# Strip test name from file name
			test_name = f[:-4]

			if check_name(test_name):     # check if file should be skipped
				continue
			else:   
				# Load FDS case data file               
				if len(test_name) < 7:  # East Test
					FDS_data_file  = test_name[:5]+'0'+test_name[5]+'_full_devc.csv'

					# set channels to skip based on structure
					plotted_channels = ['HF_A2', 'RAD_A2', 'TC_A10_1', 'TC_A10_2', 'TC_A10_3']
					for i in range(1,9):    # should be a better way to do this
						added_channels = ['BDP_A7_'+str(i), 'TC_A7_'+str(i), 'BDP_A8_'+str(i), 'TC_A8_'+str(i),
						'TC_A9_'+str(i), 'BDP_A9_'+str(i), 'TC_A10_'+str(i)]
						plotted_channels = plotted_channels + added_channels
				else:
					FDS_data_file  = test_name+'_full_devc.csv'

					# set channels to skip based on structure
					plotted_channels = ['HF_3_V', 'HF_3_H']
					for i in range(1,9):    # should be a better way to do this
						added_channels = ['BDP_A5_'+str(i), 'TC_A5_'+str(i), 'BDP_A6_'+str(i), 'TC_A6_'+str(i), 'TC_A10_',
						'BDP_A11_'+str(i), 'TC_A11_'+str(i), 'BDP_A13_'+str(i), 'TC_A13_'+str(i),
						'BDP_A14_'+str(i), 'TC_A14_'+str(i)]
						plotted_channels = plotted_channels + added_channels

				try:
					FDS_data  = pd.read_csv(FDS_data_dir+test_name[:-12]+'/'+FDS_data_file,skiprows=1)
				except:
					continue

				FDS_data = FDS_data.set_index('Time')

				# # Load original FDS data file (if applicable)
				# if orig_FDS_cases:
				# 	if len(test_name) < 7:
				# 		orig_FDS_data_file = test_name[:5]+'0'+test_name[5]+'_devc.csv'
				# 	else:
				# 		orig_FDS_data_file = test_name+'_devc.csv'

				# 	try:
				# 		orig_FDS_data = pd.read_csv(orig_FDS_data_dir+test_name[:-12]+'/'+orig_FDS_data_file,skiprows=1)
				# 	except:
				# 		continue

				# 	orig_FDS_data = orig_FDS_data.set_index('Time')

				# Load experimental data file
				exp_data = pd.read_csv(exp_data_dir + f)
				exp_data = exp_data.set_index('Time')
				exp_data = exp_data.drop('s')
				print('--- Loaded FDS & Exp Data Files for ' + test_name + ' ---')
			
			# Determine end time
			if FDS_data.index.values.astype(float)[-1] >= exp_data.index.values.astype(float)[-1]:
				x_max = exp_data.index.values.astype(float)[-1]
				FDS_data = FDS_data.iloc[:(len(exp_data.index.values.astype(float))+1),:]
			else:
				x_max = FDS_data.index.values.astype(float)[-1]
				exp_data = exp_data.iloc[:(len(FDS_data.index.values.astype(float))+1),:]

			custom_group_ls = []
			custom_set_ls = []
			custom_ncols_ls = []
			custom_loc_ls = []
			custom_adj_ls = []
			for index, row in plot_settings.iterrows():
				if row['Test'] == test_name:
					custom_group_ls.append(row['Group'])
					custom_set_ls.append(row['Set'])
					custom_ncols_ls.append(row['ncols'])
					custom_loc_ls.append(row['loc'])
					custom_adj_ls.append(row['Adjustment'])

			# iterate through exp data column headers and generate desired plots
			for column in exp_data:
				# Check if column's data type should be plotted
				if any([substring in column for substring in plot_types]):     
					# Check if channel's sensor group has been plotted
					if any([substring in column for substring in plotted_channels]):
						continue
					
					legend = True       # include legend with plot unless variable changed to False later on
					negValues = False   # some sensor channels are backwards, will plot neg of data file values if this is the case
					split_plot = False 	# split for TC and BDP plots
					customize = False

					i = 0
					for sensor_group in custom_group_ls:
						str_len = len(sensor_group)
						if sensor_group == column[:str_len]:
							customize = True
							custom_set = custom_set_ls[i]
							custom_ncols = custom_ncols_ls[i]
							custom_loc = custom_loc_ls[i]
							custom_adj = custom_adj_ls[i]
							continue
						i = i+1
					ncols_leg = 1

					# Add legend/timing information later
					if 'TC_' in column:
						group = column[:-2]
						channel_ls = []
						split_plot = True
						plot_colors = ['0.35','0.35',tableau20[6],tableau20[6],tableau20[0],tableau20[0],tableau20[5],tableau20[5]]					
						for i in range(1,9):
							channel_name = group+'_'+str(i)
							try: 
								df = exp_data[channel_name]
								channel_ls.append(channel_name)
							except KeyError:
								continue
						data_type = 'Temperature'
						y_label = r'Temperature ($^\circ$C)'
						exp_error = 0.15
						FDS_error = 0.07
						err_colors = cycle(plot_colors)
						if test_name == 'Test_5':
							ncols_leg = 2
							legend_loc = 'upper left'

						elif test_name == 'Test_25':
							ncols_leg = 2
							legend_loc = 'lower center'
						else:	
							legend_loc = 'upper right'

						if test_name == 'Test_4':
							ncols_leg = 2

					elif 'BDP_' in column:
						group = column[:-2]
						channel_ls = []
						if len(test_name) < 7:
							plot_colors = ['0.35','0.35',tableau20[6],tableau20[6],tableau20[0],tableau20[0]]
							if test_name == 'Test_5':
								legend_loc = 'upper left'
							else:
								legend_loc = 'lower left'
							ncols_leg = 3
						else:
							split_plot = True
							plot_colors = ['0.35','0.35',tableau20[6],tableau20[6],tableau20[0],tableau20[0],tableau20[5],tableau20[5]]	
							legend_loc = 'lower right'
							ncols_leg = 2

						for i in range(1,9):
							channel_name = group+'_'+str(i)
							try: 
								df = exp_data[channel_name]
								channel_ls.append(channel_name)
							except KeyError:
								continue
						data_type = 'Velocity' 
						y_label = r'Velocity (m/s)'
						exp_error = 0.18
						FDS_error = 0.08
						err_colors = cycle(plot_colors)

					elif 'HF_' in column:
						if len(test_name)<7:    # East structure
							group = column
							channel_ls = ['HF_A1', 'HF_A3', 'HF_A4', 'HF_A5']
						else:   # West structure
							group = column[:-2]
							channel_ls = ['HF_1_H', 'HF_1_V', 'HF_2_H', 'HF_2_V']
						plot_colors = ['0.35','0.35',tableau20[6],tableau20[6],tableau20[0],tableau20[0],tableau20[5],tableau20[5]]	
						err_colors = cycle(plot_colors)
						data_type = 'Heat_Flux' 
						y_label = r'Heat Flux (kW/m$^2$)'
						exp_error = 0.08
						FDS_error = 0.11
						ncols_leg = 2
						if test_name == 'Test_5' or test_name == 'Test_25' or test_name == 'Test_24' or test_name == 'Test_6':
							legend_loc = 'upper left'
						elif test_name == 'Test_22':
							legend_loc = 'upper left'
							ncols_leg = 1
						else:
							legend_loc = 'upper right'

					elif 'CO_' in column:
						group = 'CO'
						channel_ls = ['CO_A', 'CO_B']
						data_type = 'Gas_Concentration'
						plot_colors = ['k', 'r']
						y_label = 'Volume Fraction'
						exp_error = 0.12
						FDS_error = 0.08

					elif column[:4] == 'CO2_':
						group = 'CO2'
						channel_ls = ['CO2_A', 'CO2_B']
						data_type = 'Gas_Concentration'
						plot_colors = ['0.35','0.35',tableau20[6],tableau20[6]]
						y_label = 'Volume Fraction'
						exp_error = 0.12
						FDS_error = 0.08
						legend_loc = 'upper right'
						err_colors = cycle(plot_colors)

					elif column[:3] == 'O2_':
						group = 'O2'
						channel_ls = ['O2_A', 'O2_B']
						data_type = 'Gas_Concentration'
						plot_colors = ['0.35','0.35',tableau20[6],tableau20[6]]
						y_label = 'Volume Fraction'
						exp_error = 0.12
						FDS_error = 0.08
						legend_loc = 'lower right'
						err_colors = cycle(plot_colors)
					
					if split_plot:
						channel_ls1 = channel_ls[:4]
						channel_ls2 = channel_ls[4:]
						
						lower_customize = False
						upper_customize = False
						if customize:
							if custom_set == 'both':
								lower_customize = True
								upper_customize = True
							elif custom_set == 'lower':
								lower_customize = True
							elif custom_set == 'upper':
								upper_customize = True

						ax1 = setup_fig(plot_colors,y_label,x_max)

						if upper_customize:
							if custom_adj == 'None':
								offset = 0.0
							else:
								offset = float(custom_adj)
						else:
							offset = 0.0	

						# Plot data for each channel in sensor group
						for name in channel_ls1:
							# Plot experimental data
							x = exp_data.index.values.astype(float)
							y = exp_data[name].values.astype(float)
							exp_max = np.max(y)
							error_index = np.argmax(y)*10
							plt.plot(x, y, ls='-', lw=2, label='Exp '+name)
							plt.errorbar(error_index, exp_max, yerr=exp_max*exp_error, c=next(err_colors), fmt='o', ms=4, capthick=1.25,lw=1.25, capsize=8)
				
							# Plot FDS Data
							x = FDS_data.index.values.astype(float)
							y = FDS_data[name].values.astype(float)+offset
							FDS_max = np.max(y)
							error_index = np.argmax(y)*10
							plt.plot(x, y, ls='--', lw=2, label='FDS '+name)
							plt.errorbar(error_index, FDS_max, yerr=FDS_max*FDS_error, c=next(err_colors), fmt='o', ms=4, capthick=1.25,lw=1.25, capsize=8)
							plotted_channels.append(name)
						print ('    Plotted '+group+' upper channels')

						# # Plot FDS data for each channel in sensor group          
						# for name in channel_ls1:
						# 	x = FDS_data.index.values.astype(float)
						# 	y = FDS_data[name].values.astype(float)+offset
						# 	FDS_max = np.max(y)
						# 	error_index = np.argmax(y)*10
						# 	plt.plot(x, y, ls='--', lw=2, label='FDS '+name)
						# 	plt.errorbar(error_index, FDS_max, yerr=FDS_max*FDS_error, c=next(err_colors), fmt='o', ms=4, capthick=1.25,lw=1.25, capsize=8)
						# print ('    Plotted ' + group + ' upper channels FDS Data')

						plt.grid(color='0.75', linestyle='-.', linewidth=1)

						# Add vertical lines and labels for timing information (if available)
						ax3 = ax1.twiny()
						ax1_xlims = ax1.axis()[0:2]
						ax3.set_xlim(ax1_xlims)
						# Remove NaN items from event timeline
						events = all_times[test_name].dropna()
						# Ignore events that are commented starting with a pound sign
						events = events[~events.str.startswith('#')]
						[plt.axvline(_x, color='0.4', lw=1) for _x in events.index.values]
						ax3.set_xticks(events.index.values)
						plt.setp(plt.xticks()[1], rotation=45)
						ax3.set_xticklabels(events.values, fontsize=10, ha='left')
						plt.xlim([0, x_max])

						if legend:
							if upper_customize:
								if custom_loc != 'None':
									legend_loc = custom_loc
								ncols_leg = int(custom_ncols)
							handles1, labels1 = ax1.get_legend_handles_labels()
							plt.legend(handles1, labels1, loc=legend_loc, ncol=ncols_leg, fontsize=leg_font, handlelength=3)

						# Save plot to file
						print ('    => Saving ' + group + ' upper channels figure')
						print

						plt.savefig(save_dir+data_type+'/'+test_name + '_' +group+ '_upper.pdf')
						plt.close('all')

						ax1 = setup_fig(plot_colors,y_label,x_max)

						if lower_customize:
							offset = float(custom_adj)
						else:
							offset = 0.0
						# Plot data for each channel in sensor group
						for name in channel_ls2:
							# Plot Exp Data
							x = exp_data.index.values.astype(float)
							y = exp_data[name].values.astype(float)
							exp_max = np.max(y)
							error_index = np.argmax(y)*10
							plt.plot(x, y, ls='-', lw=2, label='Exp '+name)
							plt.errorbar(error_index, exp_max, yerr=exp_max*exp_error, c=next(err_colors), fmt='o', ms=4, capthick=1.25,lw=1.25, capsize=8)	

							# Plot FDS Data
							x = FDS_data.index.values.astype(float)
							y = FDS_data[name].values.astype(float)+offset
							FDS_max = np.max(y)
							error_index = np.argmax(y)*10
							plt.plot(x, y, ls='--', lw=2, label='FDS '+name)
							plt.errorbar(error_index, FDS_max, yerr=FDS_max*FDS_error, c=next(err_colors), fmt='o', ms=4, capthick=1.25,lw=1.25, capsize=8)	
							plotted_channels.append(name)
						print ('    Plotted '+group+' lower channels')

						# # Plot FDS data for each channel in sensor group          
						# for name in channel_ls2:
						# 	x = FDS_data.index.values.astype(float)
						# 	y = FDS_data[name].values.astype(float)+offset
						# 	FDS_max = np.max(y)
						# 	error_index = np.argmax(y)*10
						# 	plt.plot(x, y, ls='--', lw=2, label='FDS '+name)
						# 	plt.errorbar(error_index, exp_max, yerr=exp_max*exp_error, c=next(err_colors), fmt='o', ms=4, capthick=1.25,lw=1.25, capsize=8)	
						# print ('    Plotted ' + group + ' lower channels FDS Data')

						plt.grid(color='0.75', linestyle='-.', linewidth=1)

						# Add vertical lines and labels for timing information (if available)
						ax3 = ax1.twiny()
						ax1_xlims = ax1.axis()[0:2]
						ax3.set_xlim(ax1_xlims)
						# Remove NaN items from event timeline
						events = all_times[test_name].dropna()
						# Ignore events that are commented starting with a pound sign
						events = events[~events.str.startswith('#')]
						[plt.axvline(_x, color='0.4', lw=1) for _x in events.index.values]
						ax3.set_xticks(events.index.values)
						plt.setp(plt.xticks()[1], rotation=45)
						ax3.set_xticklabels(events.values, fontsize=10, ha='left')
						plt.xlim([0, x_max])

						if legend:              
							if lower_customize:
								if custom_loc != 'None':
									legend_loc = custom_loc
								ncols_leg = int(custom_ncols)
							handles1, labels1 = ax1.get_legend_handles_labels()
							plt.legend(handles1, labels1, loc=legend_loc, ncol=ncols_leg, fontsize=leg_font, handlelength=3)

						# Save plot to file
						print ('    => Saving ' + group + ' lower channels figure')
						print

						plt.savefig(save_dir+data_type+'/'+test_name + '_' +group+ '_lower.pdf')
						plt.close('all')

					else:
						ax1 = setup_fig(plot_colors,y_label,x_max)
						# if y_label == 'Volume Fraction':
						# 	if group != 'CO2':
						# 		ax1.set_ylim(0,0.25)
							# else:
							# 	ax1.set_ylim(0,0.15)

						# Plot data for each channel in sensor group
						for name in channel_ls:
							# Plot exp data
							x = exp_data.index.values.astype(float)
							if name == 'HF_A5':
								y = exp_data[name].values.astype(float)*-1.
							else:
								y = exp_data[name].values.astype(float)

							if name[:2] != 'O2':
								exp_max = np.max(y[:100])
								error_index = np.argmax(y[:100])*10
							else:
							 	exp_max = np.min(y)
							 	error_index = np.argmin(y)*10
							plt.plot(x, y, ls='-', lw=2, label='Exp '+name)
							plt.errorbar(error_index, exp_max, yerr=exp_max*exp_error, c=next(err_colors), fmt='o', ms=4, capthick=1.25,lw=1.25, capsize=8)

							# Plot FDS Data
							x = FDS_data.index.values.astype(float)
							y = FDS_data[name].values.astype(float)
							if name[:2] != 'O2':
								FDS_max = np.max(y[:200])
								error_index = np.argmax(y[:200])*10
							else:
								FDS_max = np.min(y)
								error_index = np.argmin(y)*10
							plt.plot(x, y, ls='--', lw=2, label='FDS '+name)
							plt.errorbar(error_index, FDS_max, yerr=FDS_max*FDS_error, c=next(err_colors), fmt='o', ms=4, ls='--',capthick=1.25,lw=1.25, capsize=8)
							plotted_channels.append(name)
						print ('    Plotted '+group+' Data')

						# err_colors = cycle(plot_colors)
						# line_colors = cycle(plot_colors)

						 
						# # Plot FDS data for each channel in sensor group          
						# for name in channel_ls:
						# 	x = FDS_data.index.values.astype(float)
						# 	y = FDS_data[name].values.astype(float)
						# 	if name[:2] != 'O2':
						# 		FDS_max = np.max(y[:200])
						# 		error_index = np.argmax(y[:200])*10
						# 	else:
						# 		FDS_max = np.min(y)
						# 		error_index = np.argmin(y)*10
						# 	plt.plot(x, y, ls='--', lw=2, label='FDS '+name)
						# 	plt.errorbar(error_index, FDS_max, yerr=FDS_max*FDS_error, c=next(err_colors), fmt='o', ms=4, ls='--',capthick=1.25,lw=1.25, capsize=8)
						# print ('    Plotted ' + group + ' FDS Data')

						plt.grid(color='0.75', linestyle='-.', linewidth=1)

						# Add vertical lines and labels for timing information (if available)
						ax3 = ax1.twiny()
						ax1_xlims = ax1.axis()[0:2]
						ax3.set_xlim(ax1_xlims)
						# Remove NaN items from event timeline
						events = all_times[test_name].dropna()
						# Ignore events that are commented starting with a pound sign
						events = events[~events.str.startswith('#')]
						[plt.axvline(_x, color='0.4', lw=1) for _x in events.index.values]
						ax3.set_xticks(events.index.values)
						plt.setp(plt.xticks()[1], rotation=45)
						ax3.set_xticklabels(events.values, fontsize=10, ha='left')
						plt.xlim([0, x_max])
						# Increase figure size for plot labels at top
						# fig.set_size_inches(10, 7)

						if legend:
							if customize:
								if custom_loc != 'None':
									legend_loc = custom_loc
								ncols_leg = int(custom_ncols)              
							handles1, labels1 = ax1.get_legend_handles_labels()
							plt.legend(handles1, labels1, loc=legend_loc, ncol=ncols_leg, fontsize=leg_font, handlelength=3)

						# Save plot to file
						if data_type == 'Heat_Flux':
							plt.savefig(save_dir+data_type+'/'+test_name + '_HFs.pdf')
							print ('    => Saving HFs Figure')					
						else:
							plt.savefig(save_dir+data_type+'/'+test_name + '_' +group+ '.pdf')
							print ('    => Saving ' + group + ' Figure')
						print
						plt.close('all')

				else:
					continue

			if plot_cjetTC:
				# set plot colors 
				plot_colors = ['0.3','0.3',tableau20[6],tableau20[6],tableau20[0],tableau20[0]]

				# Set array of TCs to plot (each row corresponds to a figure)
				if len(test_name)<7:
					channel_array = np.array([['TC_A1_1', 'TC_A3_1', 'TC_A5_1'],['TC_A2_1', 'TC_A4_1', 'None']]) 
				else:
					channel_array = np.array([['TC_A1_1', 'TC_A2_1', 'TC_A3_1'],['TC_A7_1', 'TC_A8_1', 'TC_A9_1']])

				exp_error = 0.15
				FDS_error = 0.07
				err_colors = cycle(plot_colors)

				for i in range(0,channel_array.shape[0]):
					# Setup figure to plot set of TCs
					ax1 = setup_fig(plot_colors,r'Temperature ($^\circ$C)',x_max)
					for j in range(0, channel_array.shape[1]):
						name = channel_array[i,j]
						if name != 'None':
							# plot exp + FDS data
							x = exp_data.index.values.astype(float)
							y = exp_data[name].values.astype(float)
							exp_max = np.max(y)
							error_index = np.argmax(y)*10
							plt.plot(x, y, ls='-', lw=2, label='Exp '+name)
							plt.errorbar(error_index, exp_max, yerr=exp_max*exp_error, c=next(err_colors), fmt='o', ms=4, capthick=1.25,lw=1.25, capsize=8)	
							
							x = FDS_data.index.values.astype(float)
							y = FDS_data[name].values.astype(float)
							FDS_max = np.max(y)
							FDS_index = np.argmax(y)*10	
							plt.plot(x, y, ls='--', lw=2, label='FDS '+name)
							plt.errorbar(FDS_index, FDS_max, yerr=FDS_max*FDS_error, c=next(err_colors), fmt='o', ms=4, capthick=1.25,lw=1.25, capsize=8)
							# plt.errorbar(x[0::10], y[0::10], yerr=FDS_error*y[0::10], ecolor=next(err_colors), ls='none', fmt='.')
						else:
							continue

					plt.grid(color='0.75', linestyle='-.', linewidth=1)

					# Add vertical lines and labels for timing information (if available)
					ax3 = ax1.twiny()
					ax1_xlims = ax1.axis()[0:2]
					ax3.set_xlim(ax1_xlims)
					# Remove NaN items from event timeline
					events = all_times[test_name].dropna()
					# Ignore events that are commented starting with a pound sign
					events = events[~events.str.startswith('#')]
					[plt.axvline(_x, color='0.4', lw=1) for _x in events.index.values]
					ax3.set_xticks(events.index.values)
					plt.setp(plt.xticks()[1], rotation=45)
					ax3.set_xticklabels(events.values, fontsize=10, ha='left')
					plt.xlim([0, x_max])

					if test_name == 'Test_6':
						legend_loc = 'upper left'
						if i+1 == 1:
							ncols_leg = 3
						else:
							ncols_leg = 1
					elif test_name == 'Test_5':
						if i+1 == 1:
							ncols_leg = 3
							legend_loc = 'upper center'
						else:
							ncols_leg = 1
							legend_loc = 'upper left'

					else:
						ncols_leg = 1
						legend_loc = 'upper right'

					handles1, labels1 = ax1.get_legend_handles_labels()
					plt.legend(handles1, labels1, loc=legend_loc, ncol=ncols_leg, fontsize=leg_font, handlelength=3)
					
					# Save plot to file
					print ('    Saving '+test_name+'_cjet_'+str(i+1))
					plt.savefig(save_dir+'Temperature/'+test_name+'_cjet_'+str(i+1)+'.pdf')
					plt.close('all')
				print

			# if plot_O2_CO2:
			# 	# set plot colors 
			# 	if FDS_cases:
			# 		plot_colors = ['k','k','k','r','r','r']
			# 	else:
			# 		plot_colors = ['k','k','r','r']

			# 	exp_time = exp_data.index.values.astype(float)
			# 	orig_FDS_time = orig_FDS_data.index.values.astype(float)
			# 	if FDS_cases: 
			# 		FDS_time = FDS_data.index.values.astype(float)

			# 	x_max = max(orig_FDS_time)
			# 	y_label = ''

			# 	for location in ['_A', '_B']:
			# 		ax1 = setup_fig(plot_colors,y_label,x_max) 
			# 		for species in ['O2','CO2']: 
			# 			channel = species+location
			# 			exp_gas_data = exp_data[channel].dropna().values.astype(float)
			# 			orig_FDS_gas_data = orig_FDS_data[channel].dropna().values.astype(float)
			# 			if orig_FDS_cases: 
			# 				FDS_gas_data = FDS_data[channel].dropna().values.astype(float)

			# 			if species == 'O2':
			# 				O2_exp_max = np.max(exp_gas_data)
			# 				exp_gas_data = (O2_exp_max-exp_gas_data)*(3./5.)
			# 				O2_orig_FDS_max = np.max(FDS_gas_data)
			# 				orig_FDS_gas_data = (O2_orig_FDS_max-orig_FDS_gas_data)*(3./5.)
			# 				if FDS_cases:
			# 					O2_FDS_max = np.max(FDS_gas_data)
			# 					FDS_gas_data = (O2_FDS_max-FDS_gas_data)*(3./5.)

			# 			plt.plot(exp_time[:len(exp_gas_data)], exp_gas_data, 
			# 				marker='o', markevery=int(x_max/100), mew=1.5, mec='none', ms=7,
			# 				lw=1.5, ls='-', label='Exp '+channel)
			# 			plt.plot(FDS_time[:len(FDS_gas_data)], FDS_gas_data, 
			# 					marker='s', markevery=int(x_max/100), mew=1.5, mec='none', ms=7,
			# 					lw=1.5, ls='-.', label='FDS '+channel)
			# 			if orig_FDS_cases:
			# 				plt.plot(orig_FDS_time[:len(orig_FDS_gas_data)], orig_FDS_gas_data, 
			# 					marker='^', markevery=int(x_max/100), mew=1.5, mec='none', ms=7,
			# 					lw=1.5, ls='--', label='Orig FDS '+channel)
					
			# 		# Add legend & gridlines                 
			# 		handles1, labels1 = ax1.get_legend_handles_labels()
			# 		plt.legend(handles1, labels1, loc='upper right', fontsize=8, handlelength=3)
			# 		plt.grid(True)

			# 		# Save plot to file
			# 		fig_name = 'O2'+location+'_on_CO2'+location
			# 		print ('    Saving '+fig_name+' plot')
			# 		plt.savefig(save_dir + '/Gas_Concentration/'+test_name+'_'+fig_name+'.pdf')
			# 		plt.close('all')
			# 	print

	else:
		if plot_HGL:
			# Strip test name from file name
			if f[6] == '_':
				test_name = f[:6]
				# loc = f[7:13]
				loc_ls = ['Room_1', 'Room_2', 'Room_3']
			else:
				test_name = f[:7]
				# loc = f[8:15]
				loc_ls = ['Floor_1', 'Floor_2']

			if check_name(test_name):     # check if file should be skipped
				continue

			skip_test = False
			for name in HGL_test_plots:
				if name == test_name:
					skip_test = True

			if skip_test:
				continue
			else:
				HGL_test_plots.append(test_name)

			plot_colors = ['0.35','0.35',tableau20[6],tableau20[6],tableau20[0],tableau20[0]]

			# Load modified FDS HGL and Exp HGL files
			if len(test_name) < 7:
				FDS_data_file  = test_name[:5]+'0'+f[5:7]+'full_'+loc_ls[0]+'_HGL.csv'
			else:
				FDS_data_file  = f[:8]+'full_'+loc_ls[0]+'_HGL.csv'

			try:
				FDS_data  = pd.read_csv(FDS_data_dir+FDS_data_file)
			except:
				continue

			FDS_data = FDS_data.set_index('Time')

			exp_data = pd.read_csv(exp_data_dir + f)
			exp_data = exp_data.set_index('Time')
			
			print('--- Loaded FDS & Exp HGL Files for ' + test_name + ' ---')
			
			# Determine end time
			if FDS_data.index.values.astype(float)[-1] >= exp_data.index.values.astype(float)[-1]:
				x_max = exp_data.index.values.astype(float)[-1]
				FDS_data = FDS_data.iloc[:(len(exp_data.index.values.astype(float))+1),:]
			else:
				x_max = FDS_data.index.values.astype(float)[-1]
				exp_data = exp_data.iloc[:(len(FDS_data.index.values.astype(float))+1),:]

			exp_error = 0.15
			FDS_error = 0.07
			err_colors = cycle(plot_colors)
			
			ax1 = setup_fig(plot_colors,r'Temperature ($^\circ$C)',x_max)
			overall_max = 0

			for loc in loc_ls:
				# Load modified FDS HGL and Exp HGL files
				if len(test_name) < 7:
					FDS_data_file  = test_name[:5]+'0'+f[5:7]+'full_'+loc+'_HGL.csv'
					exp_data_file  = test_name+'_'+loc+'_HGL.csv'
				else:
					FDS_data_file  = test_name+'_full_'+loc+'_HGL.csv'
					exp_data_file  = test_name+'_'+loc+'_HGL.csv'

				FDS_data  = pd.read_csv(FDS_data_dir+FDS_data_file)
				FDS_data = FDS_data.set_index('Time')

				exp_data = pd.read_csv(exp_data_dir+exp_data_file)
				exp_data = exp_data.set_index('Time')

				exp_upper_T = exp_data['T_upper'].values.astype(float)
				FDS_upper_T = FDS_data['T_upper'].values.astype(float)
				exp_time = exp_data.index.values.astype(float)
				FDS_time = FDS_data.index.values.astype(float)

				exp_max = np.max(exp_upper_T)
				FDS_max = np.max(FDS_upper_T)
				if overall_max < (exp_max+exp_max*exp_error):
					overall_max = exp_max+exp_max*exp_error

				if overall_max < (FDS_max+FDS_max*FDS_error):
					overall_max = FDS_max+FDS_max*FDS_error		
				
				plt.plot(exp_time, exp_upper_T, ls='-', lw=2, label='Exp '+loc+' HGL')
				plt.plot(FDS_time, FDS_upper_T, ls='--', lw=2, label='FDS '+loc+' HGL')
				plt.errorbar(np.argmax(exp_upper_T)*10, exp_max, yerr=exp_max*exp_error, c=next(err_colors), fmt='o', ms=4, capthick=1.25,lw=1.25, capsize=8)
				plt.errorbar(np.argmax(FDS_upper_T)*10, FDS_max, yerr=FDS_max*FDS_error, c=next(err_colors), fmt='o', ms=4, ls='--',capthick=1.25,lw=1.25, capsize=8)
			plt.grid(color='0.75', linestyle='-.', linewidth=1)

			# Add vertical lines and labels for timing information (if available)
			ax3 = ax1.twiny()
			ax1_xlims = ax1.axis()[0:2]
			ax3.set_xlim(ax1_xlims)
			# Remove NaN items from event timeline
			events = all_times[test_name].dropna()
			# Ignore events that are commented starting with a pound sign
			events = events[~events.str.startswith('#')]
			[plt.axvline(_x, color='0.4', lw=1) for _x in events.index.values]
			ax3.set_xticks(events.index.values)
			plt.setp(plt.xticks()[1], rotation=45)
			ax3.set_xticklabels(events.values, fontsize=10, ha='left')
			plt.xlim([0, x_max])
			ax1.set_ylim([0,overall_max*1.05])

			if test_name == 'Test_5' or test_name == 'Test_6':
				legend_loc = 'upper left'
			else:
				legend_loc = 'upper right'

			handles1, labels1 = ax1.get_legend_handles_labels()
			plt.legend(handles1, labels1, loc=legend_loc, fontsize=leg_font, handlelength=3)

			# Save plot to file
			fig_name = test_name+'_HGL.pdf'
			print ('    Saving '+fig_name+' plot')
			plt.savefig(save_dir+'Temperature/'+fig_name)
			plt.close('all')
		else:
			continue

# Commands to run FDS on blaze and copy files over to JMW_HDD
# qfds.sh -p 8 -o 4 -e FireModels_fork/fds/Build/mpi_intel_linux_64ib/fds_mpi_intel_linux_64ib FireModels_fork/fds/Validation/DelCo_Trainers/FDS_Input_Files/Test_03.fds

# qfds.sh -p 8 -o 4 -e FireModels_fork/fds/Build/mpi_intel_linux_64ib/fds_mpi_intel_linux_64ib Test_25_full.fds

# scp jmw3@blaze.nist.gov:Test_04* ../../Volumes/JMW_ExtHDD/jmw_files/School/Grad_School/Thesis/FDS_Output_Files/Test_24/

# Copy .fds local to blaze
# scp ../../Volumes/JMW_ExtHDD/jmw_files/School/Grad_School/Thesis/FDS_Input_Files/Modified_Cases/Test_24_full.fds jmw3@blaze.nist.gov:

# scp jmw3@blaze.nist.gov:Test_04_full_devc.csv ../../Volumes/JMW_ExtHDD/jmw_files/School/Grad_School/Thesis/FDS_Output_Files/Modified_Cases/

