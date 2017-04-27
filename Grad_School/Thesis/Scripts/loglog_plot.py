# -*- coding: utf-8 -*-
from __future__ import unicode_literals, division
#!/usr/bin/env python

# Script generates log/log plot of data and calculates 

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

# Location of formatted experimental & FDS data file directories
exp_data_dir = '../Experimental_Data/Official_Repo_Data/'
FDS_data_dir = '../FDS_Output_Files/Modified_Cases/'

# Location of file with timing information
all_times_file = '../Experimental_Data/All_Times.csv'

# Location of test description file
info_file = '../Experimental_Data/Description_of_Experiments.csv'

# Load exp. timings and description file
all_times = pd.read_csv(all_times_file)
all_times = all_times.set_index('Time')
info = pd.read_csv(info_file, index_col=3)

#  =================
#  = Specify plots =
#  =================

save_dir = '../Plots/Validation/'

# Specify plots to generate
plot_CO2    = False
plot_O2     = False
plot_TC     = False
plot_HF     = False     # determine if separate plot for RAD should be used
plot_BDP    = True
plot_HGL    = False  
plot_cjetTC = False

# Make list of column header prefixes corresponding to plot data
plot_types = []
if plot_CO2:
	plot_types.append('CO2_')
	CO2_FDS_data = []
	CO2_exp_data = []
if plot_O2:
	plot_types.append('O2_')
	O2_FDS_data = []
	O2_exp_data = []
if plot_TC:
	plot_types.append('TC_')
	TC_FDS_data = []
	TC_exp_data = []
if plot_HF:
	plot_types.append('HF_')
	HF_FDS_data = []
	HF_exp_data = []
if plot_BDP:
	plot_types.append('BDP_')
	Vel_FDS_data = []
	Vel_exp_data = []
if plot_HGL:
	HGL_FDS_data = []
	HGL_exp_data = []
if plot_cjetTC:
	cjet_FDS_data = []
	cjet_exp_data = []

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
	fig.set_size_inches(10, 6)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)

	return ax1

def loglog_plot(exp_ls, FDS_ls, Sigma_E, Plot_Min, Plot_Max, x_label, y_label, save_loc_name, plot_title):
	Measured_Values = []
	Predicted_Values = []
	if save_loc_name[-7:-4] == 'CO2':
		for n in range(0,len(exp_ls)):
			if math.isnan(exp_ls[n]) or math.isnan(FDS_ls[n]):
				continue
			elif exp_ls[n] > 0 and FDS_ls[n] > 0:
				if exp_ls[n] <= 0.1 and FDS_ls[n] <= 0.1: 
					Measured_Values.append(exp_ls[n])
					Predicted_Values.append(FDS_ls[n])
	else:
		for n in range(0,len(exp_ls)):
			if math.isnan(exp_ls[n]) or math.isnan(FDS_ls[n]):
				continue
			elif exp_ls[n] > 0 and FDS_ls[n] > 0:
				Measured_Values.append(exp_ls[n])
				Predicted_Values.append(FDS_ls[n])

	Measured_Values_Raw = Measured_Values
	Predicted_Values_Raw = Predicted_Values

	Measured_Values = []
	Predicted_Values = []
	for n in range(0, len(Measured_Values_Raw)-3):
		if (n+1)%3 == 0:
			Measured_Values.append(np.mean(Measured_Values_Raw[n-2:n+1]))
			Predicted_Values.append(np.mean(Predicted_Values_Raw[n-2:n+1]))

	Measured_Values.append(np.mean(Measured_Values_Raw[-3:]))
	Predicted_Values.append(np.mean(Predicted_Values_Raw[-3:]))	

	n_pts = len(Measured_Values)

	# Weight the data -- for each point on the scatterplot compute a "weight" to provide 
	#   sparse data with greater importance in the calculation of the accuracy statistics

	weight = np.zeros(len(Measured_Values))
	bin_weight = [] 
	Max_Measured_Value = max(Measured_Values)
	Bin_Size = Max_Measured_Value/10
	
	for ib in range(1,11):
		bin_indices = [x for x in Measured_Values if(x>(ib-1)*Bin_Size and x<=ib*Bin_Size)]
		try:
			bin_weight.append(n_pts/len(bin_indices))
		except:
			bin_weight.append(1.733170134638923)

	for iv in range(0,n_pts):
		for ib in range(0,10):
			if (Measured_Values[iv]>(ib)*Bin_Size and Measured_Values[iv]<=(ib+1)*Bin_Size): 
				weight[iv] = bin_weight[ib]

	# weight = weight.tolist()
	Measured_Values = np.asarray(Measured_Values)
	Predicted_Values = np.asarray(Predicted_Values)

	# Setup figure and plot nonzero data
	fig = plt.figure()
	plt.rc('axes')
	ax1 = plt.gca()
	ax1.set_ylim(Plot_Min,Plot_Max)
	plt.ylabel(y_label, fontsize=20)
	ax1.set_xlim(Plot_Min,Plot_Max)
	plt.xlabel(x_label, fontsize=20)
	fig.set_size_inches(8, 8)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)

	plt.loglog(Measured_Values,Predicted_Values,ls='None',
		marker='o',mew=2,mec='0.5',mfc='0.75',markersize=5,markevery=1)
	
	# Calculate statistics
	E_bar = np.sum(np.log10(Measured_Values)*weight)/np.sum(weight)
	M_bar = np.sum(np.log10(Predicted_Values)*weight)/np.sum(weight)
	Size_Measured = Measured_Values.size
	Size_Predicted = Predicted_Values.size

	if Size_Measured != Size_Measured:
		error_message('Mismatched measured and predicted arrays for scatterplot ')
	
	u2 = np.sum((((np.log10(Predicted_Values)-np.log10(Measured_Values))-(M_bar-E_bar))**2)*weight)/(np.sum(weight)-1)
	u  = math.sqrt(u2)
	Sigma_E = Sigma_E/100.
	Sigma_E = min(u/math.sqrt(2.),Sigma_E)
	Sigma_M = math.sqrt(max(0.,u*u-Sigma_E**2.))
	delta = math.exp(M_bar-E_bar+0.5*Sigma_M**2.-0.5*Sigma_E**2.)

	# Plot diagonal lines
	plt.plot([Plot_Min,Plot_Max],[Plot_Min,Plot_Max],'k-', lw=1.5)
	plt.plot([Plot_Min,Plot_Max],[Plot_Min*(1.+2.*Sigma_E),Plot_Max*(1.+2.*Sigma_E)],ls='--',c='k',lw=1.5)
	plt.plot([Plot_Min,Plot_Max],[Plot_Min/(1.+2.*Sigma_E),Plot_Max/(1.+2.*Sigma_E)],ls='--',c='k',lw=1.5)
	plt.plot([Plot_Min,Plot_Max],[Plot_Min*delta,Plot_Max*delta],ls='-',c='r',lw=1.5)
	# if delta > 2.*Sigma_M:
	plt.plot([Plot_Min,Plot_Max],[Plot_Min*delta*(1.+2.*Sigma_M),Plot_Max*delta*(1.+2.*Sigma_M)],ls='--',c='r',lw=1.5)
	plt.plot([Plot_Min,Plot_Max],[Plot_Min*delta/(1.+2.*Sigma_M),Plot_Max*delta/(1.+2.*Sigma_M)],ls='--',c='r',lw=1.5)

	if plot_title == 'Heat Flux':
		ax1.text(1.05*Plot_Min, 0.80*Plot_Max, plot_title, fontsize=12)
		ax1.text(1.05*Plot_Min, 0.65*Plot_Max, 'Exp. Rel. Std. Dev.: '+str(round(Sigma_E,2)), fontsize=10)
		ax1.text(1.05*Plot_Min, 0.54*Plot_Max, 'Model Rel. Std. Dev.: '+str(round(Sigma_M,2)), fontsize=10)
		ax1.text(1.05*Plot_Min, 0.44*Plot_Max,  'Model Bias Factor: '+str(round(delta,2)), fontsize=10)
	elif plot_title == 'Gas Velocity':
		ax1.text(1.05*Plot_Min, 0.85*Plot_Max, plot_title, fontsize=12)
		ax1.text(1.05*Plot_Min, 0.74*Plot_Max, 'Exp. Rel. Std. Dev.: '+str(round(Sigma_E,2)), fontsize=10)
		ax1.text(1.05*Plot_Min, 0.65*Plot_Max, 'Model Rel. Std. Dev.: '+str(round(Sigma_M,2)), fontsize=10)
		ax1.text(1.05*Plot_Min, 0.57*Plot_Max,  'Model Bias Factor: '+str(round(delta,2)), fontsize=10)
	elif plot_title == 'Carbon Dioxide Concentration':
		ax1.text(1.05*Plot_Min, 0.85*Plot_Max, plot_title, fontsize=12)
		ax1.text(1.05*Plot_Min, 0.74*Plot_Max, 'Exp. Rel. Std. Dev.: '+str(round(Sigma_E,2)), fontsize=10)
		ax1.text(1.05*Plot_Min, 0.64*Plot_Max, 'Model Rel. Std. Dev.: '+str(round(Sigma_M,2)), fontsize=10)
		ax1.text(1.05*Plot_Min, 0.56*Plot_Max,  'Model Bias Factor: '+str(round(delta,2)), fontsize=10)	
	elif plot_title == 'Oxygen Concentration':
		ax1.text(1.05*Plot_Min, 0.88*Plot_Max, plot_title, fontsize=12)
		ax1.text(1.05*Plot_Min, 0.81*Plot_Max, 'Exp. Rel. Std. Dev.: '+str(round(Sigma_E,2)), fontsize=10)
		ax1.text(1.05*Plot_Min, 0.75*Plot_Max, 'Model Rel. Std. Dev.: '+str(round(Sigma_M,2)), fontsize=10)
		ax1.text(1.05*Plot_Min, 0.69*Plot_Max,  'Model Bias Factor: '+str(round(delta,2)), fontsize=10)		
	else:
		ax1.text(1.05*Plot_Min, 0.85*Plot_Max, plot_title, fontsize=12)
		ax1.text(1.05*Plot_Min, 0.75*Plot_Max, 'Exp. Rel. Std. Dev.: '+str(round(Sigma_E,2)), fontsize=10)
		ax1.text(1.05*Plot_Min, 0.65*Plot_Max, 'Model Rel. Std. Dev.: '+str(round(Sigma_M,2)), fontsize=10)
		ax1.text(1.05*Plot_Min, 0.56*Plot_Max,  'Model Bias Factor: '+str(round(delta,2)), fontsize=10)
	print('    Saving '+save_loc_name)
	# print(delta)
	# print(Sigma_M)
	# print(Sigma_E)
	plt.savefig(save_loc_name)
	plt.close('all')


# # create log/log plots using all of the stored data
# if plot_CO2:
# 	Sigma_E = 8.
# 	plot_min = 0.001
# 	plot_max = 0.2
# 	x_label = 'Measured Volume Fraction'
# 	y_label = 'Predicted Volume Fraction'
# 	save_loc_name = save_dir+'Gas_Concentration/loglog_CO2.pdf'
# 	plot_title = 'Carbon Dioxide Concentration'
# 	loglog_plot(CO2_exp_data, CO2_FDS_data, Sigma_E, plot_min, plot_max,
# 		x_label, y_label, save_loc_name,plot_title)
# if plot_O2:
# 	Sigma_E = 8.
# 	plot_min = 0.01
# 	plot_max = 0.25
# 	x_label = 'Measured Volume Fraction'
# 	y_label = 'Predicted Volume Fraction'
# 	save_loc_name = save_dir+'Gas_Concentration/loglog_O2.pdf' 
# 	plot_title = 'Oxygen Concentration'
# 	loglog_plot(O2_exp_data, O2_FDS_data, Sigma_E, plot_min, plot_max,
# 		x_label, y_label, save_loc_name,plot_title)
# if plot_TC:
# 	Sigma_E = 7.
# 	plot_min = 10.
# 	plot_max = 2000.
# 	x_label = r'Measured Temperature ($^\circ$C)'
# 	y_label = r'Predicted Temperature ($^\circ$C)'
# 	save_loc_name = save_dir+'Temperature/loglog_TC_arrays.pdf' 
# 	plot_title = 'TC Array Temperatures'
# 	loglog_plot(TC_exp_data, TC_FDS_data, Sigma_E, plot_min, plot_max,
# 		x_label, y_label, save_loc_name,plot_title)

# if plot_BDP:
# 	Sigma_E = 6.
# 	plot_min = 0.1
# 	plot_max = 10.
# 	x_label = 'Measured Velocity (m/s)'
# 	y_label = 'Predicted Velocity (m/s)'
# 	save_loc_name = save_dir+'Velocity/loglog_BDPs.pdf'
# 	plot_title = 'Gas Velocity'
# 	loglog_plot(Vel_exp_data, Vel_FDS_data, Sigma_E, plot_min, plot_max,
# 		x_label, y_label, save_loc_name, plot_title)
# if plot_cjetTC:
# 	Sigma_E = 7.
# 	plot_min = 10.
# 	plot_max = 2000.
# 	x_label = r'Measured Temperature Rise ($^\circ$C)'
# 	y_label = r'Predicted Temperature Rise ($^\circ$C)'
# 	save_loc_name = save_dir+'Temperature/loglog_cjetTCs.pdf' 
# 	plot_title = 'Ceiling Jet Temperature'
# 	loglog_plot(cjet_exp_data, cjet_FDS_data, Sigma_E, plot_min, plot_max,
# 		x_label, y_label, save_loc_name, plot_title)
# if plot_HGL:
# 	Sigma_E = 7.
# 	plot_min = 10.
# 	plot_max = 2000.
# 	x_label = r'Measured Temperature Rise ($^\circ$C)'
# 	y_label = r'Predicted Temperature Rise ($^\circ$C)'
# 	save_loc_name = save_dir+'Temperature/loglog_HGL.pdf' 
# 	plot_title = 'HGL Temperature'
# 	loglog_plot(HGL_exp_data, HGL_FDS_data, Sigma_E, plot_min, plot_max,
# 		x_label, y_label, save_loc_name, plot_title)

################################
	# # Calculate statistics
	# E_bar = sum(math.log(Measured_Values).*weight)/sum(weight);
	# M_bar = sum(log(Predicted_Values).*weight)/sum(weight);
	# Size_Measured = size(Measured_Values);
	# Size_Predicted = size(Predicted_Values);

	# if Size_Measured(1) ~= Size_Predicted(1)
	#     display(['Error: Mismatched measured and predicted arrays for scatterplot ', Scatter_Plot_Title, '. Skipping scatterplot.'])
	#     continue
	# end

	# u2 = sum(    (((log(Predicted_Values)-log(Measured_Values)) - (M_bar-E_bar)).^2).*weight   )/(sum(weight)-1);
	# u  = sqrt(u2);
	# Sigma_E = Sigma_E/100;
	# Sigma_E = min(u/sqrt(2),Sigma_E);
	# Sigma_M = sqrt( max(0,u*u - Sigma_E.^2) );
	# delta = exp(M_bar-E_bar+0.5*Sigma_M.^2-0.5*Sigma_E.^2);

	# % Plot diagonal lines
	# plot([Plot_Min,Plot_Max],[Plot_Min,Plot_Max],'k-')
	# if strcmp(Model_Error, 'yes')
	#     plot([Plot_Min,Plot_Max],[Plot_Min,Plot_Max]*(1+2*Sigma_E),'k--')
	#     plot([Plot_Min,Plot_Max],[Plot_Min,Plot_Max]/(1+2*Sigma_E),'k--')
	#     plot([Plot_Min,Plot_Max],[Plot_Min,Plot_Max]*delta,'r-')
	#  %  if delta > 2*Sigma_M
	#        plot([Plot_Min,Plot_Max],[Plot_Min,Plot_Max]*delta*(1+2*Sigma_M),'r--')
	#        plot([Plot_Min,Plot_Max],[Plot_Min,Plot_Max]*delta/(1+2*Sigma_M),'r--')
	#  %  end
	# end

#  ===============================
#  = Loop through all data files =
#  ===============================
for f in os.listdir(exp_data_dir):
	if f[-7:-4] != 'HGL':
		if f.endswith('.csv'):
			# Strip test name from file name and checks if valid data file
			test_name = f[:-4]			
			if test_name[:4] != 'Test': 
				continue

			# Load FDS data
			if len(test_name) < 7:
				if int(test_name[5])<5:
					continue

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
					added_channels = ['BDP_A5_'+str(i), 'TC_A5_'+str(i), 'BDP_A6_'+str(i), 'TC_A6_'+str(i),
					'BDP_A11_'+str(i), 'TC_A11_'+str(i), 'BDP_A13_'+str(i), 'TC_A13_'+str(i),
					'BDP_A14_'+str(i), 'TC_A14_'+str(i)]
					plotted_channels = plotted_channels + added_channels
			try:
				FDS_data  = pd.read_csv(FDS_data_dir+test_name[:-12]+'/'+FDS_data_file,skiprows=1)
			except:
				continue

			FDS_data = FDS_data.set_index('Time')

			# Load exp data
			exp_data = pd.read_csv(exp_data_dir + f)
			exp_data = exp_data.set_index('Time')
			exp_data = exp_data.drop('s')
			print('--- Loaded FDS & Exp Data Files for ' + test_name + ' ---')
			
			# iterate through exp data column headers and store specified FDS and exp data
			for column in exp_data:
				# Check if column's data type should be stored
				if any([substring in column for substring in plot_types]):     
					# Check if channel's sensor group is already stored
					if any([substring in column for substring in plotted_channels]):
						continue

					negValues = False   # some sensor channels are backwards, will plot neg of data file values if this is the case

					# Generate channel list for particular sensor group
					if 'TC_' in column:
						group = column[:-2]
						channel_ls = []
						for i in range(1,9):
							channel_name = group+'_'+str(i)
							try: 
								df = exp_data[channel_name]
								channel_ls.append(channel_name)
							except KeyError:
								continue
					elif 'BDP_' in column:
						group = column[:-2]
						channel_ls = []
						for i in range(1,9):
							channel_name = group+'_'+str(i)
							try: 
								df = exp_data[channel_name]
								channel_ls.append(channel_name)
							except KeyError:
								continue
					elif 'HF_' in column:
						if len(test_name)<7:    # East structure
							group = column
							channel_ls = [group]
							if group[-1] == '5':
								negValues = True
						else:   # West structure
							group = column[:-2]
							channel_ls = [group+'_H', group+'_V']
					# elif 'RAD_' in column:
					# 	group = 'HF'+column[3:]
					# 	channel_ls = [group, column]
					elif column[0:3] == 'CO2':
						group = 'CO2'
						channel_ls = ['CO2_A', 'CO2_B']
					elif column[0:2]=='O2':
						group = 'O2'
						channel_ls = ['O2_A', 'O2_B']
					else:
						print('Skipped column '+column)
						continue

					exp_values = []
					FDS_values = []

					# Loop through group's channels and add values to lists
					if group[:3] != 'BDP':
						for name in channel_ls:
							if negValues:
								exp_channel_values = -1.*exp_data[name].values.astype(float)
							else:
								exp_channel_values = exp_data[name].values.astype(float)
							exp_channel_values = exp_channel_values.tolist()
							FDS_channel_values = FDS_data[name].values.astype(float)
							FDS_channel_values = FDS_channel_values.tolist()
							if len(FDS_channel_values) > len(exp_channel_values):
								FDS_channel_values = FDS_channel_values[:len(exp_channel_values)]
							elif len(exp_channel_values) > len(FDS_channel_values):
								exp_channel_values = exp_channel_values[:len(FDS_channel_values)]

							if test_name == 'Test_5':
								exp_channel_values = exp_channel_values[:57]+exp_channel_values[122:184]+exp_channel_values[242:299]
								FDS_channel_values = FDS_channel_values[:57]+FDS_channel_values[122:184]+FDS_channel_values[242:299]
							elif test_name == 'Test_6':
								exp_channel_values = exp_channel_values[:32]+exp_channel_values[56:86]+exp_channel_values[107:138]
								FDS_channel_values = FDS_channel_values[:32]+FDS_channel_values[56:86]+FDS_channel_values[107:138]
							elif test_name == 'Test_22':
								exp_channel_values = exp_channel_values[:65]
								FDS_channel_values = FDS_channel_values[:65]
							elif test_name == 'Test_23':
								exp_channel_values = exp_channel_values[:61]
								FDS_channel_values = FDS_channel_values[:61]
							elif test_name == 'Test_24':
								exp_channel_values = exp_channel_values[:62]
								FDS_channel_values = FDS_channel_values[:62]
							elif test_name == 'Test_25':
								exp_channel_values = exp_channel_values[:59]
								FDS_channel_values = FDS_channel_values[:59]

							exp_values = exp_values+exp_channel_values
							FDS_values = FDS_values+FDS_channel_values
							plotted_channels.append(name)
					else:
						# exp_channel_values = []
						# FDS_channel_values = []
						# i = 0
						# if len(FDS_data.index.values) > len(exp_data.index.values):					
						# 	for index, row in exp_data.iterrows():
						# 		exp_row_values = []
						# 		FDS_row_values = []
						# 		for name in channel_ls:
						# 			exp_row_values.append(float(exp_data[name].iloc[i]))
						# 			FDS_row_values.append(float(FDS_data[name].iloc[i]))
						# 		exp_channel_values.append(np.mean(exp_row_values))
						# 		FDS_channel_values.append(np.mean(FDS_row_values))
						# 		i = i+1
						# else:
						# 	for index, row in FDS_data.iterrows():
						# 		exp_row_values = []
						# 		FDS_row_values = []
						# 		for name in channel_ls:
						# 			exp_row_values.append(float(exp_data[name].iloc[i]))
						# 			FDS_row_values.append(float(FDS_data[name].iloc[i]))
						# 		exp_channel_values.append(np.mean(exp_row_values))
						# 		FDS_channel_values.append(np.mean(FDS_row_values))
						# 		i = i+1
						for name in channel_ls:
							if negValues:
								exp_channel_values = -1.*exp_data[name].values.astype(float)
							else:
								exp_channel_values = exp_data[name].values.astype(float)

							exp_channel_values = exp_channel_values.tolist()
							FDS_channel_values = FDS_data[name].values.astype(float)
							FDS_channel_values = FDS_channel_values.tolist()
							if len(FDS_channel_values) > len(exp_channel_values):
								FDS_channel_values = FDS_channel_values[:len(exp_channel_values)]
							elif len(exp_channel_values) > len(FDS_channel_values):
								exp_channel_values = exp_channel_values[:len(FDS_channel_values)]
							
							if test_name == 'Test_5':
								exp_channel_values = exp_channel_values[15:44]+exp_channel_values[134:172]+exp_channel_values[254:285]
								FDS_channel_values = FDS_channel_values[15:44]+FDS_channel_values[134:172]+FDS_channel_values[254:285]
							elif test_name == 'Test_6':
								exp_channel_values = exp_channel_values[20:32]+exp_channel_values[74:86]+exp_channel_values[129:138]
								FDS_channel_values = FDS_channel_values[20:32]+FDS_channel_values[74:86]+FDS_channel_values[129:138]
							elif test_name == 'Test_22':
								exp_channel_values = exp_channel_values[:65]
								FDS_channel_values = FDS_channel_values[:65]
							elif test_name == 'Test_23':
								exp_channel_values = exp_channel_values[:61]
								FDS_channel_values = FDS_channel_values[:61]
							elif test_name == 'Test_24':
								exp_channel_values = exp_channel_values[14:62]
								FDS_channel_values = FDS_channel_values[14:62]
							elif test_name == 'Test_25':
								exp_channel_values = exp_channel_values[11:59]
								FDS_channel_values = FDS_channel_values[11:59]

							exp_values = exp_values+exp_channel_values
							FDS_values = FDS_values+FDS_channel_values
							plotted_channels.append(name)

					# Add sensor group data to list of values of specific quantity
					if 'TC_' in column:
						TC_FDS_data = TC_FDS_data+FDS_values
						TC_exp_data = TC_exp_data+exp_values
					elif 'BDP_' in column:
						Vel_FDS_data = Vel_FDS_data+FDS_values
						Vel_exp_data = Vel_exp_data+exp_values
					elif 'HF_' in column:
						HF_FDS_data = HF_FDS_data+FDS_values
						HF_exp_data = HF_exp_data+exp_values
					# elif 'RAD_' in column:
					# 	HF_FDS_data = HF_FDS_data+FDS_values
					# 	HF_exp_data = HF_exp_data+exp_values
					elif column[0:3] == 'CO2':
						CO2_FDS_data = CO2_FDS_data+FDS_values
						CO2_exp_data = CO2_exp_data+exp_values
					elif column[0:2]=='O2':
						O2_FDS_data = O2_FDS_data+FDS_values
						O2_exp_data = O2_exp_data+exp_values
						print('Added to O2 data list')

			if plot_cjetTC:
				# Set array of TCs to plot (each row corresponds to a figure)
				if len(test_name)<7:
					channel_array = np.array([['TC_A1_1', 'TC_A3_1', 'TC_A5_1'],['TC_A2_1', 'TC_A4_1', 'None']]) 
				else:
					channel_array = np.array([['TC_A1_1', 'TC_A2_1', 'TC_A3_1'],['TC_A7_1', 'TC_A8_1', 'TC_A9_1']])

				for i in range(0,channel_array.shape[0]):
					for j in range(0, channel_array.shape[1]):
						name = channel_array[i,j]
						if name != 'None':
							exp_values = exp_data[name].values.astype(float).tolist()
							FDS_values = FDS_data[name].values.astype(float).tolist()
							if len(FDS_values) > len(exp_values):
								FDS_values = FDS_values[:len(exp_values)]
							elif len(exp_values) > len(FDS_values):
								exp_values = exp_values[:len(FDS_values)]
							cjet_exp_data = cjet_exp_data+exp_values
							cjet_FDS_data = cjet_FDS_data+FDS_values
						else:
							continue
	else:
		if plot_HGL:
			# Strip test name and location from file name
			if f[6] == '_':
				test_name = f[:6]
				loc = f[7:13]
			else:
				test_name = f[:7]
				loc = f[8:15]

			# Checks if valid data file
			if test_name[:4] != 'Test': 
				continue

			# Load modified FDS HGL and Exp HGL files
			if len(test_name) < 7:
				FDS_data_file  = test_name[:5]+'0'+f[5:7]+'full_'+loc+'_HGL.csv'
			else:
				FDS_data_file  = f[:8]+'full_'+loc+'_HGL.csv'

			try:
				FDS_data  = pd.read_csv(FDS_data_dir+FDS_data_file)
			except:
				continue
			FDS_data = FDS_data.set_index('Time')

			exp_data = pd.read_csv(exp_data_dir + f)
			exp_data = exp_data.set_index('Time')

			print('--- Loaded FDS & Exp HGL Files for '+loc+' '+test_name+' ---')
			exp_values = exp_data['T_upper'].values.astype(float).tolist()
			FDS_values = FDS_data['T_upper'].values.astype(float).tolist()
			if len(FDS_values) > len(exp_values):
				FDS_values = FDS_values[:len(exp_values)]
			elif len(exp_values) > len(FDS_values):
				exp_values = exp_values[:len(FDS_values)]

			HGL_exp_data = HGL_exp_data+exp_values
			HGL_FDS_data = HGL_FDS_data+FDS_values

		else:
			continue

# create log/log plots using all of the stored data
if plot_CO2:
	Sigma_E = 8.
	plot_min = 0.001
	plot_max = 0.2
	x_label = 'Measured Volume Fraction'
	y_label = 'Predicted Volume Fraction'
	save_loc_name = save_dir+'Gas_Concentration/loglog_CO2.pdf'
	plot_title = 'Carbon Dioxide Concentration'
	loglog_plot(CO2_exp_data, CO2_FDS_data, Sigma_E, plot_min, plot_max,
		x_label, y_label, save_loc_name,plot_title)
if plot_O2:
	Sigma_E = 8.
	plot_min = 0.01
	plot_max = 0.25
	x_label = 'Measured Volume Fraction'
	y_label = 'Predicted Volume Fraction'
	save_loc_name = save_dir+'Gas_Concentration/loglog_O2.pdf' 
	plot_title = 'Oxygen Concentration'
	loglog_plot(O2_exp_data, O2_FDS_data, Sigma_E, plot_min, plot_max,
		x_label, y_label, save_loc_name,plot_title)
if plot_TC:
	Sigma_E = 7.
	plot_min = 10.
	plot_max = 2000.
	x_label = r'Measured Temperature ($^\circ$C)'
	y_label = r'Predicted Temperature ($^\circ$C)'
	save_loc_name = save_dir+'Temperature/loglog_TC_arrays.pdf' 
	plot_title = 'TC Array Temperatures'
	loglog_plot(TC_exp_data, TC_FDS_data, Sigma_E, plot_min, plot_max,
		x_label, y_label, save_loc_name,plot_title)
if plot_HF:
	Sigma_E = 11.
	plot_min = 0.1
	plot_max = 200.
	x_label = r'Measured Heat Flux (kW/m$^2$)'
	y_label = r'Predicted Heat Flux (kW/m$^2$)'
	save_loc_name = save_dir+'Heat_Flux/loglog_HFs.pdf' 
	plot_title = 'Heat Flux'
	loglog_plot(HF_exp_data, HF_FDS_data, Sigma_E, plot_min, plot_max,
		x_label, y_label, save_loc_name, plot_title)
if plot_BDP:
	Sigma_E = 6.
	plot_min = 0.1
	plot_max = 10.
	x_label = 'Measured Velocity (m/s)'
	y_label = 'Predicted Velocity (m/s)'
	save_loc_name = save_dir+'Velocity/loglog_BDPs.pdf'
	plot_title = 'Gas Velocity'
	loglog_plot(Vel_exp_data, Vel_FDS_data, Sigma_E, plot_min, plot_max,
		x_label, y_label, save_loc_name, plot_title)
if plot_cjetTC:
	Sigma_E = 7.
	plot_min = 10.
	plot_max = 2000.
	x_label = r'Measured Temperature Rise ($^\circ$C)'
	y_label = r'Predicted Temperature Rise ($^\circ$C)'
	save_loc_name = save_dir+'Temperature/loglog_cjetTCs.pdf' 
	plot_title = 'Ceiling Jet Temperature'
	loglog_plot(cjet_exp_data, cjet_FDS_data, Sigma_E, plot_min, plot_max,
		x_label, y_label, save_loc_name, plot_title)
if plot_HGL:
	Sigma_E = 7.
	plot_min = 10.
	plot_max = 2000.
	x_label = r'Measured Temperature Rise ($^\circ$C)'
	y_label = r'Predicted Temperature Rise ($^\circ$C)'
	save_loc_name = save_dir+'Temperature/loglog_HGL.pdf' 
	plot_title = 'HGL Temperature'
	loglog_plot(HGL_exp_data, HGL_FDS_data, Sigma_E, plot_min, plot_max,
		x_label, y_label, save_loc_name, plot_title)
