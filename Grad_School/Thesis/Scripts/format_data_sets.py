# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#!/usr/bin/env python

# Commands to run FDS on blaze and copy files over to JMW_HDD
# qfds.sh -p 8 -o 4 -e FireModels_fork/fds/Build/mpi_intel_linux_64ib/fds_mpi_intel_linux_64ib FireModels_fork/fds/Validation/DelCo_Trainers/FDS_Input_Files/Test_03.fds

# qfds.sh -p 8 -e FireModels_fork/fds/Build/mpi_intel_linux_64ib/fds_mpi_intel_linux_64ib Test_04_10cm.fds

# scp jmw3@blaze.nist.gov:Test_24* ../../Volumes/JMW_ExtHDD/jmw_files/School/Grad_School/Thesis/FDS_Output_Files/Test_24/


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

# Specify name
specify_test = False
specific_name = 'Test_5_East_062614'

# Specify structure
specify_struct = False
specific_struct = 'West'

# Files to skip
skip_files = ['_times', '_reduced', 'description_','zero_','_rh','burn','helmet','ppv_']

# Duration of pre-test time for bi-directional probes and heat flux gauges (s)
pre_test_time = 60

# Location of experimental data files
data_dir = '../Experimental_Data/'

# Location of formatted data file directory
save_dir = '../Experimental_Data/Formatted_Data/'

# Location of file with timing information
all_times_file = '../Experimental_Data/All_Times.csv'

# Location of test description file
info_file = '../Experimental_Data/Description_of_Experiments.csv'

# Load exp. timings and description file
all_times = pd.read_csv(all_times_file)
all_times = all_times.set_index('Time')
info = pd.read_csv(info_file, index_col=3)

#  ==========================
#  = User defined functions =
#  ==========================

# Prints an error message and stops code
def error_message(message):
    lineno = inspect.currentframe().f_back.f_lineno
    print '[ERROR, line '+str(lineno)+']:'
    print '  ' + message
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
            if specific_struct not in test_name:
                return(True)
        elif specific_struct == 'East': 
            if 'West' in test_name:
                return(True)
        else:
            error_message('Invalid name for specific_struct')

    return(False)

#  ===============================
#  = Loop through all data files =
#  ===============================
for f in os.listdir(data_dir):
    if f.endswith('.csv'):
        # Skip files with time information or reduced data files
        if any([substring in f.lower() for substring in skip_files]):
            continue

        # Strip test name from file name
        test_name = f[:-4]

        if check_name(test_name):     # check if file should be skipped
            continue
        else:   # Load exp. data file
            data = pd.read_csv(data_dir + f)
            # data = data.drop(data.index[0], axis=1)     # drops timestamp       
            print ('--- Loaded ' + test_name + ' ---')

        # Create group and channel lists
        if 'West' in test_name:
            channel_list_file = '../DAQ_Files/DAQ_Files_2014/West_DelCo_DAQ_Channel_List.csv'
        elif 'East' in test_name:
            channel_list_file = '../DAQ_Files/DAQ_Files_2014/East_DelCo_DAQ_Channel_List.csv'
        else:
            error_message('Neither "West" nor "East" in test_name')
        
        channel_list = pd.read_csv(channel_list_file)
        channel_list = channel_list.set_index('Device Name')
        channel_groups = channel_list.groupby('Group Name')

        # Read in test times to offset plots.
        if 'West' in test_name:     # ignore first 2 time entries => 3rd entry corresponds to 1st burner ignited
            events = all_times[test_name].dropna()[2:]
            # gasA_lag_time = 12   # [s]; gas analyzer A lag time for west burner tests
            # gasB_lag_time = 12   # [s]; gas analyzer B lag time for west burner tests
        else:     # ignore first time entry => 2nd entry corresponds to 1st burner ignited 
            events = all_times[test_name].dropna()[1:]
            # gasA_lag_time = 12   # [s]; gas analyzer A lag time for east burner tests
            # gasB_lag_time = 35   # [s]; gas analyzer B lag time for east burner tests

        # adjust times so t=0 corresponds to ignition of 1st burner
        offset_time = events.index.values[0]
        new_times = events.index.values - int(offset_time)
            
        # create series of event names & new times 
        events = pd.Series(events.values, index=new_times)
        
        # adjust times in data file to correspond with t=0 being ignition of 1st burner
        data['Time'] = data['Time'].values - offset_time
        corrected_data = data.drop('Time', axis=1)
        corrected_data.insert(0, 'Time', data['Time'])
        corrected_data = corrected_data.set_index('Time')
        
        # reduce data to relevant data (60 seconds of background data before experimental data)
        # Note: row corresponding to -61 will be replaced with unit headers
        reduced_data = corrected_data.loc[-61:, :]
        
        # set up dataframe to be filled with relevant processed sensor data
        final_reduced_data = pd.DataFrame(index=reduced_data.index)

        # Process data for each quantity group
        for group in channel_groups.groups:
            # Skip excluded groups listed in test description file
            if any([substring in group for substring in info['Excluded Groups'][test_name].split('|')]):
                continue

            for channel in channel_groups.get_group(group).index.values:            
                # Skip plot quantity if channel name is blank
                if pd.isnull(channel):
                    continue 

                # Scale channel depending on quantity
                current_channel_data = reduced_data[channel]
                calibration_slope = float(channel_list['Calibration Slope'][channel])
                calibration_intercept = float(channel_list['Calibration Intercept'][channel])
                
                # Skip excluded channels listed in test description file
                if any([substring in channel for substring in info['Excluded Channels'][test_name].split('|')]):
                    current_channel_data = current_channel_data.replace(to_replace=current_channel_data, value='NaN')
                # Temperature
                elif channel_list['Measurement Type'][channel] == 'Temperature':
                    current_channel_data = current_channel_data * calibration_slope + calibration_intercept
                # Velocity
                elif channel_list['Measurement Type'][channel] == 'Velocity':
                    conv_inch_h2o = 0.4
                    conv_pascal = 248.8
                    zero_voltage = np.mean(current_channel_data.loc[-61:-5])  # Get zero voltage from pre-test data
                    pressure = conv_inch_h2o * conv_pascal * (current_channel_data - zero_voltage)  # Convert voltage to pascals
                    current_channel_data = 0.0698 * np.sqrt(np.abs(pressure) * (reduced_data['TC_' + channel[4:]] + 273.15)) * np.sign(pressure)
                # Heat Flux
                elif channel_list['Measurement Type'][channel] == 'Heat Flux':
                    zero_voltage = np.mean(current_channel_data.loc[-61:-5])  # Get zero voltage from pre-test data
                    current_channel_data = (current_channel_data - zero_voltage) * calibration_slope + calibration_intercept
                # Pressure
                elif channel_list['Measurement Type'][channel] == 'Pressure':
                    conv_inch_h2o = 0.4
                    conv_pascal = 248.8
                    zero_voltage = np.mean(current_channel_data.loc[-61:-5])  # Convert voltage to pascals
                    current_channel_data = conv_inch_h2o * conv_pascal * (current_channel_data - zero_voltage)  # Get zero voltage from pre-test data
                # Gas
                elif channel_list['Measurement Type'][channel] == 'Gas':
                    # if channel[-1] == 'A':
                    #     shift_data = gasA_lag_time
                    # elif channel[-1] == 'B':
                    #     shift_data = gasB_lag_time
                    # else:
                    #     print '[ERROR] Neither A nor B read from Gas channel '+ channel
                    #     sys.exit()
                    # Create list of data shifted according to analyzer lag time and calculate zero voltage
                    # current_channel_data = current_channel_data.loc[-61+shift_data:].values
                    # zero_voltage = np.mean(corrected_data[channel].loc[-71+shift_data:-15+shift_data])

                    zero_voltage = np.mean(current_channel_data.loc[-61:-5])

                    if 'CO' in channel:
                        current_channel_data = (current_channel_data-zero_voltage)*calibration_slope + calibration_intercept
                    else:
                        zero_voltage = zero_voltage-1.
                        current_channel_data = (current_channel_data-zero_voltage)*20.95 + calibration_intercept
                    # final_reduced_data[channel] = ''
                    # final_reduced_data[channel].iloc[0:-shift_data] = current_channel_data
                    # final_reduced_data[channel] = final_reduced_data[channel].replace(to_replace='', value='NaN')
                    # continue
                # Hose
                elif channel_list['Measurement Type'][channel] == 'Hose':
                    # Skip data other than sensors on 2.5 inch hoseline
                    if '2p5' not in channel:
                        continue
                    current_channel_data = current_channel_data * calibration_slope + calibration_intercept
                
                # Save converted channel data back to exp. dataframe
                final_reduced_data[channel] = current_channel_data

        
        units = []
        for heading in final_reduced_data.columns.values:
            if 'TC_' in heading:
                units.append('C')
            elif 'BDP_' in heading:
                units.append('m/s')
            elif 'RAD_' in heading or 'HF_' in heading:
                units.append('kW/m2')
            elif 'CO_' in heading or 'O2_' in heading or 'CO2_' in heading:
                units.append('%')
            else:
                error_message('No units found for value ' + heading)

        final_reduced_data.loc[-61, : ] = units
        final_reduced_data.to_csv(save_dir + test_name + '.csv')
        print 'Saved data set for ' + test_name
        print 
        continue

    # start_of_test = info['Start of Test'][test_name]
    # end_of_test = info['End of Test'][test_name]

    # # Offset data time to start of test
    # data['Time'] = data['Time'].values - start_of_test

    # data_copy = data.drop('Time', axis=1)
    # data_copy = pd.rolling_mean(data_copy, data_time_averaging_window, center=True)
    # data_copy.insert(0, 'Time', data['Time'])
    # data_copy = data_copy.dropna()
    # data = data_copy

