# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:25:20 2024

@author: sjgauva
"""

###############################################################################
# This code is reading csv files converted from the raw xml data (native DTS
#   output) via Silixa DTS Viewer Lite. Temperature data is calculated
#   using default parameters (e.g., differential attenuation = 0.025). 
#   Data plotted here is therefore considered uncalibrated and additional 
#   calibration is required.

# The plots:
#   Temperature variation with distance and time (colors are temperatures)

# Christian Stanciu @ SNL, Feb 2024
###############################################################################
import obspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as ticker
import statistics
import math
import pprint
from mpl_toolkits import mplot3d
import csv
import glob, os 
from matplotlib.pyplot import cm
from obspy import UTCDateTime
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
from matplotlib.lines import Line2D

#%% Get data directory (csv files)
dir_path = 'C:/Users/sjgauva/Desktop/DAS/DTS/DTS2024_csv/'

# Get all files within the directory above
file_list = []
for file in os.listdir(dir_path):
    #Check only csv files
    #if file.endswith('.csv'):
    if file.startswith('channel'):
        file_list.append(file)
        
# Grab the last 23 characters of the file name (this is for sorting)
def last_23chars(x):
    return(x[-23:])

# Sort files using the key above        
sfiles=sorted(file_list, key = last_23chars)  

# Count the number of files
count=len(sfiles)

# Empty panda data frames
allTemps = pd.DataFrame()
movAverages = pd.DataFrame()
allDistances  = pd.DataFrame()
allTimes =  pd.DataFrame()
allStokes = pd.DataFrame()
allAnti = pd.DataFrame()

# Empty lists to handle the time
timeString0 = []
timeString = []
file_utc_times = []
file_time_epochs = []
#%% Read all csv files (each file is 1 channel/waveform)

full_range = np.arange(0,count,1)
jan31_feb2 = full_range[644:781]

feb1_feb2 = full_range[675:700]
feb7_feb9 = full_range[1000:1100]

feb1_anomaly = full_range[665:710] # 685:686 originally
feb1_good = full_range[700:701]

feb13_anomaly = full_range[1270:1320] # 1292:1293 originally
feb13_good = full_range[1299:1300]

feb7_good = full_range[999:1000] 
feb7_anomaly = full_range[985:1040] # 1015:1016 originally

feb16_anomaly = full_range[1430:1483] # 1452:1453 originally
feb16_good = full_range[1442:1443]

jan13_anomaly = full_range[65:97] # 86:87 originally
jan13_good = full_range[90:91]

for n in feb16_anomaly:
    fileName=dir_path + sfiles[n]
    # Split file name string and get the time
    parts = fileName.split('_')
    timestr = parts[4] + parts[5] #get string 0 from the split
    print(n, timestr)
    
    # Store time in corresponding variables
    year = int(timestr[0:4])
    month = int(timestr[4:6])
    day = int(timestr[6:8])
    hour = int(timestr[8:10])
    minute = int(timestr[10:12])
    seconds = float(timestr[12:14])
    utc = UTCDateTime(year, month, day, hour, minute, seconds)
    file_utc_times.append(utc)
    
    # Get epoch time as oposed to UTC date time to be able to plot
    file_time_epochs.append(utc.timestamp)
    
    tmp0 = str(year) + str(month).zfill(2) + str(day).zfill(2) + \
        str(hour).zfill(2) + str(minute).zfill(2)
    tmp = str(month).zfill(2) + str(day).zfill(2) + str(hour).zfill(2) \
        + str(minute).zfill(2)
    timeString0.append(tmp0)
    timeString.append(tmp)

    # Read each temperature file
    data = pd.read_csv(dir_path + sfiles[n], encoding='ISO-8859–1', skiprows=1)   
    
    # Add distance to the data frame
    allDistances = pd.concat([allDistances, \
                              pd.DataFrame(data["distance"]/1000)], axis=1)  

    # Add temperature to the data frame
    allTemps = pd.concat([allTemps, pd.DataFrame(data["temperature"])], axis=1) 
    
    allStokes = pd.concat([allStokes, pd.DataFrame(data["forward stokes"])], axis=1) 
    
    allAnti = pd.concat([allAnti, pd.DataFrame(data[" forward antistokes"])], axis=1) 

allTimes = pd.DataFrame(np.repeat(pd.DataFrame(file_time_epochs), \
                                  len(data['distance']), axis=1))
allTimes = allTimes.transpose()
#Convert from epoch-times to float otherwise can't use the formater
for col in allTimes.columns:
    allTimes[col] = pd.to_datetime(allTimes[col], unit='s', errors='coerce')

allTemps.columns = range(len(allTemps.columns))

temp_changes = allTemps.max() - allTemps.min()
max_change_column = temp_changes.idxmax()
max_change_index = allTemps.columns.get_loc(max_change_column)
print('Greatest temperature change at index ' + str(max_change_index))
#% throw out last value if size is mismatched
# allTemps = allTemps[:-1]
# allDistances = allDistances[:-1]
# allStokes = allStokes[:-1]
# allAnti = allAnti[:-1]
#%% Calculate wavelength over distance

def calculate_antistokes_wavelength(intensity, baseline_wavelength=1465):
    scaling_factor = 0.1  # nm per unit intensity
    return baseline_wavelength + (intensity * scaling_factor)

data['antistokes_wavelength'] = data[' forward antistokes'].apply(calculate_antistokes_wavelength)

#%% Plot above

anti_wavelength = data['antistokes_wavelength']

plt.figure(figsize=(12, 6))
plt.plot(allDistances, anti_wavelength)
plt.xlabel('Fiber Distance (km)')
plt.ylabel('Anti-Stokes Wavelength (nm)')

plt.title(f'{allTimes.iloc[0, 0]}' + ' Anomaly Anti-Stokes Wavelengths', fontsize=16)
plt.grid(True)
# plt.xlim(16500, 20000)
# plt.ylim(1400, 1500)
plt.show()


#%% ---------------------------------------------------------------------------
#                                2D SPECTROGRAM PLOT
#   --------------------------------------------------------------------------- 
# Min/Max of temp, stokes, or anti

# allAnti.columns = range(allAnti.shape[1])
# allStokes.columns = range(allStokes.shape[1])

# stokes_min = np.min(allStokes)
# stokes_max = np.max(allStokes)
# anti_min = np.min(allAnti)
# anti_max = np.max(allAnti)
# vmin = -20
# vmax = 30

# # Make figure
# fig, ax = plt.subplots() 

# psd = ax.contourf(allDistances, allTimes, allTemps, 500, cmap='viridis', \
#                   vmin=vmin, vmax=vmax)

# # yticks = allTimes.iloc[:, 0].tolist()
# # yticklabels = [str(time) for time in yticks]
# # ax.set_yticks(yticks)
# # ax.set_yticklabels(yticklabels)

# ax.set_xticks(np.arange(0, 30.543, 5))
# #ax.set_yticks(np.arange([], [], 1), minor=True)

# ax.set_xlim(0,30)

# ax.set_xlabel('Distance along fiber [km]')
# fig.set_dpi(300)

# from obspy import UTCDateTime


#%% plot up each column of stokes/anti/temp
    
os.chdir(dir_path)

start_time_str = str(allTimes.iloc[0, 0]).replace(' ', '_').replace(':', '_')
end_time_str = str(allTimes.iloc[-1, -1]).replace(' ', '_').replace(':', '_')

filename = f'DTS_V5{start_time_str}_to_{end_time_str}.png'

def plot_data(allStokes, allAnti, allTemps, allTimes):
    
    # remove DC offset from allAnti
    #allAnti = allAnti.apply(lambda x: x - x.mean(), axis=0)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Plot allStokes and allAnti on the first axis
    for column in allStokes.columns:
        axes[0].plot(allStokes.index, allStokes[column])
    
    for column in allAnti.columns:
        axes[0].plot(allAnti.index, allAnti[column])

    axes[0].set_title('Stokes and Anti-Stokes')
    axes[0].grid(True)
    axes[0].legend(['Stokes', 'Anti'], loc='upper right')
    axes[0].set_xlim([16500, 20000])
    axes[0].set_ylim([33,63])

    # Plot allTemps on the second axis
    for column in allTemps.columns:
        axes[1].plot(allTemps.index, allTemps[column])

    axes[1].set_title('Temperature')
    axes[1].grid(True)
    axes[1].set_xlim([16500, 20000])
    axes[1].set_xlabel('Distance (m)')
    axes[1].set_ylim([0,40])

    fig.suptitle(f'{allTimes.iloc[0, 0]}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    #plt.savefig(filename)
    plt.show()
    
plot_data(allStokes, allAnti, allTemps, allTimes)

#%% Antistokes percentage change

anti_mean = np.mean(allAnti)
percentage_change = ((allAnti-anti_mean)/anti_mean)*100

plt.figure(figsize=(10, 6))

plt.plot(allDistances, percentage_change)
plt.xlabel('DTS Distance (km)', fontsize = 16)
plt.ylabel('Antistokes % Change from Mean', fontsize = 16)
plt.xlim([5,30])
plt.ylim([-80,60])
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)

time_string = str(allTimes.iloc[0, 0])
shortened_time = time_string[:-9]

plt.suptitle('Forward Antistokes Comparison on ' + shortened_time, fontsize=18)

point_x = 11.79
point_y = 1.3

point_x2 = 16.15
point_y2 = -36.56

point_x3 = 7.5
point_y3 = 43.6

plt.plot(point_x3, point_y3, marker='*', markersize=16, color='red', markeredgecolor='black', markeredgewidth=1.5)
plt.plot(point_x2, point_y2, marker='*', markersize=16, color='red', markeredgecolor='black', markeredgewidth=1.5)
plt.plot(point_x, point_y, marker='*', markersize=16, color='red', markeredgecolor='black', markeredgewidth=1.5)

legend_labels = [allTimes.iloc[0, i].strftime('%Y-%m-%d %H:%M') for i in range(len(allAnti.columns))]

legend_labels.append('Cable Splice')
plt.plot([], [], marker='*', markersize=16, color='red', markeredgecolor='black', markeredgewidth=1.5, label='Cable Splice', linestyle = None)

plt.legend(legend_labels, fontsize=16, loc='best')

#%% 9/5 animation/anomaly classification
anomaly_events = []

#%% Run to append new info to anomaly table

animation_data = []
is_anomaly = False
anomaly_start = None
anomaly_peak = None
anomaly_peak_value = 0
cooldown_period = pd.Timedelta(hours=0.5)
cooldown_start = None

def detect_anomaly(current_data, baseline, distances, threshold=0.2, min_anomaly_distance=2): # threshold 10 originally
    diff_percentage = ((current_data - baseline) / np.abs(baseline)) * 100
    anomalous_points = np.abs(diff_percentage) > threshold
    
    from scipy.ndimage import label
    labeled_array, num_features = label(anomalous_points)
    
    for i in range(1, num_features + 1):
        region = labeled_array == i
        region_distances = distances[region]
        if np.ptp(region_distances) >= min_anomaly_distance:
            return True
    
    return False

# Calculate baseline from first 10 points in time
baseline = percentage_change.iloc[:, :10].mean(axis=1)

for i in range(percentage_change.shape[1]):
    col_data = percentage_change.iloc[:, i]
    current_time = allTimes.iloc[0, i]
    current_anomaly = detect_anomaly(col_data, baseline, allDistances)
    
    # Calculate the maximum change from baseline across all distances
    current_deviation = np.max(col_data - baseline)
    
    is_peak = False  # Initialize is_peak for this time step
    
    if current_anomaly:
        if not is_anomaly:
            anomaly_start = current_time
            is_anomaly = True
            cooldown_start = None
            anomaly_peak = current_time
            anomaly_peak_value = current_deviation
            is_peak = True  # First point of anomaly is initially the peak
        else:
            # Update peak if current deviation is larger
            if current_deviation > anomaly_peak_value:
                anomaly_peak = current_time
                anomaly_peak_value = current_deviation
                is_peak = True  # Update peak
    
    if not current_anomaly and is_anomaly:
        if cooldown_start is None:
            cooldown_start = current_time
        elif current_time - cooldown_start >= cooldown_period:
            anomaly_end = current_time
            
            # Calculate duration in hours
            duration = (anomaly_end - anomaly_start).total_seconds() / 3600
            
            anomaly_events.append({
                'onset': anomaly_start,
                'peak': anomaly_peak,
                'offset': anomaly_end,
                'peak_value': anomaly_peak_value,
                'duration_hours': duration
            })
            
            is_anomaly = False
            anomaly_start = None
            anomaly_peak = None
            anomaly_peak_value = 0
            cooldown_start = None
    elif current_anomaly:
        cooldown_start = None
    
    animation_data.append({
        'time': current_time,
        'data': col_data,
        'is_anomaly': is_anomaly,
        'is_peak': is_peak,
        'is_onset': current_time == anomaly_start and is_anomaly
    })

# Create the animation
fig, ax = plt.subplots(figsize=(12, 6))
legend_elements = [Line2D([0], [0], color='blue', label='Normal'),
                   Line2D([0], [0], color='red', label='Anomaly')]

def animate(frame):
    ax.clear()
    data = animation_data[frame]
    
    if data['is_anomaly']:
        color = 'red'
    else:
        color = 'blue'
    
    ax.plot(allDistances, data['data'], color=color)
    
    ax.set_xlabel('DTS Distance (km)', fontsize=12)
    ax.set_ylabel('Antistokes % Change from Mean', fontsize=12)
    ax.set_xlim([5, 30])
    ax.set_ylim([-80, 60])
    ax.set_title(f"Antistokes Percentage Change at {data['time']}", fontsize=14)
    ax.legend(handles=legend_elements, loc='upper right')

ani = FuncAnimation(fig, animate, frames=len(animation_data), interval=400, repeat=False)
ani.save('dts_anomaly_animation.gif', writer='pillow')

# Print anomaly events for verification
for i, event in enumerate(anomaly_events):
    print(f"Anomaly Event {i+1}:")
    print(f"  Onset: {event['onset']}")
    print(f"  Peak: {event['peak']}")
    print(f"  Offset: {event['offset']}")
    print(f"  Peak Value: {event['peak_value']:.2f}")
    print(f"  Duration (hours): {event['duration_hours']:.2f}")
    print("---")
#%% Save table of anomaly timing information

anomaly_table = pd.DataFrame(anomaly_events)
print(anomaly_table)
anomaly_table.to_csv('dts_anomaly_table.csv', index=False)
#%% find wavelengths of anti/stokes with xml data

import re
import xml.etree.ElementTree as ET

xml_directory = 'C:/Users/sjgauva/Desktop/DAS/DTS/DTS2024_xml/'
xml_filename = fileName 

def extract_datetime_from_filename(filename):
    match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}', filename)
    if not match:
        raise ValueError("Filename does not contain a valid datetime string")
    
    datetime_str = match.group().replace('-', '').replace('_', '')
    return datetime_str

def read_xml_with_datetime(directory, filename):
    """
    Reads an XML file from the specified directory if its name contains the datetime string
    extracted from the given filename.
    
    Args:
        directory (str): The path to the directory containing the XML files.
        filename (str): The filename from which to extract the datetime string.
    
    Returns:
        xml.etree.ElementTree.Element: The root element of the parsed XML file.
    """
    # Extract the datetime string from the filename
    datetime_str = extract_datetime_from_filename(filename)
    
    # Convert to the format used in the XML files (e.g., 20240201_012936)
    xml_datetime_str = datetime_str[:8] + '_' + datetime_str[8:]
    
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter files that contain the xml_datetime_str
    matching_files = [file for file in files if xml_datetime_str in file and file.endswith('.xml')]
    
    if not matching_files:
        raise FileNotFoundError(f"No XML files found in the directory '{directory}' containing datetime string '{xml_datetime_str}'")
    
    # Read the first matching file
    xml_file_path = os.path.join(directory, matching_files[0])
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    return root


try:
    root = read_xml_with_datetime(xml_directory, xml_filename)
    #print(f"Successfully read the XML file: {ET.tostring(root, encoding='unicode')}")
except (FileNotFoundError, ValueError) as e:
    print(e)

#%% compare log data from 2 xml files

from difflib import unified_diff

directory = xml_directory
filename1 = xml_filename 
filename2 = 'C:/Users/sjgauva/Desktop/DAS/DTS/DTS2024_csv/channel 1_UTC+0000_DST0_20240201_012936.596.csv'

def extract_datetime_from_filename(filename):
    """
    Extracts the datetime string from a filename formatted as 
    'channel 1_UTC+0000_DST0_YYYYMMDD_HHMMSS.sss.csv'.
    
    Args:
        filename (str): The filename from which to extract the datetime string.
    
    Returns:
        str: The extracted datetime string in the format yearmonthday_hourminutesecond.
    """
    match = re.search(r'_(\d{8}_\d{6})\.\d+', filename)
    if not match:
        raise ValueError("Filename does not contain a valid datetime string")
    
    datetime_str = match.group(1).replace('_', '')
    return datetime_str

def read_xml_with_datetime(directory, datetime_str):
    """
    Reads an XML file from the specified directory if its name contains the given datetime string.
    
    Args:
        directory (str): The path to the directory containing the XML files.
        datetime_str (str): The datetime string to look for in the XML file names.
    
    Returns:
        xml.etree.ElementTree.Element: The root element of the parsed XML file.
    """
    # Convert to the format used in the XML files (e.g., 20240201_012936)
    xml_datetime_str = datetime_str[:8] + '_' + datetime_str[8:]
    
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter files that contain the xml_datetime_str
    matching_files = [file for file in files if xml_datetime_str in file and file.endswith('.xml')]
    
    if not matching_files:
        raise FileNotFoundError(f"No XML files found in the directory '{directory}' containing datetime string '{xml_datetime_str}'")
    
    # Read the first matching file
    xml_file_path = os.path.join(directory, matching_files[0])
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    return root

def compare_xml_files(file1_root, file2_root):
    """
    Compares the content of two XML files starting from the </logData> tag.
    
    Args:
        file1_root (xml.etree.ElementTree.Element): The root element of the first XML file.
        file2_root (xml.etree.ElementTree.Element): The root element of the second XML file.
    
    Prints:
        The differences between the two XML files.
    """
    def get_text_from_log_data(element):
        return ''.join(element.itertext())

    log_data_1 = file1_root.find('.//logData')
    log_data_2 = file2_root.find('.//logData')
    
    if log_data_1 is None or log_data_2 is None:
        raise ValueError("One of the XML files does not contain a </logData> tag.")
    
    log_data_1_text = get_text_from_log_data(log_data_1)
    log_data_2_text = get_text_from_log_data(log_data_2)

    diff = unified_diff(
        log_data_1_text.splitlines(keepends=True),
        log_data_2_text.splitlines(keepends=True),
        fromfile='file1',
        tofile='file2'
    )

    for line in diff:
        print(line, end='')

try:
    datetime_str1 = extract_datetime_from_filename(filename1)
    datetime_str2 = extract_datetime_from_filename(filename2)
    
    root1 = read_xml_with_datetime(directory, datetime_str1)
    root2 = read_xml_with_datetime(directory, datetime_str2)
    
    compare_xml_files(root1, root2)
    
except (FileNotFoundError, ValueError) as e:
    print(e)
#%%
temp_changes = allTemps.max() - allTemps.min()
max_change_column = temp_changes.idxmax()
max_change_index = allTemps.columns.get_loc(max_change_column)

#%% Add colorbar (edited 5/2)
# psd.colorbar(ScalarMappable(norm=psd.norm, cmap=psd.cmap), \
#              ticks=range(vmin, vmax+5, 5))
   
plt.colorbar(ScalarMappable(norm=psd.norm, cmap=psd.cmap), ax=ax, orientation='horizontal', label='Power spectral density')
# File name 
namestr = 'test'

# Figure title    
plt.title('Uncalibrated temperature 202401 - 202402')

# Save figure
#plt.savefig(namestr + '.png', dpi=300, bbox_inches='tight')