# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:45:11 2024

@author: sjgauva
"""
# import os
# import h5py
# import numpy as np
# #%%
# def read_hdf5_files(directory):

#     # Ensure the directory path exists
#     if not os.path.exists(directory):
#         raise FileNotFoundError(f"Directory not found: {directory}")
    
#     # Get the globals dict to store variables
#     globals_dict = globals()
    
#     # Iterate through all files in the directory
#     for filename in os.listdir(directory):
#         # Check if the file is an HDF5 file
#         if filename.endswith(('.h5', '.hdf5', '.hdf')):
#             filepath = os.path.join(directory, filename)
            
#             try:
#                 # Create variable name from filename (remove extension and invalid characters)
#                 var_name = os.path.splitext(filename)[0]
#                 var_name = ''.join(c if c.isalnum() else '_' for c in var_name)
                
#                 # Open the HDF5 file
#                 with h5py.File(filepath, 'r') as f:
#                     # Dictionary to store datasets for this file
#                     file_datasets = {}
                    
#                     # Iterate through all datasets in the file
#                     for name, dataset in f.items():
#                         try:
#                             # Special handling for seconds_since_epoch
#                             if name == 'seconds_since_epoch':
#                                 try:
#                                     if len(dataset.shape) > 1:
#                                         data = dataset[:].flatten()
#                                     else:
#                                         data = dataset[:]
#                                 except Exception as flatten_err:
#                                     print(f"Error processing {name}: {flatten_err}")
#                                     continue
#                             else:
#                                 data = dataset[:]
                            
#                             file_datasets[name] = data
                            
#                         except Exception as dataset_err:
#                             print(f"Error reading dataset {name}: {dataset_err}")
                    
#                     # Store the dictionary in a variable named after the file
#                     globals_dict[var_name] = file_datasets
#                     print(f"Successfully processed {filename} into variable '{var_name}'")
            
#             except Exception as e:
#                 print(f"Error processing file {filename}: {e}")
                
# def main():
#     input_directory = r"C:/Users/sjgauva/Desktop/Thrasher/data_2024-03-29/thrasher_data/"
    
#     read_hdf5_files(input_directory)
# #%%
# if __name__ == "__main__":
#     main()
    
#%% New script for reading in, saving variables, and plotting

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import ast

def read_and_plot_hdf5_files(directory):
    # Ensure the directory path exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    print(f"Scanning directory: {directory}")
    
    # Dictionary to store all file datasets
    all_data = {}
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is an HDF5 file
        if filename.endswith(('.h5', '.hdf5', '.hdf')):
            print(f"\nProcessing file: {filename}")
            filepath = os.path.join(directory, filename)
            
            try:
                # Create variable name from filename
                var_name = os.path.splitext(filename)[0]
                var_name = ''.join(c if c.isalnum() else '_' for c in var_name)
                
                # Open the HDF5 file
                with h5py.File(filepath, 'r') as f:
                    # Print available datasets
                    print(f"Available datasets in {filename}:")
                    print(list(f.keys()))
                    
                    # Dict to store datasets for this file
                    file_datasets = {}
                    
                    # First, read seconds_since_epoch if it exists
                    time_data = None
                    if 'seconds_since_epoch' in f:
                        try:
                            time_dataset = f['seconds_since_epoch']
                            if len(time_dataset.shape) > 1:
                                time_data = time_dataset[:].flatten()
                            else:
                                time_data = time_dataset[:]
                            print(f"Time data shape: {time_data.shape if time_data is not None else 'None'}")
                            file_datasets['seconds_since_epoch'] = time_data
                        except Exception as time_err:
                            print(f"Error reading time data: {time_err}")
                    
                    # Create a figure with subplots for this file
                    datasets = [name for name in f.keys() if name != 'seconds_since_epoch']
                    if not datasets:
                        print(f"No plottable datasets found in {filename}")
                        continue
                        
                    # Iterate through all datasets in the file
                    for name in datasets:
                        try:
                            data = f[name][:]
                            print(f"Dataset {name} shape: {data.shape}")
                            file_datasets[name] = data
                            
                            if time_data is not None and len(time_data) > 0:
                                # Convert seconds since epoch to datetime
                                timestamps = [datetime.fromtimestamp(t) for t in time_data]
                                
                                # Ensure data lengths match
                                min_len = min(len(timestamps), len(data))
                            
                        except Exception as dataset_err:
                            print(f"Error reading dataset {name}: {dataset_err}")
                    
                    # Store the dictionary in all_data
                    all_data[var_name] = file_datasets
                    print(f"Successfully processed {filename} into variable '{var_name}'")
            
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    return all_data

def main():
    input_directory = r"C:/Users/sjgauva/Desktop/Thrasher/Mesa/data_2024-03-29/thrasher_data/"
    
    # Get the data and assign it to global variables
    data_dict = read_and_plot_hdf5_files(input_directory)
    
    globals().update(data_dict)
    
    # Print available variables
    print("\nAvailable data variables:")
    for var_name in data_dict.keys():
        print(f"- {var_name}")
        if var_name in globals():
            print(f"  Contains datasets: {list(globals()[var_name].keys())}")
    return data_dict

if __name__ == "__main__":
    main()

#%% Plot accel and magnetic magnitude by changing strings to each station

# PNADO_SN7, PSAON_SN4, PSCIN_SN8, PWAOF_SN1, PWBVI_SN6, PWCES_SN2, PWCTE_SN5, PWDAM_SN3

current_string = 'PSCIN_SN8'
current_station = PSCIN_SN8

current_accel_data = current_station['E4']
H1 = current_station['H1']
H2 = current_station['H2']
H3 = current_station['H3']
#%% Updated as of 12/11
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt

def bandpass(data, lowcut, highcut, fs, order=4):
    """
    Apply a zero-phase Butterworth bandpass filter to the input data.

    Parameters:
        data (array): The input data to be filtered.
        lowcut (float): The lower frequency cutoff (Hz).
        highcut (float): The upper frequency cutoff (Hz).
        fs (float): The sampling frequency (Hz).
        order (int): The order of the Butterworth filter.

    Returns:
        array: The filtered data.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')

    filtered_data = filtfilt(b, a, data)

    return filtered_data

def plot_mag_and_accel(time, mag, current_accel_data):
    
    fs = 24000
    lowcut = 5
    highcut = 5000
    
    # Apply high-pass filter to the acceleration data
    accel_filtered = bandpass(current_accel_data, lowcut, highcut, fs, order = 4)
    
    accel_filtered = np.nan_to_num(accel_filtered, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Subtract means from acceleration
    accel_filtered -= np.mean(accel_filtered)
    
    # Calculate velocity and subtract its mean
    velocity = cumulative_trapezoid(accel_filtered, time, initial=0)
    velocity -= np.mean(velocity)
    
    # Calculate position and subtract its mean
    position = cumulative_trapezoid(velocity, time, initial=0)
    #position -= np.mean(position)
    
    # Convert position from meters to millimeters
    position_mm = position * 1000
    position_mm -= np.mean(position_mm)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # First panel: Filtered acceleration and magnetic field
    ax1.plot(time, accel_filtered*1000*(1/20), color='red', label='Filtered Acceleration (g)')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time, mag, color='blue', label='Magnetic Field Magnitude (nT)')
    ax1.set_ylabel('Acceleration (g)')
    ax1_twin.set_ylabel('Magnetic Field Magnitude (nT)', color='blue')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.set_title(current_string + ' Filtered Acceleration and Magnetic Field')
    
    # Second panel: Velocity and magnetic field
    ax2.plot(time, velocity, color='green', label='Velocity (m/s)')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(time, mag, color='blue', label='Magnetic Field Magnitude (nT)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2_twin.set_ylabel('Magnetic Field Magnitude (nT)', color='blue')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.set_title(current_string + ' Velocity and Magnetic Field')
    
    # Inflate velocity axis for better visibility
    ax2.set_ylim(np.min(velocity) * 1.2, np.max(velocity) * 1.2)
    
    # Third panel: Position and magnetic field
    ax3.plot(time, position_mm, color='purple', label='Position (mm)')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(time, mag, color='blue', label='Magnetic Field Magnitude (nT)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (mm)')
    ax3_twin.set_ylabel('Magnetic Field Magnitude (nT)', color='blue')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.set_title(current_string + ' Position and Magnetic Field')
    
    ax3.set_ylim(-0.237, -0.226) 
    
    """
    PNADO: -0.88, -0.83, PSAON:-.55, -.49, PSCIN: -0.237,-0.226, PWAOF: -0.352,-0.343
    PWBVI: -0.1485,-0.1450, PWCES: -0.0672,-0.063, PWCTE: 
        
    """
    
    plt.tight_layout()
    plt.show()
    
    return position, velocity, accel_filtered

# Calculate magnetic field magnitude
mag = np.sqrt(H1**2 + H2**2 + H3**2)

# Create time array
time = np.arange(len(current_accel_data)) * 1/24000

# Call the new plotting function
plot_mag_and_accel(time, mag, current_accel_data)

#%% plot average magnetic field magnitude for 5-10s before and after signal arrival. 
### 

def analyze_magnetic_window(time, mag, arrival_time, station_name, window_size=10):
    """
    Analyze magnetic field magnitude before and after arrival time.
    """
    # Find the index closest to arrival time
    arrival_idx = np.abs(time - arrival_time).argmin()
    
    # Calculate indices for window before and after
    fs = 24000  # sampling rate
    window_samples = int(window_size * fs)
    start_idx = max(0, arrival_idx - window_samples)
    end_idx = min(len(mag), arrival_idx + window_samples)
    
    # Extract time and magnitude within window
    window_time = time[start_idx:end_idx] - arrival_time  # Center at arrival time
    window_mag = mag[start_idx:end_idx]
    
    # Calculate rolling mean and std using the full window
    rolling_mean = np.zeros_like(window_mag)
    rolling_std = np.zeros_like(window_mag)
    
    # Use a smaller rolling window (e.g., 0.1 seconds) for the statistics
    roll_window = int(0.1 * fs)  # 0.1 second window
    
    for i in range(len(window_mag)):
        start = max(0, i - roll_window)
        end = min(len(window_mag), i + roll_window)
        rolling_mean[i] = np.mean(window_mag[start:end])
        rolling_std[i] = np.std(window_mag[start:end])
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    #plt.plot(window_time, window_mag, 'b-', alpha=0.5, label='Magnetic Field Magnitude')
    #plt.axvline(x=0, color='r', linestyle='--', label='Arrival Time')
    
    # Plot rolling mean and standard deviation
    plt.plot(window_time, rolling_mean, 'g-', label='Running Mean')
    plt.fill_between(window_time, 
                    rolling_mean - rolling_std, 
                    rolling_mean + rolling_std, 
                    color='g', 
                    alpha=0.2, 
                    label='±1 STD')
    
    plt.xlabel('Time relative to arrival (s)')
    plt.ylabel('Magnetic Field Magnitude (nT)')
    plt.title(station_name + ' Magnetic Field Running Mean & Standard Deviation')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    return np.mean(rolling_mean), np.mean(rolling_std)

arrival_times = {
    'PNADO_SN7': 181.2,
    'PSAON_SN4': 211.19,
    'PSCIN_SN8': 202.30,
    'PWAOF_SN1': 191.15,
    'PWBVI_SN6': 212.21,
    'PWCES_SN2': 219.3,
    'PWCTE_SN5': 206.45,
    'PWDAM_SN3': 210.6
}
#%%
current_station = PNADO_SN7
H1 = current_station['H1']
H2 = current_station['H2']
H3 = current_station['H3']
mag = np.sqrt(H1**2 + H2**2 + H3**2)
time = np.arange(len(mag)) * 1/24000

# Get station name from the variable name assigned to current_station
station_name = [name for name, value in globals().items() if value is current_station][0]

# Analyze the window around arrival time
mean_mag, std_mag = analyze_magnetic_window(time, mag, arrival_times[station_name], station_name)
print(f"\nStatistics for {station_name}:")
print(f"Mean magnetic field magnitude: {mean_mag:.2f} nT")
print(f"Standard deviation: {std_mag:.2f} nT")