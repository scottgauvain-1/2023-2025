# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:35:47 2024

@author: sjgauva
"""

staLat =   70.510829 #N 
staLon = -149.869864 #E

import os 
os.chdir('C:/Users/sjgauva/Desktop/DAS/code')

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from obspy.taup import TauPyModel # 
#from geopy import distance, Point
import math
import scipy.signal
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Stream, Trace
from tdms_to_obspy import *
import glob
import pyproj
from datetime import timedelta
import pickle
#%% Constants and parameters

gauge_length = 10  # meters
sampling_rate = 100 
total_fiber_length = 40000  # meters
target_distances = [500, 1000, 5000, 15000]  # Target distances in meters

# def calculate_temperature_change(st1, gauge_length, sampling_rate):
#     """
#     Calculate temperature change from DAS data with absolute normalization.
#     """
#     alpha = 0.48e-6
#     dn_dT = 10e-5
#     n = 7e-6
    
#     k = []
#     for tr in st1:
#         normalized_data = (tr.data - np.min(tr.data)) / (np.max(tr.data) - np.min(tr.data))
#         strain_rate = np.gradient(normalized_data, axis=0) / (gauge_length * sampling_rate)
#         dT = strain_rate / (2 * (alpha + dn_dT/n))
#         k.append(dT)
    
#     return k

def process_stream(st, coordinates, freqmin=0.0001, freqmax=0.001):
    """
    Process and filter the stream data.
    """
    st_processed = Stream()
    
    for tr in st:
        if np.any(~np.isnan(tr.data)):
            data = tr.data[~np.isnan(tr.data)]
            new_tr = Trace(data=data, header=tr.stats)
            new_tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)
            st_processed += new_tr

    for itr, tr in enumerate(st_processed):
        if itr < len(coordinates):
            tr.stats.stla = coordinates['Latitude'].iloc[itr]
            tr.stats.stlo = coordinates['Longitude'].iloc[itr]
            tr.stats.distance = coordinates['Distance'].iloc[itr] * 1000

    return st_processed

def find_nearest_channel(coordinates, target_distance):
    """
    Find the channel closest to the target distance.
    """ 
    return (coordinates['Distance'] * 1000 - target_distance).abs().idxmin()

def process_day_folder(folder_path, coordinates, target_distances):
    """
    Process data from a single day folder.
    """
    all_files = glob.glob(os.path.join(folder_path, "Continuous_UTC_*.tdms"))
    all_files.sort()
    
    st = Stream()
    for file in all_files[:10]:  # Process first 10 files for demonstration
        print(f"Reading file: {file}")
        st += read_TDMS_to_Stream(file)
    
    st_processed = process_stream(st, coordinates)
    
    target_streams = {dist: Stream() for dist in target_distances}
    for dist in target_distances:
        channel = find_nearest_channel(coordinates, dist)
        if channel < len(st_processed):
            target_streams[dist] += st_processed[channel]
    
    return target_streams

def read_and_process_data(root_dir, coord_file, start_date, end_date, target_distances):
    """
    Read and process data for the specified date range.
    """
    coordinates = pd.read_csv(coord_file)
    
    merged_streams = {dist: Stream() for dist in target_distances}
    
    current_date = start_date
    while current_date < end_date:
        folder_name = current_date.strftime("%Y%m%d_05")
        folder_path = os.path.join(root_dir, folder_name)
        
        if os.path.exists(folder_path):
            print(f"Processing folder: {folder_path}")
            daily_streams = process_day_folder(folder_path, coordinates, target_distances)
            
            for dist, stream in daily_streams.items():
                merged_streams[dist] += stream
        
        current_date += timedelta(days=1)
    
    return merged_streams, coordinates

def save_processed_data(data, filename):
    """
    Save processed data to a file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_processed_data(filename):
    """
    Load processed data from a file.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Data Reading and Processing
def main_data_processing():
    root_dir = "//snl//Collaborative//8861_Geophysical_RawData//astanci//PEMDATS//LF-DAS//"
    coord_file = "coordinates.csv"
    start_date = UTCDateTime("2023-11-08")
    end_date = UTCDateTime("2023-11-12")
    
    merged_streams, coordinates = read_and_process_data(root_dir, coord_file, start_date, end_date, target_distances)
    
    # Save processed data
    save_processed_data({'merged_streams': merged_streams, 'coordinates': coordinates}, 'processed_das_data.pkl')
    
    print("Data processing complete. Processed data saved.")

def calculate_temperature_change(st1, gauge_length, sampling_rate):
    """
    Temperature change calculation with corrected coefficients
    """
    alpha = 0.48e-6  # Thermal expansion coefficient
    dn_dT = 10e-5   # Corrected thermo-optic coefficient
    n = 1.468     # Absolute refractive index
    
    temp_changes = []
    strain_rates = []
    
    for tr in st1:
        # Calculate strain
        strain = np.gradient(tr.data, axis=0)
        strain_rates.append(strain)
        
        # Convert strain to temperature (ΔT = ε/(α + (1/n)*(dn/dT)))
        dT = strain / (alpha + (1/n) * dn_dT)
        temp_changes.append(dT)
    
    return np.array(temp_changes), np.array(strain_rates)

def plot_analysis_separate(merged_streams, start_date, end_date):
    distances = [0.5, 1.0, 5.0, 15.0]  # in km
    fig, axs = plt.subplots(len(distances), 2, figsize=(15, 20))
    
    for idx, dist in enumerate(distances):
        dist_m = dist * 1000  # convert to meters
        
        if dist_m in merged_streams and merged_streams[dist_m]:
            stream = merged_streams[dist_m]
            temp_changes, strain_rates = calculate_temperature_change(stream, gauge_length, sampling_rate)
            avg_temp_change = np.mean(temp_changes, axis=0)
            avg_strain = np.mean(strain_rates, axis=0)
            
            t = np.linspace(0, (end_date - start_date) / 86400, num=len(avg_temp_change))
            
            # Temperature subplot
            axs[idx, 0].plot(t, avg_temp_change, 'b-')
            axs[idx, 0].set_ylabel('Relative Temperature Change (°C)')
            axs[idx, 0].set_title(f'Temperature Change at {dist} km')
            axs[idx, 0].grid(True)
            
            # Strain rate subplot
            axs[idx, 1].plot(t, avg_strain, 'r-')
            axs[idx, 1].set_ylabel('Strain Rate (s⁻¹)')
            axs[idx, 1].set_title(f'Strain Rate at {dist} km')
            axs[idx, 1].grid(True)
    
    # Set common x-labels
    for ax in axs[-1]:
        ax.set_xlabel('Days since start')
    
    plt.tight_layout()
    plt.savefig('das_analysis_separate.png', dpi=300)
    plt.show()

def main_analysis():
    data = load_processed_data('processed_das_data.pkl')
    merged_streams = data['merged_streams']
    
    start_date = UTCDateTime("2023-11-08")
    end_date = UTCDateTime("2023-11-12")
    
    plot_analysis_separate(merged_streams, start_date, end_date)
#%%
if __name__ == "__main__":
    main_analysis()
#%%
if __name__ == "__main__":
    # Comment this out once data is processed
    #main_data_processing()
    
    main_analysis_and_plotting()