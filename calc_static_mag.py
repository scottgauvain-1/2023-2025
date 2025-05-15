# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:06:56 2025

@author: sjgauva
"""

import pandas as pd
import numpy as np
import os
import h5py
#%%
def calculate_station_midpoints(csv_path):
    """
    Calculate the midpoint position for each station from its components.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing station locations
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing station midpoint locations
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Group by station
    station_groups = df.groupby('Station')
    
    # Calculate midpoints
    midpoints = []
    for station, group in station_groups:
        midpoint = {
            'Station': station,
            'X_mid': np.mean(group['X (m)']),
            'Y_mid': np.mean(group['Y (m)']),
            'Z_mid': np.mean(group['Z (m)'])
        }
        # Calculate distance from source (assuming source is at origin)
        midpoint['distance'] = np.sqrt(
            midpoint['X_mid']**2 + 
            midpoint['Y_mid']**2 + 
            midpoint['Z_mid']**2
        )
        midpoints.append(midpoint)
    
    return pd.DataFrame(midpoints)

def read_magnetic_field_data(directory):
    """
    Read magnetic field data from HDF5 files and calculate field magnitudes.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    print(f"Processing HDF5 files in: {directory}")
    
    magnetic_data = {}
    
    for filename in os.listdir(directory):
        if filename.endswith(('.h5', '.hdf5', '.hdf')):
            print(f"Reading file: {filename}")
            filepath = os.path.join(directory, filename)
            
            try:
                station_name = os.path.splitext(filename)[0].split(' ')[0]
                
                with h5py.File(filepath, 'r') as f:
                    if all(comp in f.keys() for comp in ['H1', 'H2', 'H3']):
                        # Read magnetic field components
                        H1 = f['H1'][:]
                        H2 = f['H2'][:]
                        H3 = f['H3'][:]
                        
                        # Calculate RMS values for each component
                        H1_rms = np.sqrt(np.mean(H1**2))
                        H2_rms = np.sqrt(np.mean(H2**2))
                        H3_rms = np.sqrt(np.mean(H3**2))
                        
                        # Calculate total magnitude
                        magnitudes = np.sqrt(H1**2 + H2**2 + H3**2)
                        magnitude_mean = np.mean(magnitudes)
                        
                        magnetic_data[station_name] = {
                            'H1_mean': H1_rms,  # Using RMS instead of mean
                            'H2_mean': H2_rms,  # Using RMS instead of mean
                            'H3_mean': H3_rms,  # Using RMS instead of mean
                            'magnitude_mean': magnitude_mean
                        }
                        
                        print(f"Processed magnetic data for {station_name}")
                        print(f"RMS values - H1: {H1_rms:.3f}, H2: {H2_rms:.3f}, H3: {H3_rms:.3f}")
                        print(f"Total magnitude: {magnitude_mean:.3f}")
                        
                    else:
                        print(f"Missing magnetic field components in {filename}")
                
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    return magnetic_data

def calculate_dipole_field(locations_df, magnetic_data):
    """
    Calculate static magnetic field considering dipole field falloff with distance.
    """
    
    print("\nStations in locations_df:", locations_df['Station'].tolist())
    print("\nStations in magnetic_data:", list(magnetic_data.keys()))
    
    results = []
    
    for _, row in locations_df.iterrows():
        station = row['Station']
        if station in magnetic_data:
            data = magnetic_data[station]
            
            # Calculate 1/r³ factor for dipole field
            distance = row['distance']
            dipole_factor = 1 / (distance**3)
            
            results.append({
                'Station': station,
                'X_mid': row['X_mid'],
                'Y_mid': row['Y_mid'],
                'Z_mid': row['Z_mid'],
                'Distance_from_source': distance,
                'Vertical_Component': data['H1_mean'],
                'NS_Component': data['H2_mean'],
                'EW_Component': data['H3_mean'],
                'Measured_Field_Magnitude': data['magnitude_mean'],
                'Distance_Factor': dipole_factor,
                'Normalized_Field': data['magnitude_mean'] * distance**3  # This should be roughly constant for a dipole
            })
    
    results_df = pd.DataFrame(results)
    return results_df

def main():
    # Read and calculate station midpoints
    csv_path = "C:/Users/sjgauva/Desktop/Thrasher/Mesa/data_2024-03-29/thrasher_data/relative_station_locs.csv"
    print(f"\nCalculating station midpoints from {csv_path}")
    location_data = calculate_station_midpoints(csv_path)
    
    # Read magnetic field data
    hdf5_directory = "C:/Users/sjgauva/Desktop/Thrasher/Mesa/data_2024-03-29/thrasher_data/"
    magnetic_data = read_magnetic_field_data(hdf5_directory)
    
    # Calculate dipole fields
    results = calculate_dipole_field(location_data, magnetic_data)
    print("\nDataFrame shape:", results.shape)
    print("\nDataFrame columns:", results.columns.tolist())
    print("\nFirst few rows:")
    print(results.head())
    
    # Print results
    print("\nStatic Magnetic Field Results (Dipole Source):")
    print("-" * 80)
    for _, row in results.iterrows():
        print(f"\nStation: {row['Station']}")
        print(f"Position (m): X={row['X_mid']:.1f}, Y={row['Y_mid']:.1f}, Z={row['Z_mid']:.1f}")
        print(f"Distance from source: {row['Distance_from_source']:.1f} m")
        print(f"Magnetic Field Components (nT):")
        print(f"  Vertical: {row['Vertical_Component']:.1f}")
        print(f"  N-S:      {row['NS_Component']:.1f}")
        print(f"  E-W:      {row['EW_Component']:.1f}")
        print(f"Measured Field Magnitude: {row['Measured_Field_Magnitude']:.1f} nT")
        print(f"Normalized Field (B·r³): {row['Normalized_Field']:.1e} nT·m³")
    
    # Save results to CSV
    output_file = 'magnetic_field_results.csv'
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()