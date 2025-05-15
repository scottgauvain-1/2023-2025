# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:38:53 2024

@author: sjgauva
"""
from tabulate import tabulate
import pandas as pd
import numpy as np
import matplotlib
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
import glob
import time
from scipy.ndimage import median_filter
from collections import Counter
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.ndimage import label, binary_opening
from scipy.spatial import cKDTree
from tqdm import tqdm
import warnings
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO

warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

class TeeOutput:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be immediately visible

    def flush(self):
        for f in self.files:
            f.flush()

def read_parameters_from_csv(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            data = next(reader)
    
        constraints = {
            'min_easting': int(round(float(data['min_easting']))),
            'max_easting': int(round(float(data['max_easting']))),
            'min_northing': int(round(float(data['min_northing']))),
            'max_northing': int(round(float(data['max_northing']))),
            'min_elevation': float(data['min_elevation']),  # Changed to float
            'max_elevation': float(data['max_elevation'])   # Changed to float
        }
        
        grid_spacing = {
            'dxy': int(round(float(data['grid_spacing_dxy']))),
            'dz': float(data['grid_spacing_dz'])  # Changed to float
        }
        
        file_path = data['file_path']
        parameter_filepath = data['parameter_file_path']
        
        return constraints, grid_spacing, file_path, parameter_filepath
    
    except Exception as e:
            print(f"Error reading CSV file: {str(e)}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Attempting to read file: {file_path}")
            raise
            
def calculate_slice_indices(constraints, grid_spacing):
    total_easting_cells = (constraints['max_easting'] - constraints['min_easting']) // grid_spacing['dxy']
    total_northing_cells = (constraints['max_northing'] - constraints['min_northing']) // grid_spacing['dxy']
    
    middle_easting_index = total_easting_cells // 2
    middle_northing_index = total_northing_cells // 2
    
    return middle_easting_index, middle_northing_index

def setup_and_print_parameters(csv_file_name='GFM_user_file.csv'):
    current_directory = os.getcwd()
    csv_file_path = os.path.join(current_directory, csv_file_name)
    print(f"Full path to CSV file: {csv_file_path}")
    try:
        constraints, grid_spacing, file_path, parameter_filepath = read_parameters_from_csv(csv_file_path)
        
        mid_easting, mid_northing = calculate_slice_indices(constraints, grid_spacing)
        
        return constraints, grid_spacing, file_path, parameter_filepath, mid_easting, mid_northing
    except Exception as e:
        print(f"Failed to read or process parameters: {str(e)}")
        raise

constraints, grid_spacing, file_path, parameter_filepath, mid_easting, mid_northing = setup_and_print_parameters()


#%% Define main function

def build_gfm(file_path, constraints, grid_spacing, mid_northing, mid_easting, parameter_filepath=None):
    with open('run_info.txt', 'w') as f, StringIO() as buf:
        # Redirect stdout to both the file and the StringIO
        with redirect_stdout(TeeOutput(sys.stdout, f, buf)):
            """ 
                INPUTS: A user defined file_path, data constraints (min & max northing, easting, and 
                elevation), and grid_spacing (dxy and dz) in order to visualize GFM layers in 2D.
                Users may also add secondary constraints and spacing in order to zoom in on smaller
                sections of the model with higher resolution. 
            
                The user can populate layers with parameters given a provided parameter
                file. The model uses UTM coordinates for Easting and Northing, and meters for elevation. 
            
                OUTPUTS: a saved .csv file of the model, another .csv file of the model populated with 
                parameters, and the option to save 2D cross sections.
                
            """
            
            def read_files(file_paths, constraints, grid_spacing, apply_constraints=True, method='nearest'):
                column_names = ['Easting', 'Northing', 'Elevation', 'Surface', 'Layer']
                dat_column_names = ['Easting', 'Northing', 'Elevation', 'column', 'layer']
                data = {}
                
                for file_path in file_paths:
                    extension = os.path.splitext(file_path)[1]
                    if extension == '.xlsx':
                        metadata = pd.read_excel(file_path, nrows=11)
                        df = pd.read_excel(file_path, skiprows=11, names=column_names)
                    elif extension in ['.txt', '.asc', '.dat']:
                
                        if 'UNESE' in file_path:
                            layer_name = os.path.basename(file_path).split('.')[0]
                            column_names = ['Easting', 'Northing', 'Elevation', 'Layer']
                            df = pd.read_csv(file_path, comment='#', sep='\t', low_memory = False, usecols = [0,1,2])
                            #df = df.drop(df.columns[[-1,-2]], axis = 1)
                            df['Layer'] = layer_name
                            df.columns = column_names
                        else:
                            layer_name = os.path.basename(file_path).split('.')[0]
                        
                            # Reading the .dat file, skipping comment lines, and specifying tab delimeter
                            df = pd.read_csv(file_path, comment='#', sep='\t', names=column_names, low_memory = False)
                        
                            # Adding the layer name as a new column
                            df['Layer'] = layer_name
                        
                            # Selecting only relevant columns
                            df = df[['Easting', 'Northing', 'Elevation', 'Layer']]
                        
                    else:
                        print(f"Unsupported file type for file {file_path}.")
                        continue
                    data[file_path] = df  # Add the filtered data to the dictionary
                
                return data
            
            def get_file_paths():
                file_paths = []
            
                while True:
                    file_path = input("Please enter a file path, or 'done' to finish: ")
                    if file_path.lower() == 'done':
                        break
                    else:
                        file_paths.append(file_path.strip())
            
                return file_paths
            
            def create_model(constraints, grid_spacing, air_value=0):
                # Determine the size of the model in each dimension
                x_size = int((constraints['max_easting'] - constraints['min_easting']) / grid_spacing['dxy'])
                y_size = int((constraints['max_northing'] - constraints['min_northing']) / grid_spacing['dxy'])
                
                z_size = int((constraints['max_elevation'] - constraints['min_elevation']) / grid_spacing['dz'])
                model = np.full((x_size, y_size, z_size), air_value)
            
                return model
            
            def grid_layer(data, constraints, grid_spacing, method='linear', distance_threshold=5*grid_spacing['dxy']): # was 5
        
                easting = np.arange(constraints['min_easting'], constraints['max_easting'], grid_spacing['dxy'])
                northing = np.arange(constraints['min_northing'], constraints['max_northing'], grid_spacing['dxy'])
                
                grid_easting, grid_northing = np.meshgrid(easting, northing)
                
                # Interpolate
                grid_elevation = griddata((data['Easting'], data['Northing']), data['Elevation'], 
                                          (grid_easting, grid_northing), method=method, fill_value=np.nan)
                
                # Create a KDTree to increase speed
                tree = cKDTree(np.vstack((data['Easting'], data['Northing'])).T)
                
                # Flatten grid coordinates
                grid_points = np.vstack((grid_easting.ravel(), grid_northing.ravel())).T
                
                # Query the KDTree for distances to the nearest neighbor (k=1 nearest neighbor)
                distances, _ = tree.query(grid_points, k=1)
                
                distances = distances.reshape(grid_easting.shape)
                mask = distances <= distance_threshold
                grid_elevation[~mask] = np.nan
                
                return grid_easting, grid_northing, grid_elevation
            
            def map_label_to_property(label, property_column):
                if label in property_column.index:
                    return property_column[label]
                else:
                    return np.nan
            
            def drop_nan_values(data):
                cleaned_data = {}
                for key, df in data.items():
                    cleaned_data[key] = df.dropna()
                return cleaned_data
            
            os.chdir(file_path)
        
            file_names = glob.glob('*.dat')
            file_paths = [os.path.join(file_path, file_name) for file_name in file_names]
        
            data = read_files(file_paths, constraints, grid_spacing, apply_constraints=False)
            cleaned_data = drop_nan_values(data)
            model = create_model(constraints, grid_spacing)
            print(constraints)
            print(grid_spacing)
            print('Model dimensions (X,Y,Z) : ' + str(model.shape))
            
            for layer in data.values(): # Redo column order if they're messed up
                
                if layer.index[0]!= 0:
                    easting = layer.index
                    layer.rename(columns = {'Elevation':'Ind'}, inplace = True)
                    layer.rename(columns = {'Northing':'Elevation'}, inplace = True)
                    layer.rename(columns = {'Easting':'Northing'}, inplace = True)
                    layer.insert(0, 'Easting', easting, True)
                    layer.drop('Ind', axis = 1)
            
            def create_gridded_data(cleaned_data, constraints, grid_spacing):
                 gridded_data = {}
                 for file_path, layer_data in cleaned_data.items():
                     grid_easting, grid_northing, grid_elevation = grid_layer(layer_data, constraints, grid_spacing)
                     gridded_data[file_path] = (grid_easting, grid_northing, grid_elevation)
                 return gridded_data
            
            gridded_data = create_gridded_data(cleaned_data, constraints, grid_spacing)
                 
            unique_layers = pd.concat(data.values())['Layer'].unique() # all layer names
            print('Unique layers in model: ' + str(unique_layers))
            colors = list(mcolors.TABLEAU_COLORS.keys())
            color_map = dict(zip(unique_layers, colors))
            colors = list(mcolors.TABLEAU_COLORS.keys()) * (len(unique_layers) // len(colors) + 1)
            color_map = dict(zip(unique_layers, colors)) # assign each layer a color
            
            for file_path, (grid_easting, grid_northing, grid_elevation) in gridded_data.items():
                  layer = data[file_path]['Layer'].iloc[0]
                  print(f"{layer}, Max elevation: {np.round(np.nanmax(grid_elevation))}, Min elevation: {np.round(np.nanmin(grid_elevation))}")
            
            del data # Saves memory
            
            def get_unique_layers(model):
                return list(set(model.flatten()))
        
            def create_prioritized_layers(gridded_data, z_values, grid_easting, grid_northing):
                layer_paths = list(gridded_data.keys())
                # Sort layers by priority (assuming lower layers have higher priority)
                layer_paths.sort(key=lambda x: np.nanmin(gridded_data[x][2]))
                
                model = np.full((grid_easting.shape[0], grid_northing.shape[1], len(z_values)), 'Air', dtype=object)
                
                for layer_path in layer_paths:
                    layer_name = os.path.splitext(os.path.basename(layer_path))[0]
                    _, _, elevations = gridded_data[layer_path]
                    
                    for i in range(model.shape[0]):
                        for j in range(model.shape[1]):
                            if np.isfinite(elevations[i, j]):
                                # Interpolate to get smoother boundaries
                                interp_func = interp1d([elevations[i, j], z_values[-1]], [1, 0], kind='linear', bounds_error=False, fill_value=(1, 0))
                                layer_values = interp_func(z_values)
                                
                                # Fill down approach
                                layer_mask = (layer_values > 0.5) & (model[i, j, :] == 'Air')
                                model[i, j, layer_mask] = layer_name
                
                return model
        
            def enforce_boundaries(model):
                enforced_model = model.copy()
                
                for i in range(model.shape[0]):
                    for j in range(model.shape[1]):
                        current_layer = 'Air'
                        for k in range(model.shape[2]):
                            if model[i, j, k] != 'Air' and model[i, j, k] != current_layer:
                                current_layer = model[i, j, k]
                            enforced_model[i, j, k] = current_layer
                
                return enforced_model
        
            def smooth_boundaries(model, sigma=10): # changed from 3 to 10
                
                chunk_size = grid_spacing['dxy']*10
                unique_layers = list(set(model.flatten()))
                smoothed_model = model.copy()
                
                for layer in unique_layers:
                    if layer == 'Air':
                        continue

                    for i in range(0, model.shape[0], chunk_size):
                        chunk = model[i:i+chunk_size]
                        layer_mask = (chunk == layer).astype(float)
                        smoothed_mask = gaussian_filter(layer_mask, sigma=sigma)
                        smoothed_model[i:i+chunk_size][smoothed_mask > 0.5] = layer
                
                return smoothed_model
        
            def correct_thin_vertical_sections(model, min_thickness=5, span_threshold=0.9):
                northing, easting, elevation = model.shape
                corrected_model = model.copy()
        
                # Identify layers that span most of the northing grid
                for d in range(northing):
                    slice_2d = model[d,:,:]
                    unique_layers = np.unique(slice_2d)
                    
                    for layer in unique_layers:
                        if layer == 'Air':
                            continue
                        
                        layer_mask = (slice_2d == layer)
                        spanning_columns = np.sum(layer_mask, axis=0) >= northing * span_threshold
                        
                        if np.any(spanning_columns):
                            # This layer spans most of the northing grid in at least one column
                            labeled, num_features = label(~layer_mask)
                            
                            for feature in range(1, num_features + 1):
                                feature_mask = labeled == feature
                                feature_width = np.max(np.sum(feature_mask, axis=0))
                                
                                if feature_width < min_thickness:
                                    # This is a thin vertical section, replace it with the spanning layer
                                    corrected_model[d,:,:][feature_mask] = layer
        
                return corrected_model
        
            def remove_thin_horizontal_layers(model, min_thickness=2):
                cleaned_model = model.copy()
                northing, easting, elevation = model.shape
                
                for i in range(northing):
                    for j in range(easting):
                        current_layer = model[i, j, 0]
                        layer_start = 0
                        for k in range(1, elevation):
                            if model[i, j, k] != current_layer:
                                layer_thickness = k - layer_start
                                if layer_thickness < min_thickness:
                                    if layer_start == 0:  # Top layer
                                        cleaned_model[i, j, layer_start:k] = model[i, j, k]
                                    else:  # Middle or bottom layers
                                        cleaned_model[i, j, layer_start:k] = model[i, j, layer_start-1]
                                layer_start = k
                                current_layer = model[i, j, k]
                        
                        # Check the last layer
                        layer_thickness = elevation - layer_start
                        if layer_thickness < min_thickness and layer_start > 0:
                            cleaned_model[i, j, layer_start:] = model[i, j, layer_start-1]
                
                return cleaned_model
            
            def create_geological_model(gridded_data, constraints, grid_spacing):
                z_values = np.arange(constraints['max_elevation'], constraints['min_elevation'], -grid_spacing['dz'])
                grid_easting, grid_northing = next(iter(gridded_data.values()))[:2]
                
                # Create initial model with prioritized layers and interpolation
                initial_model = create_prioritized_layers(gridded_data, z_values, grid_easting, grid_northing)
        
                smoothed_model = smooth_boundaries(initial_model, sigma=10)
            
                final_model = enforce_boundaries(smoothed_model)
                
                return final_model
            
            def create_clean_geological_model(gridded_data, constraints, grid_spacing):
                print("Creating interpolated model...")
                initial_model = create_geological_model(gridded_data, constraints, grid_spacing)
                print('Done with interpolated model')
                print('Smoothing model...')
                cleaned_model = remove_thin_horizontal_layers(initial_model, min_thickness=2)
                corrected_model = correct_thin_vertical_sections(cleaned_model, min_thickness=grid_spacing['dxy']*10, span_threshold=0.9)
                print('Done smoothing model')
                print("Enforcing final boundaries...")
                final_model = enforce_boundaries(corrected_model)
                print('Final model complete')
                return final_model
            
            model = create_clean_geological_model(gridded_data, constraints, grid_spacing)
            model = model[5:,:,:]
            
            unpopulated_model = []
            
            for layer_path, (grid_easting, grid_northing, grid_elevation) in gridded_data.items():
                layer_name = os.path.splitext(os.path.basename(layer_path))[0]
                layer_data = np.column_stack((grid_easting.flatten(), 
                                              grid_northing.flatten(), 
                                              grid_elevation.flatten(), 
                                              np.full(grid_easting.size, layer_name)))
                unpopulated_model.append(layer_data)
            
            unpopulated_model = np.vstack(unpopulated_model)
            unpopulated_df = pd.DataFrame(unpopulated_model, columns=['Easting', 'Northing', 'Elevation', 'Layer'])
            unpopulated_df.to_csv(os.path.join(parameter_filepath, 'GFM_original.csv'), index=False)

            def read_and_combine_csv_files(directory_path):
                csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
                if not csv_files:
                    print(f"No CSV files found in {directory_path}")
                    return None
                
                combined_df = pd.DataFrame(columns=['Unit'])
                files_to_ignore = ['GFM_original.csv', 'GFM_populated.csv', 'GFM_combined_parameters.csv']
                
                for file in csv_files:
                    filename = os.path.basename(file)
                    
                    # Skip files that should be ignored
                    if filename in files_to_ignore:
                        continue
                    
                    try:
                        df = pd.read_csv(file, usecols=['Geologic_Unit', 'Avg'])
                        
                        if df.empty or df.shape[1] != 2:
                            print(f"Skipping {file}: File is empty or does not have the required columns")
                            continue
                        
                        parameter = filename.rsplit('.', 1)[0].split('_')[-1].upper()
                        df.columns = ['Unit', parameter]
                        
                        if combined_df.empty:
                            combined_df = df
                        else:
                            combined_df = pd.merge(combined_df, df, on='Unit', how='outer')
                        
                        output_csv_file = 'GFM_combined_parameters.csv'
                        combined_df.to_csv(os.path.join(directory_path, output_csv_file), index=False)
                        print(f"Combined parameters have been saved to {output_csv_file}")
                    
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
                
                return combined_df

                        
            def merge_params_to_model(params, model_pop):
                merged_data = model_pop.merge(params, left_on='Layer', right_on='Unit', how='left')
                columns_to_append = properties
                for col in columns_to_append:
                    model_pop[col] = merged_data[col].fillna(-9999)
                return model_pop
            
            if parameter_filepath and os.path.exists(parameter_filepath):
                try:
                    # Use the existing function to read and combine CSV files
                    combined_params = read_and_combine_csv_files(parameter_filepath)
                    
                    if combined_params is not None and not combined_params.empty:
                        combined_params_file = os.path.join(parameter_filepath, 'GFM_combined_parameters.csv')
                        
                        print("\nCurrent Parameter Table:")
                        print(tabulate(combined_params, headers='keys', tablefmt='psql'))
                        
                        print('Manually edit and save the parameter values if desired.')
                        input("\nAfter editing the file, press Enter to continue with model population...")
                        
                        # Re-read the file to capture any manual changes
                        combined_params = pd.read_csv(combined_params_file)
                        
                        print("\nUpdated Parameter Table:")
                        print(tabulate(combined_params, headers='keys', tablefmt='psql'))
                        
                        properties = list(combined_params.columns[1:])
                        model_pop_original = unpopulated_df.copy()
                        model_pop_original = merge_params_to_model(combined_params, model_pop_original)
                
                        output_file = os.path.join(parameter_filepath, 'GFM_populated.csv')
                        model_pop_original.to_csv(output_file, index=False)
                        print(f'Populated CSV model saved to {output_file}')
                    else:
                        print("No data to populate the model.")
                        model_pop_original = None
                except Exception as e:
                    print(f"Error while processing and populating model: {str(e)}")
                    model_pop_original = None
                else:
                    #print("No valid parameter filepath provided. Skipping model population.")
                    model_pop_original = None
            
            print('CSV models saved.')
            def plot_2d_slices(model, constraints, grid_spacing, file_names, mid_northing, mid_easting):
    
                xz_slice = model[:, mid_easting, :]
                yz_slice = model[mid_northing, :, :]
            
                def process_slice(slice_data):
                    unique_layers = ['Air']  # Start with 'Air' at the top
                    for row in slice_data.T:  # Transpose to iterate over columns (top to bottom)
                        for layer in row:
                            if layer not in unique_layers:
                                unique_layers.append(layer)
                    
                    layer_to_index = {layer: i for i, layer in enumerate(reversed(unique_layers))}
                    int_slice = np.vectorize(lambda x: layer_to_index[x])(slice_data).T
                    return int_slice, unique_layers
            
                int_xz_slice, xz_sorted_layers = process_slice(xz_slice)
                int_yz_slice, yz_sorted_layers = process_slice(yz_slice)
            
                elev_ticks = np.arange(constraints['max_elevation'], constraints['min_elevation'], -grid_spacing['dz'])
                north_ticks = np.arange(constraints['max_northing'], constraints['min_northing'], -grid_spacing['dxy'])
                east_ticks = np.arange(constraints['min_easting'], constraints['max_easting'], grid_spacing['dxy'])
                elev_str = [str(int(tick)) for tick in elev_ticks]
                north_str = [str(int(tick)) for tick in north_ticks]
                east_str = [str(int(tick)) for tick in east_ticks]
            
                # North-South Plane (XZ slice)
                fig1, ax1 = plt.subplots(figsize=(12, 8))
                cax1 = ax1.imshow(int_xz_slice, cmap=plt.cm.get_cmap('jet', len(xz_sorted_layers)))
                ax1.set_xlabel('Northing (UTM)', fontsize=10)
                ax1.set_ylabel('Elevation (m)', fontsize=10)
                ax1.set_title(f'North-South Plane (Easting Index {mid_easting} of {model.shape[1]}))')
            
                cbar1 = fig1.colorbar(cax1, ax=ax1, ticks=range(len(xz_sorted_layers)))
                cbar1.set_ticklabels(reversed(xz_sorted_layers))
            
                ax1.set_xticks(range(len(north_str))[::40])
                ax1.set_xticklabels(north_str[::40], fontsize=10, rotation=45)
                ax1.set_yticks(range(len(elev_str))[::20])
                ax1.set_yticklabels(elev_str[::20], fontsize=10)
                plt.tight_layout()
                plt.show()
            
                # East-West Plane (YZ slice)
                fig2, ax2 = plt.subplots(figsize=(12, 8))
                cax2 = ax2.imshow(int_yz_slice, cmap=plt.cm.get_cmap('jet', len(yz_sorted_layers)))
                ax2.set_xlabel('Easting (UTM)', fontsize=10)
                ax2.set_ylabel('Elevation (m)', fontsize=10)
                ax2.set_title(f'East-West Plane (Northing Index {mid_northing} of {model.shape[0]}))')
            
                cbar2 = fig2.colorbar(cax2, ax=ax2, ticks=range(len(yz_sorted_layers)))
                cbar2.set_ticklabels(reversed(yz_sorted_layers))
            
                ax2.set_xticks(range(len(east_str))[::40])
                ax2.set_xticklabels(east_str[::40], fontsize=10, rotation=45)
                ax2.set_yticks(range(len(elev_str))[::20])
                ax2.set_yticklabels(elev_str[::20], fontsize=10)
            
                plt.tight_layout()
                plt.show()
            
                return int_xz_slice, int_yz_slice, north_ticks, east_ticks, elev_ticks
            
            int_xz_slice, int_yz_slice, north_ticks, east_ticks, elev_ticks = plot_2d_slices(model, constraints, grid_spacing, file_names, mid_northing, mid_easting)
    return model, unique_layers, int_xz_slice, int_yz_slice, east_ticks, elev_ticks, model_pop_original

#%% Run the main function to create models
start = time.time()
model, unique_layers, int_xz_slice, int_yz_slice, east_ticks, elev_ticks, model_pop_original = build_gfm(file_path, constraints, grid_spacing, mid_northing, mid_easting, parameter_filepath)

end = time.time()
rtime = (end - start)
print(f'Time taken to compute, smooth, and plot models: {rtime/60:.1f} minutes')

#%% Adjust the user_index to view more slices of the models

# user_index = 1

# os.chdir(file_path)

# file_names = glob.glob('*.dat')
# file_paths = [os.path.join(file_path, file_name) for file_name in file_names]
# user_twodmod = new_model[:,user_index,:] # 2d model slice in xz
# user_unique_2mod = np.unique(user_twodmod)

# # loop through and assign numbers to layers
# for i,j in enumerate(user_unique_2mod):
#     user_twodmod[user_twodmod==str(j)] = i
    
# # plot numbered matrix of layers
# user_int_2dmod = user_twodmod.astype(int).T
# plt.figure()

# cax3 = plt.imshow(user_int_2dmod, cmap=plt.cm.get_cmap('jet',len(file_paths)))

# ticks = range(0,len(user_unique_2mod))
# colorbar = plt.colorbar(cax3)
# colorbar.set_ticks(ticks)
# colorbar.set_ticklabels(user_unique_2mod)
# colorbar.ax.invert_yaxis() # argsort unique_2mod by elevations to have accurate colorbar
# plt.xlabel('Northing (UTM)')
# plt.ylabel('Elevation (m)')
# plt.title('Easting Index ' + str(user_index) + ' of ' + str(new_model.shape[1]))
# plt.show()

# # Here, make sure constraints or new_constraints match which slice you view
# elev_ticks = (np.arange(new_constraints['max_elevation'], new_constraints['min_elevation'], -new_grid_spacing['dz']))
# east_ticks = (np.arange(new_constraints['max_northing'], new_constraints['min_northing'], -new_grid_spacing['dxy']))
# elev_str = []
# east_str = []
# for tick in elev_ticks:
#     elev_str.append(str(tick))
# for tick in east_ticks:
#     east_str.append(str(tick))

# plt.xticks(range(len(east_str))[::200], east_str[::200])
# plt.yticks(range(len(elev_str))[::50], elev_str[::50])

