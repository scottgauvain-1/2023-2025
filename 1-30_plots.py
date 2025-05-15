# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:08:30 2025

@author: sjgauva
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Read the CSV file
data = 'C:/Users/sjgauva/Desktop/AtmoSense/campaign_2/0.1-20_Hz_final_results.csv'
df = pd.read_csv(data)

# Define save directory
save_dir = 'C:/Users/sjgauva/Desktop/AtmoSense/campaign_2/'
# Create directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Define columns to plot
y_columns = [
    'Observed-predicted travel time [s]',
    'Signal duration [s]',
    'Dominant frequency [Hz] (based on max PSD)',
    'Signal period [s] (full cycle)',
    'Signal period [s] (half cycle)',
    'Signal period [s] (all zero crossings)',
    'Signal max amplitude [Pa]',
    'Signal peak to peak amplitude [Pa]',
    'Signal min amplitude [Pa]',
    'Duration [s]'
]

# Set color palette
colors = sns.color_palette("husl", n_colors=len(y_columns)+1)

# Create plots vs Distance
for idx, col in enumerate(y_columns):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Distance [km] '], df[col], color=colors[idx], alpha=0.6)
    plt.title(f'Distance vs {col}')
    plt.xlabel('Distance [km]')
    plt.ylabel(col)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create safe filename by replacing problematic characters
    safe_filename = col.replace('[', '').replace(']', '').replace('/', '_').replace(' ', '_')
    safe_filename = ''.join(c for c in safe_filename if c.isalnum() or c in ('_', '-'))
    
    # Save figure with safe filename
    plt.savefig(os.path.join(save_dir, f'distance_vs_{safe_filename}.png'), 
                dpi=300, bbox_inches='tight')
    #plt.close()

# Create plots vs Celerity
for idx, col in enumerate(y_columns):
    if col != 'Signal celerity [km/s]':  # Skip celerity vs celerity plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df['Signal celerity [km/s]'], df[col], color=colors[idx], alpha=0.6)
        plt.title(f'Celerity vs {col}')
        plt.xlabel('Signal celerity [km/s]')
        plt.ylabel(col)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Create safe filename
        safe_filename = col.replace('[', '').replace(']', '').replace('/', '_').replace(' ', '_')
        safe_filename = ''.join(c for c in safe_filename if c.isalnum() or c in ('_', '-'))
        
        # Save figure
        plt.savefig(os.path.join(save_dir, f'celerity_vs_{safe_filename}.png'), 
                    dpi=300, bbox_inches='tight')

# Create separate figure for Signal class vs Distance
plt.figure(figsize=(10, 6))
signal_classes = df.groupby('Signal class (Silber & Brown (2014))')['Distance [km] '].apply(list)
plt.boxplot(signal_classes.values, labels=signal_classes.index)
plt.title('Distance Distribution by Signal Class')
plt.xlabel('Signal Class')
plt.ylabel('Distance [km]')
plt.xticks(rotation=45)
plt.tight_layout()

# Save signal class figure
plt.savefig(os.path.join(save_dir, 'signal_class_distribution.png'), 
            dpi=300, bbox_inches='tight')


# Create boxplot for Signal class vs Celerity
plt.figure(figsize=(10, 6))
celerity_classes = df.groupby('Signal class (Silber & Brown (2014))')['Signal celerity [km/s]'].apply(list)
plt.boxplot(celerity_classes.values, labels=celerity_classes.index)
plt.title('Celerity Distribution by Signal Class')
plt.xlabel('Signal Class')
plt.ylabel('Signal celerity [km/s]')
plt.xticks(rotation=45)
plt.tight_layout()

# Save celerity class figure
plt.savefig(os.path.join(save_dir, 'celerity_class_distribution.png'), 
            dpi=300, bbox_inches='tight')
