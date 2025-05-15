import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read, Stream, UTCDateTime
from obspy.signal.filter import bandpass
from obspy.signal.trigger import classic_sta_lta
from scipy.signal import welch, spectrogram
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt
from math import radians, sin, cos, sqrt, atan2
#%%
# ADJUST read_distance_data depending on which shot, and read_sac_files to get a specific file

def find_filter_cutoffs(f, Pxx):
    return 0.1,20

def great_circle_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    return 6371 * (
        np.arccos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2))
    )
def calculate_celerity(distance_km, arrival_seconds, shot_time, trace):
    """Calculate celerity in m/s"""
    # arrival_seconds is the time from trace start to first zero crossing
    # Convert to absolute time for comparison with shot time
    arrival_time = trace.stats.starttime + arrival_seconds
    time_diff = float(arrival_time - shot_time)
    
    # For debugging
    print(f"Distance (km): {distance_km}")
    print(f"Arrival time: {arrival_time}")
    print(f"Shot time: {shot_time}")
    print(f"Time diff (s): {time_diff}")
    
    return (distance_km * 1000) / time_diff if time_diff > 0 else None

def calculate_psd_segment(trace, start_time, end_time, label):
    """Calculate PSD following the format shown in the paper"""
    try:
        # Extract the segment
        segment = trace.slice(start_time, end_time)
        
        # Calculate PSD using Welch's method
        fs = segment.stats.sampling_rate
        segment_length = len(segment.data)
        nperseg = min(int(fs * 10), segment_length)  # 10-second segments
        noverlap = nperseg // 2  # 50% overlap
        
        f, Pxx = welch(segment.data, fs, window='hann',
                      nperseg=nperseg, noverlap=noverlap,
                      detrend='constant', scaling='density')
        
        # The key is not converting to dB here - we'll plot the raw PSD values
        return f, Pxx, label
        
    except Exception as e:
        print(f"Error in PSD calculation for {label}: {e}")
        return None, None, label
    
def update_excel_with_measurements(measurements, excel_path):
    """Update Excel file with new measurements"""
    try:
        # Read existing Excel file
        df = pd.read_excel(excel_path)
        
        # For each set of measurements
        for meas in measurements:
            station = meas['station']
            
            # Find the row for this station
            station_mask = df['Stn ID'] == station
            
            if not any(station_mask):
                print(f"Warning: Station {station} not found in Excel file")
                continue
                
            # Update the measurements we have
            updates = {
                'Signal travel time [s]': meas.get('first_crossing_time'),
                '3rd zero crossing time': meas.get('third_crossing_time'),
                '4th zero crossing time': meas.get('fourth_crossing_time'),
                'Signal period [s] (full cycle)': meas.get('full_cycle_period'),
                'Signal period [s] (half cycle)': meas.get('half_cycle_period'),
                'Signal period [s] (all zero crossings)': meas.get('total_period'),
                'Signal max amplitude [Pa]': meas.get('amplitude_max'),
                'Signal peak to peak amplitude [Pa]': meas.get('peak_to_peak'),
                'Signal min amplitude [Pa]': meas.get('amplitude_min')
            }
            
            # Only update celerity if we have both distance and travel time
            for shot_id in SHOTS.keys():
                celerity_key = f'celerity_{shot_id}'
                if celerity_key in meas and meas[celerity_key] is not None:
                    updates['Signal celerity [km/s]'] = meas[celerity_key] / 1000  # Convert m/s to km/s

            # Update each column if we have the data
            for col, value in updates.items():
                if value is not None and col in df.columns:
                    df.loc[station_mask, col] = value
                    
        # Save the updated DataFrame back to Excel
        df.to_excel(excel_path, index=False)
        print(f"Successfully updated measurements in {excel_path}")
        
    except Exception as e:
        print(f"Error updating Excel file: {e}")
        
def plot_psd_analysis(trace, first_crossing_time, fourth_crossing_time):
    """Plot PSDs matching the paper's format"""
    # Close any existing figures
    plt.close('all')
    
    first_utc = trace.stats.starttime + first_crossing_time
    fourth_utc = trace.stats.starttime + fourth_crossing_time
    
    # Define time windows to match the paper
    segments = [
        (first_utc - 180, first_utc, "Pre-signal Noise (3 min prior, 1 min)"),
        (first_utc, fourth_utc + 3, "Signal PSD (3s)"),
        (fourth_utc, fourth_utc + 60, "Post-signal Noise (1 min after)")
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each segment
    for start, end, label in segments:
        f, psd, _ = calculate_psd_segment(trace, start, end, label)
        if f is not None and psd is not None:
            ax.loglog(f, psd, label=label, linewidth=1.5)
    
    # Set axes and labels to match the paper
    ax.set_xlim(0.1, 100)
    ax.set_ylim(1e-18, 1)  # Adjust these limits to match your data range
    ax.grid(True, which='both', ls='-', alpha=0.2)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power Spectral Density [dB]')
    ax.set_title(f'Log-Log PSD Plot (Signal + Noise) for {trace.stats.station}')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

class ShotInfo:
    def __init__(self, lat, lon, time):
        self.lat = lat
        self.lon = lon
        self.time = UTCDateTime(time)

# Define shots
SHOTS = {
    "1A": ShotInfo(lat=34.0721618, lon=-106.983103, time="2024-10-16T17:29:00"), # shot 2: 2024-10-16T
    "1B": ShotInfo(lat=34.071569, lon=-106.984415, time="2024-10-16T17:32:00"),
    "2A": ShotInfo(lat=34.0722001, lon=-106.9830746, time="2024-10-16T19:07:00"),
    "2B": ShotInfo(lat=34.07157701, lon=-106.984484, time="2024-10-16T19:10:00"),
    "3": ShotInfo(lat=34.0693092, lon=-107.0050267, time="2024-10-16T20:29:00")
}

def read_sac_files(directory):
    st = Stream()
    valid_hours = [17,19,20]
    valid_stations = ['BXCYN']
    
    for file in os.listdir(directory):
        if file.endswith('.SAC'):
            trace = read(os.path.join(directory, file))[0]
            hour = trace.stats.starttime.hour
            station = trace.stats.station
            if hour in valid_hours and station in valid_stations:
                st += Stream([trace])
    return st

def read_metadata(file_path):
    return pd.read_excel(file_path)

def read_distance_data(file_paths):
    """Read distance data from CSV files, focusing on the first file for distances"""
    # Read only the first file for distances
    distance_df = pd.read_csv(file_paths[2])
    
    # Print the first few rows and columns to verify data
    print("\nDistance data sample:")
    print(distance_df[['Station code', 'Distance [km]']].head())
    
    return distance_df

def merge_streams(stream):
    return stream.merge(method=1, fill_value='interpolate')

def calculate_psd(trace):
    # Calculate PSD using Welch's method
    fs = trace.stats.sampling_rate
    f, Pxx = welch(trace.data, fs, nperseg=int(fs*10), detrend = 'constant')  # 10-second segments
    return f, Pxx

def isolate_signal(trace, low_cut, high_cut):
    
    # Remove trend
    trace = trace.detrend()
    
    # Nyquist frequency
    nyq = 0.5 * trace.stats.sampling_rate
    
    # Normalize frequencies by Nyquist
    low = low_cut / nyq
    high = high_cut / nyq
    
    # Design 2nd order Butterworth bandpass filter
    b, a = butter(2, [low, high], btype='band')
    
    # Apply zero-phase filtering
    filtered_data = filtfilt(b, a, trace.data)
    
    # Create new trace with filtered data
    filtered_trace = trace.copy()
    filtered_trace.data = filtered_data
    
    return filtered_trace

def calculate_arrival_time(trace):
    # Using STA/LTA 
    cft = classic_sta_lta(trace.data, int(5 * trace.stats.sampling_rate), int(10 * trace.stats.sampling_rate))
    triggers = np.where(cft > 1.5)[0]
    return trace.stats.starttime + triggers[0] / trace.stats.sampling_rate if len(triggers) > 0 else None

def calculate_max_amplitude(trace):
    return np.max(np.abs(trace.data))

def calculate_peak_to_peak_amplitude(trace):
    return np.ptp(trace.data)

def calculate_zero_crossing_period(trace):
    zero_crossings = np.where(np.diff(np.sign(trace.data)))[0] 
    if len(zero_crossings) >= 2:
        return np.mean(np.diff(zero_crossings)) / trace.stats.sampling_rate
    return None

def calculate_psd_period(f, Pxx):
    idx_max = np.argmax(Pxx)
    return 1 / f[idx_max] if f[idx_max] != 0 else None

def calculate_spectrogram(trace, start_time, end_time):
    """Calculate spectrogram for a specific time window"""
    segment = trace.slice(start_time, end_time)
    fs = segment.stats.sampling_rate
    
    # Calculate spectrogram
    f, t, Sxx = spectrogram(segment.data, fs=fs, nperseg=int(fs), noverlap=int(fs*0.75))
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx)
    
    return f, t, Sxx_db

def plot_psd_analysis(trace, first_crossing_time, fourth_crossing_time):
    """Plot PSDs using Silber & Brown (2014) methodology with original time windows"""
    first_utc = trace.stats.starttime + first_crossing_time
    fourth_utc = trace.stats.starttime + fourth_crossing_time
    
    # Adjust time windows to ensure sufficient data
    segments = [
        (first_utc - 60, fourth_utc, "1min before to 4th crossing"),  # Reduced from 180s to 60s
        (first_utc, fourth_utc + 3, "1st to 4th crossing + 3 seconds"),
        (fourth_utc, fourth_utc + 30, "30sec after 4th crossing")  # Reduced from 60s to 30s
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each segment with error handling
    for start, end, label in segments:
        f, psd_db, _ = calculate_psd_segment(trace, start, end, label)
        if f is not None and psd_db is not None:
            ax.loglog(f, psd_db, label=label)
    
    # Set axis limits and labels
    ax.set_xlim(0.1, 100)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power Spectral Density [dB re Pa²/Hz]')
    ax.set_title(f'PSD Analysis for {trace.stats.station}')
    ax.grid(True, which='both', ls='-', alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.show()

    
def plot_spectrogram(trace, first_crossing_time, fourth_crossing_time):
    """Plot spectrogram with robust error handling"""
    try:
        # Calculate window times
        first_utc = trace.stats.starttime + first_crossing_time
        fourth_utc = trace.stats.starttime + fourth_crossing_time
        
        # Window from 10s before to 10s after (shorter window)
        start_time = first_utc - 10
        end_time = fourth_utc + 10
        
        # Calculate spectrogram
        segment = trace.slice(start_time, end_time)
        fs = segment.stats.sampling_rate
        
        # Adjust nperseg based on data length
        segment_length = len(segment.data)
        nperseg = min(int(fs), segment_length // 4)  # Use shorter segments
        if nperseg < 2:
            print("Segment too short for spectrogram analysis")
            return
            
        noverlap = min(nperseg // 2, nperseg - 1)
        
        f, t, Sxx = spectrogram(segment.data, fs=fs, nperseg=nperseg,
                               noverlap=noverlap, scaling='density')
        
        # Convert to dB with handling of zero/negative values
        Sxx = np.maximum(Sxx, np.finfo(float).tiny)
        Sxx_db = 10 * np.log10(Sxx)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert time axis to seconds from start
        t_offset = t
        
        im = ax.pcolormesh(t_offset, f, Sxx_db,
                          shading='gouraud',
                          cmap='viridis')
        
        ax.set_ylim(0, 50)
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [s]')
        ax.set_title(f'Spectrogram for {trace.stats.station}')
        
        plt.colorbar(im, ax=ax, label='Power [dB]')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error generating spectrogram: {e}")

class PointCalculator:
    def __init__(self):
        # Initialize measurements dictionary first
        self.measurements = {
            'station': None,
            'amplitude_min': None,
            'amplitude_max': None,
            'peak_to_peak': None,
            'half_cycle_period': None,
            'full_cycle_period': None,
            'total_period': None,
            'signal_duration': None,
            'third_crossing_time': None,
            'fourth_crossing_time': None
        }
        # Add distance and celerity fields for each shot
        for shot_id in SHOTS.keys():
            self.measurements[f'distance_{shot_id}'] = None
            self.measurements[f'celerity_{shot_id}'] = None
            
        # Then initialize other attributes
        self.points = []
        self.current_phase = 'amplitude'
        self.amp_points = []
        self.period_points = []
        self.zero_crossing_line = None
        
    def reset(self):
        # Reset all values in measurements dict to None
        for key in self.measurements:
            self.measurements[key] = None
            
        # Reset other attributes
        self.points = []
        self.current_phase = 'amplitude'
        self.amp_points = []
        self.period_points = []
        self.zero_crossing_line = None

def create_click_handler(ax, calculator, all_measurements, station, trace, isolated_signal):
    def click_handler(event):
        if event.inaxes == ax:
            # Remove the check for exactly 6 clicks
            on_double_click(event, ax, calculator, all_measurements, station, trace, isolated_signal)
    return click_handler

# def check_distance(lat1, lon1, lat2, lon2):
#     """Simple distance check in km"""
#     km_per_deg_lat = 111.0  # approximate
#     km_per_deg_lon = 111.0 * cos(radians((lat1 + lat2)/2))  # at average latitude
    
#     dlat = abs(lat2 - lat1)
#     dlon = abs(lon2 - lon1)
    
#     print(f"\nSimple distance check:")
#     print(f"N-S distance: {dlat * km_per_deg_lat:.2f} km")
#     print(f"E-W distance: {dlon * km_per_deg_lon:.2f} km")
    
    # return sqrt((dlat * km_per_deg_lat)**2 + (dlon * km_per_deg_lon)**2)

def process_and_plot(stream, metadata, distance_files):
    merged_stream = merge_streams(stream)
    all_measurements = []
    amp_info = []
    calculator = PointCalculator()
    
    # Create dictionaries to store distances for each station and shot
    station_shot_distances = {}
    
    # Read distances from each shot file
    for shot_id, shot_file in zip(SHOTS.keys(), distance_files):
        shot_data = pd.read_csv(shot_file)
        for _, row in shot_data.iterrows():
            station = row['Station code']
            if station not in station_shot_distances:
                station_shot_distances[station] = {}
            station_shot_distances[station][shot_id] = row['Distance [km]']
    
    print("\nStation-shot distances:")
    for station in station_shot_distances:
        print(f"\n{station}:")
        for shot_id, distance in station_shot_distances[station].items():
            print(f"  Shot {shot_id}: {distance:.2f} km")
    
    for tr in merged_stream:
        station = tr.stats.station
        amp_info.append(tr.stats)
        
        calculator.reset()
        calculator.measurements['station'] = station
        
        # Store distances for each shot separately
        if station in station_shot_distances:
            for shot_id, distance in station_shot_distances[station].items():
                print(f"\nUsing distance for station {station}, shot {shot_id}: {distance:.2f} km")
                calculator.measurements[f'distance_{shot_id}'] = distance
        else:
            print(f"Warning: No distance data found for station {station}")
        
        time = np.arange(tr.stats.npts) / tr.stats.sampling_rate
        start_time = tr.stats.starttime
        
        f, Pxx = calculate_psd(tr)
        low_cut, high_cut = find_filter_cutoffs(f, Pxx)
        
        isolated_signal = isolate_signal(tr, low_cut, high_cut)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(time, isolated_signal.data)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        ax.set_title(f'Station: {station} ({low_cut:.1f}-{high_cut:.1f} Hz)\nStart Time (UTC): {start_time}')
        ax.set_xlabel(f'Time (s) from {start_time}')
        ax.set_ylabel('Amplitude (Pa)')
        
        handler = create_click_handler(ax, calculator, all_measurements, station, tr, isolated_signal)
        fig.canvas.mpl_connect('button_press_event', handler)
        
        plt.show()
    
    return all_measurements, amp_info

def on_double_click(event, ax, calculator, all_measurements, current_station, trace, isolated_signal):
    """Modified double-click handler that combines functionality from both scripts"""
    if not hasattr(calculator, 'click_count'):
        calculator.click_count = 0
    calculator.click_count += 1
    
    if calculator.click_count == 7:
        if calculator.zero_crossing_line is not None:
            calculator.zero_crossing_line.remove()
            calculator.zero_crossing_line = None
        calculator.click_count = 1
        
    if event.dblclick and event.xdata is not None and event.ydata is not None:
        if calculator.current_phase == 'amplitude' and len(calculator.amp_points) == 0:
            if calculator.zero_crossing_line is not None:
                calculator.zero_crossing_line.remove()
                calculator.zero_crossing_line = None
                ax.figure.canvas.draw()

        calculator.measurements['station'] = current_station
        
        if calculator.current_phase == 'amplitude':
            calculator.amp_points.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'r*', markersize=10, label='Amplitude point')
            
            if len(calculator.amp_points) == 2:
                amp_y_values = [p[1] for p in calculator.amp_points]
                calculator.measurements['amplitude_min'] = min(amp_y_values)
                calculator.measurements['amplitude_max'] = max(amp_y_values)
                calculator.measurements['peak_to_peak'] = max(amp_y_values) - min(amp_y_values)
                
                calculator.current_phase = 'period'
                print(f"\nSwitching to period points for station {current_station}")
                
        elif calculator.current_phase == 'period':
            calculator.period_points.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'b*', markersize=10, label='Period point')
            
            if len(calculator.period_points) == 1:
                calculator.first_crossing_time = event.xdata
                print(f"First zero crossing time: {event.xdata:.4f} s")
                
                # Calculate celerities for each shot using the correct station's distance
                for shot_id, shot in SHOTS.items():
                    distance_key = f'distance_{shot_id}'
                    if distance_key in calculator.measurements:
                        distance = calculator.measurements[distance_key]
                        print(f"\nCalculating celerity for {current_station}, shot {shot_id}")
                        print(f"Using distance: {distance} km")
                        celerity = calculate_celerity(distance, event.xdata, shot.time, trace)
                        calculator.measurements[f'celerity_{shot_id}'] = celerity
                        if celerity is not None:
                            print(f"Celerity for shot {shot_id}: {celerity:.2f} m/s")
                        else:
                            print(f"Could not calculate celerity for shot {shot_id} (invalid time difference)")
                            
            if len(calculator.period_points) == 1:
                if calculator.zero_crossing_line is not None:
                    calculator.zero_crossing_line.remove()
                calculator.zero_crossing_line = ax.axhline(y=event.ydata, color='g', linestyle='--', alpha=0.5)
            
            if len(calculator.period_points) == 4:
                calculator.fourth_crossing_time = event.xdata
                print(f"Fourth zero crossing time: {event.xdata:.4f} s")
                
                x_values = [p[0] for p in calculator.period_points]
                
                plt.figure()  
                plot_psd_analysis(isolated_signal, calculator.first_crossing_time, calculator.fourth_crossing_time)
                
                plt.figure() 
                plot_spectrogram(isolated_signal, calculator.first_crossing_time, calculator.fourth_crossing_time)
                
                calculator.measurements.update({
                    'zero_crossing_times': x_values,
                    'first_crossing_time': calculator.first_crossing_time,
                    'fourth_crossing_time': calculator.fourth_crossing_time,
                    'signal_duration': x_values[-1] - x_values[0],
                    'half_cycle_period': abs(x_values[1] - x_values[0]),
                    'full_cycle_period': abs(x_values[2] - x_values[0]),
                    'total_period': abs(x_values[3] - x_values[0])
                })
                
                print(f"\nMeasurements for station {current_station}")
                print("--------------------")
                for key, value in calculator.measurements.items():
                    if value is not None and key != 'zero_crossing_times':
                        print(f"{key}: {value}")
                
                all_measurements.append(calculator.measurements.copy())
                
                if calculator.zero_crossing_line is not None:
                    calculator.zero_crossing_line.remove()
                    calculator.zero_crossing_line = None
                
                calculator.current_phase = 'amplitude'
                calculator.period_points = []
                calculator.amp_points = []
                calculator.reset()
        
        ax.figure.canvas.draw()

def main():
    sac_directory = 'C:/Users/sjgauva/Desktop/AtmoSense/campaign_2/documents_20241218/shot_day_ground_gem'
    metadata_file = 'C:/Users/sjgauva/Desktop/AtmoSense/campaign_2/0.1-20_Hz_final_results.xlsx'
    distance_files = [
        'C:/Users/sjgauva/Desktop/AtmoSense/campaign_2/event_data_20241016_172900.csv',
        'C:/Users/sjgauva/Desktop/AtmoSense/campaign_2/event_data_20241016_173200.csv',
        'C:/Users/sjgauva/Desktop/AtmoSense/campaign_2/event_data_20241016_190700.csv',
        'C:/Users/sjgauva/Desktop/AtmoSense/campaign_2/event_data_20241016_191000.csv',
        'C:/Users/sjgauva/Desktop/AtmoSense/campaign_2/event_data_20241016_202900.csv'
    ]
    
    stream = read_sac_files(sac_directory)
    metadata = read_metadata(metadata_file)
    distance_data = read_distance_data(distance_files)
    
    results, amp_info = process_and_plot(stream, metadata, distance_files)
    
    # Convert results to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv('AtmoSense_analysis_results.csv', index=False)
    
    return stream, metadata, distance_data, results, amp_info

if __name__ == "__main__":
    stream, metadata, distance_data, results, amp_info = main()
