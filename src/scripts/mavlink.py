from pymavlink import mavutil
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timezone
import numpy as np
from scipy.spatial import KDTree

# Load the log file (.bin, .log, or .tlog)
def load_ardupilot_log(log_file_path):
    """Load ArduPilot log file"""
    try:
        # Connect to log file
        mlog = mavutil.mavlink_connection(log_file_path)
        print(f"Successfully loaded log: {log_file_path}")
        return mlog
    except Exception as e:
        print(f"Error loading log: {e}")
        return None

# Extract GPS data
def extract_gps_data(mlog):
    """Extract GPS coordinates and timestamps"""
    gps_data = []
    
    # Reset to beginning of log
    mlog.rewind()
    
    while True:
        msg = mlog.recv_match(type='GPS', blocking=False)
        if msg is None:
            break
            
        if msg.get_type() == 'GPS':
            gps_data.append({
                'timestamp': msg._timestamp,
                'time_usec': msg.TimeUS if hasattr(msg, 'TimeUS') else None,
                'gps_instance': msg.I if hasattr(msg, 'I') else 0,
                'status': msg.Status if hasattr(msg, 'Status') else None,
                'gps_week': msg.GWk if hasattr(msg, 'GWk') else None,
                'gps_ms': msg.GMS if hasattr(msg, 'GMS') else None,
                'lat': msg.Lat if hasattr(msg, 'Lat') and msg.Lat else None, 
                'lon': msg.Lng if hasattr(msg, 'Lng') and msg.Lng else None,  
                'alt': msg.Alt if hasattr(msg, 'Alt') and msg.Alt else None, 
                'hdop': msg.HDop if hasattr(msg, 'HDop') else None,  
                'satellites': msg.NSats if hasattr(msg, 'NSats') else None,
                'ground_speed': msg.Spd if hasattr(msg, 'Spd') else None, 
                'ground_course': msg.GCrs if hasattr(msg, 'GCrs') else None, 
                'vertical_velocity': msg.VZ if hasattr(msg, 'VZ') else None,  
                'yaw': msg.Yaw if hasattr(msg, 'Yaw') else None, 
                'usage_flags': msg.U if hasattr(msg, 'U') else None
            })
    
    return pd.DataFrame(gps_data)

# Extract attitude data (roll, pitch, yaw)
def extract_attitude_data(mlog):
    """Extract attitude information"""
    attitude_data = []
    
    mlog.rewind()
    
    while True:
        msg = mlog.recv_match(type='ATT', blocking=False)
        if msg is None:
            break
            
        if msg.get_type() == 'ATT':
            attitude_data.append({
                'timestamp': msg._timestamp,
                'time_boot_ms': msg.TimeUS,
                'roll': msg.Roll,  # degrees
                'pitch': msg.Pitch,  # degrees
                'yaw': msg.Yaw,  # degrees
            })
    
    return pd.DataFrame(attitude_data)

# Extract battery data
def extract_battery_data(mlog):
    """Extract battery information"""
    battery_data = []
    
    mlog.rewind()
    
    while True:
        msg = mlog.recv_match(type='BAT', blocking=False)
        if msg is None:
            break
            
        if msg.get_type() == 'BAT':
            battery_data.append({
                'timestamp': msg._timestamp,
                'voltage': msg.VoltR if msg.VoltR != -1 else None,  # Convert to volts
                'current': msg.Curr if msg.Curr != -1 else None,  # Convert to amps
                'battery_remaining': msg.RemPct / 100.0 # Coverts to 0-1
            })
    
    return pd.DataFrame(battery_data)

# Extract rangefinder data
def extract_rangefinder_data(mlog):
    """Extract rangefinder distance information"""
    rangefinder_data = []
    
    mlog.rewind()
    
    while True:
        msg = mlog.recv_match(type=['RFND'], blocking=False)
        if msg is None:
            break
            
        rangefinder_data.append({
            'timestamp': msg._timestamp,
            'distance': msg.Dist  # Distance in meters
        })

    
    return pd.DataFrame(rangefinder_data)

# Extract all message types
def get_message_types(mlog):
    """Get all available message types in the log"""
    mlog.rewind()
    message_types = set()
    
    while True:
        msg = mlog.recv_match(blocking=False)
        if msg is None:
            break
        message_types.add(msg.get_type())
    
    return sorted(list(message_types))

# Complete analysis function
def analyze_ardupilot_log(log_file_path):
    """Complete log analysis"""
    # Load log
    mlog = load_ardupilot_log(log_file_path)
    if not mlog:
        return None
    
    # Get available message types
    print("Available message types:")
    msg_types = get_message_types(mlog)
    for msg_type in msg_types[:10]:  # Show first 10
        print(f"  - {msg_type}")
    print(f"  ... and {len(msg_types)-10} more")
    
    # Extract data
    print("\nExtracting GPS data...")
    gps_df = extract_gps_data(mlog)
    print(f"Found {len(gps_df)} GPS records")
    
    print("Extracting attitude data...")
    attitude_df = extract_attitude_data(mlog)
    print(f"Found {len(attitude_df)} attitude records")
    
    print("Extracting battery data...")
    battery_df = extract_battery_data(mlog)
    print(f"Found {len(battery_df)} battery records")
    
    print("Extracting rangefinder data...")
    rangefinder_df = extract_rangefinder_data(mlog)
    print(f"Found {len(rangefinder_df)} rangefinder records")
    
    return {
        'gps': gps_df,
        'attitude': attitude_df,
        'battery': battery_df,
        'rangefinder': rangefinder_df,
        'message_types': msg_types,
        'mlog': mlog
    }

# Plotting functions
def plot_flight_path(gps_df, save_path=None):
    """Plot the flight path"""
    if gps_df.empty or gps_df['lat'].isna().all():
        print("No valid GPS data for plotting")
        return
    
    plt.figure(figsize=(10, 8))
    plt.plot(gps_df['lon'], gps_df['lat'], 'b-', linewidth=1, alpha=0.7)
    plt.scatter(gps_df['lon'].iloc[0], gps_df['lat'].iloc[0], color='green', s=100, label='Start')
    plt.scatter(gps_df['lon'].iloc[-1], gps_df['lat'].iloc[-1], color='red', s=100, label='End')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Flight Path')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Flight path saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_altitude_profile(gps_df, rangefinder_df=None, save_path=None):
    """Plot altitude over time with rangefinder/LiDAR distance on single y-axis"""
    if gps_df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot altitude
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Height above Ground (m)')
    
    # Filter out invalid altitude values
    valid_alt_mask = (gps_df['alt'].notna()) & (gps_df['alt'] > 0)
    valid_indices = gps_df.index[valid_alt_mask].tolist()
    valid_altitudes = gps_df.loc[valid_alt_mask, 'alt'].tolist()
    
    if valid_altitudes:
        ax.plot(valid_indices, valid_altitudes, 'b-', linewidth=2, label='GPS Altitude (MSL)', alpha=0.8)
        ax.grid(True, alpha=0.3)
    
    # Add rangefinder data if available
    rangefinder_plotted = False
    if rangefinder_df is not None and not rangefinder_df.empty:
        print(f"Found {len(rangefinder_df)} rangefinder records")
        
        # Align rangefinder data with GPS timestamps
        rf_distances = []
        rf_indices = []
        
        for idx, gps_row in gps_df.iterrows():
            if pd.notna(gps_row['timestamp']):
                # Find closest rangefinder timestamp
                time_diffs = abs(rangefinder_df['timestamp'] - gps_row['timestamp'])
                if len(time_diffs) > 0:
                    closest_idx = time_diffs.idxmin()
                    
                    # Only use if within 2 second tolerance
                    if time_diffs[closest_idx] <= 2.0:
                        distance = rangefinder_df.loc[closest_idx, 'distance']
                        if pd.notna(distance) and distance > 0:
                            rf_distances.append(distance)
                            rf_indices.append(idx)
        
        # Plot rangefinder data on same axis
        if rf_distances:
            ax.plot(rf_indices, rf_distances, 'r-', linewidth=2, alpha=0.7, label='LiDAR Height above Ground')
            rangefinder_plotted = True
            
            # Print statistics
            print(f"LiDAR data: min={min(rf_distances):.2f}m, max={max(rf_distances):.2f}m, avg={sum(rf_distances)/len(rf_distances):.2f}m")
            print(f"LiDAR coverage: {len(rf_distances)}/{len(gps_df)} GPS points ({len(rf_distances)/len(gps_df)*100:.1f}%)")
        else:
            print("No valid rangefinder distances found after alignment")
    
    # Set title and legend
    if rangefinder_plotted:
        plt.title('Flight Height Profile: GPS Altitude (MSL) vs LiDAR Height (AGL)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
    else:
        plt.title('GPS Altitude Profile (Mean Sea Level)', fontsize=14, fontweight='bold')
        ax.legend()
        
        # Print debug info about rangefinder data
        if rangefinder_df is not None:
            print(f"Rangefinder DataFrame info:")
            print(f"  Shape: {rangefinder_df.shape}")
            print(f"  Columns: {list(rangefinder_df.columns)}")
            if not rangefinder_df.empty:
                print(f"  Distance range: {rangefinder_df['distance'].min():.2f} - {rangefinder_df['distance'].max():.2f}m")
                print(f"  Valid distances: {rangefinder_df['distance'].notna().sum()}/{len(rangefinder_df)}")
        else:
            print("No rangefinder data provided to plotting function")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Altitude profile saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_gps_quality(gps_df, save_path=None):
    """Plot GPS quality metrics"""
    if gps_df.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Satellite count
    axes[0, 0].plot(gps_df['satellites'], 'g-', linewidth=1)
    axes[0, 0].set_title('Satellite Count')
    axes[0, 0].set_ylabel('Number of Satellites')
    axes[0, 0].grid(True)
    
    # HDOP (already converted by pymavlink)
    axes[0, 1].plot(gps_df['hdop'], 'r-', linewidth=1)
    axes[0, 1].set_title('HDOP (Horizontal Dilution of Precision)')
    axes[0, 1].set_ylabel('HDOP')
    axes[0, 1].grid(True)
    
    # GPS Status
    status_counts = gps_df['status'].value_counts().sort_index()
    status_labels = [interpret_gps_status(s) for s in status_counts.index]
    axes[1, 0].bar(range(len(status_counts)), status_counts.values)
    axes[1, 0].set_title('GPS Fix Status Distribution')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_xticks(range(len(status_counts)))
    axes[1, 0].set_xticklabels(status_labels, rotation=45)
    axes[1, 0].grid(True)
    
    # Ground Speed (already converted by pymavlink)
    axes[1, 1].plot(gps_df['ground_speed'], 'purple', linewidth=1)
    axes[1, 1].set_title('Ground Speed')
    axes[1, 1].set_ylabel('Speed (m/s)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"GPS quality plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def process_mavlink_log_for_webapp(file_paths, data_type, msgs_synced_file, existing_df, existing_paths):
    """Process MAVLink logs and extract relevant data for web app"""
    
    gps_df_total = pd.DataFrame()
    
    # Convert existing_paths to a set of rounded timestamps if it's not already
    if existing_df is not None and not existing_df.empty:
        if 'timestamp' in existing_df.columns:
            # Round existing timestamps to 6 decimal places (microsecond precision)
            existing_timestamps = set(round(ts, 6) for ts in existing_df['timestamp'].values)
        elif 'time' in existing_df.columns:
            # Convert time strings back to timestamps if timestamp column doesn't exist
            existing_timestamps = set()
            for time_str in existing_df['time'].values:
                try:
                    # Parse the extended timestamp format with microseconds
                    dt = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S.%f')
                    existing_timestamps.add(round(dt.timestamp(), 6))
                except ValueError:
                    # Fallback for old format without microseconds
                    try:
                        dt = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
                        existing_timestamps.add(round(dt.timestamp(), 6))
                    except ValueError:
                        continue
        else:
            existing_timestamps = set()
    else:
        existing_timestamps = set(existing_paths) if existing_paths else set()
    
    for file_path in file_paths:
        try:
            print(f"Processing MAVLink log: {file_path}")
            
            # Load the log
            mlog = load_ardupilot_log(file_path)
            if not mlog:
                continue
                
            # Extract GPS, rangefinder, and attitude data
            gps_df = extract_gps_data(mlog)
            rangefinder_df = extract_rangefinder_data(mlog)
            attitude_df = extract_attitude_data(mlog)
            
            # Build KD-Tree for fast rangefinder timestamp lookup
            rangefinder_kdtree = None
            rangefinder_timestamps = None
            rangefinder_distances = None
            
            if not rangefinder_df.empty:
                print(f"Building KD-Tree for {len(rangefinder_df)} rangefinder records...")
                # Round timestamps to 6 decimal places for consistency
                rangefinder_timestamps = np.array([round(ts, 6) for ts in rangefinder_df['timestamp'].values])
                rangefinder_distances = rangefinder_df['distance'].values
                
                # Create 1D KD-Tree for timestamps
                rangefinder_kdtree = KDTree(rangefinder_timestamps.reshape(-1, 1))
                print(f"KD-Tree built successfully")
            
            # Build KD-Tree for attitude data
            attitude_kdtree = None
            attitude_timestamps = None
            attitude_rolls = None
            attitude_pitches = None
            attitude_yaws = None
            
            if not attitude_df.empty:
                print(f"Building KD-Tree for {len(attitude_df)} attitude records...")
                # Round timestamps to 6 decimal places for consistency
                attitude_timestamps = np.array([round(ts, 6) for ts in attitude_df['timestamp'].values])
                attitude_rolls = attitude_df['roll'].values
                attitude_pitches = attitude_df['pitch'].values
                attitude_yaws = attitude_df['yaw'].values
                
                # Create 1D KD-Tree for timestamps
                attitude_kdtree = KDTree(attitude_timestamps.reshape(-1, 1))
                print(f"Attitude KD-Tree built successfully")
            
            if not gps_df.empty:
                # Convert to webapp format (msgs_synced.csv format)
                webapp_data = []
                new_data_count = 0
                duplicate_count = 0
                
                print(f"Processing {len(gps_df)} GPS records...")
                
                for _, row in gps_df.iterrows():
                    if row['lat'] is not None and row['lon'] is not None:
                        # Round timestamp to 6 decimal places for consistent comparison
                        rounded_timestamp = round(row['timestamp'], 6)
                        
                        # Check if this rounded timestamp already exists
                        if rounded_timestamp not in existing_timestamps:
                            timestamp = datetime.fromtimestamp(row['timestamp'], tz=timezone.utc)
                            
                            # Find closest rangefinder distance using KD-Tree
                            rangefinder_distance = None
                            if rangefinder_kdtree is not None:
                                try:
                                    # Query KD-Tree for nearest neighbor
                                    distances, indices = rangefinder_kdtree.query([[rounded_timestamp]], k=1)
                                    
                                    # Handle scalar vs array results
                                    if np.isscalar(distances):
                                        closest_distance = distances
                                        closest_idx = indices
                                    else:
                                        closest_distance = distances[0] if len(distances) > 0 else float('inf')
                                        closest_idx = indices[0] if len(indices) > 0 else -1
                                    
                                    # Only use if within 1 second tolerance and valid index
                                    if closest_distance <= 1.0 and closest_idx >= 0 and closest_idx < len(rangefinder_distances):
                                        rangefinder_distance = rangefinder_distances[closest_idx]
                                        
                                except Exception as kdtree_error:
                                    print(f"KD-Tree query error: {kdtree_error}")
                                    # Fallback to simple timestamp matching
                                    time_diffs = abs(rangefinder_df['timestamp'] - row['timestamp'])
                                    if len(time_diffs) > 0:
                                        closest_idx = time_diffs.idxmin()
                                        if time_diffs[closest_idx] <= 1.0:
                                            rangefinder_distance = rangefinder_df.loc[closest_idx, 'distance']
                            
                            # Find closest attitude data using KD-Tree
                            roll, pitch, yaw = None, None, None
                            if attitude_kdtree is not None:
                                try:
                                    # Query KD-Tree for nearest neighbor
                                    distances, indices = attitude_kdtree.query([[rounded_timestamp]], k=1)
                                    
                                    # Handle scalar vs array results
                                    if np.isscalar(distances):
                                        closest_distance = distances
                                        closest_idx = indices
                                    else:
                                        closest_distance = distances[0] if len(distances) > 0 else float('inf')
                                        closest_idx = indices[0] if len(indices) > 0 else -1
                                    
                                    # Only use if within 1 second tolerance and valid index
                                    if closest_distance <= 1.0 and closest_idx >= 0 and closest_idx < len(attitude_rolls):
                                        roll = attitude_rolls[closest_idx]
                                        pitch = attitude_pitches[closest_idx]
                                        yaw = attitude_yaws[closest_idx]
                                        
                                except Exception as kdtree_error:
                                    print(f"Attitude KD-Tree query error: {kdtree_error}")
                                    # Fallback to simple timestamp matching
                                    time_diffs = abs(attitude_df['timestamp'] - row['timestamp'])
                                    if len(time_diffs) > 0:
                                        closest_idx = time_diffs.idxmin()
                                        if time_diffs[closest_idx] <= 1.0:
                                            att_row = attitude_df.loc[closest_idx]
                                            roll = att_row['roll']
                                            pitch = att_row['pitch']
                                            yaw = att_row['yaw']
                            
                            # Determine altitude and source
                            if rangefinder_distance is not None and rangefinder_distance > 0:
                                # Use rangefinder distance as altitude (height above ground)
                                altitude = round(rangefinder_distance, 2)
                                alt_source = "rangefinder"
                            elif row['alt'] is not None:
                                # Use GPS altitude as fallback
                                altitude = round(row['alt'], 2)
                                alt_source = "gps"
                            else:
                                altitude = None
                                alt_source = None
                            
                            webapp_row = {
                                'image_path': file_path,
                                'time': timestamp.strftime('%Y:%m:%d %H:%M:%S.%f %z'),  # Extended format with microseconds
                                'timestamp': rounded_timestamp,  # Store rounded timestamp for next iteration
                                'lat': round(row['lat'], 8),  # Round coordinates to 8 decimal places (~1cm precision)
                                'lon': round(row['lon'], 8),
                                'alt': altitude,  # Use rangefinder distance if available, GPS altitude otherwise
                                'alt_source': alt_source,  # Source of altitude data
                                'gps_alt': round(row['alt'], 2) if row['alt'] is not None else None,  # Keep original GPS altitude
                                'rangefinder_distance': round(rangefinder_distance, 2) if rangefinder_distance is not None else None,  # Rangefinder distance in meters
                                'roll': round(roll, 2) if roll is not None else None,  
                                'pitch': round(pitch, 2) if pitch is not None else None,  
                                'yaw': round(yaw, 2) if yaw is not None else None, 
                            }
                            
                            webapp_data.append(webapp_row)
                            existing_timestamps.add(rounded_timestamp)  # Add rounded timestamp to existing set
                            new_data_count += 1
                        else:
                            duplicate_count += 1
                
                if webapp_data:
                    webapp_df = pd.DataFrame(webapp_data)
                    gps_df_total = pd.concat([gps_df_total, webapp_df], ignore_index=True)
                    print(f"Added {new_data_count} new GPS points from {file_path}")
                    if duplicate_count > 0:
                        print(f"Skipped {duplicate_count} duplicate points")
                    
                    # Print altitude source statistics
                    rf_count = len([r for r in webapp_data if r['alt_source'] == 'rangefinder'])
                    gps_count = len([r for r in webapp_data if r['alt_source'] == 'gps'])
                    print(f"Altitude sources: {rf_count} rangefinder ({rf_count/new_data_count*100:.1f}%), {gps_count} GPS ({gps_count/new_data_count*100:.1f}%)")
                    
                    # Print attitude data coverage
                    attitude_count = len([r for r in webapp_data if r['roll'] is not None])
                    print(f"Attitude data coverage: {attitude_count}/{new_data_count} GPS points ({attitude_count/new_data_count*100:.1f}%)")
                else:
                    print(f"No new data found in {file_path} (all {len(gps_df)} points already exist)")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save to CSV if new data was extracted
    if not gps_df_total.empty:
        if existing_df is not None and not existing_df.empty:
            gps_df_total.to_csv(msgs_synced_file, mode='a', header=False, index=False)
        else:
            gps_df_total.to_csv(msgs_synced_file, mode='w', header=True, index=False)
        
        print(f"Saved {len(gps_df_total)} new MAVLink records to {msgs_synced_file}")
        
        # Print altitude source summary
        alt_sources = gps_df_total['alt_source'].value_counts()
        print(f"Final altitude source distribution:")
        for source, count in alt_sources.items():
            print(f"  {source}: {count} records ({count/len(gps_df_total)*100:.1f}%)")
        
        # Print altitude ranges by source
        for source in alt_sources.index:
            source_data = gps_df_total[gps_df_total['alt_source'] == source]['alt'].dropna()
            if not source_data.empty:
                print(f"{source.capitalize()} altitude: min={source_data.min():.2f}m, max={source_data.max():.2f}m, avg={source_data.mean():.2f}m")
        
        # Print attitude data summary
        attitude_data = gps_df_total[['roll', 'pitch', 'yaw']].dropna()
        if not attitude_data.empty:
            print(f"Attitude data summary:")
            print(f"  Roll: min={attitude_data['roll'].min():.1f}°, max={attitude_data['roll'].max():.1f}°, avg={attitude_data['roll'].mean():.1f}°")
            print(f"  Pitch: min={attitude_data['pitch'].min():.1f}°, max={attitude_data['pitch'].max():.1f}°, avg={attitude_data['pitch'].mean():.1f}°")
            print(f"  Yaw: min={attitude_data['yaw'].min():.1f}°, max={attitude_data['yaw'].max():.1f}°, avg={attitude_data['yaw'].mean():.1f}°")
            print(f"  Coverage: {len(attitude_data)}/{len(gps_df_total)} records ({len(attitude_data)/len(gps_df_total)*100:.1f}%)")
        else:
            print("No attitude data found in this log")
    else:
        print("No new data to save")
    
    return gps_df_total

def interpret_gps_status(status):
    """Interpret GPS status code"""
    status_map = {
        0: "No GPS",
        1: "No Fix", 
        2: "2D Fix",
        3: "3D Fix",
        4: "DGPS",
        5: "RTK Float",
        6: "RTK Fixed"
    }
    return status_map.get(status, f"Unknown ({status})")

# Main execution
if __name__ == "__main__":
    log_file_path = "/home/heesup/GEMINI-App-Data/Raw/2025/GEMINI/Davis/Legumes/2025-06-24/Drone-5m/iPhoneRGB/Metadata/2025-06-24_10-14-05.bin"
    
    # Create output directory for plots and CSV
    import os
    output_dir = os.path.join(os.path.dirname(log_file_path), "analysis_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze the log
    results = analyze_ardupilot_log(log_file_path)
    
    if results:
        gps_df = results['gps']
        rangefinder_df = results['rangefinder']
        if not gps_df.empty:
            # Save plots as images
            base_name = os.path.splitext(os.path.basename(log_file_path))[0]
            
            plot_flight_path(gps_df, 
                           save_path=os.path.join(output_dir, f"{base_name}_flight_path.png"))
            
            print(f"\nCreating altitude profile with rangefinder data...")
            plot_altitude_profile(gps_df, rangefinder_df,
                                save_path=os.path.join(output_dir, f"{base_name}_altitude.png"))
            
            plot_gps_quality(gps_df, 
                           save_path=os.path.join(output_dir, f"{base_name}_gps_quality.png"))
            
            # Print summary (values are already converted)
            print(f"\nFlight Summary:")
            print(f"Duration: {len(gps_df)} GPS samples")
            print(f"Max Altitude: {gps_df['alt'].max():.1f}m")
            print(f"Min Altitude: {gps_df['alt'].min():.1f}m")
            print(f"Average satellites: {gps_df['satellites'].mean():.1f}")
            print(f"Average HDOP: {gps_df['hdop'].mean():.2f}")
            print(f"Average ground speed: {gps_df['ground_speed'].mean():.2f} m/s")
            
            print(f"\nPlots saved in: {output_dir}")
            
            # Test the process_mavlink_log_for_webapp function
            print("\n" + "="*50)
            print("TESTING process_mavlink_log_for_webapp function")
            print("="*50)
            
            # Set up test parameters
            file_paths = [log_file_path]
            data_type = "mavlink"
            msgs_synced_file = os.path.join(output_dir, f"{base_name}_drone_msgs.csv")
            existing_df = None  # No existing data for first run
            existing_paths = set()  # No existing paths
            
            # Test first run (should create new CSV)
            print(f"\nTest 1: Processing MAVLink log for webapp (first run)")
            result_df = process_mavlink_log_for_webapp(
                file_paths=file_paths,
                data_type=data_type,
                msgs_synced_file=msgs_synced_file,
                existing_df=existing_df,
                existing_paths=existing_paths
            )
            
            if not result_df.empty:
                print(f"✅ Successfully processed {len(result_df)} records")
                print(f"CSV saved to: {msgs_synced_file}")
                
                # Display sample of the data
                print(f"\nSample of processed data:")
                print(result_df[['time', 'lat', 'lon', 'alt', 'alt_source', 'gps_alt', 'rangefinder_distance']].head())
                
                # Test second run (should detect duplicates)
                print(f"\nTest 2: Processing same log again (should detect duplicates)")
                existing_df_test = pd.read_csv(msgs_synced_file)
                
                result_df2 = process_mavlink_log_for_webapp(
                    file_paths=file_paths,
                    data_type=data_type,
                    msgs_synced_file=msgs_synced_file,
                    existing_df=existing_df_test,
                    existing_paths=set()
                )
                
                if result_df2.empty:
                    print("✅ Successfully detected all records as duplicates")
                else:
                    print(f"⚠️ Unexpected: Found {len(result_df2)} new records on second run")
                
                # Show final CSV stats
                final_df = pd.read_csv(msgs_synced_file)
                print(f"\nFinal CSV statistics:")
                print(f"Total records: {len(final_df)}")
                print(f"Records with rangefinder data: {final_df['rangefinder_distance'].notna().sum()}")
                print(f"Date range: {final_df['time'].min()} to {final_df['time'].max()}")
                
                # Show rangefinder data summary
                rf_data = final_df['rangefinder_distance'].dropna()
                if not rf_data.empty:
                    print(f"Rangefinder data summary:")
                    print(f"  Min distance: {rf_data.min():.2f}m")
                    print(f"  Max distance: {rf_data.max():.2f}m")
                    print(f"  Average distance: {rf_data.mean():.2f}m")
                    print(f"  Coverage: {len(rf_data)}/{len(final_df)} records ({len(rf_data)/len(final_df)*100:.1f}%)")
                else:
                    print("No rangefinder data found in this log")
                    
            else:
                print("❌ No data was processed")
                
        else:
            print("No GPS data found in log file")
    else:
        print("Failed to analyze log file")