from pymavlink import mavutil
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

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
        msg = mlog.recv_match(type='ATTITUDE', blocking=False)
        if msg is None:
            break
            
        if msg.get_type() == 'ATTITUDE':
            attitude_data.append({
                'timestamp': msg._timestamp,
                'time_boot_ms': msg.time_boot_ms,
                'roll': msg.roll,  # radians
                'pitch': msg.pitch,  # radians
                'yaw': msg.yaw,  # radians
                'rollspeed': msg.rollspeed,
                'pitchspeed': msg.pitchspeed,
                'yawspeed': msg.yawspeed
            })
    
    return pd.DataFrame(attitude_data)

# Extract battery data
def extract_battery_data(mlog):
    """Extract battery information"""
    battery_data = []
    
    mlog.rewind()
    
    while True:
        msg = mlog.recv_match(type='BATTERY_STATUS', blocking=False)
        if msg is None:
            break
            
        if msg.get_type() == 'BATTERY_STATUS':
            battery_data.append({
                'timestamp': msg._timestamp,
                'voltage': msg.voltages[0] / 1000.0 if msg.voltages[0] != -1 else None,  # Convert to volts
                'current': msg.current_battery / 100.0 if msg.current_battery != -1 else None,  # Convert to amps
                'battery_remaining': msg.battery_remaining
            })
    
    return pd.DataFrame(battery_data)

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
    
    return {
        'gps': gps_df,
        'attitude': attitude_df,
        'battery': battery_df,
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

def plot_altitude_profile(gps_df, save_path=None):
    """Plot altitude over time"""
    if gps_df.empty:
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(gps_df)), gps_df['alt'], 'b-', linewidth=1)
    plt.xlabel('Time (samples)')
    plt.ylabel('Altitude (m)')
    plt.title('Altitude Profile')
    plt.grid(True)
    
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
                
            # Reuse the extract_gps_data function
            gps_df = extract_gps_data(mlog)
            
            if not gps_df.empty:
                # Convert to webapp format (msgs_synced.csv format)
                webapp_data = []
                new_data_count = 0
                duplicate_count = 0
                
                for _, row in gps_df.iterrows():
                    if row['lat'] is not None and row['lon'] is not None:
                        # Round timestamp to 6 decimal places for consistent comparison
                        rounded_timestamp = round(row['timestamp'], 6)
                        
                        # Check if this rounded timestamp already exists
                        if rounded_timestamp not in existing_timestamps:
                            timestamp = datetime.fromtimestamp(row['timestamp'])
                            
                            webapp_row = {
                                'image_path': file_path,
                                'time': timestamp.strftime('%Y:%m:%d %H:%M:%S.%f'),  # Extended format with microseconds
                                'timestamp': rounded_timestamp,  # Store rounded timestamp for next iteration
                                'lat': round(row['lat'], 8),  # Round coordinates to 8 decimal places (~1cm precision)
                                'lon': round(row['lon'], 8),
                                'alt': round(row['alt'], 2) if row['alt'] is not None else None,  # Round altitude to cm
                                'naturalWidth': None,
                                'naturalHeight': None
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
                else:
                    print(f"No new data found in {file_path} (all {len(gps_df)} points already exist)")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save to CSV if new data was extracted
    if not gps_df_total.empty:
        # Drop the timestamp column before saving (keep only webapp format)
        # gps_df_save = gps_df_total.drop('timestamp', axis=1, errors='ignore')
        
        if existing_df is not None and not existing_df.empty:
            gps_df_total.to_csv(msgs_synced_file, mode='a', header=False, index=False)
        else:
            gps_df_total.to_csv(msgs_synced_file, mode='w', header=True, index=False)
        
        print(f"Saved {len(gps_df_total)} new MAVLink records to {msgs_synced_file}")
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
    
    # Create output directory for plots
    import os
    output_dir = os.path.join(os.path.dirname(log_file_path), "analysis_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze the log
    results = analyze_ardupilot_log(log_file_path)
    
    if results:
        gps_df = results['gps']
        
        if not gps_df.empty:
            # Save plots as images
            base_name = os.path.splitext(os.path.basename(log_file_path))[0]
            
            plot_flight_path(gps_df, 
                           save_path=os.path.join(output_dir, f"{base_name}_flight_path.png"))
            
            plot_altitude_profile(gps_df, 
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
        else:
            print("No GPS data found in log file")
    else:
        print("Failed to analyze log file")