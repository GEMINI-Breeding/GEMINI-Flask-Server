# Standard library imports
import os
import pandas as pd
import zipfile
import shutil
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app, send_file
from pyproj import Geod
import json
import numpy as np
from scipy.spatial import KDTree

plot_marking_bp = Blueprint('plot_marking', __name__)

def check_csv_integrity(csv_path):
    """Check CSV file integrity and report issues"""
    try:
        # Try to read with error detection
        df = pd.read_csv(csv_path, on_bad_lines='warn')
        expected_cols = 33  # Expected number of columns based on header # TODO: Sometimes not true
        
        # Check if we have the expected number of columns
        if len(df.columns) != expected_cols:
            print(f"WARNING: CSV has {len(df.columns)} columns, expected {expected_cols}")
        
        # Check for any rows that might have been skipped
        with open(csv_path, 'r') as f:
            total_lines = sum(1 for line in f) - 1  # Subtract header
        
        if len(df) != total_lines:
            print(f"WARNING: CSV parsing skipped {total_lines - len(df)} rows due to formatting issues")
            
        return True, df
    except Exception as e:
        print(f"ERROR: CSV integrity check failed: {e}")
        return False, None

def update_plot_index(directory, start_image_name, end_image_name, plot_index, camera, stitch_direction=None, filter=False, shift_all=False, shift_amount=0, original_start_image_index=None):
    data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
    image_dir_path = os.path.join(data_root_dir_path, directory)
    
    metadata_dir = os.path.abspath(os.path.join(image_dir_path, '..', '..', 'Metadata'))
    csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')
    start_image_name = "/top/" + start_image_name
    end_image_name = "/top/" + end_image_name
    
     # check if metadata directory exists
    if not os.path.exists(metadata_dir):
           
        metadata_dir = os.path.abspath(os.path.join(image_dir_path, '..')) # folder above Metadata
        csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')
        
        start_image_name = start_image_name.replace('/top/', '')
        end_image_name = end_image_name.replace('/top/', '')

        if not os.path.exists(csv_path):
            print(f"ERROR: CSV file not found at {csv_path}")
            return False
        
    # if not os.path.exists(csv_path):
    #     print(f"ERROR: Metadata CSV not found at {csv_path}")
    #     return False
    
    # Check CSV integrity before processing
    is_valid, df = check_csv_integrity(csv_path)
    if not is_valid:
        print(f"ERROR: CSV integrity check failed")
        return False
        
    df.columns = df.columns.str.strip() # FIX: Sanitize column names
    if 'plot_index' not in df.columns:
        df['plot_index'] = -1
    if 'stitch_direction' not in df.columns:
        df['stitch_direction'] = None
    
    # Check if this plot_index already exists (indicating an edit/update operation)
    is_update = plot_index in df['plot_index'].values
    image_column = f'/{camera}/rgb_file'
    if image_column not in df.columns:
        
        image_column = '/top/rgb_file'  # Fallback to default column
        
        if image_column not in df.columns:
            print(f"ERROR: Image column '{image_column}' not found in CSV")
            return False
        
    try:
        start_row_index = df.index[df[image_column] == start_image_name].tolist()[0]
        end_row_index = df.index[df[image_column] == end_image_name].tolist()[0]
    except IndexError:
        print("ERROR: Start or end image not found in metadata")
        return False
    current_plot_index = int(plot_index)
    
    # Save stitch direction to dedicated file for dataset-wide persistence
    if stitch_direction and not filter:
        save_stitch_direction_to_file(image_dir_path, stitch_direction)
    
    # Create or update plot_borders.csv
    plot_borders_path = os.path.abspath(os.path.join(image_dir_path, '../../../../..', 'plot_borders.csv'))
    # puts plot_borders.csv in $data_root_dir$/Raw/$year$/$experiment$/$location$/$population$/ {image_dir_path is Raw/$year$/$experiment$/$location$/$population$/$platform$/$sensor$/Images/top}
    if os.path.exists(plot_borders_path):
        try:
            borders_df = pd.read_csv(plot_borders_path, on_bad_lines='skip')
        except Exception as e:
            print(f"WARNING: Failed to read plot_borders.csv, creating new one: {e}")
            borders_df = pd.DataFrame(columns=["plot_index", "start_lat", "start_lon", "end_lat", "end_lon", "stitch_direction"])
            update_pb = True
        else:
            borders_df.columns = borders_df.columns.str.strip() # FIX: Sanitize column names
            if plot_index not in borders_df['plot_index'].values:
                update_pb = True
            else:
                update_pb = False
    else:
        borders_df = pd.DataFrame(columns=["plot_index", "start_lat", "start_lon", "end_lat", "end_lon", "stitch_direction"])
        update_pb = True
    if not filter and update_pb:
        try:
            start_lat = df.loc[start_row_index, 'lat']
            start_lon = df.loc[start_row_index, 'lon']
            end_lat = df.loc[end_row_index, 'lat']
            end_lon = df.loc[end_row_index, 'lon']

            new_border_data = {
                "plot_index": current_plot_index,
                "start_lat": start_lat,
                "start_lon": start_lon,
                "end_lat": end_lat,
                "end_lon": end_lon,
                "stitch_direction": stitch_direction
            }
            # Check if plot_index already exists and update it, otherwise append
            if current_plot_index in borders_df['plot_index'].values:
                idx = borders_df.index[borders_df['plot_index'] == current_plot_index].tolist()[0]
                borders_df.loc[idx] = new_border_data
            else:
                borders_df = pd.concat([borders_df, pd.DataFrame([new_border_data])], ignore_index=True)

            borders_df.to_csv(plot_borders_path, index=False)

        except Exception as e:
            print(f"ERROR: Could not update plot_borders.csv: {e}")
    # If this is an update, clear the old plot index entries first
    if is_update:
        # Store the mask before modifying the dataframe
        old_plot_mask = df['plot_index'] == plot_index
        df.loc[old_plot_mask, 'plot_index'] = -1
        df.loc[old_plot_mask, 'stitch_direction'] = None
    
    # Assign plot index and stitch dir to msgs_synced.csv
    df.loc[start_row_index:end_row_index, 'plot_index'] = current_plot_index
    df.loc[start_row_index:end_row_index, 'stitch_direction'] = stitch_direction
    
    if shift_all and shift_amount != 0 and is_update:
        unique_plots_to_shift = df[df['plot_index'] > current_plot_index]['plot_index'].unique()
        unique_plots_to_shift = unique_plots_to_shift[unique_plots_to_shift != -1]  # Exclude unassigned plots
        
        if len(unique_plots_to_shift) > 0:
            plot_assignments = {}
            for plot_idx in unique_plots_to_shift:
                plot_rows = df[df['plot_index'] == plot_idx].index.tolist()
                plot_assignments[plot_idx] = {
                    'rows': plot_rows,
                    'stitch_direction': df.loc[plot_rows[0], 'stitch_direction'] if plot_rows else None
                }
                # Store the mask before modifying the dataframe
                plot_mask = df['plot_index'] == plot_idx
                df.loc[plot_mask, 'plot_index'] = -1
                df.loc[plot_mask, 'stitch_direction'] = None
        
            for plot_idx, assignment in plot_assignments.items():
                old_rows = assignment['rows']
                stitch_dir = assignment['stitch_direction']
                
                new_rows = []
                for old_row in old_rows:
                    new_row = old_row - shift_amount  
                    if 0 <= new_row < len(df):
                        new_rows.append(new_row)
                
                # Assign the plot to the new rows
                if new_rows:
                    df.loc[new_rows, 'plot_index'] = plot_idx
                    df.loc[new_rows, 'stitch_direction'] = stitch_dir
                else:
                    print(f"WARNING: Plot {plot_idx} could not be shifted - new position out of bounds")
    
    # Use atomic write to prevent CSV corruption
    try:
        backup_path = csv_path + '.backup'
        # Create backup before writing
        import shutil
        shutil.copy2(csv_path, backup_path)
        
        # Write to temporary file first
        temp_path = csv_path + '.tmp'
        df.to_csv(temp_path, index=False)
        
        # Verify the temporary file can be read
        verification_df = pd.read_csv(temp_path, on_bad_lines='skip')
        if len(verification_df) > 0:
            # Atomic move to replace original
            shutil.move(temp_path, csv_path)
            
            # Remove backup after successful write
            if os.path.exists(backup_path):
                os.remove(backup_path)
        else:
            print(f"ERROR: Verification failed - temporary file is empty")
            # Restore from backup
            shutil.move(backup_path, csv_path)
            return False
            
    except Exception as e:
        print(f"ERROR: Failed to save CSV safely: {e}")
        # Restore from backup if it exists
        backup_path = csv_path + '.backup'
        if os.path.exists(backup_path):
            shutil.move(backup_path, csv_path)
        return False
    
    return True

def save_stitch_direction_to_file(image_dir_path, stitch_direction):
    """Save the stitch direction to a dedicated file for dataset-wide persistence"""
    try:
        # Save to the population directory level
        stitch_dir_file = os.path.abspath(os.path.join(image_dir_path, '../../../../..', 'stitch_direction.json'))
        stitch_data = {
            "stitch_direction": stitch_direction
        }
        with open(stitch_dir_file, 'w') as f:
            json.dump(stitch_data, f)
    except Exception as e:
        print(f"ERROR: Failed to save stitch direction: {e}")

def get_stitch_direction_from_file(image_dir_path):
    """Get the saved stitch direction from file"""
    try:
        stitch_dir_file = os.path.abspath(os.path.join(image_dir_path, '../../../../..', 'stitch_direction.json'))
        if os.path.exists(stitch_dir_file):
            with open(stitch_dir_file, 'r') as f:
                stitch_data = json.load(f)
                return stitch_data.get('stitch_direction')
        return None
    except Exception as e:
        print(f"ERROR: Failed to read stitch direction: {e}")
        return None

@plot_marking_bp.route('/mark_plot', methods=['POST'])
def mark_plot():
    data = request.get_json()
    directory = data.get('directory')
    start_image_name = data.get('start_image_name')
    end_image_name = data.get('end_image_name')
    plot_index = data.get('plot_index')
    camera = data.get('camera')
    stitch_direction = data.get('stitch_direction')
    shift_all = data.get('shift_all', False)
    shift_amount = data.get('shift_amount', 0)
    original_start_image_index = data.get('original_start_image_index')

    # if not all([directory, start_image_name, end_image_name, plot_index is not None, camera]):
    if not all([directory, start_image_name, end_image_name, plot_index is not None]):
        print(f"ERROR: Missing required parameters: {directory}, {start_image_name}, {end_image_name}, {plot_index}")
        return jsonify({"error": "Missing required parameters"}), 400

    success = update_plot_index(directory, start_image_name, end_image_name, plot_index, camera, stitch_direction, filter=False, shift_all=shift_all, shift_amount=shift_amount, original_start_image_index=original_start_image_index)
    if success:
        return jsonify({"status": "success", "message": f"Plot {plot_index} marked successfully."})
    else:
        print(f"ERROR: Failed to mark plot {plot_index}")
        return jsonify({"error": "Failed to mark plot."}), 500

@plot_marking_bp.route('/get_stitch_direction', methods=['POST'])
def get_stitch_direction():
    """Get the saved stitch direction for a dataset"""
    data = request.get_json()
    directory = data.get('directory')
    
    if not directory:
        return jsonify({'error': 'Missing directory'}), 400
    
    try:
        data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
        image_dir_path = os.path.join(data_root_dir_path, directory)
        
        stitch_direction = get_stitch_direction_from_file(image_dir_path)
        
        return jsonify({'stitch_direction': stitch_direction}), 200
        
    except Exception as e:
        print(f"ERROR: Failed to get stitch direction: {e}")
        return jsonify({'error': str(e)}), 500

@plot_marking_bp.route('/save_stitch_direction', methods=['POST'])
def save_stitch_direction():
    """Save the stitch direction for a dataset"""
    data = request.get_json()
    directory = data.get('directory')
    stitch_direction = data.get('stitch_direction')
    
    if not directory or not stitch_direction:
        return jsonify({'error': 'Missing directory or stitch_direction'}), 400
    
    try:
        data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
        image_dir_path = os.path.join(data_root_dir_path, directory)
        
        save_stitch_direction_to_file(image_dir_path, stitch_direction)
        
        return jsonify({'status': 'success', 'message': 'Stitch direction saved successfully'}), 200
        
    except Exception as e:
        print(f"ERROR: Failed to save stitch direction: {e}")
        return jsonify({'error': str(e)}), 500

@plot_marking_bp.route('/get_max_plot_index', methods=['POST'])
def get_max_plot_index():
    data = request.get_json()
    directory = data.get('directory')
    data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
    dir_path = os.path.join(data_root_dir_path, directory)
    metadata_dir = os.path.abspath(os.path.join(dir_path, '..', '..', 'Metadata'))
    if not os.path.exists(metadata_dir):
        metadata_dir = os.path.abspath(os.path.join(dir_path, '..'))
    csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')
    if not directory:
        return jsonify({'error': 'Missing directory'}), 400
    if not os.path.exists(csv_path):
        # If the CSV doesn't exist, no plots have been marked.
        return jsonify({'max_plot_index': -1}), 200

    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except Exception as e:
        print(f"ERROR: Failed to read CSV file: {e}")
        return jsonify({'error': f'Failed to read CSV: {str(e)}'}), 500
        
    df.columns = df.columns.str.strip() # FIX: Sanitize column names
    if 'plot_index' in df.columns and not df['plot_index'].empty:
        max_index = df['plot_index'].max()
        return jsonify({'max_plot_index': int(max_index)}), 200
    else:
        # If column doesn't exist or is empty, no plots marked.
        return jsonify({'max_plot_index': -1}), 200


@plot_marking_bp.route('/get_image_plot_index', methods=['POST'])
def get_image_plot_index():
    data = request.get_json()
    directory = data.get('directory')
    image_name = data.get('image_name')
    image_name = "/top/" + image_name
    data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
    image_dir_path = os.path.join(data_root_dir_path, directory)
    metadata_dir = os.path.abspath(os.path.join(image_dir_path, '..', '..', 'Metadata'))
    csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')
    
    print(f"Looking for image '{image_name}' in directory '{directory}'")
    # check if image_name is in directory
    if not os.path.exists(image_dir_path):
        return jsonify({'error': 'Image directory not found'}), 404
    if not os.path.exists(os.path.join(image_dir_path, image_name)):
        
        # try another path (remove top/)
        image_name = image_name.replace('/top/', '')
        print(f"Looking for image '{image_name}' in directory '{directory}'")
        
        if not os.path.exists(os.path.join(image_dir_path, image_name)):
            print(f"ERROR: Image file '{image_name}' not found in directory '{image_dir_path}'")
            return jsonify({'error': 'Image file not found'}), 404

    # check if metadata directory exists
    if not os.path.exists(metadata_dir):
           
        metadata_dir = os.path.abspath(os.path.join(image_dir_path, '..')) # folder above Metadata
        csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')

        if not os.path.exists(csv_path):
            print(f"ERROR: CSV file not found at {csv_path}")
            return jsonify({'error': 'CSV file not found in alternative location'}), 404
            
    if not directory or not image_name:
        print(f"ERROR: Missing directory or image name: {directory}, {image_name}")
        return jsonify({'error': f'Missing directory or image name: {directory}, {image_name}'}), 400
    
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found at {csv_path}")
        return jsonify({'error': 'CSV file not found'}), 404

    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except Exception as e:
        print(f"ERROR: Failed to read CSV file: {e}")
        return jsonify({'error': f'Failed to read CSV: {str(e)}'}), 500
        
    df.columns = df.columns.str.strip() # FIX: Sanitize column names
    
    # Find the correct image column
    image_col = None
    for col in df.columns:
        if col.endswith('_file'):
            if image_name in df[col].values:
                image_col = col
                break
    
    if not image_col:
        print(f"ERROR: Image column not found for image '{image_name}'")
        return jsonify({'plot_index': -1, 'lat': None, 'lon': None}), 200

    # Get the row for this image
    row = df[df[image_col] == image_name]

    if not row.empty:
        # Handle plot_index
        plot_index = -1
        if 'plot_index' in df.columns:
            plot_idx_val = row['plot_index'].iloc[0]
            # Handle NaN or invalid values
            if pd.notna(plot_idx_val) and plot_idx_val != -1:
                try:
                    plot_index = int(plot_idx_val)
                except (ValueError, TypeError) as e:
                    plot_index = -1
        
        # Handle lat/lon
        lat_val = None
        lon_val = None
        if 'lat' in df.columns and 'lon' in df.columns:
            lat = row['lat'].iloc[0]
            lon = row['lon'].iloc[0]
            
            # Check for valid lat/lon values
            if pd.notna(lat) and pd.notna(lon):
                try:
                    lat_val = float(lat)
                    lon_val = float(lon)
                except (ValueError, TypeError):
                    lat_val = None
                    lon_val = None

        # Get Plot and Accession data from plot_borders.csv if plot_index is valid
        plot_name = None
        accession = None
        if plot_index > -1:
            try:
                # Path to plot_borders.csv at population level
                plot_borders_path = os.path.abspath(os.path.join(image_dir_path, '../../../../..', 'plot_borders.csv'))
                if os.path.exists(plot_borders_path):
                    borders_df = pd.read_csv(plot_borders_path, on_bad_lines='skip')
                    borders_df.columns = borders_df.columns.str.strip()
                    
                    # Find the row for this plot_index
                    plot_row = borders_df[borders_df['plot_index'] == plot_index]
                    if not plot_row.empty:
                        if 'Plot' in borders_df.columns:
                            plot_val = plot_row['Plot'].iloc[0]
                            if pd.notna(plot_val):
                                plot_name = str(plot_val)
                        
                        if 'Accession' in borders_df.columns:
                            acc_val = plot_row['Accession'].iloc[0]
                            if pd.notna(acc_val):
                                accession = str(acc_val)
                        
                    else:
                        pass
                else:
                    pass
            except Exception as e:
                print(f"ERROR: Failed to read plot_borders.csv: {e}")
                pass

        return jsonify({
            'plot_index': plot_index,
            'lat': lat_val,
            'lon': lon_val,
            'plot_name': plot_name,
            'accession': accession
        }), 200
    else:
        print(f"ERROR: No data found for image '{image_name}'")
        return jsonify({'plot_index': -1, 'lat': None, 'lon': None, 'plot_name': None, 'accession': None}), 200

@plot_marking_bp.route('/get_gps_data', methods=['POST'])
def get_gps_data():
    data = request.get_json()
    directory = data.get('directory')
    data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
    dir_path = os.path.join(data_root_dir_path, directory)
    metadata_dir = os.path.abspath(os.path.join(dir_path, '..', '..', 'Metadata'))
    if not os.path.exists(metadata_dir):
        metadata_dir = os.path.abspath(os.path.join(dir_path, '..'))
    csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')

    if not directory:
        return jsonify({'error': 'Missing directory'}), 400

    if not os.path.exists(csv_path):
        return jsonify({'error': 'msgs_synced.csv not found'}), 404

    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except Exception as e:
        print(f"ERROR: Failed to read CSV file for GPS data: {e}")
        return jsonify({'error': f'Failed to read CSV: {str(e)}'}), 500
        
    df.columns = df.columns.str.strip() # FIX: Sanitize column names
    
    if 'lat' in df.columns and 'lon' in df.columns:
        gps_data = df[['lat', 'lon']].to_dict('records')
        return jsonify(gps_data)
    else:
        return jsonify({'error': "'lat' or 'lon' columns not found in csv"}), 400

@plot_marking_bp.route('/delete_plot', methods=['POST'])
def delete_plot():
    data = request.get_json()
    directory = data.get('directory')
    data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
    dir_path = os.path.join(data_root_dir_path, directory)
    metadata_dir = os.path.abspath(os.path.join(dir_path, '..', '..', 'Metadata'))
    if not os.path.exists(metadata_dir):
        metadata_dir = os.path.abspath(os.path.join(dir_path, '..'))
    csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')
    plot_index = data.get('plot_index')
    
    if not os.path.exists(csv_path):
        return jsonify({"error": "CSV file not found."}), 404

    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except Exception as e:
        print(f"ERROR: Failed to read CSV file for delete plot: {e}")
        return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 500
        
    df.columns = df.columns.str.strip() # FIX: Sanitize column names

    if 'plot_index' not in df.columns:
        return jsonify({"error": "'plot_index' column not found in csv"}), 500
    
    # Create a boolean mask to identify the rows to delete *before* any changes are made.
    rows_to_delete = df['plot_index'] == plot_index
    
    # Use the mask to update both columns. This prevents the logical error.
    df.loc[rows_to_delete, 'plot_index'] = -1
    df.loc[rows_to_delete, 'stitch_direction'] = None
    
    # Use atomic write to prevent CSV corruption
    try:
        import shutil
        backup_path = csv_path + '.backup'
        # Create backup before writing
        shutil.copy2(csv_path, backup_path)
        
        # Write to temporary file first
        temp_path = csv_path + '.tmp'
        df.to_csv(temp_path, index=False)
        
        # Verify the temporary file can be read
        verification_df = pd.read_csv(temp_path, on_bad_lines='skip')
        if len(verification_df) > 0:
            # Atomic move to replace original
            shutil.move(temp_path, csv_path)
            
            # Remove backup after successful write
            if os.path.exists(backup_path):
                os.remove(backup_path)
        else:
            print(f"ERROR: Verification failed during delete - temporary file is empty")
            # Restore from backup
            shutil.move(backup_path, csv_path)
            return jsonify({"error": "Failed to delete plot - CSV verification failed"}), 500
            
    except Exception as e:
        print(f"ERROR: Failed to save CSV safely during delete: {e}")
        # Restore from backup if it exists
        backup_path = csv_path + '.backup'
        if os.path.exists(backup_path):
            shutil.move(backup_path, csv_path)
        return jsonify({"error": f"Failed to delete plot: {str(e)}"}), 500

    return jsonify({"status": "success", "message": f"Plot {plot_index} deleted successfully."})

@plot_marking_bp.route('/get_plot_data', methods=['POST'])
def get_plot_data():
    data = request.get_json()
    directory = data.get('directory')
    
    if not directory:
        return jsonify([])
    
    data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
    dir_path = os.path.join(data_root_dir_path, directory)
    metadata_dir = os.path.abspath(os.path.join(dir_path, '..', '..', 'Metadata'))
    if not os.path.exists(metadata_dir):
        metadata_dir = os.path.abspath(os.path.join(dir_path, '..'))
    csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')
    
    if not os.path.exists(csv_path):
        return jsonify([])
    
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except Exception as e:
        print(f"ERROR: Failed to read msgs_synced.csv: {e}")
        return jsonify([])
        
    df.columns = df.columns.str.strip() # FIX: Sanitize column names
    
    # Check if plot_index column exists
    if 'plot_index' not in df.columns:
        return jsonify([])
    
    # Filter for plots that have been marked (plot_index > -1)
    marked_plots_df = df[df['plot_index'] > -1].copy()
    
    if marked_plots_df.empty:
        return jsonify([])
    
    # Find the correct image column
    image_col = '/top/rgb_file'
    if image_col not in df.columns:
        # Try to find an alternative image column
        image_cols = [col for col in df.columns if col.endswith('_file')]
        if image_cols:
            image_col = image_cols[0]  # Use the first available _file column
        else:
            return jsonify([])
    
    # Group by plot_index to get the first image for each plot
    start_plots_df = marked_plots_df.groupby('plot_index').first().reset_index()
    
    # Prepare the basic plot data from msgs_synced.csv
    plot_data = []
    for _, row in start_plots_df.iterrows():
        plot_info = {
            'plot_index': int(row['plot_index']),
            'image_name': row[image_col]
        }
        plot_data.append(plot_info)
    
    # Now try to enhance with Plot and Accession data from plot_borders.csv
    plot_borders_path = os.path.abspath(os.path.join(dir_path, '../../../../..', 'plot_borders.csv'))
    
    has_plot_metadata = False
    if os.path.exists(plot_borders_path):
        try:
            borders_df = pd.read_csv(plot_borders_path, on_bad_lines='skip')
            borders_df.columns = borders_df.columns.str.strip()
            
            # Check if Plot and Accession columns exist
            if 'Plot' in borders_df.columns and 'Accession' in borders_df.columns:
                has_plot_metadata = True
                
                # Merge Plot and Accession data into our plot_data
                for plot_info in plot_data:
                    plot_index = plot_info['plot_index']
                    border_row = borders_df[borders_df['plot_index'] == plot_index]
                    if not border_row.empty:
                        # Handle NaN values properly
                        plot_val = border_row.iloc[0]['Plot']
                        accession_val = border_row.iloc[0]['Accession']
                        
                        # Convert NaN to None (which becomes null in JSON)
                        plot_info['Plot'] = None if pd.isna(plot_val) else plot_val
                        plot_info['Accession'] = None if pd.isna(accession_val) else accession_val
                    else:
                        plot_info['Plot'] = None
                        plot_info['Accession'] = None
                        
        except Exception as e:
            print(f"ERROR: Failed to read plot_borders.csv: {e}")

    # Add metadata availability flag to each plot
    for plot_info in plot_data:
        plot_info['has_plot_metadata'] = has_plot_metadata

    return jsonify(plot_data)

@plot_marking_bp.route('/debug_plot_indices', methods=['POST'])
def debug_plot_indices():
    """Debug endpoint to check plot index assignments in a range"""
    data = request.get_json()
    directory = data.get('directory')
    start_index = data.get('start_index', 0)
    end_index = data.get('end_index', 50)
    
    if not directory:
        return jsonify({'error': 'Missing directory'}), 400
    
    data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
    dir_path = os.path.join(data_root_dir_path, directory)
    metadata_dir = os.path.abspath(os.path.join(dir_path, '..', '..', 'Metadata'))
    if not os.path.exists(metadata_dir):
        metadata_dir = os.path.abspath(os.path.join(dir_path, '..'))
    csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')
    
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        
        # Get a range of rows to debug
        debug_rows = df.iloc[start_index:end_index]
        
        result = {
            'total_rows': len(df),
            'debug_range': f"{start_index}-{end_index}",
            'plot_assignments': []
        }
        
        if '/top/rgb_file' in debug_rows.columns and 'plot_index' in debug_rows.columns:
            for idx, row in debug_rows.iterrows():
                image_file = row['/top/rgb_file']
                plot_idx = row['plot_index']
                result['plot_assignments'].append({
                    'row_index': idx,
                    'image_file': image_file,
                    'plot_index': plot_idx
                })
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': f'Debug failed: {str(e)}'}), 500

@plot_marking_bp.route('/test_image_lookup', methods=['POST'])
def test_image_lookup():
    """Test endpoint to verify specific image lookup"""
    data = request.get_json()
    directory = data.get('directory')
    image_name = data.get('image_name')
    
    if not directory or not image_name:
        return jsonify({'error': 'Missing directory or image_name'}), 400
    
    # Add the /top/ prefix if not present
    if not image_name.startswith('/top/'):
        image_name = "/top/" + image_name
    
    data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
    image_dir_path = os.path.join(data_root_dir_path, directory)
    metadata_dir = os.path.abspath(os.path.join(image_dir_path, '..', '..', 'Metadata'))
    csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')
    
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        
        result = {
            'csv_path': csv_path,
            'csv_shape': df.shape,
            'image_name': image_name,
            'found': False,
            'row_index': None,
            'plot_index': None,
            'context_rows': []
        }
        
        # Check all _file columns
        file_columns = [col for col in df.columns if col.endswith('_file')]
        result['file_columns'] = file_columns
        
        for col in file_columns:
            if image_name in df[col].values:
                result['found'] = True
                result['found_in_column'] = col
                
                # Get the row index
                row_indices = df.index[df[col] == image_name].tolist()
                if row_indices:
                    row_idx = row_indices[0]
                    result['row_index'] = row_idx
                    
                    # Get plot index for this row
                    if 'plot_index' in df.columns:
                        plot_idx = df.loc[row_idx, 'plot_index']
                        result['plot_index'] = plot_idx
                    
                    # Get context (surrounding rows)
                    start_ctx = max(0, row_idx - 2)
                    end_ctx = min(len(df), row_idx + 3)
                    context_df = df.iloc[start_ctx:end_ctx]
                    
                    for idx, ctx_row in context_df.iterrows():
                        result['context_rows'].append({
                            'row_index': idx,
                            'image_file': ctx_row.get(col, 'N/A'),
                            'plot_index': ctx_row.get('plot_index', 'N/A'),
                            'is_target': idx == row_idx
                        })
                break
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': f'Test lookup failed: {str(e)}'}), 500

@plot_marking_bp.route('/filter_plot_borders', methods=['POST'])
def filter_plot_borders():
    geod = Geod(ellps='WGS84')
    data = request.get_json()
    year = data.get('year')
    experiment = data.get('experiment')
    location = data.get('location')
    population = data.get('population')
    date = data.get('date')
    data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
    population_path = os.path.join(data_root_dir_path, 'Raw', year, experiment, location, population)
    metadata_dir = os.path.abspath(os.path.join(population_path, date, 'rover', 'RGB', 'Metadata'))
    if not os.path.exists(metadata_dir):
        metadata_dir = os.path.join(population_path, date, 'Amiga', 'RGB', 'Metadata')
    msgs_synced_path = os.path.join(metadata_dir, 'msgs_synced.csv')
    plot_borders_path = os.path.join(population_path, 'plot_borders.csv')
    if not os.path.exists(plot_borders_path):
        return jsonify({"error": "plot_borders.csv not found"}), 500
    else:
        try:
            pb_df = pd.read_csv(plot_borders_path, on_bad_lines='skip')
            print(f"DEBUG: Plot borders CSV loaded successfully with {len(pb_df)} rows")
        except Exception as e:
            print(f"ERROR: Failed to read plot_borders.csv: {e}")
            return jsonify({"error": f"Failed to read plot_borders.csv: {str(e)}"}), 500
            
        pb_df.columns = pb_df.columns.str.strip() # FIX: Sanitize column names
        
        try:
            msgs_df = pd.read_csv(msgs_synced_path, on_bad_lines='skip')
            print(f"DEBUG: Messages CSV loaded successfully with {len(msgs_df)} rows")
        except Exception as e:
            print(f"ERROR: Failed to read msgs_synced.csv: {e}")
            return jsonify({"error": f"Failed to read msgs_synced.csv: {str(e)}"}), 500
            
        msgs_df.columns = msgs_df.columns.str.strip() # FIX: Sanitize column names
        if 'plot_index' in msgs_df.columns and msgs_df['plot_index'].max() > -1:
            # no filtering
            print("DEBUG: No filtering applied, plot borders already exist.")
            return jsonify({"status": "success", "message": "No filtering applied, plot borders already exist."}), 200
        image_coords = msgs_df[['lat', 'lon']].values
        start_coords = pb_df[['start_lat', 'start_lon']].values
        end_coords = pb_df[['end_lat', 'end_lon']].values

        tree = KDTree(image_coords)
        
        start_distances_euclidean, start_indices = tree.query(start_coords)
        end_distances_euclidean, end_indices = tree.query(end_coords)
        
        pb_df['start_img_idx'] = start_indices
        pb_df['end_img_idx'] = end_indices

        for _, border_row in pb_df.iterrows():
            plot_idx = border_row['plot_index']
            
            # Get the actual closest image's data using the index from KDTree
            start_img_row = msgs_df.iloc[border_row['start_img_idx']]
            end_img_row = msgs_df.iloc[border_row['end_img_idx']]

            # Perform accurate geodetic distance check on the identified closest points
            _, _, actual_start_dist = geod.inv(border_row['start_lon'], border_row['start_lat'], start_img_row['lon'], start_img_row['lat'])
            _, _, actual_end_dist = geod.inv(border_row['end_lon'], border_row['end_lat'], end_img_row['lon'], end_img_row['lat'])
            
            if actual_start_dist > 0.25 or actual_end_dist > 0.25:
                print(f"WARNING: Closest match for plot {plot_idx} is too far (start: {actual_start_dist:.2f}m, end: {actual_end_dist:.2f}m). Skipping.")
                continue
            
            start_image = start_img_row['/top/rgb_file']
            end_image = end_img_row['/top/rgb_file']

            directory_path = os.path.join(metadata_dir, '..', 'Images', 'top')
            camera = 'top'
            start_image_name = start_image.replace('/top/', '')
            end_image_name = end_image.replace('/top/', '')
            
            success = update_plot_index(directory_path, start_image_name, end_image_name, plot_idx, camera, stitch_direction=border_row['stitch_direction'], filter=True)
            if not success:
                return jsonify({"error": f"Failed to update plot index for plot {plot_idx}"}), 500
        return jsonify({"status": "success", "message": f"{len(pb_df)} plots were filtered and updated successfully."}), 200


@plot_marking_bp.route('/download_amiga_images', methods=['POST'])
def download_amiga_images():
    """
    Downloads images extracted from an Amiga binary.

    If a 'plot_index' column with marked plots (> -1) exists in the metadata,
    only those marked images are zipped. Otherwise, all images from the directory are zipped.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        # --- 1. Get request data and construct paths using pathlib ---
        year = data.get('year')
        experiment = data.get('experiment')
        location = data.get('location')
        population = data.get('population')
        date = data.get('date')
        camera = data.get('camera')
        print(f"DEBUG: Received data: {data}")
        # Using pathlib for cleaner and more robust path manipulation
        base_path = Path(current_app.config['DATA_ROOT_DIR']) / 'Raw' / year / experiment / location / population / date / 'rover'
        if not base_path.is_dir():
            base_path = Path(current_app.config['DATA_ROOT_DIR']) / 'Raw' / year / experiment / location / population / date / 'Amiga'
        images_path = base_path / 'RGB' / 'Images' / camera
        csv_path = base_path / 'RGB' / 'Metadata' / 'msgs_synced.csv'

        if not images_path.is_dir():
            return jsonify({"error": f"Images directory not found at: {images_path}"}), 404
        if not csv_path.is_file():
            return jsonify({"error": f"Metadata CSV not found at: {csv_path}"}), 404

        # --- 2. Image Selection Logic ---
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip')
            print(f"DEBUG: Download images CSV loaded successfully with {len(df)} rows")
        except Exception as e:
            print(f"ERROR: Failed to read CSV file for download: {e}")
            return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 500
            
        df.columns = df.columns.str.strip() # FIX: Sanitize column names
        marked_images_filenames = set()
        download_all = True

        # Check if plots have been marked and the corresponding column exists
        plot_index_col = 'plot_index'
        image_file_col = f'/{camera}/rgb_file'

        if plot_index_col in df.columns and image_file_col in df.columns:
            # Filter for rows where a plot has been marked
            marked_plots_df = df[df[plot_index_col] > -1]
            if not marked_plots_df.empty:
                print("Marked plots found. Preparing to download specific images.")
                download_all = False
                marked_images_filenames = set(
                    Path(f).name for f in marked_plots_df[image_file_col].dropna()
                )

        # --- 2.5. Load plot borders data for custom naming ---
        plot_data = {}
        plot_borders_path = os.path.join(base_path, '..', '..', 'plot_borders.csv')
        if os.path.exists(plot_borders_path):
            try:
                borders_df = pd.read_csv(plot_borders_path)
                borders_df.columns = borders_df.columns.str.strip()
                for _, row in borders_df.iterrows():
                    plot_index = row.get('plot_index')
                    plot_label = row.get('Plot')
                    accession = row.get('Accession')
                    
                    if not pd.isna(plot_index):
                        plot_data[int(plot_index)] = {
                            'plot_label': plot_label if not pd.isna(plot_label) else None,
                            'accession': accession if not pd.isna(accession) else None
                        }
                print(f"DEBUG: Loaded plot borders data for {len(plot_data)} plots")
            except Exception as e:
                print(f"WARNING: Error reading plot borders for custom naming: {e}")

        if download_all:
            print("No marked plots found. Preparing to download all images.")

        # --- 3. Zipping and Sending the File ---
        if download_all:
            zip_filename = f"{year}_{experiment}_{location}_{population}_{date}_Amiga_RGB_All.zip"
        else:
            zip_filename = f"{year}_{experiment}_{location}_{population}_{date}_Amiga_RGB_Marked_Plots.zip"
        # Place the zip file in a temporary or cache directory if possible
        zip_path = images_path.parent / zip_filename

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through the images directory
            for file in os.listdir(images_path):
                file_path = images_path / file
                if file_path.is_file():
                    # If downloading all, or if the file is in our marked set, add it to the zip
                    if download_all or file in marked_images_filenames:
                        if not download_all and file in marked_images_filenames:
                            # Find the row in the DataFrame matching this file by comparing just the file name
                            df_row = marked_plots_df[marked_plots_df[image_file_col].apply(lambda x: Path(x).name) == file]
                            if not df_row.empty:
                                plot_index = df_row['plot_index'].iloc[0]
                                
                                # Get plot metadata for custom naming
                                metadata = plot_data.get(plot_index, {})
                                plot_label = metadata.get('plot_label')
                                accession = metadata.get('accession')
                                
                                # Create custom filename by prepending plot/accession info to original filename
                                if plot_label and accession:
                                    custom_filename = f"plot_{plot_label}_accession_{accession}_{file}"
                                elif plot_label:
                                    custom_filename = f"plot_{plot_label}_{file}"
                                else:
                                    # Fallback to plot index if no plot label available
                                    custom_filename = f"plot_{plot_index}_{file}"
                                
                                print(f"DEBUG: Adding to zip: {file} -> {custom_filename}")
                                zipf.write(file_path, arcname=custom_filename)
                            else:
                                # Fallback to original filename if row not found
                                zipf.write(file_path, arcname=file)
                        else:
                            # Use 'arcname' to keep the zip file flat (no parent directories)
                            zipf.write(file_path, arcname=file)

        # Send the created zip file to the user for download
        return send_file(zip_path, as_attachment=True)

    except FileNotFoundError:
        return jsonify({"error": "A specified file or directory was not found."}), 404
    except pd.errors.EmptyDataError:
        return jsonify({"error": "Metadata CSV file is empty."}), 400
    except KeyError as e:
        return jsonify({"error": f"Missing expected key in data: {e}"}), 400
    except Exception as e:
        # Generic error handler for unexpected issues
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500


@plot_marking_bp.route('/save_stitch_mask', methods=['POST'])
def save_stitch_mask():
    data = request.get_json()
    mask = data.get('mask')
    directory = data.get('directory')
    mask_file = os.path.join(current_app.config['DATA_ROOT_DIR'], directory, '../../../../..', 'stitch_mask.json')
    mask_data = {
        "mask": mask
    }
    # write mask_data to mask_file
    with open(mask_file, 'w') as f:
        json.dump(mask_data, f)
    return jsonify({"status": "success", "message": "Mask saved successfully."}), 200

@plot_marking_bp.route('/check_mask', methods=['POST'])
def check_mask():
    data = request.get_json()
    year = data.get('year')
    experiment = data.get('experiment')
    location = data.get('location')
    population = data.get('population')
    
    # Validate required parameters
    if not all([year, experiment, location, population]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        directory = os.path.join(current_app.config['DATA_ROOT_DIR'], 'Raw', year, experiment, location, population)
        mask_file = os.path.join(directory, 'stitch_mask.json')
        
        if not os.path.exists(mask_file):
            # Return success with null mask instead of error
            return jsonify({"mask": None}), 200
            
        with open(mask_file, 'r') as f:
            mask_data = json.load(f)
            mask = mask_data.get('mask')
            
        return jsonify({"mask": mask}), 200
        
    except Exception as e:
        print(f"Error checking mask: {e}")
        return jsonify({"error": f"Error reading mask file: {str(e)}"}), 500


@plot_marking_bp.route('/set_gps_reference', methods=['POST'])
def set_gps_reference():
    """Set GPS reference point for a dataset"""
    data = request.get_json()
    directory = data.get('directory')
    lat = data.get('lat')
    lon = data.get('lon')
    
    if not all([directory, lat is not None, lon is not None]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
        image_dir_path = os.path.join(data_root_dir_path, directory)
        
        # Save to the population directory level
        gps_ref_file = os.path.abspath(os.path.join(image_dir_path, '../../../../..', 'gps_reference.json'))
        gps_ref_data = {
            "reference_lat": lat,
            "reference_lon": lon
        }
        
        with open(gps_ref_file, 'w') as f:
            json.dump(gps_ref_data, f)
        
        return jsonify({"status": "success", "message": "GPS reference saved successfully"}), 200
        
    except Exception as e:
        print(f"ERROR: Failed to save GPS reference: {e}")
        return jsonify({"error": f"Failed to save GPS reference: {e}"}), 500


@plot_marking_bp.route('/get_gps_reference', methods=['POST'])
def get_gps_reference():
    """Get GPS reference point for a dataset"""
    data = request.get_json()
    directory = data.get('directory')
    
    if not directory:
        return jsonify({"error": "Missing directory parameter"}), 400
    
    try:
        data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
        image_dir_path = os.path.join(data_root_dir_path, directory)
        
        gps_ref_file = os.path.abspath(os.path.join(image_dir_path, '../../../../..', 'gps_reference.json'))
        
        if os.path.exists(gps_ref_file):
            with open(gps_ref_file, 'r') as f:
                gps_ref_data = json.load(f)
            return jsonify(gps_ref_data), 200
        else:
            return jsonify({"reference_lat": None, "reference_lon": None}), 200
            
    except Exception as e:
        print(f"ERROR: Failed to get GPS reference: {e}")
        return jsonify({"error": f"Failed to get GPS reference: {e}"}), 500


@plot_marking_bp.route('/shift_gps', methods=['POST'])
def shift_gps():
    """Shift all GPS coordinates in the dataset based on reference point"""
    data = request.get_json()
    directory = data.get('directory')
    current_lat = data.get('current_lat')
    current_lon = data.get('current_lon')
    
    if not all([directory, current_lat is not None, current_lon is not None]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
        image_dir_path = os.path.join(data_root_dir_path, directory)
        metadata_dir = os.path.abspath(os.path.join(image_dir_path, '..', '..', 'Metadata'))
        csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')
        
        # Get GPS reference
        gps_ref_file = os.path.abspath(os.path.join(image_dir_path, '../../../../..', 'gps_reference.json'))
        if not os.path.exists(gps_ref_file):
            return jsonify({"error": "No GPS reference set for this dataset"}), 400
            
        with open(gps_ref_file, 'r') as f:
            gps_ref_data = json.load(f)
        
        ref_lat = gps_ref_data.get('reference_lat')
        ref_lon = gps_ref_data.get('reference_lon')
        
        if ref_lat is None or ref_lon is None:
            return jsonify({"error": "Invalid GPS reference data"}), 400
        
        # Calculate shift amounts
        lat_shift = ref_lat - current_lat
        lon_shift = ref_lon - current_lon
        
        print(f"DEBUG: GPS shift calculation - Reference: ({ref_lat}, {ref_lon}), Current: ({current_lat}, {current_lon}), Shift: ({lat_shift}, {lon_shift})")
        
        # Load and process CSV
        if not os.path.exists(csv_path):
            return jsonify({"error": "CSV file not found"}), 404
            
        # Check CSV integrity before processing
        is_valid, df = check_csv_integrity(csv_path)
        if not is_valid:
            return jsonify({"error": "CSV integrity check failed"}), 500
            
        df.columns = df.columns.str.strip()
        
        if 'lat' not in df.columns or 'lon' not in df.columns:
            return jsonify({"error": "Lat/Lon columns not found in CSV"}), 400
        
        # Save original coordinates before shifting
        gps_shift_backup_file = os.path.abspath(os.path.join(image_dir_path, '../../../../..', 'gps_shift_backup.json'))
        
        # Only create backup if this is the first shift
        if not os.path.exists(gps_shift_backup_file):
            backup_data = {
                "original_coordinates": df[['lat', 'lon']].to_dict('records'),
                "shift_applied": {
                    "lat_shift": lat_shift,
                    "lon_shift": lon_shift,
                    "reference_lat": ref_lat,
                    "reference_lon": ref_lon
                }
            }
            with open(gps_shift_backup_file, 'w') as f:
                json.dump(backup_data, f)
        
        # Apply shift to all coordinates
        df['lat'] = df['lat'] + lat_shift
        df['lon'] = df['lon'] + lon_shift
        
        # Use atomic write to prevent CSV corruption
        try:
            backup_path = csv_path + '.backup'
            shutil.copy2(csv_path, backup_path)
            
            temp_path = csv_path + '.tmp'
            df.to_csv(temp_path, index=False)
            
            verification_df = pd.read_csv(temp_path, on_bad_lines='skip')
            if len(verification_df) > 0:
                shutil.move(temp_path, csv_path)
                print(f"DEBUG: GPS shift applied successfully to {csv_path}")
                
                if os.path.exists(backup_path):
                    os.remove(backup_path)
            else:
                print(f"ERROR: Verification failed - temporary file is empty")
                shutil.move(backup_path, csv_path)
                return jsonify({"error": "Failed to apply GPS shift"}), 500
                
        except Exception as e:
            print(f"ERROR: Failed to save CSV safely during GPS shift: {e}")
            backup_path = csv_path + '.backup'
            if os.path.exists(backup_path):
                shutil.move(backup_path, csv_path)
            return jsonify({"error": f"Failed to apply GPS shift: {e}"}), 500
        
        return jsonify({
            "status": "success", 
            "message": f"GPS coordinates shifted by ({lat_shift:.6f}, {lon_shift:.6f})",
            "shift_applied": {
                "lat_shift": lat_shift,
                "lon_shift": lon_shift
            }
        }), 200
        
    except Exception as e:
        print(f"ERROR: Failed to shift GPS coordinates: {e}")
        return jsonify({"error": f"Failed to shift GPS coordinates: {e}"}), 500


@plot_marking_bp.route('/check_gps_shift_status', methods=['POST'])
def check_gps_shift_status():
    """Check if GPS shift has been applied to a dataset"""
    data = request.get_json()
    directory = data.get('directory')
    
    if not directory:
        return jsonify({"error": "Missing directory parameter"}), 400
    
    try:
        data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
        image_dir_path = os.path.join(data_root_dir_path, directory)
        
        # Check for GPS shift backup file
        gps_shift_backup_file = os.path.abspath(os.path.join(image_dir_path, '../../../../..', 'gps_shift_backup.json'))
        
        if os.path.exists(gps_shift_backup_file):
            with open(gps_shift_backup_file, 'r') as f:
                backup_data = json.load(f)
            
            shift_info = backup_data.get('shift_applied', {})
            return jsonify({
                "has_shift": True,
                "shift_applied": shift_info
            }), 200
        else:
            return jsonify({
                "has_shift": False,
                "shift_applied": None
            }), 200
            
    except Exception as e:
        print(f"ERROR: Failed to check GPS shift status: {e}")
        return jsonify({"error": f"Failed to check GPS shift status: {e}"}), 500


@plot_marking_bp.route('/undo_gps_shift', methods=['POST'])
def undo_gps_shift():
    """Undo GPS coordinate shift by restoring original coordinates"""
    data = request.get_json()
    directory = data.get('directory')
    
    if not directory:
        return jsonify({"error": "Missing directory parameter"}), 400
    
    try:
        data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
        image_dir_path = os.path.join(data_root_dir_path, directory)
        metadata_dir = os.path.abspath(os.path.join(image_dir_path, '..', '..', 'Metadata'))
        csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')
        
        # Check for backup file
        gps_shift_backup_file = os.path.abspath(os.path.join(image_dir_path, '../../../../..', 'gps_shift_backup.json'))
        if not os.path.exists(gps_shift_backup_file):
            return jsonify({"error": "No GPS shift backup found"}), 400
            
        with open(gps_shift_backup_file, 'r') as f:
            backup_data = json.load(f)
        
        original_coords = backup_data.get('original_coordinates')
        if not original_coords:
            return jsonify({"error": "Invalid backup data"}), 400
        
        # Load current CSV
        if not os.path.exists(csv_path):
            return jsonify({"error": "CSV file not found"}), 404
            
        is_valid, df = check_csv_integrity(csv_path)
        if not is_valid:
            return jsonify({"error": "CSV integrity check failed"}), 500
            
        df.columns = df.columns.str.strip()
        
        # Restore original coordinates
        original_df = pd.DataFrame(original_coords)
        if len(original_df) != len(df):
            return jsonify({"error": "Backup data size mismatch with current CSV"}), 400
            
        df['lat'] = original_df['lat']
        df['lon'] = original_df['lon']
        
        # Use atomic write to prevent CSV corruption
        try:
            backup_path = csv_path + '.backup'
            shutil.copy2(csv_path, backup_path)
            
            temp_path = csv_path + '.tmp'
            df.to_csv(temp_path, index=False)
            
            verification_df = pd.read_csv(temp_path, on_bad_lines='skip')
            if len(verification_df) > 0:
                shutil.move(temp_path, csv_path)
                print(f"DEBUG: GPS coordinates restored successfully to {csv_path}")
                
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                    
                # Remove backup file after successful restore
                os.remove(gps_shift_backup_file)
                
            else:
                print(f"ERROR: Verification failed - temporary file is empty")
                shutil.move(backup_path, csv_path)
                return jsonify({"error": "Failed to restore GPS coordinates"}), 500
                
        except Exception as e:
            print(f"ERROR: Failed to save CSV safely during GPS restore: {e}")
            backup_path = csv_path + '.backup'
            if os.path.exists(backup_path):
                shutil.move(backup_path, csv_path)
            return jsonify({"error": f"Failed to restore GPS coordinates: {e}"}), 500
        
        return jsonify({
            "status": "success", 
            "message": "GPS coordinates restored to original values"
        }), 200
        
    except Exception as e:
        print(f"ERROR: Failed to undo GPS shift: {e}")
        return jsonify({"error": f"Failed to undo GPS shift: {e}"}), 500
