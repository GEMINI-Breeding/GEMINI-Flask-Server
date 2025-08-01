# Standard library imports
import os
import pandas as pd
import zipfile
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app, send_file
from pyproj import Geod
import json

plot_marking_bp = Blueprint('plot_marking', __name__)

def update_plot_index(directory, start_image_name, end_image_name, plot_index, camera, stitch_direction=None, original_plot_index=None, filter=False, shift_all=False, shift_amount=0, original_start_image_index=None):
    print("DEBUG: Starting update_plot_index with parameters:")
    print(f"DEBUG: directory={directory}, start_image_name={start_image_name}, end_image_name={end_image_name}, plot_index={plot_index}, camera={camera}, stitch_direction={stitch_direction}, original_plot_index={original_plot_index}, filter={filter}")
    data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
    image_dir_path = os.path.join(data_root_dir_path, directory)
    metadata_dir = os.path.abspath(os.path.join(image_dir_path, '..', '..', 'Metadata'))
    csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')
    start_image_name = "/top/" + start_image_name
    end_image_name = "/top/" + end_image_name
    print(f"DEBUG: CSV path is {csv_path}")
    if not os.path.exists(csv_path):
        print(f"ERROR: Metadata CSV not found at {csv_path}")
        return False
    df = pd.read_csv(csv_path)
    if 'plot_index' not in df.columns:
        df['plot_index'] = -1

    # If this is an update, clear the old plot index entries first
    if original_plot_index is not None:
        print(f"DEBUG: Clearing old plot index: {original_plot_index}")
        df.loc[df['plot_index'] == original_plot_index, 'plot_index'] = -1
        df.loc[df['plot_index'] == original_plot_index, 'stitch_direction'] = None

    image_column = f'/{camera}/rgb_file'
    if image_column not in df.columns:
        print(f"ERROR: Image column '{image_column}' not found in CSV")
        return False
    try:
        start_row_index = df.index[df[image_column] == start_image_name].tolist()[0]
        end_row_index = df.index[df[image_column] == end_image_name].tolist()[0]
        print(f"DEBUG: Start row index: {start_row_index}, End row index: {end_row_index}")
    except IndexError:
        print("ERROR: Start or end image not found in metadata")
        return False
    current_plot_index = int(plot_index)
    df.loc[start_row_index:end_row_index, 'plot_index'] = current_plot_index
    df.loc[start_row_index:end_row_index, 'stitch_direction'] = stitch_direction
    print(f"DEBUG: shift_amount: {shift_amount}, original_start_image_index: {original_start_image_index}")
    
    if shift_all and shift_amount != 0 and original_plot_index is not None:
        print(f"DEBUG: Applying shift_all logic. Shift amount: {shift_amount}")
        
        unique_plots_to_shift = df[df['plot_index'] > current_plot_index]['plot_index'].unique()
        unique_plots_to_shift = unique_plots_to_shift[unique_plots_to_shift != -1]  # Exclude unassigned plots
        
        if len(unique_plots_to_shift) > 0:
            print(f"DEBUG: Found {len(unique_plots_to_shift)} plots to shift: {unique_plots_to_shift}")
            plot_assignments = {}
            for plot_idx in unique_plots_to_shift:
                plot_rows = df[df['plot_index'] == plot_idx].index.tolist()
                plot_assignments[plot_idx] = {
                    'rows': plot_rows,
                    'stitch_direction': df.loc[plot_rows[0], 'stitch_direction'] if plot_rows else None
                }
                df.loc[df['plot_index'] == plot_idx, 'plot_index'] = -1
                df.loc[df['plot_index'] == plot_idx, 'stitch_direction'] = None
        
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
                    print(f"DEBUG: Shifted plot {plot_idx} from rows {old_rows} to rows {new_rows}")
                else:
                    print(f"WARNING: Plot {plot_idx} could not be shifted - new position out of bounds")
    
    df.to_csv(csv_path, index=False)

    # Create or update plot_borders.csv
    if not filter and plot_index not in df['plot_index'].values:
        try:
            start_lat = df.loc[start_row_index, 'lat']
            start_lon = df.loc[start_row_index, 'lon']
            end_lat = df.loc[end_row_index, 'lat']
            end_lon = df.loc[end_row_index, 'lon']

            plot_borders_path = os.path.abspath(os.path.join(image_dir_path, '../../../../..', 'plot_borders.csv'))
            # puts plot_borders.csv in $data_root_dir$/Raw/$year$/$experiment$/$location$/$population$/ {image_dir_path is Raw/$year$/$experiment$/$location$/$population$/$platform$/$sensor$/Images/top}
            if os.path.exists(plot_borders_path):
                borders_df = pd.read_csv(plot_borders_path)
            else:
                borders_df = pd.DataFrame(columns=["plot_index", "start_lat", "start_lon", "end_lat", "end_lon", "stitch_direction"])

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
            print(f"DEBUG: Successfully updated {plot_borders_path}")

        except Exception as e:
            print(f"ERROR: Could not update plot_borders.csv: {e}")
    return True

@plot_marking_bp.route('/mark_plot', methods=['POST'])
def mark_plot():
    data = request.get_json()
    directory = data.get('directory')
    start_image_name = data.get('start_image_name')
    end_image_name = data.get('end_image_name')
    plot_index = data.get('plot_index')
    camera = data.get('camera')
    stitch_direction = data.get('stitch_direction')
    original_plot_index = data.get('original_plot_index') # Can be None
    shift_all = data.get('shift_all', False)
    shift_amount = data.get('shift_amount', 0)
    original_start_image_index = data.get('original_start_image_index')

    if not all([directory, start_image_name, end_image_name, plot_index is not None, camera]):
        return jsonify({"error": "Missing required parameters"}), 400

    success = update_plot_index(directory, start_image_name, end_image_name, plot_index, camera, stitch_direction, original_plot_index, filter=False, shift_all=shift_all, shift_amount=shift_amount, original_start_image_index=original_start_image_index)
    if success:
        return jsonify({"status": "success", "message": f"Plot {plot_index} marked successfully."})
    else:
        return jsonify({"error": "Failed to mark plot."}), 500


@plot_marking_bp.route('/get_max_plot_index', methods=['POST'])
def get_max_plot_index():
    data = request.get_json()
    directory = data.get('directory')
    data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
    dir_path = os.path.join(data_root_dir_path, directory)
    metadata_dir = os.path.abspath(os.path.join(dir_path, '..', '..', 'Metadata'))
    csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')
    print(f"DEBUG: CSV path is {csv_path}")
    if not directory:
        return jsonify({'error': 'Missing directory'}), 400
    if not os.path.exists(csv_path):
        # If the CSV doesn't exist, no plots have been marked.
        print(f"DEBUG: msgs_synced.csv not found at {csv_path}")
        return jsonify({'max_plot_index': -1}), 200

    try:
        df = pd.read_csv(csv_path)
        if 'plot_index' in df.columns and not df['plot_index'].empty:
            max_index = df['plot_index'].max()
            print(f"DEBUG: Max plot index found: {max_index}")
            return jsonify({'max_plot_index': int(max_index)}), 200
        else:
            # If column doesn't exist or is empty, no plots marked.
            print("DEBUG: No plot_index column found or it is empty.")
            return jsonify({'max_plot_index': -1}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
    if not directory or not image_name:
        return jsonify({'error': 'Missing directory or image name'}), 400

    try:
        df = pd.read_csv(csv_path)
        
        image_col = None
        for col in df.columns:
            if col.endswith('_file'):
                # Use .str.contains as a fallback if exact match fails, though exact is better
                if image_name in df[col].values:
                    image_col = col
                    break
        
        if not image_col:
            return jsonify({'plot_index': -1, 'lat': None, 'lon': None}), 200

        row = df[df[image_col] == image_name]

        if not row.empty:
            if 'plot_index' not in row.columns:
                if 'lat' not in row.columns or 'lon' not in row.columns:
                    # If no plot_index, lat, or lon columns, return -1 for plot_index and None for lat/lon
                    return jsonify({'plot_index': -1, 'lat': None, 'lon': None}), 200
                lat = row['lat'].iloc[0] 
                lon = row['lon'].iloc[0] 
                lat_val = float(lat)
                lon_val = float(lon)
                return jsonify({'plot_index': -1, 'lat': lat_val, 'lon': lon_val}), 200
            plot_index = row['plot_index'].iloc[0]
            lat = row['lat'].iloc[0] 
            lon = row['lon'].iloc[0]
            lat_val = float(lat) 
            lon_val = float(lon) 

            return jsonify({
                'plot_index': int(plot_index),
                'lat': lat_val,
                'lon': lon_val
            }), 200
        else:
            return jsonify({'plot_index': -1, 'lat': None, 'lon': None}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@plot_marking_bp.route('/get_gps_data', methods=['POST'])
def get_gps_data():
    data = request.get_json()
    directory = data.get('directory')
    data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
    dir_path = os.path.join(data_root_dir_path, directory)
    metadata_dir = os.path.abspath(os.path.join(dir_path, '..', '..', 'Metadata'))
    csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')

    if not directory:
        return jsonify({'error': 'Missing directory'}), 400

    if not os.path.exists(csv_path):
        return jsonify({'error': 'msgs_synced.csv not found'}), 404

    try:
        df = pd.read_csv(csv_path)
        
        if 'lat' in df.columns and 'lon' in df.columns:
            gps_data = df[['lat', 'lon']].to_dict('records')
            return jsonify(gps_data)
        else:
            return jsonify({'error': "'lat' or 'lon' columns not found in csv"}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@plot_marking_bp.route('/delete_plot', methods=['POST'])
def delete_plot():
    data = request.get_json()
    directory = data.get('directory')
    data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
    dir_path = os.path.join(data_root_dir_path, directory)
    metadata_dir = os.path.abspath(os.path.join(dir_path, '..', '..', 'Metadata'))
    csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')
    plot_index = data.get('plot_index')
    df = pd.read_csv(csv_path)
    if 'plot_index' not in df.columns:
        return jsonify({"error": "'plot_index' column not found in csv"}), 500
    df.loc[df['plot_index'] == plot_index, 'plot_index'] = -1
    df.loc[df['plot_index'] == plot_index, 'stitch_direction'] = None
    df.to_csv(csv_path, index=False)

    return jsonify({"status": "success", "message": f"Plot {plot_index} deleted successfully."})


@plot_marking_bp.route('/get_plot_data', methods=['POST'])
def get_plot_data():
    data = request.get_json()
    directory = data.get('directory')
    data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
    dir_path = os.path.join(data_root_dir_path, directory)
    metadata_dir = os.path.abspath(os.path.join(dir_path, '..', '..', 'Metadata'))
    csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')
    try:
        df = pd.read_csv(csv_path)
        if 'plot_index' not in df.columns:
            return jsonify([])
        # Filter for plots that have been marked
        marked_plots_df = df[df['plot_index'] > -1].copy()
        if marked_plots_df.empty:
            return jsonify([])
        image_col = '/top/rgb_file'
        start_plots_df = marked_plots_df.groupby('plot_index').first().reset_index()
        # Rename column to match what frontend expects
        start_plots_df = start_plots_df.rename(columns={image_col: 'image_name'})

        start_plots = start_plots_df[['plot_index', 'image_name']].to_dict('records')

        return jsonify(start_plots)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
    msgs_synced_path = os.path.join(metadata_dir, 'msgs_synced.csv')
    plot_borders_path = os.path.join(population_path, 'plot_borders.csv')
    if not os.path.exists(plot_borders_path):
        return jsonify({"error": "plot_borders.csv not found"}), 500
    else:
        pb_df = pd.read_csv(plot_borders_path)
        msgs_df = pd.read_csv(msgs_synced_path)
        if 'plot_index' in msgs_df.columns and msgs_df['plot_index'].max() > -1:
            # no filtering
            print("DEBUG: No filtering applied, plot borders already exist.")
            return jsonify({"status": "success", "message": "No filtering applied, plot borders already exist."}), 200
        for _, border_row in pb_df.iterrows():
            plot_idx = border_row['plot_index']
            target_start_lat = border_row['start_lat']
            target_start_lon = border_row['start_lon']
            target_end_lat = border_row['end_lat']
            target_end_lon = border_row['end_lon']

            min_start_distance = float('inf')
            min_end_distance = float('inf')
            start_image = None
            end_image = None

            for _, msg_row in msgs_df.iterrows():
                curr_lat = msg_row['lat']
                curr_lon = msg_row['lon']

                # Calculate distance from target start point
                _, _, dist_start = geod.inv(target_start_lon, target_start_lat, curr_lon, curr_lat)
                if dist_start < min_start_distance:
                    min_start_distance = dist_start
                    start_image = msg_row['/top/rgb_file']  # column expected in msgs_synced.csv

                # Calculate distance from target end point
                _, _, dist_end = geod.inv(target_end_lon, target_end_lat, curr_lon, curr_lat)
                if dist_end < min_end_distance:
                    min_end_distance = dist_end
                    end_image = msg_row['/top/rgb_file']

            # Assume camera is 'top' and construct a relative directory as used in update_plot_index
            directory = os.path.join('Raw', year, experiment, location, population, date, 'rover', 'RGB', 'Images', 'top')
            camera = 'top'
            # Call update_plot_index to update the plot index entries in msgs_synced.csv
            start_image = start_image.replace('/top/', '')  # Remove '/top/' prefix for consistency
            end_image = end_image.replace('/top/', '')  # Remove '/top/' prefix for consistency
            success = update_plot_index(directory, start_image, end_image, plot_idx, camera, stitch_direction=border_row['stitch_direction'], filter=True)

    return jsonify({"status": "success", "message": "Plot borders filtered and updated successfully."}), 200


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
        images_path = base_path / 'RGB' / 'Images' / camera
        csv_path = base_path / 'RGB' / 'Metadata' / 'msgs_synced.csv'

        if not images_path.is_dir():
            return jsonify({"error": f"Images directory not found at: {images_path}"}), 404
        if not csv_path.is_file():
            return jsonify({"error": f"Metadata CSV not found at: {csv_path}"}), 404

        # --- 2. Image Selection Logic ---
        df = pd.read_csv(csv_path)
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

        if download_all:
            print("No marked plots found. Preparing to download all images.")

        # --- 3. Zipping and Sending the File ---
        zip_filename = f"{year}_{experiment}_{location}_{population}_{date}_Amiga_RGB.zip"
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
                            df_row = marked_plots_df[marked_plots_df[image_file_col].apply(lambda x: Path(x).name) == {file}]
                            if not df_row.empty:
                                plot_index = df_row['plot_index'].iloc[0]
                                new_filename = f"plot{plot_index}-{file}"
                            else:
                                new_filename = file
                            zipf.write(file_path, arcname=new_filename)
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
    print(f"DEBUG: Saving stitch mask to {mask_file}")
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

