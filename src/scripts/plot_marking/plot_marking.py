# Standard library imports
import os
import pandas as pd
import zipfile
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app, send_file

plot_marking_bp = Blueprint('plot_marking', __name__)

def update_plot_index(directory, start_image_name, end_image_name, plot_index, camera, stitch_direction=None, original_plot_index=None):
    data_root_dir_path = os.path.abspath(current_app.config['DATA_ROOT_DIR'])
    image_dir_path = os.path.join(data_root_dir_path, directory)
    metadata_dir = os.path.abspath(os.path.join(image_dir_path, '..', '..', 'Metadata'))
    csv_path = os.path.join(metadata_dir, 'msgs_synced.csv')
    start_image_name = "/top/" + start_image_name  
    end_image_name = "/top/" + end_image_name
    print(f"DEBUG: CSV path is {csv_path}")
    if not os.path.exists(csv_path):
        return jsonify({"error": f"Metadata CSV not found at {csv_path}"}), 404
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
        return jsonify({"error": f"Image column '{image_column}' not found in CSV"}), 400
    try:
        start_row_index = df.index[df[image_column] == start_image_name].tolist()[0]
        end_row_index = df.index[df[image_column] == end_image_name].tolist()[0]
    except IndexError:
        return jsonify({"error": "Start or end image not found in metadata"}), 404
    current_plot_index = int(plot_index)
    df.loc[start_row_index:end_row_index, 'plot_index'] = current_plot_index
    df.loc[start_row_index:end_row_index, 'stitch_direction'] = stitch_direction
    df.to_csv(csv_path, index=False)

    # Create or update plot_borders.csv
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
            borders_df = pd.DataFrame(columns=["plot_index", "start_lat", "start_lon", "end_lat", "end_lon"])

        new_border_data = {
            "plot_index": current_plot_index,
            "start_lat": start_lat,
            "start_lon": start_lon,
            "end_lat": end_lat,
            "end_lon": end_lon
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
    
    return jsonify({"status": "success", "message": f"Plot {plot_index} marked successfully."})

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

    if not all([directory, start_image_name, end_image_name, plot_index is not None, camera]):
        return jsonify({"error": "Missing required parameters"}), 400

    return update_plot_index(directory, start_image_name, end_image_name, plot_index, camera, stitch_direction, original_plot_index)


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
        
        # Find the column that contains the image name
        image_col = None
        for col in df.columns:
            if col.endswith('_file'):
                if image_name in df[col].values:
                    image_col = col
                    break
        
        if not image_col:
            return jsonify({'plot_index': -1}), 200 # Image not in msgs_synced, so no index

        row = df[df[image_col] == image_name]

        if not row.empty:
            plot_index = row['plot_index'].iloc[0]
            return jsonify({'plot_index': int(plot_index)}), 200
        else:
            return jsonify({'plot_index': -1}), 200 # Image not in msgs_synced, so no index
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
        base_path = Path(data_root_dir) / 'Raw' / year / experiment / location / population / date / 'rover'
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