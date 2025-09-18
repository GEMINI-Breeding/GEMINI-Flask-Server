# Standard library imports
import os
import re
import shutil
import tempfile
import json
import pandas as pd
import numpy as np
import sys
from flask import Blueprint, jsonify, request, send_file, current_app

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# Local application/library specific imports
from scripts.orthomosaic_generation import convert_tif_to_png

postprocessing_bp = Blueprint('postprocessing', __name__)

@postprocessing_bp.route('/get_ortho_progress', methods=['GET'])
def get_ortho_progress():
    latest_data = current_app.config['LATEST_DATA']
    return jsonify(latest_data)

@postprocessing_bp.route('/get_ortho_metadata', methods=['GET'])
def get_ortho_metadata():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    date = request.args.get('date')
    platform = request.args.get('platform')
    sensor = request.args.get('sensor')
    year = request.args.get('year')
    experiment = request.args.get('experiment')
    location = request.args.get('location')
    population = request.args.get('population')
    
    metadata_path = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform, sensor, 'ortho_metadata.json')
    
    if not os.path.exists(metadata_path):
        return jsonify({"error": "Metadata file not found"}), 404
    
    try:
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        return jsonify({
            "quality": metadata.get("quality", "N/A"),
            "timestamp": metadata.get("timestamp", "N/A")
        })
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in metadata file"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    
@postprocessing_bp.route('/download_ortho', methods=['POST'])
def download_ortho():
    data = request.get_json()
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    try:
        file_path = os.path.join(
            data_root_dir,
            'Processed',
            data['year'],
            data['experiment'],
            data['location'],
            data['population'],
            data['date'],
            data['platform'],
            data['sensor'],
            f"{data['date']}-RGB.png"
        )
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            try:
                print("Attempting to convert TIF to PNG...")
                
                # convert png to tif
                tif_path = file_path.replace('.png', '.tif')
                
                # convert tif to png

                if os.path.exists(tif_path):
                    convert_tif_to_png(tif_path)
                    
                else:
                    print(f"File not found: {tif_path}")
                    return jsonify({'error': f'File not found: {file_path}'}), 404
            except Exception as e:
                print(f"An error occurred while converting the file: {str(e)}")
                return jsonify({'error': str(e)}), 500

        # Returns the file as an attachment so that the browser downloads it.
        print(f"Sending file: {file_path}")
        return send_file(
            file_path,
            mimetype="image/png",
            as_attachment=True,
            download_name=os.path.basename(file_path)  # For Flask 2.0+
        )
        
    except Exception as e:
        print(f"An error occurred while downloading the ortho: {str(e)}")
        return jsonify({'error': str(e)}), 500

@postprocessing_bp.route('/download_plot_ortho', methods=['POST'])
def download_plot_ortho():
    data = request.get_json()
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    try:
        # Build the path to the plot images directory
        plot_dir = os.path.join(
            data_root_dir,
            'Processed',
            data['year'],
            data['experiment'],
            data['location'],
            data['population'],
            data['date'],
            data['platform'],
            data['sensor'],
            data['agrowstitchDir']
        )
        
        if not os.path.exists(plot_dir):
            return jsonify({'error': 'Plot directory not found'}), 404
        
        # Find all plot PNG files
        plot_files = [f for f in os.listdir(plot_dir) 
                     if f.startswith('full_res_mosaic_temp_plot_') and f.endswith('.png')]
        
        if not plot_files:
            return jsonify({'error': 'No plot files found'}), 404
        
        # Get plot borders data for custom naming
        plot_borders_path = os.path.join(
            data_root_dir, "Raw", data['year'], data['experiment'], 
            data['location'], data['population'], "plot_borders.csv"
        )
        
        plot_data = {}
        if os.path.exists(plot_borders_path):
            try:
                borders_df = pd.read_csv(plot_borders_path)
                for _, row in borders_df.iterrows():
                    plot_index = row.get('plot_index')
                    plot_label = row.get('Plot')
                    accession = row.get('Accession')
                    
                    if not pd.isna(plot_index):
                        plot_data[int(plot_index)] = {
                            'plot_label': plot_label if not pd.isna(plot_label) else None,
                            'accession': accession if not pd.isna(accession) else None
                        }
            except Exception as e:
                print(f"Error reading plot borders for zip download: {e}")
        
        # Create a temporary zip file
        import tempfile
        import zipfile
        
        temp_dir = tempfile.mkdtemp()
        zip_filename = f"{data['date']}-{data['platform']}-{data['sensor']}-plots-with-metadata.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for plot_file in sorted(plot_files):
                file_path = os.path.join(plot_dir, plot_file)
                
                # Extract plot index and create custom filename
                plot_match = re.search(r'temp_plot_(\d+)', plot_file)
                if plot_match:
                    plot_index = int(plot_match.group(1))
                    extension = os.path.splitext(plot_file)[1]
                    
                    # Get plot metadata
                    metadata = plot_data.get(plot_index, {})
                    plot_label = metadata.get('plot_label')
                    accession = metadata.get('accession')
                    
                    # Create custom filename using the requested format
                    if plot_label and accession:
                        custom_filename = f"plot_{plot_label}_accession_{accession}{extension}"
                    elif plot_label:
                        custom_filename = f"plot_{plot_label}{extension}"
                    else:
                        custom_filename = f"plot_{plot_index}{extension}"
                    
                    # Add file to zip with custom name
                    zipf.write(file_path, custom_filename)
                    print(f"Adding to zip: {plot_file} -> {custom_filename}")
                else:
                    # Fallback to original filename if pattern doesn't match
                    zipf.write(file_path, plot_file)
        
        print(f"Sending zip file: {zip_path}")
        return send_file(
            zip_path,
            mimetype="application/zip",
            as_attachment=True,
            download_name=zip_filename
        )
    
    except Exception as e:
        print(f"An error occurred while downloading plot ortho: {str(e)}")
        return jsonify({'error': str(e)}), 500
        

@postprocessing_bp.route('/delete_ortho', methods=['POST'])
def delete_ortho():
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    data = request.json
    year = data.get('year')
    experiment = data.get('experiment')
    location = data.get('location')
    population = data.get('population')
    date = data.get('date')
    platform = data.get('platform')
    sensor = data.get('sensor')
    delete_type = data.get('deleteType', 'ortho')  # 'ortho' or 'agrowstitch'
    
    base_path = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform, sensor)
    
    try:
        if delete_type == 'agrowstitch':
            # Delete specific AgRowStitch version directory
            agrowstitch_dir = data.get('agrowstitchDir')
            if agrowstitch_dir:
                agrowstitch_path = os.path.join(base_path, agrowstitch_dir)
                if os.path.exists(agrowstitch_path):
                    shutil.rmtree(agrowstitch_path)
                    print(f"Deleted AgRowStitch directory: {agrowstitch_path}")
                else:
                    print(f"AgRowStitch directory not found: {agrowstitch_path}")
            else:
                return jsonify({"error": "AgRowStitch directory name not provided"}), 400
        else:
            # Delete specific orthomosaic file
            file_name = data.get('fileName')
            if file_name:
                file_path = os.path.join(base_path, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                else:
                    print(f"File not found: {file_path}")
            else:
                # For backward compatibility, try to find and delete standard orthomosaic files
                # Look for common orthomosaic file patterns
                ortho_patterns = [
                    f"{date}-RGB-Pyramid.tif",
                    f"{date}-RGB.tif", 
                    f"{date}-RGB.png",
                    f"{date}-DEM-Pyramid.tif",
                    f"{date}-DEM.tif"
                ]
                
                deleted_files = []
                for pattern in ortho_patterns:
                    file_path = os.path.join(base_path, pattern)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        deleted_files.append(pattern)
                        print(f"Deleted file: {file_path}")
                
                if not deleted_files:
                    return jsonify({"error": "No orthomosaic files found to delete"}), 404
                
    except FileNotFoundError:
        print("Path not found during deletion")
        return jsonify({"error": "File or directory not found"}), 404
    except PermissionError:
        print("Permission denied during deletion")
        return jsonify({"error": "Permission denied"}), 403
    except Exception as e:
        print(f"An error occurred during deletion: {str(e)}")
        return jsonify({"error": f"Deletion failed: {str(e)}"}), 500
        
    return jsonify({"message": "Ortho deleted successfully"}), 200

@postprocessing_bp.route('/associate_plots_with_boundaries', methods=['POST'])
def associate_plots_with_boundaries():
    """
    Associate AgRowStitch plots with boundary polygons and update plot_borders.csv with plot labels
    """
    import geopandas as gpd
    from shapely.geometry import Point
    import pandas as pd
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    data = request.json
    year = data.get('year')
    experiment = data.get('experiment')
    location = data.get('location')
    population = data.get('population')
    date = data.get('date')
    platform = data.get('platform')
    sensor = data.get('sensor')
    agrowstitch_dir = data.get('agrowstitchDir')
    boundaries = data.get('boundaries')  # GeoJSON FeatureCollection
    
    if not all([year, experiment, location, population, date, platform, sensor, agrowstitch_dir, boundaries]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        # Paths
        msgs_synced_path = os.path.join(
            data_root_dir, "Raw", year, experiment, location, population,
            date, platform, sensor, "Metadata", "msgs_synced.csv"
        )
        plot_borders_path = os.path.join(
            data_root_dir, "Raw", year, experiment, location, population,
            "plot_borders.csv"
        )
        agrowstitch_path = os.path.join(
            data_root_dir, "Processed", year, experiment, location, population,
            date, platform, sensor, agrowstitch_dir
        )
        
        # Check if msgs_synced.csv exists
        if not os.path.exists(msgs_synced_path):
            return jsonify({'error': f'msgs_synced.csv not found at {msgs_synced_path}'}), 404
            
        # Check if plot_borders.csv exists
        if not os.path.exists(plot_borders_path):
            return jsonify({'error': f'plot_borders.csv not found at {plot_borders_path}'}), 404
            
        # Check if AgRowStitch directory exists
        if not os.path.exists(agrowstitch_path):
            return jsonify({'error': f'AgRowStitch directory not found at {agrowstitch_path}'}), 404
            
        # Load msgs_synced.csv
        msgs_df = pd.read_csv(msgs_synced_path)
        
        # Load plot_borders.csv
        borders_df = pd.read_csv(plot_borders_path)
        
        # Check if plot_index column exists
        if 'plot_index' not in msgs_df.columns:
            return jsonify({'error': 'plot_index column not found in msgs_synced.csv'}), 400
            
        if 'plot_index' not in borders_df.columns:
            return jsonify({'error': 'plot_index column not found in plot_borders.csv'}), 400
            
        # Get unique plot indices (excluding unassigned)
        plot_indices = [idx for idx in msgs_df['plot_index'].unique() if idx > 0 and not pd.isna(idx)]
        
        if len(plot_indices) == 0:
            return jsonify({'error': 'No plot indices found in msgs_synced.csv'}), 400
            
        # Create GeoDataFrame from boundaries
        boundaries_gdf = gpd.GeoDataFrame.from_features(boundaries['features'])
        boundaries_gdf.set_crs(epsg=4326, inplace=True)
        
        # Initialize Plot and Accession columns if they don't exist
        if 'Plot' not in borders_df.columns:
            borders_df['Plot'] = None
        if 'Accession' not in borders_df.columns:
            borders_df['Accession'] = None
            
        # Track associations to prevent duplicates
        plot_associations = {}
        
        # For each plot index, find its center point and associate with boundary
        for plot_idx in plot_indices:
            plot_data = msgs_df[msgs_df['plot_index'] == plot_idx]
            
            if plot_data.empty:
                continue
                
            # Calculate center point of plot based on GPS coordinates
            center_lat = plot_data['lat'].mean()
            center_lon = plot_data['lon'].mean()
            center_point = Point(center_lon, center_lat)
            
            # Find which boundary contains this point
            for _, boundary in boundaries_gdf.iterrows():
                if boundary.geometry.contains(center_point):
                    # Get plot and accession from boundary properties
                    boundary_plot = boundary.get('plot', boundary.get('Plot'))
                    boundary_accession = boundary.get('accession', boundary.get('Accession'))
                    
                    # Check if this boundary is already associated with another plot
                    boundary_key = f"{boundary_plot}_{boundary_accession}"
                    if boundary_key in plot_associations:
                        print(f"Warning: Boundary {boundary_key} already associated with plot {plot_associations[boundary_key]}")
                        continue
                        
                    # Update plot_borders.csv with Plot and Accession
                    borders_df.loc[borders_df['plot_index'] == plot_idx, 'Plot'] = boundary_plot
                    borders_df.loc[borders_df['plot_index'] == plot_idx, 'Accession'] = boundary_accession
                    
                    # Track association
                    plot_associations[boundary_key] = int(plot_idx)
                    
                    print(f"Associated plot index {plot_idx} with boundary {boundary_key} -> Plot: {boundary_plot}, Accession: {boundary_accession}")
                    break
                    
        # Save updated plot_borders.csv
        borders_df.to_csv(plot_borders_path, index=False)
        
        # Return association summary
        return jsonify({
            'message': 'Plot associations completed successfully',
            'associations': int(len(plot_associations)),
            'plot_associations': plot_associations
        }), 200
        
    except Exception as e:
        print(f"Error in plot association: {e}")
        return jsonify({'error': str(e)}), 500

@postprocessing_bp.route('/get_agrowstitch_plot_associations', methods=['POST'])
def get_agrowstitch_plot_associations():
    """
    Get current plot associations for AgRowStitch plots from plot_borders.csv
    """
    data = request.json
    year = data.get('year')
    experiment = data.get('experiment')
    location = data.get('location')
    population = data.get('population')
    date = data.get('date')
    platform = data.get('platform')
    sensor = data.get('sensor')
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    if not all([year, experiment, location, population, date, platform, sensor]):
        return jsonify({'error': 'Missing required parameters'}), 400
        
    try:
        # Path to msgs_synced.csv (for plot indices)
        msgs_synced_path = os.path.join(
            data_root_dir, "Raw", year, experiment, location, population,
            date, platform, sensor, "Metadata", "msgs_synced.csv"
        )
        
        # Path to plot_borders.csv (for plot labels)
        plot_borders_path = os.path.join(
            data_root_dir, "Raw", year, experiment, location, population,
            "plot_borders.csv"
        )
        
        if not os.path.exists(msgs_synced_path):
            return jsonify({'error': 'msgs_synced.csv not found'}), 404
            
        if not os.path.exists(plot_borders_path):
            return jsonify({'error': 'plot_borders.csv not found'}), 404
            
        # Load both files
        msgs_df = pd.read_csv(msgs_synced_path)
        borders_df = pd.read_csv(plot_borders_path)
        
        # Check required columns
        if 'plot_index' not in msgs_df.columns:
            return jsonify({'error': 'plot_index column not found in msgs_synced.csv'}), 400
            
        if 'plot_index' not in borders_df.columns:
            return jsonify({'error': 'plot_index column not found in plot_borders.csv'}), 400
            
        # Get plot associations from plot_borders.csv
        associations = {}
        for _, row in borders_df.iterrows():
            plot_idx = row['plot_index']
            if plot_idx > 0 and (pd.notna(row.get('Plot')) or pd.notna(row.get('Accession'))):
                # Get center coordinates from msgs_synced.csv
                plot_data = msgs_df[msgs_df['plot_index'] == plot_idx]
                if not plot_data.empty:
                    center_lat = plot_data['lat'].mean()
                    center_lon = plot_data['lon'].mean()
                    
                    # Create plot label
                    plot_value = row.get('Plot')
                    accession_value = row.get('Accession')
                    
                    plot_label = f"Plot_{plot_value if pd.notna(plot_value) else 'Unknown'}"
                    if pd.notna(accession_value):
                        plot_label += f"_Acc_{accession_value}"
                    
                    associations[str(int(plot_idx))] = {
                        'plot_label': plot_label,
                        'center_lat': float(center_lat) if pd.notna(center_lat) else None,
                        'center_lon': float(center_lon) if pd.notna(center_lon) else None
                    }
                    
        return jsonify({
            'associations': associations,
            'total_plots': int(len([idx for idx in msgs_df['plot_index'].unique() if idx > 0 and not pd.isna(idx)]))
        }), 200
        
    except Exception as e:
        print(f"Error getting plot associations: {e}")
        return jsonify({'error': str(e)}), 500

@postprocessing_bp.route('/get_plot_borders_data', methods=['POST'])
def get_plot_borders_data():
    """
    Get plot borders data with plot labels and accessions
    """
    data = request.json
    year = data.get('year')
    experiment = data.get('experiment') 
    location = data.get('location')
    population = data.get('population')
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    if not all([year, experiment, location, population]):
        return jsonify({'error': 'Missing required parameters'}), 400
        
    try:
        # Path to plot_borders.csv
        plot_borders_path = os.path.join(
            data_root_dir, "Raw", year, experiment, location, population,
            "plot_borders.csv"
        )
        
        if not os.path.exists(plot_borders_path):
            return jsonify({'error': 'plot_borders.csv not found'}), 404
            
        # Load plot_borders.csv
        borders_df = pd.read_csv(plot_borders_path)
        
        # Create a dictionary mapping plot_index to plot labels and accessions
        plot_data = {}
        for _, row in borders_df.iterrows():
            plot_idx = row.get('plot_index')
            if pd.notna(plot_idx) and plot_idx > 0:
                plot_data[int(plot_idx)] = {
                    'plot': row.get('Plot') if pd.notna(row.get('Plot')) else None,
                    'accession': row.get('Accession') if pd.notna(row.get('Accession')) else None
                }
                
        return jsonify({'plot_data': plot_data}), 200
        
    except Exception as e:
        print(f"Error getting plot borders data: {e}")
        return jsonify({'error': str(e)}), 500

@postprocessing_bp.route('/download_single_plot', methods=['POST'])
def download_single_plot():
    """
    Download a single plot image with plot label and accession in filename
    """
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    data = request.get_json()
    try:
        year = data['year']
        experiment = data['experiment']
        location = data['location']
        population = data['population']
        date = data['date']
        platform = data['platform']
        sensor = data['sensor']
        agrowstitch_dir = data['agrowstitchDir']
        plot_filename = data['plotFilename']
        
        # Extract plot index from filename
        plot_match = re.search(r'temp_plot_(\d+)', plot_filename)
        if not plot_match:
            return jsonify({'error': 'Could not extract plot index from filename'}), 400
        
        plot_index = int(plot_match.group(1))
        
        # Get plot borders data
        plot_borders_path = os.path.join(
            data_root_dir, "Raw", year, experiment, location, population,
            "plot_borders.csv"
        )
        
        plot_label = None
        accession = None
        
        if os.path.exists(plot_borders_path):
            try:
                borders_df = pd.read_csv(plot_borders_path)
                plot_row = borders_df[borders_df['plot_index'] == plot_index]
                if not plot_row.empty:
                    plot_label = plot_row.iloc[0].get('Plot')
                    accession = plot_row.iloc[0].get('Accession')
                    if pd.isna(plot_label):
                        plot_label = None
                    if pd.isna(accession):
                        accession = None
            except Exception as e:
                print(f"Error reading plot borders: {e}")
        
        # Build original file path
        file_path = os.path.join(
            data_root_dir, 'Processed', year, experiment, location, population,
            date, platform, sensor, agrowstitch_dir, plot_filename
        )
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'Plot file not found'}), 404
        
        # Create new filename with plot label and accession in the requested format
        base_name = os.path.splitext(plot_filename)[0]
        extension = os.path.splitext(plot_filename)[1]
        
        # Use the format: plot_{plot}_accession_{accession}
        if plot_label and accession:
            new_filename = f"plot_{plot_label}_accession_{accession}{extension}"
        elif plot_label:
            new_filename = f"plot_{plot_label}{extension}"
        else:
            # Fallback to plot index if no plot label available
            new_filename = f"plot_{plot_index}{extension}"
        
        print(f"Download debug - Original: {plot_filename}, New format: {new_filename}")
        print(f"Plot data - plot_index: {plot_index}, plot_label: {plot_label}, accession: {accession}")

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, new_filename)
        
        try:
            # Copy original file to temp location with new name
            shutil.copy2(file_path, temp_file_path)
            print(f"Created temporary file: {temp_file_path}")
            
            # Send the temporary file with the enhanced filename
            response = send_file(
                temp_file_path,
                as_attachment=True,
                download_name=new_filename,
                mimetype='image/png'
            )
            
            # Clean up temp file after sending (Flask handles this automatically)
            print(f"Sending file with custom format name: {new_filename}")
            return response
            
        except Exception as temp_error:
            # Clean up temp directory if error occurs
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise temp_error
        
    except Exception as e:
        print(f"Error downloading single plot: {str(e)}")
        return jsonify({'error': str(e)}), 500
    

@postprocessing_bp.route('/split_orthomosaics', methods=['POST'])
def split_orthomosaics():
    """
    Split orthomosaics into individual plot images based on plot boundaries
    """
    data_root_dir = current_app.config['DATA_ROOT_DIR']
    try:
        data = request.get_json()
        year = data['year']
        experiment = data['experiment']
        location = data['location']
        population = data['population']
        date = data['date']
        boundaries = data['boundaries']
        
        # Construct paths
        base_path = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date)
        intermediate_path = os.path.join(data_root_dir, 'Intermediate', year, experiment, location, population)
        
        # Find orthomosaic
        orthomosaic_path = None
        platform_name = None
        sensor_name = None
        
        for platform in os.listdir(base_path):
            platform_path = os.path.join(base_path, platform)
            if not os.path.isdir(platform_path):
                continue
                
            for sensor in os.listdir(platform_path):
                sensor_path = os.path.join(platform_path, sensor)
                if not os.path.isdir(sensor_path):
                    continue
                    
                for file in os.listdir(sensor_path):
                    if file.endswith('-RGB.tif'):
                        orthomosaic_path = os.path.join(sensor_path, file)
                        platform_name = platform
                        sensor_name = sensor
                        break
                        
                if orthomosaic_path:
                    break
            if orthomosaic_path:
                break
        
        if not orthomosaic_path:
            return jsonify({"error": "No RGB orthomosaic found for the specified date"}), 404
        
        # Create output directory for plot images
        output_dir = os.path.join(intermediate_path, 'plot_images', date)
        os.makedirs(output_dir, exist_ok=True)
        
        plots_processed = 0
        
        # Process the orthomosaic
        try:
            import rasterio
            from rasterio.windows import from_bounds
            from rasterio.warp import transform_bounds
            import numpy as np
            from PIL import Image
            
            with rasterio.open(orthomosaic_path) as src:
                print(f"Opened orthomosaic: {orthomosaic_path}")
                print(f"CRS: {src.crs}, Shape: {src.shape}, Transform: {src.transform}")
                
                # Process each plot boundary
                for feature in boundaries['features']:
                    properties = feature['properties']
                    geometry = feature['geometry']
                    
                    # Get plot and accession info
                    plot = properties.get('plot', properties.get('Plot', 'unknown'))
                    accession = properties.get('accession', 'unknown')
                    
                    if plot == 'unknown' or accession == 'unknown':
                        print(f"Skipping feature - plot: {plot}, accession: {accession}")
                        continue
                        
                    # Validate geometry
                    if not geometry or geometry.get('type') != 'Polygon':
                        print(f"Invalid geometry for plot {plot}")
                        continue
                        
                    # Convert geometry coordinates to image coordinates
                    coords = geometry['coordinates'][0]  # Polygon exterior ring
                    
                    if len(coords) < 4:  # A polygon needs at least 4 points (including closing point)
                        print(f"Invalid polygon for plot {plot}: only {len(coords)} coordinates")
                        continue
                    
                    # Get bounding box from polygon
                    lons = [coord[0] for coord in coords]
                    lats = [coord[1] for coord in coords]
                    min_lon, max_lon = min(lons), max(lons)
                    min_lat, max_lat = min(lats), max(lats)
                    
                    print(f"Plot {plot}: bounds = ({min_lon}, {min_lat}, {max_lon}, {max_lat})")
                    
                    # Transform geographic coordinates to the orthomosaic's coordinate system
                    try:
                        # Transform bounds from WGS84 (EPSG:4326) to the orthomosaic's CRS
                        transformed_bounds = transform_bounds(
                            'EPSG:4326',  # source CRS (WGS84)
                            src.crs,      # destination CRS (orthomosaic's CRS)
                            min_lon, min_lat, max_lon, max_lat
                        )
                        min_x, min_y, max_x, max_y = transformed_bounds
                        
                        print(f"Plot {plot}: transformed bounds = ({min_x}, {min_y}, {max_x}, {max_y})")
                        
                        # Validate transformed bounds
                        if min_x >= max_x or min_y >= max_y:
                            print(f"Invalid transformed bounds for plot {plot}: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")
                            continue
                        
                        # Add small buffer to ensure non-zero area (in projected coordinates)
                        x_buffer = max(0.1, (max_x - min_x) * 0.1)  # 0.1 meter minimum buffer
                        y_buffer = max(0.1, (max_y - min_y) * 0.1)
                        min_x -= x_buffer
                        max_x += x_buffer
                        min_y -= y_buffer
                        max_y += y_buffer
                        
                        # Create window from transformed bounds
                        window = from_bounds(min_x, min_y, max_x, max_y, src.transform)
                        
                        print(f"Plot {plot}: window = {window} (width={window.width}, height={window.height})")
                        
                        # Validate window dimensions
                        if window.width <= 0 or window.height <= 0:
                            print(f"Invalid window dimensions for plot {plot}: width={window.width}, height={window.height}")
                            continue
                        
                        # Read the windowed data
                        data = src.read(window=window)
                        
                        # Create new transform for the windowed data
                        window_transform = src.window_transform(window)
                        
                        # Create filename
                        filename = f"plot_{plot}_accession_{accession}.tif"
                        temp_tif_path = os.path.join(output_dir, filename)
                        
                        # Write cropped TIF
                        with rasterio.open(
                            temp_tif_path,
                            'w',
                            driver='GTiff',
                            height=data.shape[1],
                            width=data.shape[2],
                            count=data.shape[0],
                            dtype=data.dtype,
                            crs=src.crs,
                            transform=window_transform,
                        ) as dst:
                            dst.write(data)
                        
                        # Convert TIF to PNG
                        png_filename = f"plot_{plot}_accession_{accession}.png"
                        png_path = os.path.join(output_dir, png_filename)
                        
                        # Open TIF and convert to PNG
                        with rasterio.open(temp_tif_path) as tif_src:
                            # Read all bands
                            data = tif_src.read()
                            
                            # Handle different band configurations
                            if data.shape[0] >= 3:  # RGB or RGBA
                                # Take first 3 bands for RGB
                                rgb_data = data[:3]
                                # Transpose from (bands, height, width) to (height, width, bands)
                                rgb_data = np.transpose(rgb_data, (1, 2, 0))
                                
                                # Normalize to 0-255 if needed
                                if rgb_data.dtype == np.uint16:
                                    rgb_data = (rgb_data / 65535.0 * 255).astype(np.uint8)
                                elif rgb_data.dtype == np.float32 or rgb_data.dtype == np.float64:
                                    rgb_data = (np.clip(rgb_data, 0, 1) * 255).astype(np.uint8)
                                
                                # Create PIL image and save as PNG
                                image = Image.fromarray(rgb_data)
                                image.save(png_path)
                                
                                plots_processed += 1
                        
                        # Remove temporary TIF file
                        os.remove(temp_tif_path)
                        
                    except Exception as e:
                        print(f"Error processing plot {plot}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error opening orthomosaic {orthomosaic_path}: {e}")
            return jsonify({"error": f"Error opening orthomosaic: {e}"}), 500
        
        return jsonify({
            "message": f"Successfully processed {plots_processed} plots",
            "plots_processed": plots_processed,
            "output_directory": output_dir
        }), 200
        
    except Exception as e:
        print(f"Error in split_orthomosaics: {e}")
        return jsonify({"error": str(e)}), 500