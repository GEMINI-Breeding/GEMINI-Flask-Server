# Standard library imports
import os
# Third-party library imports
from flask import current_app, Blueprint, jsonify, request
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

fetching_bp = Blueprint('fetching', __name__)
data_root_dir = current_app.config['DATA_ROOT_DIR']

@fetching_bp.route('/get_data_options', methods=['GET'])
def get_data_options():
    """
    Get all available years from the data directory
    """
    try:
        years = []
        raw_dir = os.path.join(data_root_dir, 'Raw')
        
        if os.path.exists(raw_dir):
            years = [item for item in os.listdir(raw_dir) 
                    if os.path.isdir(os.path.join(raw_dir, item)) and item.isdigit()]
            years.sort()
        
        return jsonify({'years': years}), 200
        
    except Exception as e:
        print(f"Error getting data options: {e}")
        return jsonify({'error': str(e)}), 500

@fetching_bp.route('/get_experiments', methods=['POST'])
def get_experiments():
    """
    Get available experiments for a given year
    """
    try:
        data = request.json
        year = data.get('year')
        
        if not year:
            return jsonify({'error': 'Year is required'}), 400
            
        experiments = []
        year_dir = os.path.join(data_root_dir, 'Raw', year)
        
        if os.path.exists(year_dir):
            experiments = [item for item in os.listdir(year_dir) 
                          if os.path.isdir(os.path.join(year_dir, item))]
            experiments.sort()
        
        return jsonify({'experiments': experiments}), 200
        
    except Exception as e:
        print(f"Error getting experiments: {e}")
        return jsonify({'error': str(e)}), 500

@fetching_bp.route('/get_locations', methods=['POST'])
def get_locations():
    """
    Get available locations for a given year and experiment
    """
    try:
        data = request.json
        year = data.get('year')
        experiment = data.get('experiment')
        
        if not all([year, experiment]):
            return jsonify({'error': 'Year and experiment are required'}), 400
            
        locations = []
        exp_dir = os.path.join(data_root_dir, 'Raw', year, experiment)
        
        if os.path.exists(exp_dir):
            locations = [item for item in os.listdir(exp_dir) 
                        if os.path.isdir(os.path.join(exp_dir, item))]
            locations.sort()
        
        return jsonify({'locations': locations}), 200
        
    except Exception as e:
        print(f"Error getting locations: {e}")
        return jsonify({'error': str(e)}), 500

@fetching_bp.route('/get_populations', methods=['POST'])
def get_populations():
    """
    Get available populations for a given year, experiment, and location
    """
    try:
        data = request.json
        year = data.get('year')
        experiment = data.get('experiment')
        location = data.get('location')
        
        if not all([year, experiment, location]):
            return jsonify({'error': 'Year, experiment, and location are required'}), 400
            
        populations = []
        loc_dir = os.path.join(data_root_dir, 'Raw', year, experiment, location)
        
        if os.path.exists(loc_dir):
            populations = [item for item in os.listdir(loc_dir) 
                          if os.path.isdir(os.path.join(loc_dir, item))]
            populations.sort()
        
        return jsonify({'populations': populations}), 200
        
    except Exception as e:
        print(f"Error getting populations: {e}")
        return jsonify({'error': str(e)}), 500

@fetching_bp.route('/get_dates', methods=['POST'])
def get_dates():
    """
    Get available dates for a given year, experiment, location, and population
    """
    try:
        data = request.json
        year = data.get('year')
        experiment = data.get('experiment')
        location = data.get('location')
        population = data.get('population')
        
        if not all([year, experiment, location, population]):
            return jsonify({'error': 'Year, experiment, location, and population are required'}), 400
            
        dates = []
        # Check both Raw and Processed directories for dates
        raw_pop_dir = os.path.join(data_root_dir, 'Raw', year, experiment, location, population)
        processed_pop_dir = os.path.join(data_root_dir, 'Processed', year, experiment, location, population)
        
        date_set = set()
        
        # Check Raw directory
        if os.path.exists(raw_pop_dir):
            for item in os.listdir(raw_pop_dir):
                item_path = os.path.join(raw_pop_dir, item)
                if os.path.isdir(item_path) and item not in ['plot_borders.csv']:
                    date_set.add(item)
        
        # Check Processed directory
        if os.path.exists(processed_pop_dir):
            for item in os.listdir(processed_pop_dir):
                item_path = os.path.join(processed_pop_dir, item)
                if os.path.isdir(item_path):
                    date_set.add(item)
        
        dates = sorted(list(date_set))
        
        return jsonify({'dates': dates}), 200
        
    except Exception as e:
        print(f"Error getting dates: {e}")
        return jsonify({'error': str(e)}), 500

@fetching_bp.route('/get_platforms', methods=['POST'])
def get_platforms():
    """
    Get available platforms for a given dataset
    """
    try:
        data = request.json
        year = data.get('year')
        experiment = data.get('experiment')
        location = data.get('location')
        population = data.get('population')
        date = data.get('date')
        
        if not all([year, experiment, location, population, date]):
            return jsonify({'error': 'All parameters are required'}), 400
            
        platforms = []
        # Check Processed directory for platforms
        date_dir = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date)
        
        if os.path.exists(date_dir):
            platforms = [item for item in os.listdir(date_dir) 
                        if os.path.isdir(os.path.join(date_dir, item))]
            platforms.sort()
        
        return jsonify({'platforms': platforms}), 200
        
    except Exception as e:
        print(f"Error getting platforms: {e}")
        return jsonify({'error': str(e)}), 500

@fetching_bp.route('/get_sensors', methods=['POST'])
def get_sensors():
    """
    Get available sensors for a given dataset and platform
    """
    try:
        data = request.json
        year = data.get('year')
        experiment = data.get('experiment')
        location = data.get('location')
        population = data.get('population')
        date = data.get('date')
        platform = data.get('platform')
        
        if not all([year, experiment, location, population, date, platform]):
            return jsonify({'error': 'All parameters are required'}), 400
            
        sensors = []
        # Check Processed directory for sensors
        platform_dir = os.path.join(data_root_dir, 'Processed', year, experiment, location, population, date, platform)
        
        if os.path.exists(platform_dir):
            sensors = [item for item in os.listdir(platform_dir) 
                      if os.path.isdir(os.path.join(platform_dir, item))]
            sensors.sort()
        
        return jsonify({'sensors': sensors}), 200
        
    except Exception as e:
        print(f"Error getting sensors: {e}")
        return jsonify({'error': str(e)}), 500