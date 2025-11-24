"""
One-time migration script to scan filesystem and populate database.
Scans Raw/ directory structure and creates database records.
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import get_db_session, init_db
from database.models import Experiment, DataCollection, RawData
from database.config import DatabaseConfig


def scan_directory(root_dir: str) -> dict:
    """
    Scan Raw/ directory structure and extract hierarchy.
    Returns dict with structure: {year: {experiment: {location: {population: [dates]}}}}
    """
    if not os.path.exists(root_dir):
        raise ValueError(f"Data root directory not found: {root_dir}")
    
    structure = {}
    raw_path = os.path.join(root_dir, "Raw")
    
    if not os.path.exists(raw_path):
        raise ValueError(f"Raw directory not found: {raw_path}")
    
    print(f"\n📂 Scanning directory: {raw_path}")
    
    # Walk through Raw/{year}/{experiment}/{location}/{population}/{date}/{platform}/{sensor}
    for year in os.listdir(raw_path):
        year_path = os.path.join(raw_path, year)
        if not os.path.isdir(year_path):
            continue
        
        structure[year] = {}
        
        for experiment in os.listdir(year_path):
            exp_path = os.path.join(year_path, experiment)
            if not os.path.isdir(exp_path):
                continue
            
            structure[year][experiment] = {}
            
            for location in os.listdir(exp_path):
                loc_path = os.path.join(exp_path, location)
                if not os.path.isdir(loc_path):
                    continue
                
                structure[year][experiment][location] = {}
                
                for population in os.listdir(loc_path):
                    pop_path = os.path.join(loc_path, population)
                    if not os.path.isdir(pop_path):
                        continue
                    
                    structure[year][experiment][location][population] = []
                    
                    for date_dir in os.listdir(pop_path):
                        date_path = os.path.join(pop_path, date_dir)
                        if not os.path.isdir(date_path):
                            continue
                        
                        # Collect platform/sensor combinations
                        collections = []
                        for platform in os.listdir(date_path):
                            plat_path = os.path.join(date_path, platform)
                            if not os.path.isdir(plat_path):
                                continue
                            
                            for sensor in os.listdir(plat_path):
                                sensor_path = os.path.join(plat_path, sensor)
                                if not os.path.isdir(sensor_path):
                                    continue
                                
                                collections.append({
                                    'date': date_dir,
                                    'platform': platform,
                                    'sensor': sensor,
                                    'path': sensor_path
                                })
                        
                        if collections:
                            structure[year][experiment][location][population].append({
                                'date': date_dir,
                                'collections': collections
                            })
    
    return structure


def populate_database(structure: dict):
    """
    Populate database with scanned structure.
    Creates Experiment and DataCollection records.
    """
    print("\n📝 Populating database...")
    
    with get_db_session() as session:
        exp_count = 0
        coll_count = 0
        
        for year, experiments in structure.items():
            for exp_name, locations in experiments.items():
                for location, populations in locations.items():
                    for population, date_data in populations.items():
                        # Create or get experiment
                        experiment = session.query(Experiment).filter_by(
                            year=year,
                            name=exp_name,
                            location=location,
                            population=population
                        ).first()
                        
                        if not experiment:
                            experiment = Experiment(
                                year=year,
                                name=exp_name,
                                location=location,
                                population=population
                            )
                            session.add(experiment)
                            session.flush()  # Get ID
                            exp_count += 1
                        
                        # Create data collections
                        for date_entry in date_data:
                            for coll in date_entry['collections']:
                                # Check if collection exists
                                existing = session.query(DataCollection).filter_by(
                                    experiment_id=experiment.id,
                                    date=coll['date'],
                                    platform=coll['platform'],
                                    sensor=coll['sensor']
                                ).first()
                                
                                if not existing:
                                    collection = DataCollection(
                                        experiment_id=experiment.id,
                                        date=coll['date'],
                                        platform=coll['platform'],
                                        sensor=coll['sensor'],
                                        extra_metadata={'path': coll['path']}
                                    )
                                    session.add(collection)
                                    coll_count += 1
                                    
                                    # Note: raw_data records will be created on upload
                                    # with data_type and data_path populated then
        
        # Commit all changes
        session.commit()
        
        print(f"\n✅ Migration complete!")
        print(f"   - Created {exp_count} experiments")
        print(f"   - Created {coll_count} data collections")
        
        # Show summary stats
        total_exp = session.query(Experiment).count()
        total_coll = session.query(DataCollection).count()
        print(f"\n📊 Database summary:")
        print(f"   - Total experiments: {total_exp}")
        print(f"   - Total data collections: {total_coll}")


def cleanup_deleted_entries(root_dir: str):
    """
    Remove database entries for experiments and collections that no longer exist in filesystem.
    Also removes raw_data entries for files that no longer exist.
    """
    print("\n🧹 Cleaning up deleted entries...")
    
    raw_path = os.path.join(root_dir, "Raw")
    deleted_experiments = 0
    deleted_collections = 0
    deleted_files = 0
    
    with get_db_session() as session:
        # Get all experiments from database
        all_experiments = session.query(Experiment).all()
        
        for exp in all_experiments:
            # Check if experiment directory still exists
            exp_path = os.path.join(raw_path, exp.year, exp.name, exp.location, exp.population)
            
            if not os.path.exists(exp_path):
                # Experiment directory deleted - remove from database
                print(f"   🗑️  Removing experiment: {exp.year}/{exp.name}/{exp.location}/{exp.population}")
                session.delete(exp)
                deleted_experiments += 1
                continue  # Skip collection check since experiment is gone
            
            # Check collections for this experiment
            for collection in exp.collections:
                collection_path = os.path.join(exp_path, collection.date, collection.platform, collection.sensor)
                
                if not os.path.exists(collection_path):
                    # Collection directory deleted - remove from database
                    print(f"   🗑️  Removing collection: {exp.year}/{exp.name}/{exp.location}/{exp.population}/{collection.date}/{collection.platform}/{collection.sensor}")
                    session.delete(collection)
                    deleted_collections += 1
                    continue  # Skip file check since collection is gone
                
                # Check raw_data files for this collection
                for raw_data in collection.raw_data:
                    # data_path is relative to root_dir, like: Raw/2022/GEMINI/Davis/Legumes/2022-06-27/Drone/RGB/Images/IMG_001.JPG
                    file_path = os.path.join(root_dir, raw_data.data_path)
                    
                    if not os.path.exists(file_path):
                        # File deleted - remove from database
                        session.delete(raw_data)
                        deleted_files += 1
        
        # Commit deletions
        session.commit()
    
    if deleted_experiments > 0 or deleted_collections > 0 or deleted_files > 0:
        print(f"\n🧹 Cleanup complete:")
        print(f"   - Removed {deleted_experiments} experiments")
        print(f"   - Removed {deleted_collections} data collections")
        print(f"   - Removed {deleted_files} raw data entries")
    else:
        print("   ✅ No stale entries found - database is clean!")


def main():
    """Main migration workflow"""
    print("=" * 60)
    print("🚀 GEMINI Database Migration - Hierarchy Scanner")
    print("=" * 60)
    
    # Get data root from config
    data_root = DatabaseConfig.DATA_ROOT_DIR
    
    if not data_root:
        print("\n❌ ERROR: DATA_ROOT_DIR not set in .env.database")
        print("   Set DATA_ROOT_DIR=/path/to/your/data")
        sys.exit(1)
    
    print(f"\n⚙️  Configuration:")
    print(f"   Data root: {data_root}")
    print(f"   Database: {DatabaseConfig.POSTGRES_DB}")
    
    try:
        # Initialize database (create tables if needed)
        print("\n🔧 Initializing database...")
        init_db()
        
        # Scan filesystem
        start_time = datetime.now()
        structure = scan_directory(data_root)
        scan_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   Scan completed in {scan_time:.2f}s")
        
        # Populate database
        start_time = datetime.now()
        populate_database(structure)
        populate_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   Population completed in {populate_time:.2f}s")
        
        print("\n" + "=" * 60)
        print("✅ Migration successful!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Verify data in pgAdmin (localhost:5050)")
        print("2. Run test_migration.py to benchmark performance")
        print("3. Update Flask endpoints to use data_service")
        print("4. Set USE_DATABASE=true in .env.database")
        print("5. Set ENABLE_CACHING=true for Redis caching")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
