#!/usr/bin/env python3
"""Clear all data from the database (keeps table structure)"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import init_db, get_db_session
from database.models import Experiment, DataCollection, RawData
from database.config import DatabaseConfig


def clear_database():
    """Clear all data from database tables"""
    
    if not DatabaseConfig.USE_DATABASE:
        print("❌ Error: USE_DATABASE is not enabled")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("⚠️  WARNING: This will delete ALL data from the database!")
    print("="*60)
    
    response = input("\nAre you sure you want to continue? (type 'yes' to confirm): ")
    
    if response.lower() != 'yes':
        print("❌ Cancelled - no data was deleted")
        return
    
    print("\nInitializing database connection...")
    init_db()
    
    with get_db_session() as session:
        # Delete in order (child tables first due to foreign keys)
        print("\nDeleting data...")
        
        raw_count = session.query(RawData).count()
        session.query(RawData).delete()
        print(f"  ✅ Deleted {raw_count} raw_data records")
        
        coll_count = session.query(DataCollection).count()
        session.query(DataCollection).delete()
        print(f"  ✅ Deleted {coll_count} data_collection records")
        
        exp_count = session.query(Experiment).count()
        session.query(Experiment).delete()
        print(f"  ✅ Deleted {exp_count} experiment records")
        
        session.commit()
    
    print("\n" + "="*60)
    print("✅ Database cleared successfully!")
    print("="*60)
    print("\nThe tables still exist but are empty.")
    print("Restart the Flask server to sync new data.\n")


if __name__ == "__main__":
    clear_database()
