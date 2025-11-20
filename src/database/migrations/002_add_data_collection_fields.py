"""
Migration: Add data_type, has_raw_files, and data_path fields to data_collections table

Run this migration after the database has been initialized with the base schema.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import text
from database.connection import init_db, get_db_session
from database.config import DatabaseConfig


def upgrade():
    """Add new columns to data_collections table"""
    print("=" * 60)
    print("Migration: Adding fields to data_collections table")
    print("=" * 60)
    
    # Initialize database connection
    init_db()
    
    migrations = [
        # Add data_type column
        """
        ALTER TABLE data_collections 
        ADD COLUMN IF NOT EXISTS data_type VARCHAR(50);
        """,
        
        # Add has_raw_files column
        """
        ALTER TABLE data_collections 
        ADD COLUMN IF NOT EXISTS has_raw_files BOOLEAN DEFAULT TRUE;
        """,
        
        # Add data_path column
        """
        ALTER TABLE data_collections 
        ADD COLUMN IF NOT EXISTS data_path VARCHAR(1000);
        """,
        
        # Create index on data_type
        """
        CREATE INDEX IF NOT EXISTS idx_collection_data_type 
        ON data_collections(data_type);
        """,
        
        # Create index on has_raw_files
        """
        CREATE INDEX IF NOT EXISTS idx_collection_has_raw_files 
        ON data_collections(has_raw_files);
        """,
    ]
    
    with get_db_session() as session:
        for i, migration_sql in enumerate(migrations, 1):
            try:
                print(f"\n[{i}/{len(migrations)}] Executing migration...")
                print(migration_sql.strip())
                session.execute(text(migration_sql))
                print("✅ Success")
            except Exception as e:
                print(f"❌ Error: {e}")
                raise
        
        print("\n" + "=" * 60)
        print("✅ Migration completed successfully!")
        print("=" * 60)


def downgrade():
    """Remove the added columns (rollback)"""
    print("=" * 60)
    print("Migration Rollback: Removing fields from data_collections table")
    print("=" * 60)
    
    init_db()
    
    rollback_sql = [
        "DROP INDEX IF EXISTS idx_collection_has_raw_files;",
        "DROP INDEX IF EXISTS idx_collection_data_type;",
        "ALTER TABLE data_collections DROP COLUMN IF EXISTS data_path;",
        "ALTER TABLE data_collections DROP COLUMN IF EXISTS has_raw_files;",
        "ALTER TABLE data_collections DROP COLUMN IF EXISTS data_type;",
    ]
    
    with get_db_session() as session:
        for sql in rollback_sql:
            try:
                print(f"\nExecuting: {sql}")
                session.execute(text(sql))
                print("✅ Success")
            except Exception as e:
                print(f"❌ Error: {e}")
                raise
        
        print("\n" + "=" * 60)
        print("✅ Rollback completed successfully!")
        print("=" * 60)


def populate_data_path():
    """
    Helper function to populate data_path for existing records
    based on experiment hierarchy and collection metadata
    """
    print("\n" + "=" * 60)
    print("Populating data_path for existing records...")
    print("=" * 60)
    
    from database.models import DataCollection, Experiment
    
    with get_db_session() as session:
        collections = session.query(DataCollection).join(Experiment).all()
        
        updated_count = 0
        for collection in collections:
            # Construct data_path from hierarchy
            data_path = f"Raw/{collection.experiment.year}/{collection.experiment.name}/{collection.experiment.location}/{collection.experiment.population}/{collection.date}/{collection.platform}/{collection.sensor}"
            
            collection.data_path = data_path
            updated_count += 1
            
            if updated_count % 100 == 0:
                print(f"  Updated {updated_count} records...")
        
        print(f"\n✅ Updated data_path for {updated_count} collections")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run database migration")
    parser.add_argument('action', choices=['upgrade', 'downgrade', 'populate'], 
                       help='Migration action to perform')
    args = parser.parse_args()
    
    # Ensure database is enabled
    if not DatabaseConfig.USE_DATABASE:
        print("❌ Error: USE_DATABASE is not enabled in configuration")
        print("   Set USE_DATABASE=true in your environment")
        sys.exit(1)
    
    if args.action == 'upgrade':
        upgrade()
    elif args.action == 'downgrade':
        downgrade()
    elif args.action == 'populate':
        populate_data_path()
