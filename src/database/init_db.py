#!/usr/bin/env python3
"""Initialize database schema - creates all tables"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import Base, init_db
from database.models import Experiment, DataCollection, RawData


def create_tables():
    """Create all database tables"""
    print("\n" + "="*60)
    print("GEMINI Database Initialization")
    print("="*60)
    
    print("\nInitializing database connection...")
    engine, _ = init_db()
    
    print("\nCreating database tables...")
    Base.metadata.create_all(bind=engine)
    
    print("\n✅ Database tables created successfully!")
    print("\nCreated tables:")
    for table in Base.metadata.sorted_tables:
        print(f"  - {table.name}")
        # Show columns
        for column in table.columns:
            nullable = "NULL" if column.nullable else "NOT NULL"
            print(f"      {column.name}: {column.type} {nullable}")
    
    print("\n" + "="*60)
    print("✅ Database initialization complete!")
    print("="*60)
    print("\nYou can now:")
    print("  1. View tables in pgAdmin: http://localhost:5050")
    print("  2. Run migration script: python scripts/migrate_hierarchy.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    create_tables()
