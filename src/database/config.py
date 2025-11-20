"""Database configuration for GEMINI application"""
import os


class DatabaseConfig:
    """Database configuration from environment variables"""
    
    # ============================================
    # PostgreSQL Configuration
    # ============================================
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'gemini_user')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'gemini_password')
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'gemini_db')
    
    # Database URL for SQLAlchemy
    DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    
    # SQLAlchemy settings
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = os.getenv('SQLALCHEMY_ECHO', 'false').lower() == 'true'
    
    # ============================================
    # Redis Configuration
    # ============================================
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = os.getenv('REDIS_PORT', '6379')
    REDIS_DB = os.getenv('REDIS_DB', '0')
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    
    # ============================================
    # Feature Flags
    # ============================================
    USE_DATABASE = os.getenv('USE_DATABASE', 'false').lower() == 'true'
    ENABLE_CACHING = os.getenv('ENABLE_CACHING', 'false').lower() == 'true'
    
    # ============================================
    # Data Directory
    # ============================================
    DATA_ROOT_DIR = os.getenv('DATA_ROOT_DIR', os.getenv('REACT_APP_APP_DATA', '/home/gemini-data'))
    
    @classmethod
    def print_config(cls):
        """Print current configuration (for debugging)"""
        print("="*60)
        print("Database Configuration")
        print("="*60)
        print(f"Database URL: {cls.DATABASE_URL}")
        print(f"Redis URL: {cls.REDIS_URL}")
        print(f"USE_DATABASE: {cls.USE_DATABASE}")
        print(f"ENABLE_CACHING: {cls.ENABLE_CACHING}")
        print(f"DATA_ROOT_DIR: {cls.DATA_ROOT_DIR}")
        print("="*60)
