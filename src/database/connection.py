"""Database connection management"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
import redis
from .config import DatabaseConfig

# Base class for all models
Base = declarative_base()

# Global connections (initialized by init_db())
engine = None
SessionLocal = None
redis_client = None


def init_db():
    """
    Initialize database connections.
    Call this once at application startup.
    """
    global engine, SessionLocal, redis_client
    
    print("Initializing database connections...")
    
    # Initialize PostgreSQL engine
    if engine is None:
        engine = create_engine(
            DatabaseConfig.DATABASE_URL,
            echo=DatabaseConfig.SQLALCHEMY_ECHO,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,   # Recycle connections after 1 hour
        )
        
        SessionLocal = scoped_session(
            sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=engine
            )
        )
        print(f"✅ PostgreSQL engine created: {DatabaseConfig.POSTGRES_HOST}:{DatabaseConfig.POSTGRES_PORT}")
        
        # Run auto-migrations for schema changes
        _run_auto_migrations()
    
    # Initialize Redis client (if caching enabled)
    if redis_client is None and DatabaseConfig.ENABLE_CACHING:
        try:
            redis_client = redis.from_url(
                DatabaseConfig.REDIS_URL,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            # Test connection
            redis_client.ping()
            print(f"✅ Redis client connected: {DatabaseConfig.REDIS_HOST}:{DatabaseConfig.REDIS_PORT}")
        except Exception as e:
            print(f"⚠️  Redis connection failed: {e}")
            print("   Caching will be disabled")
            redis_client = None
    
    return engine, SessionLocal


def _run_auto_migrations():
    """
    Run automatic schema migrations on startup.
    Adds missing columns to existing tables.
    """
    print("Checking for schema updates...")
    
    migrations = [
        # Expand year column to support longer names (test data, etc.)
        "ALTER TABLE experiments ALTER COLUMN year TYPE VARCHAR(50)",
        
        # Create raw_data table
        """
        CREATE TABLE IF NOT EXISTS raw_data (
            id SERIAL PRIMARY KEY,
            collection_id INTEGER NOT NULL REFERENCES data_collections(id) ON DELETE CASCADE,
            data_type VARCHAR(50),
            data_path VARCHAR(1000),
            file_count INTEGER DEFAULT 0,
            total_size_bytes INTEGER DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE
        )
        """,
        
        # Create indexes for raw_data
        "CREATE INDEX IF NOT EXISTS idx_raw_data_collection ON raw_data(collection_id)",
        "CREATE INDEX IF NOT EXISTS idx_raw_data_type ON raw_data(data_type)",
        
        # Remove old columns from data_collections (if they exist)
        "ALTER TABLE data_collections DROP COLUMN IF EXISTS data_type",
        "ALTER TABLE data_collections DROP COLUMN IF EXISTS data_path",
        "ALTER TABLE data_collections DROP COLUMN IF EXISTS has_raw_files",
        "DROP INDEX IF EXISTS idx_collection_data_type",
        "DROP INDEX IF EXISTS idx_collection_has_raw_files",
    ]
    
    try:
        with engine.connect() as conn:
            for sql in migrations:
                conn.execute(text(sql))
                conn.commit()
        print("✅ Schema up to date")
    except Exception as e:
        print(f"⚠️  Schema migration warning: {e}")
        # Don't fail startup if migration has issues


def get_db():
    """
    Get database session (for use with Flask dependency injection).
    
    Usage:
        from database.connection import get_db
        
        @app.route('/something')
        def route():
            db = next(get_db())
            # use db
            db.close()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session():
    """
    Context manager for database sessions with automatic commit/rollback.
    
    Usage:
        from database.connection import get_db_session
        
        with get_db_session() as session:
            result = session.query(Experiment).all()
            # session commits automatically if no exception
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_redis():
    """
    Get Redis client instance.
    Returns None if Redis is not configured or unavailable.
    """
    return redis_client


def close_db():
    """
    Close database connections.
    Call this at application shutdown.
    """
    if SessionLocal:
        SessionLocal.remove()
        print("✅ Database sessions closed")
    if engine:
        engine.dispose()
        print("✅ Database engine disposed")


def check_db_connection():
    """
    Test database connection.
    Returns True if connected, False otherwise.
    """
    try:
        with get_db_session() as session:
            session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def check_redis_connection():
    """
    Test Redis connection.
    Returns True if connected, False otherwise.
    """
    try:
        if redis_client:
            redis_client.ping()
            return True
        return False
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False
