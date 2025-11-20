"""Database models for GEMINI application"""
from sqlalchemy import (
    Column, Integer, String, DateTime, Float, 
    Boolean, Text, ForeignKey, Index, UniqueConstraint,
    JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .connection import Base


class Experiment(Base):
    """Experiment metadata (year/experiment/location/population hierarchy)"""
    __tablename__ = 'experiments'
    
    id = Column(Integer, primary_key=True)
    year = Column(String(50), nullable=False, index=True)  # Allow longer year names (supports test data)
    name = Column(String(255), nullable=False, index=True)
    location = Column(String(255), nullable=False, index=True)
    population = Column(String(255), nullable=False, index=True)
    
    # Metadata
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    data_collections = relationship("DataCollection", back_populates="experiment", cascade="all, delete-orphan")
    
    # Unique constraint and indexes
    __table_args__ = (
        UniqueConstraint('year', 'name', 'location', 'population', name='uix_experiment'),
        Index('idx_experiment_year_name', 'year', 'name'),
        Index('idx_experiment_location', 'location'),
    )
    
    def __repr__(self):
        return f"<Experiment {self.year}/{self.name}/{self.location}/{self.population}>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'year': self.year,
            'name': self.name,
            'location': self.location,
            'population': self.population,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class DataCollection(Base):
    """Data collection session (date/platform/sensor)"""
    __tablename__ = 'data_collections'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Collection metadata
    date = Column(String(50), nullable=False, index=True)  # Date folder name (e.g., YYYY-MM-DD, may have suffixes)
    platform = Column(String(100), nullable=False, index=True)  # Drone, Rover, etc.
    sensor = Column(String(100), nullable=False, index=True)    # RGB, Thermal, etc.
    
    # Status tracking
    status = Column(String(50), default='uploaded', index=True)  # uploaded, processing, complete, failed
    file_count = Column(Integer, default=0)
    total_size_bytes = Column(Integer, default=0)
    
    # Additional metadata as JSON (renamed from 'metadata' to avoid SQLAlchemy conflict)
    extra_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    experiment = relationship("Experiment", back_populates="data_collections")
    raw_data = relationship("RawData", back_populates="data_collection", cascade="all, delete-orphan")
    
    # Unique constraint and indexes
    __table_args__ = (
        UniqueConstraint('experiment_id', 'date', 'platform', 'sensor', name='uix_collection'),
        Index('idx_collection_date', 'date'),
        Index('idx_collection_status', 'status'),
    )
    
    def __repr__(self):
        return f"<DataCollection {self.date}/{self.platform}/{self.sensor}>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'experiment_id': self.experiment_id,
            'date': self.date,
            'platform': self.platform,
            'sensor': self.sensor,
            'status': self.status,
            'file_count': self.file_count,
            'total_size_bytes': self.total_size_bytes,
            'extra_metadata': self.extra_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class RawData(Base):
    """Raw data files associated with a data collection"""
    __tablename__ = 'raw_data'
    
    id = Column(Integer, primary_key=True)
    collection_id = Column(Integer, ForeignKey('data_collections.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Data classification
    data_type = Column(String(50), nullable=True, index=True)  # 'image', 'metadata', 'gps', etc.
    data_path = Column(String(1000), nullable=True)  # Relative path to the data file/directory
    
    # File information
    file_count = Column(Integer, default=0)
    total_size_bytes = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    data_collection = relationship("DataCollection", back_populates="raw_data")
    
    # Indexes
    __table_args__ = (
        Index('idx_raw_data_collection', 'collection_id'),
        Index('idx_raw_data_type', 'data_type'),
    )
    
    def __repr__(self):
        return f"<RawData collection_id={self.collection_id} type={self.data_type}>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'collection_id': self.collection_id,
            'data_type': self.data_type,
            'data_path': self.data_path,
            'file_count': self.file_count,
            'total_size_bytes': self.total_size_bytes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
