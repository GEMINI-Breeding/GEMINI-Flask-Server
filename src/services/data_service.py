"""
Data service layer for database operations.
Provides a clean interface between Flask routes and database models.
"""
from typing import List, Optional
from sqlalchemy import func
from database.connection import get_db_session, get_redis
from database.models import Experiment, DataCollection
from database.config import DatabaseConfig
import json


class DataService:
    """Service for data operations with caching support"""
    
    def __init__(self):
        self.use_cache = DatabaseConfig.ENABLE_CACHING
        self._redis = None
        self._redis_initialized = False
        self.cache_ttl = 300  # 5 minutes default TTL
    
    @property
    def redis(self):
        """Lazy initialization of Redis client"""
        if not self._redis_initialized:
            self._redis = get_redis() if self.use_cache else None
            self._redis_initialized = True
        return self._redis
    
    def _cache_get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if not self.redis:
            return None
        try:
            return self.redis.get(key)
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    def _cache_set(self, key: str, value: str, ttl: int = None):
        """Set value in cache with TTL"""
        if not self.redis:
            return
        try:
            self.redis.setex(key, ttl or self.cache_ttl, value)
        except Exception as e:
            print(f"Cache set error: {e}")
    
    def get_years(self) -> List[str]:
        """Get all available years"""
        cache_key = "years"
        
        # Try cache first
        cached = self._cache_get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Query database
        with get_db_session() as session:
            years = session.query(Experiment.year)\
                .distinct()\
                .order_by(Experiment.year.desc())\
                .all()
            
            result = [year[0] for year in years]
            
            # Cache result
            self._cache_set(cache_key, json.dumps(result))
            
            return result
    
    def get_experiments(self, year: str) -> List[str]:
        """Get experiments for a given year"""
        cache_key = f"experiments:{year}"
        
        # Try cache
        cached = self._cache_get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Query database
        with get_db_session() as session:
            experiments = session.query(Experiment.name)\
                .filter(Experiment.year == year)\
                .distinct()\
                .order_by(Experiment.name)\
                .all()
            
            result = [exp[0] for exp in experiments]
            
            # Cache result
            self._cache_set(cache_key, json.dumps(result))
            
            return result
    
    def get_locations(self, year: str, experiment: str) -> List[str]:
        """Get locations for a given year and experiment"""
        cache_key = f"locations:{year}:{experiment}"
        
        cached = self._cache_get(cache_key)
        if cached:
            return json.loads(cached)
        
        with get_db_session() as session:
            locations = session.query(Experiment.location)\
                .filter(
                    Experiment.year == year,
                    Experiment.name == experiment
                )\
                .distinct()\
                .order_by(Experiment.location)\
                .all()
            
            result = [loc[0] for loc in locations]
            self._cache_set(cache_key, json.dumps(result))
            return result
    
    def get_populations(self, year: str, experiment: str, location: str) -> List[str]:
        """Get populations for given year, experiment, and location"""
        cache_key = f"populations:{year}:{experiment}:{location}"
        
        cached = self._cache_get(cache_key)
        if cached:
            return json.loads(cached)
        
        with get_db_session() as session:
            populations = session.query(Experiment.population)\
                .filter(
                    Experiment.year == year,
                    Experiment.name == experiment,
                    Experiment.location == location
                )\
                .distinct()\
                .order_by(Experiment.population)\
                .all()
            
            result = [pop[0] for pop in populations]
            self._cache_set(cache_key, json.dumps(result))
            return result
    
    def get_dates(self, year: str, experiment: str, location: str, population: str) -> List[str]:
        """Get dates for given experiment/location/population"""
        cache_key = f"dates:{year}:{experiment}:{location}:{population}"
        
        cached = self._cache_get(cache_key)
        if cached:
            return json.loads(cached)
        
        with get_db_session() as session:
            # Get experiment
            exp = session.query(Experiment).filter_by(
                year=year,
                name=experiment,
                location=location,
                population=population
            ).first()
            
            if not exp:
                return []
            
            # Get dates from collections
            dates = session.query(DataCollection.date)\
                .filter(DataCollection.experiment_id == exp.id)\
                .distinct()\
                .order_by(DataCollection.date.desc())\
                .all()
            
            result = [date[0] for date in dates]
            self._cache_set(cache_key, json.dumps(result))
            return result
    
    def get_platforms(self, year: str, experiment: str, location: str, 
                     population: str, date: str) -> List[str]:
        """Get platforms for given date"""
        cache_key = f"platforms:{year}:{experiment}:{location}:{population}:{date}"
        
        cached = self._cache_get(cache_key)
        if cached:
            return json.loads(cached)
        
        with get_db_session() as session:
            exp = session.query(Experiment).filter_by(
                year=year,
                name=experiment,
                location=location,
                population=population
            ).first()
            
            if not exp:
                return []
            
            platforms = session.query(DataCollection.platform)\
                .filter(
                    DataCollection.experiment_id == exp.id,
                    DataCollection.date == date
                )\
                .distinct()\
                .order_by(DataCollection.platform)\
                .all()
            
            result = [plat[0] for plat in platforms]
            self._cache_set(cache_key, json.dumps(result))
            return result
    
    def get_sensors(self, year: str, experiment: str, location: str, 
                   population: str, date: str, platform: str) -> List[str]:
        """Get sensors for given platform"""
        cache_key = f"sensors:{year}:{experiment}:{location}:{population}:{date}:{platform}"
        
        cached = self._cache_get(cache_key)
        if cached:
            return json.loads(cached)
        
        with get_db_session() as session:
            exp = session.query(Experiment).filter_by(
                year=year,
                name=experiment,
                location=location,
                population=population
            ).first()
            
            if not exp:
                return []
            
            sensors = session.query(DataCollection.sensor)\
                .filter(
                    DataCollection.experiment_id == exp.id,
                    DataCollection.date == date,
                    DataCollection.platform == platform
                )\
                .distinct()\
                .order_by(DataCollection.sensor)\
                .all()
            
            result = [sens[0] for sens in sensors]
            self._cache_set(cache_key, json.dumps(result))
            return result
    
    def get_all_data(self) -> List[dict]:
        """Get all experiments with their data collections"""
        cache_key = "all_data"
        
        cached = self._cache_get(cache_key)
        if cached:
            return json.loads(cached)
        
        with get_db_session() as session:
            # Join experiments with data_collections
            results = session.query(
                Experiment.year,
                Experiment.name.label('experiment'),
                Experiment.location,
                Experiment.population,
                DataCollection.date,
                DataCollection.platform,
                DataCollection.sensor
            ).join(
                DataCollection,
                Experiment.id == DataCollection.experiment_id
            ).order_by(
                Experiment.year,
                Experiment.name,
                Experiment.location,
                Experiment.population,
                DataCollection.date,
                DataCollection.platform,
                DataCollection.sensor
            ).all()
            
            # Transform to list of dictionaries
            data = []
            for row in results:
                item = {
                    'year': row.year,
                    'experiment': row.experiment,
                    'location': row.location,
                    'population': row.population,
                    'date': row.date,
                    'platform': row.platform,
                    'sensor': row.sensor,
                    'cameras': []  # TODO: Add camera detection for Amiga/rover platforms
                }
                
                # For Amiga/rover platforms, check for camera folders
                if row.platform in ['Amiga', 'rover']:
                    # You could add logic here to check filesystem for camera folders
                    # For now, just add default cameras
                    item['cameras'] = ['top', 'left', 'right']
                
                data.append(item)
            
            self._cache_set(cache_key, json.dumps(data))
            return data
    
    def get_images(self, year: str, experiment: str, location: str, population: str, 
                   date: str, platform: str, sensor: str, camera: Optional[str] = None) -> List[str]:
        """Get image paths from raw_data for a specific data collection"""
        from database.models import RawData
        
        cache_key = f"images:{year}:{experiment}:{location}:{population}:{date}:{platform}:{sensor}:{camera or ''}"
        
        cached = self._cache_get(cache_key)
        if cached:
            return json.loads(cached)
        
        with get_db_session() as session:
            # Find the experiment
            exp = session.query(Experiment).filter_by(
                year=year,
                name=experiment,
                location=location,
                population=population
            ).first()
            
            if not exp:
                return []
            
            # Find the data collection
            collection = session.query(DataCollection).filter_by(
                experiment_id=exp.id,
                date=date,
                platform=platform,
                sensor=sensor
            ).first()
            
            if not collection:
                return []
            
            # Get raw_data entries for images
            raw_data_entries = session.query(RawData.data_path)\
                .filter(
                    RawData.collection_id == collection.id,
                    RawData.data_type == 'images'
                )\
                .order_by(RawData.data_path)\
                .all()
            
            # Extract just the filenames from the full paths
            # Path format: Raw/2022/GEMINI/Davis/Legumes/2022-06-27/Drone/RGB/Images/IMG_0001.JPG
            image_paths = []
            for entry in raw_data_entries:
                path = entry[0]
                # Extract filename from path
                filename = path.split('/')[-1]
                image_paths.append(filename)
            
            self._cache_set(cache_key, json.dumps(image_paths))
            return image_paths
    
    def clear_cache(self):
        """Clear all cached data"""
        if self.redis:
            try:
                self.redis.flushdb()
                print("✅ Cache cleared")
            except Exception as e:
                print(f"❌ Failed to clear cache: {e}")


# Global instance
data_service = DataService()
