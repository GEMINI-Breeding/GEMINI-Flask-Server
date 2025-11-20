"""
Performance benchmarking script.
Compares filesystem scanning vs database queries.
"""
import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.config import DatabaseConfig
from services.data_service import DataService
from database.connection import init_db


def benchmark_filesystem(data_root: str) -> dict:
    """
    Benchmark filesystem scanning performance.
    Times how long it takes to list directories at each level.
    """
    print("\n📁 Filesystem Benchmark")
    print("-" * 40)
    
    raw_path = os.path.join(data_root, "Raw")
    results = {}
    
    # Benchmark: List years
    start = time.time()
    years = [d for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))]
    years.sort(reverse=True)
    results['years'] = {
        'time_ms': (time.time() - start) * 1000,
        'count': len(years)
    }
    print(f"Years: {len(years)} items in {results['years']['time_ms']:.2f}ms")
    
    if not years:
        return results
    
    # Benchmark: List experiments (for first year)
    year = years[0]
    year_path = os.path.join(raw_path, year)
    start = time.time()
    experiments = [d for d in os.listdir(year_path) 
                   if os.path.isdir(os.path.join(year_path, d))]
    experiments.sort()
    results['experiments'] = {
        'time_ms': (time.time() - start) * 1000,
        'count': len(experiments)
    }
    print(f"Experiments ({year}): {len(experiments)} items in {results['experiments']['time_ms']:.2f}ms")
    
    if not experiments:
        return results
    
    # Benchmark: List locations
    experiment = experiments[0]
    exp_path = os.path.join(year_path, experiment)
    start = time.time()
    locations = [d for d in os.listdir(exp_path) 
                 if os.path.isdir(os.path.join(exp_path, d))]
    locations.sort()
    results['locations'] = {
        'time_ms': (time.time() - start) * 1000,
        'count': len(locations)
    }
    print(f"Locations ({year}/{experiment}): {len(locations)} items in {results['locations']['time_ms']:.2f}ms")
    
    if not locations:
        return results
    
    # Benchmark: List populations
    location = locations[0]
    loc_path = os.path.join(exp_path, location)
    start = time.time()
    populations = [d for d in os.listdir(loc_path) 
                   if os.path.isdir(os.path.join(loc_path, d))]
    populations.sort()
    results['populations'] = {
        'time_ms': (time.time() - start) * 1000,
        'count': len(populations)
    }
    print(f"Populations ({year}/{experiment}/{location}): {len(populations)} items in {results['populations']['time_ms']:.2f}ms")
    
    if not populations:
        return results
    
    # Benchmark: List dates
    population = populations[0]
    pop_path = os.path.join(loc_path, population)
    start = time.time()
    dates = [d for d in os.listdir(pop_path) 
             if os.path.isdir(os.path.join(pop_path, d))]
    dates.sort(reverse=True)
    results['dates'] = {
        'time_ms': (time.time() - start) * 1000,
        'count': len(dates)
    }
    print(f"Dates ({year}/{experiment}/{location}/{population}): {len(dates)} items in {results['dates']['time_ms']:.2f}ms")
    
    return results


def benchmark_database(service: DataService, filesystem_results: dict) -> dict:
    """
    Benchmark database query performance.
    Uses same hierarchy path as filesystem benchmark for comparison.
    """
    print("\n💾 Database Benchmark")
    print("-" * 40)
    
    results = {}
    
    # Benchmark: Get years
    start = time.time()
    years = service.get_years()
    results['years'] = {
        'time_ms': (time.time() - start) * 1000,
        'count': len(years)
    }
    print(f"Years: {len(years)} items in {results['years']['time_ms']:.2f}ms")
    
    if not years:
        return results
    
    # Benchmark: Get experiments (first run - uncached)
    year = years[0]
    start = time.time()
    experiments = service.get_experiments(year)
    results['experiments_uncached'] = {
        'time_ms': (time.time() - start) * 1000,
        'count': len(experiments)
    }
    print(f"Experiments ({year}, uncached): {len(experiments)} items in {results['experiments_uncached']['time_ms']:.2f}ms")
    
    # Benchmark: Get experiments (cached)
    start = time.time()
    experiments_cached = service.get_experiments(year)
    results['experiments_cached'] = {
        'time_ms': (time.time() - start) * 1000,
        'count': len(experiments_cached)
    }
    print(f"Experiments ({year}, cached): {len(experiments_cached)} items in {results['experiments_cached']['time_ms']:.2f}ms")
    
    if not experiments:
        return results
    
    # Benchmark: Get locations
    experiment = experiments[0]
    start = time.time()
    locations = service.get_locations(year, experiment)
    results['locations'] = {
        'time_ms': (time.time() - start) * 1000,
        'count': len(locations)
    }
    print(f"Locations ({year}/{experiment}): {len(locations)} items in {results['locations']['time_ms']:.2f}ms")
    
    if not locations:
        return results
    
    # Benchmark: Get populations
    location = locations[0]
    start = time.time()
    populations = service.get_populations(year, experiment, location)
    results['populations'] = {
        'time_ms': (time.time() - start) * 1000,
        'count': len(populations)
    }
    print(f"Populations ({year}/{experiment}/{location}): {len(populations)} items in {results['populations']['time_ms']:.2f}ms")
    
    if not populations:
        return results
    
    # Benchmark: Get dates
    population = populations[0]
    start = time.time()
    dates = service.get_dates(year, experiment, location, population)
    results['dates'] = {
        'time_ms': (time.time() - start) * 1000,
        'count': len(dates)
    }
    print(f"Dates ({year}/{experiment}/{location}/{population}): {len(dates)} items in {results['dates']['time_ms']:.2f}ms")
    
    return results


def compare_results(filesystem: dict, database: dict):
    """
    Compare filesystem vs database performance.
    Calculate speedup factors.
    """
    print("\n📊 Performance Comparison")
    print("=" * 60)
    
    total_fs_time = 0
    total_db_time = 0
    
    for key in filesystem.keys():
        if key in database:
            fs_time = filesystem[key]['time_ms']
            db_time = database[key]['time_ms']
            speedup = fs_time / db_time if db_time > 0 else 0
            
            total_fs_time += fs_time
            total_db_time += db_time
            
            print(f"\n{key.upper()}:")
            print(f"  Filesystem: {fs_time:.2f}ms")
            print(f"  Database:   {db_time:.2f}ms")
            print(f"  Speedup:    {speedup:.1f}x")
    
    # Check if cached experiment query exists
    if 'experiments_cached' in database:
        db_cached = database['experiments_cached']['time_ms']
        fs_exp = filesystem.get('experiments', {}).get('time_ms', 0)
        if fs_exp > 0:
            speedup_cached = fs_exp / db_cached if db_cached > 0 else 0
            print(f"\nEXPERIMENTS (CACHED):")
            print(f"  Filesystem: {fs_exp:.2f}ms")
            print(f"  Database:   {db_cached:.2f}ms")
            print(f"  Speedup:    {speedup_cached:.1f}x")
    
    # Overall comparison
    if total_db_time > 0:
        overall_speedup = total_fs_time / total_db_time
        print(f"\n{'='*60}")
        print(f"OVERALL:")
        print(f"  Total filesystem time: {total_fs_time:.2f}ms")
        print(f"  Total database time:   {total_db_time:.2f}ms")
        print(f"  Overall speedup:       {overall_speedup:.1f}x")
        print(f"{'='*60}")
        
        if overall_speedup >= 20:
            print("✅ Target performance achieved (20x+ improvement)!")
        elif overall_speedup >= 10:
            print("⚠️  Good performance (10x+ improvement)")
        else:
            print("⚠️  Performance below target (<10x improvement)")


def main():
    """Main benchmarking workflow"""
    print("=" * 60)
    print("🏁 GEMINI Performance Benchmark")
    print("=" * 60)
    
    data_root = DatabaseConfig.DATA_ROOT_DIR
    
    if not data_root:
        print("\n❌ ERROR: DATA_ROOT_DIR not set in .env.database")
        sys.exit(1)
    
    print(f"\nData root: {data_root}")
    print(f"Caching enabled: {DatabaseConfig.ENABLE_CACHING}")
    
    try:
        # Initialize database connection
        init_db()
        
        # Run filesystem benchmark
        fs_results = benchmark_filesystem(data_root)
        
        # Run database benchmark
        service = DataService()
        db_results = benchmark_database(service, fs_results)
        
        # Compare results
        compare_results(fs_results, db_results)
        
        print("\n✅ Benchmark complete!")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
