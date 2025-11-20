"""
Benchmark Flask hierarchy endpoints - Database vs Filesystem
Compares performance of hierarchy queries with USE_DATABASE on vs off
"""
import requests
import time
import statistics
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
BASE_URL = "http://localhost:5058/flask_app"  # Flask backend port with mount point
NUM_ITERATIONS = 10  # Number of times to run each query

# Test data - use actual values from your database
TEST_DATA = {
    'year': '2025',
    'experiment': 'GEMINI',
    'location': 'Davis',
    'population': 'Legumes',
    'date': '2025-06-17',  # Use a date with multiple platforms
    'platform': 'Drone',
}


def benchmark_endpoint(method, url, data=None, iterations=NUM_ITERATIONS):
    """Benchmark a single endpoint"""
    times = []
    
    for i in range(iterations):
        start = time.time()
        
        if method == 'GET':
            response = requests.get(url)
        else:  # POST
            response = requests.post(url, json=data)
        
        elapsed = (time.time() - start) * 1000  # Convert to ms
        
        if response.status_code == 200:
            times.append(elapsed)
        else:
            print(f"  ❌ Request failed: {response.status_code}")
            return None
    
    return {
        'min': min(times),
        'max': max(times),
        'avg': statistics.mean(times),
        'median': statistics.median(times),
        'count': len(response.json().get('years', response.json().get('experiments', response.json().get('locations', response.json().get('populations', response.json().get('dates', response.json().get('platforms', response.json().get('sensors', []))))))))
    }


def run_benchmarks():
    """Run all benchmarks"""
    print("=" * 70)
    print("🏁 GEMINI Hierarchy Endpoints Benchmark")
    print("=" * 70)
    print(f"Base URL: {BASE_URL}")
    print(f"Iterations: {NUM_ITERATIONS}")
    print()
    
    endpoints = [
        {
            'name': 'Get Years',
            'method': 'GET',
            'url': f'{BASE_URL}/get_data_options',
            'data': None
        },
        {
            'name': 'Get Experiments',
            'method': 'POST',
            'url': f'{BASE_URL}/get_experiments',
            'data': {'year': TEST_DATA['year']}
        },
        {
            'name': 'Get Locations',
            'method': 'POST',
            'url': f'{BASE_URL}/get_locations',
            'data': {
                'year': TEST_DATA['year'],
                'experiment': TEST_DATA['experiment']
            }
        },
        {
            'name': 'Get Populations',
            'method': 'POST',
            'url': f'{BASE_URL}/get_populations',
            'data': {
                'year': TEST_DATA['year'],
                'experiment': TEST_DATA['experiment'],
                'location': TEST_DATA['location']
            }
        },
        {
            'name': 'Get Dates',
            'method': 'POST',
            'url': f'{BASE_URL}/get_dates',
            'data': {
                'year': TEST_DATA['year'],
                'experiment': TEST_DATA['experiment'],
                'location': TEST_DATA['location'],
                'population': TEST_DATA['population']
            }
        },
        {
            'name': 'Get Platforms',
            'method': 'POST',
            'url': f'{BASE_URL}/get_platforms',
            'data': {
                'year': TEST_DATA['year'],
                'experiment': TEST_DATA['experiment'],
                'location': TEST_DATA['location'],
                'population': TEST_DATA['population'],
                'date': TEST_DATA['date']
            }
        },
        {
            'name': 'Get Sensors',
            'method': 'POST',
            'url': f'{BASE_URL}/get_sensors',
            'data': {
                'year': TEST_DATA['year'],
                'experiment': TEST_DATA['experiment'],
                'location': TEST_DATA['location'],
                'population': TEST_DATA['population'],
                'date': TEST_DATA['date'],
                'platform': TEST_DATA['platform']
            }
        }
    ]
    
    results = []
    
    for endpoint in endpoints:
        print(f"📊 Testing: {endpoint['name']}")
        print(f"   URL: {endpoint['url']}")
        
        result = benchmark_endpoint(
            endpoint['method'],
            endpoint['url'],
            endpoint['data']
        )
        
        if result:
            print(f"   ✅ Avg: {result['avg']:.2f}ms | Min: {result['min']:.2f}ms | Max: {result['max']:.2f}ms | Median: {result['median']:.2f}ms")
            print(f"   📦 Items returned: {result['count']}")
            results.append({
                'name': endpoint['name'],
                **result
            })
        else:
            print(f"   ❌ Benchmark failed")
        
        print()
    
    # Summary
    if results:
        print("=" * 70)
        print("📈 SUMMARY")
        print("=" * 70)
        print(f"{'Endpoint':<20} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-" * 70)
        
        total_avg = 0
        for r in results:
            print(f"{r['name']:<20} {r['avg']:>10.2f}   {r['min']:>10.2f}   {r['max']:>10.2f}")
            total_avg += r['avg']
        
        print("-" * 70)
        print(f"{'TOTAL':<20} {total_avg:>10.2f}ms")
        print("=" * 70)
        
        return results
    
    return None


def main():
    try:
        # Test connection
        print("Testing connection to Flask backend...")
        response = requests.get(f'{BASE_URL}/get_data_options')
        if response.status_code != 200:
            print(f"❌ Failed to connect to {BASE_URL}")
            print("   Make sure the Flask server is running")
            sys.exit(1)
        
        print("✅ Connected successfully\n")
        
        # Run benchmarks
        results = run_benchmarks()
        
        if results:
            print("\n✅ Benchmark complete!")
            print("\nTo compare with database enabled:")
            print("1. Set USE_DATABASE=true in .env.database")
            print("2. Restart container: docker-compose restart app")
            print("3. Run this script again")
        else:
            print("\n❌ Benchmark failed")
            sys.exit(1)
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {BASE_URL}")
        print("   Make sure the Flask server is running")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
