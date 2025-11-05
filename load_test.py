import requests
import time
import json

def load_test(num_requests=10):
    """Test app with multiple simultaneous requests"""
    
    url = 'http://127.0.0.1:5000/'
    
    test_data = [
        {'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2},
        {'sepal_length': 7.0, 'sepal_width': 3.2, 'petal_length': 4.7, 'petal_width': 1.4},
        {'sepal_length': 6.3, 'sepal_width': 3.3, 'petal_length': 6.0, 'petal_width': 2.5},
    ]
    
    print(f"Starting load test with {num_requests} requests...\n")
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    for i in range(num_requests):
        try:
            data = test_data[i % len(test_data)]
            response = requests.post(url, data=data, timeout=5)
            
            if response.status_code == 200:
                successful += 1
                print(f"✓ Request {i+1}: Success (Status 200)")
            else:
                failed += 1
                print(f"✗ Request {i+1}: Failed (Status {response.status_code})")
        except Exception as e:
            failed += 1
            print(f"✗ Request {i+1}: Error - {str(e)}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*50}")
    print(f"Load Test Results:")
    print(f"{'='*50}")
    print(f"Total Requests: {num_requests}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(successful/num_requests)*100:.2f}%")
    print(f"Total Time: {duration:.2f} seconds")
    print(f"Avg Time per Request: {(duration/num_requests)*1000:.2f} ms")
    print(f"{'='*50}")

if __name__ == '__main__':
    # Make sure Flask app is running before executing this
    load_test(num_requests=20)
