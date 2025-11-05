import requests

url = 'http://127.0.0.1:5000/'

print("Test 1: Setosa Prediction")
print("="*50)
response = requests.post(url, data={
    'sepal_length': 5.1,
    'sepal_width': 3.5,
    'petal_length': 1.4,
    'petal_width': 0.2
})
print(f"Status Code: {response.status_code}")
if b'setosa' in response.content:
    print("✓ Setosa prediction found!")
else:
    print("✗ Prediction not found")

print("\n" + "="*50)
print("Test 2: Versicolor Prediction")
print("="*50)
response = requests.post(url, data={
    'sepal_length': 7.0,
    'sepal_width': 3.2,
    'petal_length': 4.7,
    'petal_width': 1.4
})
print(f"Status Code: {response.status_code}")
if b'versicolor' in response.content:
    print("✓ Versicolor prediction found!")
else:
    print("✗ Prediction not found")

print("\n" + "="*50)
print("Test 3: Virginica Prediction")
print("="*50)
response = requests.post(url, data={
    'sepal_length': 6.3,
    'sepal_width': 3.3,
    'petal_length': 6.0,
    'petal_width': 2.5
})
print(f"Status Code: {response.status_code}")
if b'virginica' in response.content:
    print("✓ Virginica prediction found!")
else:
    print("✗ Prediction not found")

print("\n" + "="*50)
print("Test 4: Home Page GET Request")
print("="*50)
response = requests.get(url)
print(f"Status Code: {response.status_code}")
if b'Iris Flower Classifier' in response.content:
    print("✓ Home page loaded successfully!")
else:
    print("✗ Home page not found")

