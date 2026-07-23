import requests
import json
import time

# Update this URL to your live AWS App Runner URL once deployed
API_URL = "http://127.0.0.1:8000/predict"

# Construct a mock multimodal payload matching the api/schemas.py Pydantic model
payload = {
    # 128-d tokenized text sequence (Mock DistilBERT input)
    "text_input_ids": [101, 2023, 2003, 1037, 3185, 2028] + [0] * 122, 
    
    # 30-day historical sales volume (Mock M5 Time-Series data)
    "temporal_history": [
        12.0, 15.0, 14.0, 18.0, 22.0, 25.0, 20.0, 19.0, 21.0, 23.0,
        28.0, 35.0, 30.0, 25.0, 20.0, 18.0, 15.0, 12.0, 14.0, 16.0,
        19.0, 22.0, 24.0, 28.0, 31.0, 33.0, 29.0, 25.0, 21.0, 19.0
    ],
    
    # 15-d tabular metadata (Mock pricing, discount, and cluster IDs)
    "tabular_features": [
        19.99, 0.15, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 
        0.0, 0.0, 0.5, 0.8, 1.2, 0.9, 1.1
    ]
}

print(f"Sending POST request to {API_URL}...")
start_time = time.time()

try:
    response = requests.post(
        API_URL, 
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    response.raise_for_status()
    
    latency = time.time() - start_time
    print(f"\n✅ Success! (Round-trip time: {latency * 1000:.2f} ms)")
    print(json.dumps(response.json(), indent=4))
    
except requests.exceptions.ConnectionError:
    print("❌ Error: Could not connect to API. Ensure Uvicorn or your Docker container is running.")
except requests.exceptions.HTTPError as e:
    print(f"❌ HTTP Error: {e}")
    print(response.text)