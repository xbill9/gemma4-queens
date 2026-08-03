import requests
import json

def test_inference():
    url = "http://18.246.7.176:8080/v1/chat/completions"
    payload = {
        "model": "google/gemma-4-E4B-it",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 50,
        "temperature": 0.0
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response JSON:")
            print(json.dumps(response.json(), indent=2))
        else:
            print("Error Response:")
            print(response.text)
    except Exception as e:
        print(f"Exception during request: {e}")

if __name__ == "__main__":
    test_inference()
