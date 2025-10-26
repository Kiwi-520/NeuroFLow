import requests
import json

def test_emotion_analysis():
    url = "http://localhost:8000/api/emotions/analyze"
    headers = {"Content-Type": "application/json"}
    data = {"text": "I am happy"}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_emotion_analysis()