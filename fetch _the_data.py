import requests
import json

# Prometheus query URL
prometheus_url = 'http://localhost:9090/api/v1/query'
query = 'your_prometheus_query_here'  # Modify this to match your Prometheus query

# ML model API URL
ml_model_url = 'http://localhost:8000/predict'

def query_prometheus(query):
    response = requests.get(prometheus_url, params={'query': query})
    if response.status_code == 200:
        result = response.json()['data']['result']
        return result
    else:
        raise Exception('Failed to query Prometheus')

def send_data_to_ml_model(data):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(ml_model_url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception('Failed to get prediction from ML model')

def main():
    try:
        prometheus_data = query_prometheus(query)
        
        # Assuming prometheus_data is a list of dictionaries with keys corresponding to sensor names
        sensor_data = {
            "sensor1": prometheus_data[0]['value'][1],
            "sensor2": prometheus_data[1]['value'][1],
            "sensor3": prometheus_data[2]['value'][1],
            "temp": prometheus_data[3]['value'][1],
            "humidity": prometheus_data[4]['value'][1],
        }
        
        prediction = send_data_to_ml_model(sensor_data)
        print("Prediction:", prediction)
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    main()
