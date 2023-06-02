import pytest
import requests

def test_server_running():
    response = requests.get('http://localhost:5002')
    assert response.status_code == 200


def test_predict_route():
    data = {'data': [5.1, 3.5, 1.4, 0.2]}  # example data
    response = requests.post('http://localhost:5002/predict', json=data)
    assert response.status_code == 200