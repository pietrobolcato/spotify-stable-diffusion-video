# -*- coding: utf-8 -*-
"""This module include tests for the fastapi server"""

from fastapi.testclient import TestClient
from inference_server.app import app
import pytest


@pytest.fixture
def client():
    # use "with" statement to run "startup" event of FastAPI
    with TestClient(app) as c:
        yield c


def test_wrong_request(client):
    """Tests a request with wrong json format"""

    headers = {}
    body = {"wrong_key": "wrong_request"}

    response = client.post("/api/v1/predict", headers=headers, json=body)

    try:
        assert response.status_code == 422
    except AssertionError:
        print(response.status_code)
        print(response.json())
        raise


def test_correct_request(client):
    """Tests a request with correct json format"""

    headers = {}
    body = {
        "preset": "as_it_was",
        "init_image": "https://i.ibb.co/7zm8Bw2/spotify-img-test.jpg",
    }

    response = client.post("/api/v1/predict", headers=headers, json=body)

    try:
        assert response.status_code == 200
        assert "video" in response.json()
        assert "elapsed_time" in response.json()
    except AssertionError:
        print(response.status_code)
        print(response.json())
        raise
