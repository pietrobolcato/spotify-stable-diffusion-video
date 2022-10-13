# -*- coding: utf-8 -*-
"""This module include tests for the flask server"""

from flask_server.app import app


def test_wrong_request():
    """Tests a request with wrong json format"""

    with app.test_client() as test_client:
        response = test_client.post("/api", json={"wrong_key": "wrong_request"})

        assert response.status_code == 400


def test_correct_request():
    """Tests a request with the correct json format"""

    with app.test_client() as test_client:
        response = test_client.post(
            "/api",
            json={
                "preset": "as_it_was",
                "init_image": "https://i.ibb.co/7zm8Bw2/spotify-img-test.jpg",
                "dev_mode": True,
            },
        )

        assert response.status_code == 200
        assert "video" in response.get_json(force=True)
        assert "elapsed_time" in response.get_json(force=True)
