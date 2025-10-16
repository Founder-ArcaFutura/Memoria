from __future__ import annotations


def test_blueprint_registration_and_legacy_routes(sample_client):
    client, _ = sample_client
    app = client.application
    assert isinstance(app.blueprints, dict)

    resp = client.get("/memory/search", query_string={"q": "alpha"})
    assert resp.status_code == 200
    assert "memories" in resp.get_json()
