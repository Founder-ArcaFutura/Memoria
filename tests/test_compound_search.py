from __future__ import annotations


def test_compound_filters(sample_client):
    client, ids = sample_client
    resp = client.get(
        "/memory/search",
        query_string={
            "keyword": "alpha",
            "category": "work",
            "x": "0",
            "y": "0",
            "z": "0",
            "max_distance": "5",
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()["memories"]
    returned_ids = {m["memory_id"] for m in data}
    assert ids["work_recent"] in returned_ids
    assert ids["personal"] not in returned_ids
    assert ids["work_old"] not in returned_ids
