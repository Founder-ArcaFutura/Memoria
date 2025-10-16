
import pytest


@pytest.fixture()
def admin_client(tmp_path, monkeypatch):
    import json
    from datetime import datetime

    from sqlalchemy import text

    db_path = tmp_path / "admin.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("MEMORIA_API_KEY", "secret")

    from memoria_server.api.app_factory import create_app

    app = create_app()
    app.config["TESTING"] = True
    memoria = app.config["memoria"]

    # Seed sample data across namespaces for filtering and search tests
    entries = [
        ("ops-1", "ops", "ops escalation record"),
        ("support-1", "support", "support desk handbook"),
        ("default-1", "default", "default knowledge base entry"),
    ]
    created_at = datetime.utcnow().isoformat()
    with memoria.db_manager.get_connection() as conn:
        for memory_id, namespace, text_content in entries:
            conn.execute(
                text(
                    """
                    INSERT INTO long_term_memory (
                        memory_id,
                        processed_data,
                        importance_score,
                        category_primary,
                        retention_type,
                        namespace,
                        timestamp,
                        created_at,
                        searchable_content,
                        summary,
                        novelty_score,
                        relevance_score,
                        actionability_score,
                        classification,
                        memory_importance,
                        extraction_timestamp
                    )
                    VALUES (
                        :memory_id,
                        :processed_data,
                        :importance_score,
                        :category_primary,
                        :retention_type,
                        :namespace,
                        :timestamp,
                        :created_at,
                        :searchable_content,
                        :summary,
                        :novelty_score,
                        :relevance_score,
                        :actionability_score,
                        :classification,
                        :memory_importance,
                        :extraction_timestamp
                    )
                    """
                ),
                {
                    "memory_id": memory_id,
                    "processed_data": json.dumps({"text": text_content}),
                    "importance_score": 0.5,
                    "category_primary": "general",
                    "retention_type": "long_term",
                    "namespace": namespace,
                    "timestamp": created_at,
                    "created_at": created_at,
                    "searchable_content": text_content,
                    "summary": f"summary of {text_content}",
                    "novelty_score": 0.5,
                    "relevance_score": 0.5,
                    "actionability_score": 0.5,
                    "classification": "conversational",
                    "memory_importance": "medium",
                    "extraction_timestamp": created_at,
                },
            )
        conn.commit()

    with app.test_client() as client:
        yield client, memoria


def _headers():
    return {"X-API-Key": "secret"}


def test_admin_tables_lists_metadata(admin_client):
    client, _ = admin_client
    response = client.get("/admin/tables", headers=_headers())
    assert response.status_code == 200
    data = response.get_json()
    tables = data.get("tables", [])
    names = {table["name"] for table in tables}
    assert "long_term_memory" in names
    long_term = next(table for table in tables if table["name"] == "long_term_memory")
    assert any(column["name"] == "namespace" for column in long_term.get("columns", []))
    assert long_term.get("row_count") is not None
    assert "default" in long_term.get("namespaces", [])


def test_admin_rows_supports_filters_and_search(admin_client):
    client, _ = admin_client
    namespace_resp = client.get(
        "/admin/tables/long_term_memory/rows?page=1&page_size=5&namespace=ops",
        headers=_headers(),
    )
    assert namespace_resp.status_code == 200
    namespace_data = namespace_resp.get_json()
    assert namespace_data["applied_filters"]["namespace"] == ["ops"]
    assert all(row["namespace"] == "ops" for row in namespace_data["rows"])

    search_resp = client.get(
        "/admin/tables/long_term_memory/rows?search=handbook&page_size=5",
        headers=_headers(),
    )
    assert search_resp.status_code == 200
    search_data = search_resp.get_json()
    assert any(
        "handbook" in str(value).lower()
        for row in search_data["rows"]
        for value in row.values()
    )


def test_admin_rows_missing_table_returns_404(admin_client):
    client, _ = admin_client
    response = client.get("/admin/tables/missing_table/rows", headers=_headers())
    assert response.status_code == 404


def test_session_endpoint_validates_key(admin_client):
    client, _ = admin_client
    ok = client.post("/session", json={"api_key": "secret"})
    assert ok.status_code == 200
    unauthorized = client.post("/session", json={"api_key": "wrong"})
    assert unauthorized.status_code == 401


def test_admin_row_mutations(admin_client):
    import json
    from urllib.parse import quote

    from sqlalchemy import text

    client, memoria = admin_client
    with memoria.db_manager.get_connection() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS admin_demo (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    namespace TEXT DEFAULT 'default',
                    notes TEXT
                )
                """
            )
        )
        conn.commit()

    create_resp = client.post(
        "/admin/tables/admin_demo/rows",
        headers=_headers(),
        json={"values": {"name": "alpha", "namespace": "ops"}},
    )
    assert create_resp.status_code == 200
    created = create_resp.get_json()
    assert created["status"] == "ok"
    pk = created.get("primary_key", {})
    assert isinstance(pk, dict)
    assert pk.get("id")

    list_resp = client.get(
        "/admin/tables/admin_demo/rows?page_size=10", headers=_headers()
    )
    assert list_resp.status_code == 200
    listed = list_resp.get_json()
    assert any(row["name"] == "alpha" for row in listed["rows"])

    pk_segment = quote(json.dumps({"id": pk["id"]}))
    update_resp = client.patch(
        f"/admin/tables/admin_demo/rows/{pk_segment}",
        headers=_headers(),
        json={"values": {"notes": "updated"}, "primary_key": {"id": pk["id"]}},
    )
    assert update_resp.status_code == 200
    assert update_resp.get_json()["updated"] == 1

    verify_resp = client.get(
        "/admin/tables/admin_demo/rows?page_size=10", headers=_headers()
    )
    assert any(row["notes"] == "updated" for row in verify_resp.get_json()["rows"])

    delete_resp = client.delete(
        f"/admin/tables/admin_demo/rows/{pk_segment}",
        headers=_headers(),
        json={"primary_key": {"id": pk["id"]}},
    )
    assert delete_resp.status_code == 200
    assert delete_resp.get_json()["deleted"] == 1

    final_resp = client.get(
        "/admin/tables/admin_demo/rows?page_size=10", headers=_headers()
    )
    assert all(row["id"] != pk["id"] for row in final_resp.get_json()["rows"])


def test_admin_row_datetime_parsing(admin_client):
    import json
    from urllib.parse import quote

    from sqlalchemy import text

    client, memoria = admin_client
    with memoria.db_manager.get_connection() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS admin_time_demo (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred DATETIME NOT NULL
                )
                """
            )
        )
        conn.commit()

    create_resp = client.post(
        "/admin/tables/admin_time_demo/rows",
        headers=_headers(),
        json={"values": {"occurred": "2024-01-02T03:04:05Z"}},
    )
    assert create_resp.status_code == 200
    created = create_resp.get_json()
    assert created["status"] == "ok"
    pk = created.get("primary_key", {})
    assert pk and "id" in pk

    rows_resp = client.get(
        "/admin/tables/admin_time_demo/rows?page_size=5", headers=_headers()
    )
    assert rows_resp.status_code == 200
    rows = rows_resp.get_json()["rows"]
    inserted = next(row for row in rows if row["id"] == pk["id"])
    assert inserted["occurred"] == "2024-01-02T03:04:05"

    pk_segment = quote(json.dumps({"id": pk["id"]}))
    update_resp = client.patch(
        f"/admin/tables/admin_time_demo/rows/{pk_segment}",
        headers=_headers(),
        json={
            "primary_key": {"id": pk["id"]},
            "values": {"occurred": "2025-06-07T08:09:10Z"},
        },
    )
    assert update_resp.status_code == 200
    assert update_resp.get_json()["updated"] == 1

    updated_rows_resp = client.get(
        "/admin/tables/admin_time_demo/rows?page_size=5", headers=_headers()
    )
    updated_rows = updated_rows_resp.get_json()["rows"]
    updated = next(row for row in updated_rows if row["id"] == pk["id"])
    assert updated["occurred"] == "2025-06-07T08:09:10"


def test_team_routes_manage_members(admin_client):
    client, memoria = admin_client
    memoria.team_memory_enabled = True
    memoria.team_enforce_membership = True
    memoria.storage_service.configure_team_policy(enforce_membership=True)

    headers = {"X-API-Key": "secret", "X-Memoria-User": "alice"}

    create_resp = client.post(
        "/memory/teams",
        headers=headers,
        json={
            "team_id": "ops",
            "members": ["alice"],
            "share_by_default": True,
            "include_members": True,
        },
    )
    assert create_resp.status_code == 200
    created = create_resp.get_json()
    assert created["team"]["namespace"] == "team:ops"
    assert "alice" in created["team"]["members"]

    list_resp = client.get("/memory/teams?include_members=1", headers=headers)
    assert list_resp.status_code == 200
    teams = list_resp.get_json()["teams"]
    assert any(team["team_id"] == "ops" for team in teams)

    add_resp = client.post(
        "/memory/teams/ops/members",
        headers=headers,
        json={"members": ["bob"], "role": "admin", "include_members": True},
    )
    assert add_resp.status_code == 200
    assert "bob" in add_resp.get_json()["team"]["admins"]

    remove_resp = client.delete(
        "/memory/teams/ops/members/bob?include_members=1", headers=headers
    )
    assert remove_resp.status_code == 200
    assert "bob" not in remove_resp.get_json()["team"]["members"]

    activate_resp = client.post("/memory/teams/ops/activate", headers=headers, json={})
    assert activate_resp.status_code == 200
    assert activate_resp.get_json()["active_team"] == "ops"

    namespaces_resp = client.get("/memory/teams/namespaces", headers=headers)
    assert namespaces_resp.status_code == 200
    namespaces = namespaces_resp.get_json()["namespaces"]
    assert "team:ops" in namespaces

    clear_resp = client.delete("/memory/teams/active", headers=headers)
    assert clear_resp.status_code == 200
    assert clear_resp.get_json()["active_team"] is None
