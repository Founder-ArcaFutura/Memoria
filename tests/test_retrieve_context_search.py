from memoria.storage.service import StorageService


class DummyDBManager:
    pass


class DummySearchEngine:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def execute_search(self, query, db_manager, namespace, limit):
        self.calls.append({"query": query, "limit": limit})
        return self.results[:limit]


class RecordingDBManager:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def search_memories(self, query, namespace, limit, team_id=None, **kwargs):
        self.calls.append(
            {
                "query": query,
                "namespace": namespace,
                "team_id": team_id,
                "limit": limit,
            }
        )
        key = (team_id, namespace)
        results = self.responses.get(key, [])
        return [dict(item) for item in results[:limit]]


def test_retrieve_context_combines_and_limits(monkeypatch):
    search_results = [
        {"memory_id": "s1"},
        {"memory_id": "e1"},
        {"memory_id": "s2"},
    ]
    search_engine = DummySearchEngine(search_results)
    service = StorageService(
        db_manager=DummyDBManager(),
        search_engine=search_engine,
        conscious_ingest=True,
    )

    monkeypatch.setattr(
        service,
        "get_essential_conversations",
        lambda limit: [{"memory_id": "e1"}],
    )

    context = service.retrieve_context("query", limit=3)

    assert search_engine.calls[0]["limit"] == 2
    ids = [item["memory_id"] for item in context]
    assert ids == ["e1", "s1"]
    assert len(ids) <= 3
    assert len(set(ids)) == len(ids)


def test_retrieve_context_respects_limit_with_multiple_essentials(monkeypatch):
    search_engine = DummySearchEngine([{"memory_id": "s1"}])
    service = StorageService(
        db_manager=DummyDBManager(),
        search_engine=search_engine,
        conscious_ingest=True,
    )

    captured_limits = []

    def fake_essentials(limit):
        captured_limits.append(limit)
        return [
            {"memory_id": "e1"},
            {"memory_id": "e2"},
            {"memory_id": "e3"},
        ]

    monkeypatch.setattr(service, "get_essential_conversations", fake_essentials)

    context = service.retrieve_context("query", limit=2)

    assert captured_limits == [2]
    assert len(context) == 2
    assert all(item["memory_id"].startswith("e") for item in context)


def test_retrieve_context_uses_team_specific_namespace():
    responses = {
        (None, "personal"): [{"memory_id": "personal-1"}],
        ("team-alpha", "team-space-alpha"): [{"memory_id": "team-alpha-1"}],
    }
    db_manager = RecordingDBManager(responses)
    service = StorageService(db_manager=db_manager, namespace="personal")
    service.register_team_space(
        "team-alpha",
        namespace="team-space-alpha",
        members=["member"],
        share_by_default=True,
    )

    personal_context = service.retrieve_context("topic", user_id="member")
    assert personal_context == [{"memory_id": "personal-1"}]
    assert db_manager.calls[-1]["namespace"] == "personal"
    assert db_manager.calls[-1]["team_id"] is None

    team_context = service.retrieve_context(
        "topic", team_id="team-alpha", user_id="member"
    )
    assert team_context == [{"memory_id": "team-alpha-1"}]
    assert db_manager.calls[-1]["namespace"] == "team-space-alpha"
    assert db_manager.calls[-1]["team_id"] == "team-alpha"


def test_retrieve_context_defaults_to_service_team_id():
    responses = {
        ("team-default", "personal"): [{"memory_id": "team-default-1"}],
    }
    db_manager = RecordingDBManager(responses)
    service = StorageService(
        db_manager=db_manager,
        namespace="personal",
        team_id="team-default",
    )

    context = service.retrieve_context("topic")

    assert context == [{"memory_id": "team-default-1"}]
    assert db_manager.calls[-1]["namespace"] == "personal"
    assert db_manager.calls[-1]["team_id"] == "team-default"
