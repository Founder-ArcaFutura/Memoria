import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memoria.core.memory import Memoria


@pytest.mark.parametrize(
    "db_url",
    [
        "sqlite:///:memory:",
        pytest.param(
            os.getenv("TEST_POSTGRES_URL"),
            marks=pytest.mark.skipif(
                os.getenv("TEST_POSTGRES_URL") is None,
                reason="TEST_POSTGRES_URL not set",
            ),
        ),
        pytest.param(
            os.getenv("TEST_MYSQL_URL"),
            marks=pytest.mark.skipif(
                os.getenv("TEST_MYSQL_URL") is None,
                reason="TEST_MYSQL_URL not set",
            ),
        ),
    ],
)
def test_spatial_retrieval_with_anchor(db_url: str) -> None:
    namespace = f"anchor_cross_db_{uuid.uuid4().hex}"
    mem = Memoria(database_connect=db_url, namespace=namespace)
    mem.db_manager.clear_memory(namespace)
    mem.store_memory(
        anchor="a",
        text="near",
        tokens=1,
        x_coord=0.0,
        y=0.0,
        z=0.0,
        symbolic_anchors=["A"],
    )
    mem.store_memory(
        anchor="a",
        text="far",
        tokens=1,
        x_coord=10.0,
        y=10.0,
        z=10.0,
        symbolic_anchors=["A"],
    )
    mem.store_memory(
        anchor="b",
        text="other",
        tokens=1,
        x_coord=1.0,
        y=1.0,
        z=1.0,
        symbolic_anchors=["B"],
    )

    results = mem.retrieve_memories_near(
        0.0, 0.0, 0.0, max_distance=5.0, anchor=["A", "B"]
    )
    texts = {r["text"] for r in results}
    assert texts == {"near", "other"}
    mem.db_manager.clear_memory(namespace)
    mem.db_manager.close()
