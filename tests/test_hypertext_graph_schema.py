from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from memoria.schemas import HYPERTEXT_GRAPH_SCHEMA, load_hypertext_graph_schema

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "memoria"
    / "schemas"
    / "hypertext_graph.json"
)


def test_hypertext_graph_schema_is_valid_and_exported() -> None:
    raw_schema = json.loads(SCHEMA_PATH.read_text())
    loaded_schema = load_hypertext_graph_schema()

    # The JSON Schema should validate under the 2020-12 draft specification.
    Draft202012Validator.check_schema(loaded_schema)

    # Ensure exported helpers match the package resource and keep a safe copy.
    assert loaded_schema == raw_schema
    assert HYPERTEXT_GRAPH_SCHEMA == raw_schema
    assert load_hypertext_graph_schema() is not loaded_schema

    # Structural smoke-tests for the most important sections.
    assert set(loaded_schema["required"]) == {"nodes", "edges"}

    nodes = loaded_schema["properties"]["nodes"]
    assert nodes["items"] == {"$ref": "#/$defs/Node"}

    node_schema = loaded_schema["$defs"]["Node"]
    assert set(node_schema["required"]) == {"id", "label", "kind"}

    coordinates = node_schema["properties"]["coordinates"]
    assert set(coordinates["required"]) == {"x", "y", "z"}
    assert coordinates["additionalProperties"] is False

    edge_schema = loaded_schema["$defs"]["Edge"]
    assert set(edge_schema["required"]) == {"id", "source", "target", "relation"}

    metadata_schema = loaded_schema["$defs"]["Metadata"]
    assert "generated_at" in metadata_schema["required"]
    assert loaded_schema["properties"]["metadata"]["$ref"] == "#/$defs/Metadata"
