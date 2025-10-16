import json
import os
from pathlib import Path


def load_inputs(json_file_path: str, limit: int | None = None) -> list[str]:
    """
    Load test inputs from JSON file and return as a list of strings.

    Args:
        json_file_path: Path to the JSON file
        limit: Optional limit on number of inputs to load (None = load all)

    Returns:
        list of user input strings
    """
    with open(json_file_path) as f:
        data = json.load(f)

    user_inputs = data.get("user_input", {})

    # Sort by numeric key and return just the values
    sorted_keys = sorted(user_inputs.keys(), key=lambda x: int(x))

    # Apply limit if specified
    if limit is not None and limit > 0:
        sorted_keys = sorted_keys[:limit]

    return [user_inputs[key] for key in sorted_keys]


def load_test_config(config_path: Path | None = None) -> dict[str, str]:
    """Load test configuration.

    The configuration provides default values for tests such as the LLM model
    name. Values can be overridden by environment variables when running tests
    locally.

    Args:
        config_path: Optional path to the configuration file. Defaults to
            ``tests/config.json`` relative to this file.

    Returns:
        Dictionary containing configuration values with environment variable
        overrides applied.
    """

    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.json"

    with open(config_path) as f:
        config = json.load(f)

    config["model"] = os.getenv("TEST_MODEL", config.get("model", "gpt-4o-mini"))
    return config


def get_test_model(config_path: Path | None = None) -> str:
    """Convenience helper to fetch the configured test model name."""
    return load_test_config(config_path)["model"]
