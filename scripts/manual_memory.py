from __future__ import annotations

import os

import requests
from memoria.schemas import MemoryEntry, validate_memory_entry
from memoria.utils.pydantic_compat import model_dump


def _prompt_float(label: str) -> float | None:
    val = input(f"{label}: ").strip()
    return float(val) if val else None


def main() -> None:
    """Interactive CLI wizard to store a memory via the API."""
    anchor = input("Anchor: ").strip()
    text = input("Text: ").strip()
    x_coord = _prompt_float("x_coord (days offset, optional)")
    y_coord = _prompt_float("y_coord (-15 to 15, optional)")
    z_coord = _prompt_float("z_coord (-15 to 15, optional)")
    symbolic_raw = input("Symbolic anchors (comma separated, optional): ").strip()
    symbolic = [s.strip() for s in symbolic_raw.split(",") if s.strip()] or None

    data = {
        "anchor": anchor,
        "text": text,
        "tokens": len(text.split()),
        "x_coord": x_coord,
        "y_coord": y_coord,
        "z_coord": z_coord,
        "symbolic_anchors": symbolic,
    }

    entry = validate_memory_entry(data)

    api_url = os.getenv("MEMORIA_API_URL", "http://localhost:8080/memory")
    headers = {}
    api_key = os.getenv("MEMORIA_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key

    resp = requests.post(api_url, json=model_dump(entry), headers=headers)
    try:
        print(resp.json())
    except Exception:
        print(resp.text)


if __name__ == "__main__":
    main()
