from collections.abc import Iterable

from memoria.core.memory import Memoria


def memory_entry(
    anchor: str,
    text: str,
    *,
    tokens: int = 1,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    symbolic_anchors: list[str] | None = None,
    emotional_intensity: float | None = None,
) -> dict:
    """Create a memory parameter dictionary for tests."""
    return {
        "anchor": anchor,
        "text": text,
        "tokens": tokens,
        "x_coord": x,
        "y": y,
        "z": z,
        "symbolic_anchors": symbolic_anchors,
        "emotional_intensity": emotional_intensity,
    }


def memoria_from_entries(entries: Iterable[dict]) -> Memoria:
    """Create a Memoria instance and populate it with provided entries."""
    mem = Memoria(database_connect="sqlite:///:memory:")
    for params in entries:
        mem.store_memory(
            anchor=params["anchor"],
            text=params["text"],
            tokens=params.get("tokens", 1),
            x_coord=params.get("x_coord"),
            y=params.get("y"),
            z=params.get("z"),
            symbolic_anchors=params.get("symbolic_anchors"),
            emotional_intensity=params.get("emotional_intensity"),
        )
    return mem
