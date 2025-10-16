"""Starter datasets packaged with Memoria's evaluation harness."""

from importlib import resources

__all__ = ["list_datasets"]


def list_datasets() -> list[str]:
    """Return dataset filenames available in the starter collection."""

    package = __name__
    with resources.as_file(resources.files(package)) as dataset_root:
        return sorted(
            entry.name
            for entry in dataset_root.iterdir()
            if entry.is_file() and entry.suffix in {".jsonl", ".json"}
        )
