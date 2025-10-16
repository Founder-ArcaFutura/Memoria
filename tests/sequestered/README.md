This folder now participates in the default pytest run. The suite primarily exercises
heuristics, spatial queries, and other behaviours that were previously skipped. Two
files still rely on optional infrastructure and therefore self-skip when the
environment is not configured:

- `test_anchor_cross_db.py` requires PostgreSQL and MySQL URLs.
- `test_benchmark_tools.py` requires the optional `pyarrow` dependency.
- `test_spatial_anchor_parsing.py` (and its shared fixture module) require `flask` and
  will self-skip if the API stack is not installed.

All other tests run against the default SQLite configuration and should be kept
green.
