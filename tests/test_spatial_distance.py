import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.utils.factories import memoria_from_entries, memory_entry


def test_spatial_distance_calculation_and_filtering():
    mem = memoria_from_entries(
        [
            memory_entry("origin", "origin", x=0.0, y=0.0, z=0.0),
            memory_entry("neg", "neg coords", x=-3.0, y=-4.0, z=0.0),
            memory_entry("threshold", "at threshold", x=3.0, y=4.0, z=0.0),
            memory_entry("just_out", "just out", x=3.0, y=4.0, z=1.0),
            memory_entry("far", "far away", x=10.0, y=0.0, z=0.0),
        ]
    )

    # Negative coordinate query
    neg_res = mem.retrieve_memories_near(-3.0, -4.0, 0.0, max_distance=0.1)
    assert len(neg_res) == 1
    neg = neg_res[0]
    expected_neg = math.sqrt(
        (neg["x"] + 3.0) ** 2 + (neg["y"] + 4.0) ** 2 + (neg["z"] - 0.0) ** 2
    )
    assert neg["distance"] == pytest.approx(expected_neg)

    # Axis offset query along z-axis
    offset_res = mem.retrieve_memories_near(0.0, 0.0, 1.0, max_distance=1.1)
    assert {r["text"] for r in offset_res} == {"origin"}
    assert offset_res[0]["distance"] == pytest.approx(1.0)

    # Near-threshold query includes boundary but excludes beyond
    near_res = mem.retrieve_memories_near(0.0, 0.0, 0.0, max_distance=5.0)
    texts = {r["text"] for r in near_res}
    assert texts == {"origin", "neg coords", "at threshold"}
    for r in near_res:
        expected = math.sqrt(r["x"] ** 2 + r["y"] ** 2 + r["z"] ** 2)
        assert r["distance"] == pytest.approx(expected)


def test_spatial_distance_2d_mode_ignores_temporal_axis():
    mem = memoria_from_entries(
        [
            memory_entry("origin", "origin", x=0.0, y=0.0, z=0.0),
            memory_entry("time", "time offset", x=4.0, y=0.0, z=0.0),
            memory_entry("mix", "spatial mix", x=0.0, y=0.6, z=0.8),
            memory_entry("far", "spatial far", x=0.0, y=3.0, z=4.0),
        ]
    )

    max_distance = 1.1
    three_d = mem.retrieve_memories_near(0.0, 0.0, 0.0, max_distance=max_distance)
    assert {r["text"] for r in three_d} == {"origin", "spatial mix"}

    for r in three_d:
        expected = math.sqrt(r["x"] ** 2 + r["y"] ** 2 + r["z"] ** 2)
        assert r["distance"] == pytest.approx(expected)

    two_d = mem.retrieve_memories_near(
        0.0, 0.0, 0.0, max_distance=max_distance, dimensions="2d"
    )
    assert {r["text"] for r in two_d} == {"origin", "time offset", "spatial mix"}
    assert len(two_d) == 3

    time_entry = next(r for r in two_d if r["text"] == "time offset")
    assert time_entry["distance"] == pytest.approx(0.0)

    mix_entry = next(r for r in two_d if r["text"] == "spatial mix")
    assert mix_entry["distance"] == pytest.approx(math.sqrt(0.6**2 + 0.8**2))

    helper_two_d = mem.retrieve_memories_near_2d(0.0, 0.0, max_distance=max_distance)
    assert {r["text"] for r in helper_two_d} == {"origin", "time offset", "spatial mix"}
    helper_time = next(r for r in helper_two_d if r["text"] == "time offset")
    assert helper_time["distance"] == pytest.approx(0.0)
    helper_mix = next(r for r in helper_two_d if r["text"] == "spatial mix")
    assert helper_mix["distance"] == pytest.approx(math.sqrt(0.6**2 + 0.8**2))

    trailing_whitespace = mem.retrieve_memories_near(
        0.0, 0.0, 0.0, max_distance=max_distance, dimensions="2d "
    )
    assert {r["text"] for r in trailing_whitespace} == {
        "origin",
        "time offset",
        "spatial mix",
    }
