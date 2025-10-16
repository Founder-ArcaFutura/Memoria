import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memoria.utils.gravity import GravityScorer


def test_closer_embedding_outweighs_mass():
    scorer = GravityScorer()
    query_embedding = [0.0, 0.0]

    heavy_far = {
        "id": "heavy_far",
        "semantic_mass": 2.0,
        "embedding": [1.0, 0.0],
        "anchors": [],
        "emotional_intensity": 0.0,
    }
    light_close = {
        "id": "light_close",
        "semantic_mass": 1.0,
        "embedding": [0.1, 0.0],
        "anchors": [],
        "emotional_intensity": 0.0,
    }

    ranked = scorer.score(query_embedding, [heavy_far, light_close])
    assert ranked[0]["id"] == "light_close"


def test_heavier_mass_beats_equal_distance():
    scorer = GravityScorer()
    query_embedding = [0.0, 0.0]

    heavy = {
        "id": "heavy",
        "semantic_mass": 2.0,
        "embedding": [0.5, 0.0],
        "anchors": [],
        "emotional_intensity": 0.0,
    }
    light = {
        "id": "light",
        "semantic_mass": 1.0,
        "embedding": [0.5, 0.0],
        "anchors": [],
        "emotional_intensity": 0.0,
    }

    ranked = scorer.score(query_embedding, [light, heavy])
    assert ranked[0]["id"] == "heavy"


def test_anchor_overlap_boosts_score():
    scorer = GravityScorer(anchor_bonus=5.0)
    query_embedding = [0.0, 0.0]

    base = {
        "id": "base",
        "semantic_mass": 1.0,
        "embedding": [0.0, 0.0],
        "anchors": [],
        "emotional_intensity": 0.0,
    }
    anchored = {
        "id": "anchored",
        "semantic_mass": 1.0,
        "embedding": [0.0, 0.0],
        "anchors": ["alpha"],
        "emotional_intensity": 0.0,
    }

    ranked = scorer.score(query_embedding, [base, anchored], query_anchors=["alpha"])
    assert ranked[0]["id"] == "anchored"


def test_emotional_intensity_influences_ranking():
    scorer = GravityScorer(emotion_weight=1.0)
    query_embedding = [0.0, 0.0]

    calm = {
        "id": "calm",
        "semantic_mass": 1.0,
        "embedding": [0.0, 0.0],
        "anchors": [],
        "emotional_intensity": 0.1,
    }
    intense = {
        "id": "intense",
        "semantic_mass": 1.0,
        "embedding": [0.0, 0.0],
        "anchors": [],
        "emotional_intensity": 0.9,
    }

    ranked = scorer.score(query_embedding, [calm, intense])
    assert ranked[0]["id"] == "intense"
