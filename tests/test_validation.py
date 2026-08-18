from src.inference.pipeline import (
    AUTO_ACCEPT_THRESHOLD,
    EVIDENCE_REQUIRED,
    PipelineConfig,
    REJECT_THRESHOLD,
    REVIEW_THRESHOLD,
    score_candidates,
    stable_edge_id,
    stable_semantic_id,
    validate_relation,
)


def test_stable_ids_are_deterministic():
    a = stable_semantic_id("PostgreSQL", "technology", "database")
    b = stable_semantic_id(" postgresql ", "TECHNOLOGY", "database")
    assert a == b
    assert a.startswith("n_") and len(a) == 66


def test_edge_id_is_deterministic():
    assert stable_edge_id("n_a", "uses", "n_b") == stable_edge_id("n_a", "uses", "n_b")


def test_canonical_weights_and_thresholds():
    config = PipelineConfig()
    config.validate()
    assert config.weights == {
        "semantic": 0.40,
        "lexical": 0.10,
        "structural": 0.20,
        "type_compatibility": 0.15,
        "evidence": 0.15,
    }
    assert REJECT_THRESHOLD == 0.70
    assert REVIEW_THRESHOLD == 0.80
    assert AUTO_ACCEPT_THRESHOLD == 0.92


def test_candidate_score_requires_auto_accept_at_092():
    source = {"id": "n_a"}
    candidate = {
        "id": "n_b",
        "semantic": 1,
        "lexical": 1,
        "structural": 1,
        "type_compatibility": 1,
        "evidence": 1,
    }
    result = score_candidates(source, [candidate], config=PipelineConfig())[0]
    assert result["overall"] == 1.0
    assert result["decision"] == "auto_accept"


def test_required_evidence_edge_is_rejected_without_evidence():
    ok, reason = validate_relation(
        {"id": "n_a"},
        {"id": "n_b"},
        "causes",
        evidence_ids=[],
        known_edges=[],
    )
    assert not ok
    assert reason == "evidence_required"
    assert "causes" in EVIDENCE_REQUIRED


def test_self_reference_rejected():
    ok, reason = validate_relation(
        {"id": "n_a"},
        {"id": "n_a"},
        "related_to",
        evidence_ids=[],
        known_edges=[],
    )
    assert not ok
    assert reason == "self_reference"
