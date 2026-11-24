"""Strict acceptance tests for the risk scoring engine."""

import copy

import pytest

from engine.risk.score import (
    decide,
    explain,
    missing_policy,
    normalize_feature,
    risk_batch,
    score,
    validate_model,
    weighted_sum,
)


MODEL = {
    "features": {
        "transaction_risk": {"low": 0.0, "high": 100.0, "direction": "up"},
        "device_risk": {"low": 0.0, "high": 10.0, "direction": "up"},
        "account_age": {"low": 0.0, "high": 5.0, "direction": "down"},
    },
    "weights": {
        "transaction_risk": 3.0,
        "device_risk": 2.0,
        "account_age": 1.0,
    },
    "missing": {
        "transaction_risk": "worst",
        "device_risk": "neutral",
        "account_age": "best",
    },
    "thresholds": [
        {"max": 0.3, "action": "allow"},
        {"max": 0.7, "action": "review"},
        {"max": 1.0, "action": "block"},
    ],
}


def test_normalize_respects_direction():
    up = {"low": 0.0, "high": 100.0, "direction": "up"}
    down = {"low": 0.0, "high": 100.0, "direction": "down"}

    assert normalize_feature(20.0, up) == pytest.approx(0.2)
    assert normalize_feature(80.0, up) == pytest.approx(0.8)
    assert normalize_feature(80.0, up) > normalize_feature(20.0, up)

    assert normalize_feature(20.0, down) == pytest.approx(0.8)
    assert normalize_feature(80.0, down) == pytest.approx(0.2)
    assert normalize_feature(20.0, down) > normalize_feature(80.0, down)


def test_normalize_clamps_to_bounds():
    spec = {"low": 10.0, "high": 20.0, "direction": "up"}

    assert normalize_feature(-100.0, spec) == 0.0
    assert normalize_feature(1_000.0, spec) == 1.0
    assert normalize_feature(30.0, spec) == 1.0

    down = {"low": 10.0, "high": 20.0, "direction": "down"}
    assert normalize_feature(-100.0, down) == 1.0
    assert normalize_feature(1_000.0, down) == 0.0


def test_weighted_sum_normalizes_weights():
    assert weighted_sum({"a": 1.0}, {"a": 5.0}) == pytest.approx(1.0)

    aggregate = weighted_sum({"a": 1.0, "b": 1.0}, {"a": 2.0, "b": 3.0})
    assert aggregate == pytest.approx(1.0)

    assert weighted_sum({"a": 1.0}, {"a": 2.0}) == weighted_sum(
        {"a": 1.0}, {"a": 7.0}
    )


def test_missing_policy_does_not_treat_missing_as_zero():
    filled = missing_policy(
        {"transaction_risk": None, "device_risk": 0.2},
        {"transaction_risk": "worst", "device_risk": "neutral"},
    )

    assert filled["transaction_risk"] == pytest.approx(1.0)
    assert filled["transaction_risk"] != 0.0
    assert filled["device_risk"] == pytest.approx(0.2)


def test_contributions_use_normalized_weighted_values():
    raw = {
        "transaction_risk": 50.0,
        "device_risk": 5.0,
        "account_age": 2.5,
    }
    result = score(raw, MODEL)
    contributions = result["contributions"]

    assert contributions["transaction_risk"] == pytest.approx(0.25)
    assert contributions["device_risk"] == pytest.approx(1.0 / 6.0)
    assert contributions["account_age"] == pytest.approx(1.0 / 12.0)
    assert contributions["transaction_risk"] != pytest.approx(25.0)
    assert sum(contributions.values()) == pytest.approx(result["score"], abs=1e-9)


def test_threshold_boundary_does_not_fall_through():
    thresholds = MODEL["thresholds"]

    assert decide(0.3, thresholds) == "review"
    assert decide(0.7, thresholds) == "block"
    assert decide(1.0, thresholds) == "block"


def test_explain_sorts_by_absolute_contribution():
    contributions = {"small": 0.2, "negative": -0.9, "medium": 0.5}

    assert explain(contributions, 2) == [("negative", -0.9), ("medium", 0.5)]
    assert explain(contributions, 10) == [
        ("negative", -0.9),
        ("medium", 0.5),
        ("small", 0.2),
    ]


def test_validate_model_detects_non_monotonic_thresholds():
    bad = copy.deepcopy(MODEL)
    bad["thresholds"] = [
        {"max": 0.7, "action": "allow"},
        {"max": 0.3, "action": "review"},
        {"max": 1.0, "action": "block"},
    ]

    assert "thresholds_non_monotonic" in validate_model(bad)


def test_validate_model_detects_non_positive_weights():
    bad = copy.deepcopy(MODEL)
    bad["weights"]["transaction_risk"] = -0.5

    assert "weights_non_positive" in validate_model(bad)


def test_batch_does_not_reuse_first_batch_normalization():
    first = {
        "transaction_risk": 10.0,
        "device_risk": 5.0,
        "account_age": 2.5,
    }
    second = {
        "transaction_risk": 90.0,
        "device_risk": 5.0,
        "account_age": 2.5,
    }

    result = risk_batch({"first": first, "second": second}, MODEL)

    assert result["second"]["contributions"]["transaction_risk"] > (
        result["first"]["contributions"]["transaction_risk"]
    )
    assert result["second"]["score"] > result["first"]["score"] + 0.1


def test_total_score_is_non_decreasing_for_each_risk_feature():
    base = {
        "transaction_risk": 10.0,
        "device_risk": 2.0,
        "account_age": 3.0,
    }
    risk_increasing_values = {
        "transaction_risk": [0.0, 25.0, 50.0, 75.0, 100.0],
        "device_risk": [0.0, 2.5, 5.0, 7.5, 10.0],
        "account_age": [5.0, 4.0, 3.0, 2.0, 0.0],
    }

    for feature, values in risk_increasing_values.items():
        previous = float("-inf")
        previous_contribution = float("-inf")
        for value in values:
            raw = dict(base)
            raw[feature] = value
            result = score(raw, MODEL)
            current = result["score"]
            contribution = result["contributions"][feature]
            assert current + 1e-12 >= previous
            assert contribution + 1e-12 >= previous_contribution
            previous = current
            previous_contribution = contribution


def test_contributions_sum_to_total():
    raw = {
        "transaction_risk": 40.0,
        "device_risk": 3.0,
        "account_age": 1.0,
    }
    result = score(raw, MODEL)

    assert set(result["contributions"]) == set(MODEL["features"])
    assert abs(sum(result["contributions"].values()) - result["score"]) <= 1e-9


def test_missing_policy_applies_configured_policy():
    policy = {
        "transaction_risk": "worst",
        "device_risk": "best",
        "account_age": {"policy": "constant", "value": 0.65},
    }
    filled = missing_policy(
        {
            "transaction_risk": None,
            "device_risk": None,
            "account_age": None,
        },
        policy,
    )

    assert filled["transaction_risk"] == 1.0
    assert filled["device_risk"] == 0.0
    assert filled["account_age"] == pytest.approx(0.65)
    assert missing_policy({"x": None}, {"x": "neutral"})["x"] == 0.5

    all_missing_score = score(
        {
            "transaction_risk": None,
            "device_risk": None,
            "account_age": None,
        },
        MODEL,
    )["score"]
    assert all_missing_score == pytest.approx(2.0 / 3.0)


def test_threshold_boundary_ownership_is_fixed():
    thresholds = [
        {"max": 0.25, "action": "low"},
        {"max": 0.5, "action": "mid"},
        {"max": 1.0, "action": "high"},
    ]

    assert decide(0.0, thresholds) == "low"
    assert decide(0.2499, thresholds) == "low"
    assert decide(0.25, thresholds) == "mid"
    assert decide(0.4999, thresholds) == "mid"
    assert decide(0.5, thresholds) == "high"
    assert decide(1.0, thresholds) == "high"


def test_batches_match_independent_scores():
    first = {
        "transaction_risk": 5.0,
        "device_risk": 1.0,
        "account_age": 4.0,
    }
    second = {
        "transaction_risk": 95.0,
        "device_risk": 9.0,
        "account_age": 0.5,
    }

    batched = risk_batch({"first": first, "second": second}, MODEL)
    independent_first = score(first, MODEL)
    independent_second = score(second, MODEL)

    assert batched["first"]["score"] == pytest.approx(
        independent_first["score"], abs=1e-12
    )
    assert batched["second"]["score"] == pytest.approx(
        independent_second["score"], abs=1e-12
    )
    assert batched["first"]["contributions"] == pytest.approx(
        independent_first["contributions"], abs=1e-12
    )
    assert batched["second"]["contributions"] == pytest.approx(
        independent_second["contributions"], abs=1e-12
    )