"""Risk scoring engine.

The public surface is the eight functions in this module.  The task is to
repair their behaviour without changing the tests.
"""

# Policy names that map directly to a normalized-scale fill value.
_POLICY_VALUES = {
    "worst": 1.0,
    "best": 0.0,
    "neutral": 0.5,
}

# Problem codes in their stable, canonical order.
_PROBLEM_CODES = (
    "features_empty",
    "weights_keys_mismatch",
    "weights_non_positive",
    "bounds_invalid",
    "direction_missing",
    "missing_policy_invalid",
    "thresholds_invalid",
    "thresholds_non_monotonic",
)


def _is_real_number(value) -> bool:
    """Return True for plain int/float values (booleans excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _clamp_unit(value: float) -> float:
    """Clamp a value onto the closed interval [0.0, 1.0]."""
    return min(1.0, max(0.0, float(value)))


def _policy_fill_value(rule) -> float:
    """Resolve one missing-value policy to a normalized-scale number.

    Accepted rules are the strings ``"worst"``/``"best"``/``"neutral"``, a
    number in [0.0, 1.0], or ``{"policy": "constant", "value": number}``.
    Numeric constants are clamped to [0.0, 1.0] so an out-of-range policy
    can never inflate a total score.
    """
    if isinstance(rule, str):
        return _POLICY_VALUES[rule]
    if isinstance(rule, dict):
        value = rule["value"]
    else:
        value = rule
    return _clamp_unit(value)


def _is_valid_policy(rule) -> bool:
    """Check whether a missing-policy entry is well formed."""
    if isinstance(rule, str):
        return rule in _POLICY_VALUES
    if isinstance(rule, dict):
        if rule.get("policy") != "constant":
            return False
        value = rule.get("value")
        return _is_real_number(value) and 0.0 <= float(value) <= 1.0
    if _is_real_number(rule):
        return 0.0 <= float(rule) <= 1.0
    return False


def normalize_feature(value: float, spec: dict) -> float:
    """Normalize one raw feature into the closed interval [0.0, 1.0].

    ``spec`` has ``low``, ``high`` and ``direction`` keys.  ``high`` must be
    greater than ``low``.  ``direction`` is ``"up"`` when a larger raw value
    means more risk, and ``"down"`` when a smaller raw value means more risk.
    Values outside the configured bounds are clamped.  Missing values are
    routed through :func:`missing_policy`, never through this function.
    """
    low = spec["low"]
    high = spec["high"]

    # Linear position inside [low, high]; may fall outside [0, 1].
    normalized = (float(value) - low) / (high - low)

    # Defect 1 fix: honor the risk direction.  A "down" feature inverts the
    # linear position so that smaller raw values mean higher risk.
    if spec.get("direction") == "down":
        normalized = 1.0 - normalized

    # Defect 2 fix: clamp out-of-bounds values instead of letting them expand
    # the weighted total.
    return _clamp_unit(normalized)


def weighted_sum(features: dict, weights: dict) -> float:
    """Return the weighted sum after normalizing weights to sum to 1.0.

    ``features`` contains normalized feature values and ``weights`` contains
    a positive weight for every feature.  The same feature set is required on
    both sides.
    """
    if set(features) != set(weights):
        raise ValueError("features and weights must define the same feature set")

    # Defect 3 fix: divide by the total weight so results stay comparable as
    # the number of features changes.
    total_weight = sum(weights[name] for name in features)
    if total_weight <= 0:
        raise ValueError("weights must have a positive total")

    return sum(features[name] * weights[name] for name in features) / total_weight


def missing_policy(features: dict, policy: dict) -> dict:
    """Return a copy of ``features`` with missing values filled by ``policy``.

    Missing means ``None``.  A policy entry may be ``"worst"`` (1.0),
    ``"best"`` (0.0), ``"neutral"`` (0.5), a number in [0.0, 1.0], or a
    mapping of the form ``{"policy": "constant", "value": number}``.  The
    replacement is on the normalized scale, so a missing risk feature must
    not silently become zero unless the policy explicitly says so.
    """
    filled = dict(features)
    for name, value in filled.items():
        if value is not None:
            continue
        # Defect 4 fix: missing is not zero.  An explicit policy wins; with
        # no entry the safe default is the worst (highest risk) value.
        rule = policy.get(name, "worst") if policy else "worst"
        filled[name] = _policy_fill_value(rule)
    return filled


def score(features: dict, model: dict) -> dict:
    """Score one raw feature mapping and return score plus contributions.

    The return value is ``{"score": float, "contributions": dict}``.  Each
    contribution is the normalized feature value multiplied by its normalized
    weight, and the contributions must add up to the score.  A missing raw
    feature is filled according to ``model["missing"]``; when no entry exists
    for that feature, the safe default is ``"worst"``.
    """
    feature_specs = model["features"]
    raw_weights = model["weights"]
    missing_rules = model.get("missing") or {}

    total_weight = sum(raw_weights[name] for name in feature_specs)

    contributions = {}
    for name, spec in feature_specs.items():
        raw_value = features.get(name) if features is not None else None

        if raw_value is None:
            rule = missing_rules.get(name, "worst")
            normalized_value = _policy_fill_value(rule)
        else:
            normalized_value = normalize_feature(raw_value, spec)

        normalized_weight = raw_weights[name] / total_weight

        # Defect 5 fix: contributions are normalized value * normalized
        # weight, i.e. exactly the piece each feature adds to the score.
        contributions[name] = normalized_value * normalized_weight

    total = sum(contributions.values())
    return {"score": total, "contributions": contributions}


def decide(score: float, thresholds: list) -> str:
    """Map a score in [0.0, 1.0] to an action using ordered thresholds.

    Each threshold is ``{"max": number, "action": str}`` with strictly
    increasing ``max`` values.  Intervals are closed on the right for the
    final bucket and half-open on the left for every bucket: a score exactly
    equal to an intermediate ``max`` belongs to the next bucket, while a score
    equal to the final ``max`` belongs to the final bucket.  Every valid score
    must therefore receive one defined action.
    """
    last_index = len(thresholds) - 1
    for index, threshold in enumerate(thresholds):
        # Defect 6 fix: buckets are [previous_max, current_max), so equality
        # with an intermediate max falls through to the next bucket.  The
        # final bucket is closed on the right and always returns an action.
        if index == last_index or score < threshold["max"]:
            return threshold["action"]
    return thresholds[-1]["action"]


def explain(contributions: dict, top_k: int) -> list:
    """Return the top ``top_k`` contributions ordered by absolute value.

    The result is a list of ``(feature, contribution)`` tuples.  Descending
    absolute contribution wins; ties are broken by feature name ascending.
    """
    # Defect 7 fix: rank by magnitude (absolute value), not signed value, so
    # large negative risk drivers are not pushed to the end.
    ordered = sorted(
        contributions.items(),
        key=lambda item: (-abs(item[1]), item[0]),
    )
    if top_k <= 0:
        return []
    return ordered[:top_k]


def validate_model(model: dict) -> list[str]:
    """Return stable problem codes for an invalid model.

    Supported codes are ``features_empty``, ``weights_keys_mismatch``,
    ``weights_non_positive``, ``bounds_invalid``, ``direction_missing``,
    ``missing_policy_invalid``, ``thresholds_invalid`` and
    ``thresholds_non_monotonic``.  A valid model returns an empty list.
    """
    problems = []

    if not isinstance(model, dict):
        return list(_PROBLEM_CODES)

    raw_features = model.get("features")
    if not isinstance(raw_features, dict) or not raw_features:
        problems.append("features_empty")
        feature_specs = {}
    else:
        feature_specs = raw_features

    raw_weights = model.get("weights")
    weights = raw_weights if isinstance(raw_weights, dict) else {}

    if set(weights.keys()) != set(feature_specs.keys()):
        problems.append("weights_keys_mismatch")

    # Defect 9 fix: zero or negative weights would zero out or reverse the
    # risk direction.
    if any(
        (not _is_real_number(weight)) or weight <= 0
        for weight in weights.values()
    ):
        problems.append("weights_non_positive")

    bounds_invalid = False
    direction_invalid = False
    for spec in feature_specs.values():
        if not isinstance(spec, dict):
            bounds_invalid = True
            direction_invalid = True
            continue

        low = spec.get("low")
        high = spec.get("high")
        if (
            not _is_real_number(low)
            or not _is_real_number(high)
            or high <= low
        ):
            bounds_invalid = True

        if spec.get("direction") not in ("up", "down"):
            direction_invalid = True

    if bounds_invalid:
        problems.append("bounds_invalid")
    if direction_invalid:
        problems.append("direction_missing")

    missing_rules = model.get("missing", {})
    if missing_rules is None:
        missing_rules = {}
    if not isinstance(missing_rules, dict) or any(
        not _is_valid_policy(rule) for rule in missing_rules.values()
    ):
        problems.append("missing_policy_invalid")

    # Defect 8 fix: reject malformed threshold sets and non-strictly
    # increasing boundaries before they can make segmentation ambiguous.
    thresholds = model.get("thresholds")
    thresholds_valid = isinstance(thresholds, list) and len(thresholds) > 0
    max_values = []
    if thresholds_valid:
        for threshold in thresholds:
            if not isinstance(threshold, dict):
                thresholds_valid = False
                break
            max_value = threshold.get("max")
            action = threshold.get("action")
            if not _is_real_number(max_value) or not isinstance(action, str) or not action:
                thresholds_valid = False
                break
            max_values.append(float(max_value))

    if not thresholds_valid:
        problems.append("thresholds_invalid")
    else:
        if any(
            max_values[index] >= max_values[index + 1]
            for index in range(len(max_values) - 1)
        ):
            problems.append("thresholds_non_monotonic")

    return problems


def risk_batch(batches: dict, model: dict) -> dict:
    """Score each named batch independently and return named score results.

    ``batches`` maps a batch id to a raw feature mapping.  The result maps the
    same ids to the corresponding :func:`score` result.  No normalized value
    may leak from one batch into another.
    """
    # Defect 10 fix: every batch gets a fresh, independent score() call; no
    # normalized feature map is cached or shared across iterations.
    return {
        batch_id: score(batch_features, model)
        for batch_id, batch_features in batches.items()
    }
