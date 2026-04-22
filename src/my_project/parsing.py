"""Shared parsing utilities for Pega ADM decision JSON exports."""
import json
import math

import pandas as pd

# All luggage add-on weight variants scored by Pega ADM
LUGGAGE_VARIANTS: frozenset[str] = frozenset(
    {"L5B15", "L5B20", "L5B25", "L5B30", "L5B40", "L5B50", "CLUG"}
)


def safe_float(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None


def parse_common_inputs(record: dict) -> dict:
    common_inputs = {}
    for ci in record.get("pxCommonInputs", []) or []:
        raw = ci.get("pxCommonInput")
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            for k, v in parsed.get("values", {}).items():
                common_inputs[k] = v.get("value")
        except Exception:
            continue
    return common_inputs


def extract_l5b15_rows(
    df,
    target_names: list[str] | str | None = None,
) -> list[dict]:
    """Extract flat scoring-event rows from a raw Polars DataFrame of decision records.

    Parameters
    ----------
    df : polars.DataFrame
        Raw decision records as loaded from the JSON export.
    target_names : list[str] | str | None
        Product name(s) to extract (``pyName`` in the partition).
        ``None``  → all entries in ``LUGGAGE_VARIANTS`` (default).
        ``"L5B15"`` or ``["L5B15"]`` → single-variant back-compat mode.
    """
    if target_names is None:
        _targets = LUGGAGE_VARIANTS
    elif isinstance(target_names, str):
        _targets = {target_names}
    else:
        _targets = set(target_names)

    rows = []
    for record in df.to_dicts():
        interaction_id = record.get("pxInteractionID")
        subject_id = record.get("pxSubjectID")
        common_inputs = parse_common_inputs(record)

        # Build decision-time lookup: pyName → pxDecisionTime (ISO string)
        # pxDecisionTime lives in pxDecisionResults, not pxModelExecutionResults.
        decision_times: dict[str, str] = {}
        for dr in record.get("pxDecisionResults", []) or []:
            name = dr.get("pyName")
            dt = dr.get("pxDecisionTime")
            if name and dt:
                decision_times[name] = dt

        for me in record.get("pxModelExecutionResults", []) or []:
            raw_me = me.get("pxModelExecutionResults")
            if not raw_me:
                continue
            try:
                parsed_me = json.loads(raw_me)
            except Exception:
                continue

            partition = parsed_me.get("partition", {}) or {}
            name = partition.get("pyName")
            if name not in _targets:
                continue

            values = parsed_me.get("values", {}) or {}
            param_preds = ((parsed_me.get("parameterPredictors") or {}).get("values") or {})

            row = {
                "pxInteractionID": interaction_id,
                "pxSubjectID": subject_id,
                "modelExecutionID": parsed_me.get("modelExecutionID"),
                "modelVersion": parsed_me.get("modelVersion"),
                "pxDecisionTime": decision_times.get(name),
                "pyName": name,
                "pyChannel": partition.get("pyChannel"),
                "pyDirection": partition.get("pyDirection"),
                "pyGroup": partition.get("pyGroup"),
                "pyIssue": partition.get("pyIssue"),
                "TreatmentID": partition.get("TreatmentID"),
                "propensity": safe_float((values.get("pyPropensity") or {}).get("value")),
                "modelPerformance": safe_float((values.get("pyModelPerformance") or {}).get("value")),
                "modelEvidence": safe_float((values.get("pyModelEvidence") or {}).get("value")),
                "modelTechnique": (values.get("pyModelTechnique") or {}).get("value"),
                **common_inputs,
            }
            for k, v in param_preds.items():
                row[f"param::{k}"] = v.get("value")

            rows.append(row)

    return rows


def extract_filtered_model(
    df,
    pyname: str | None = None,
    pychannel: str | None = None,
    pydirection: str | None = None,
    pygroup: str | None = None,
    pyissue: str | None = None,
    model_version: str | None = None,
) -> pd.DataFrame:
    """Filter and extract decision rows matching the given partition criteria."""
    rows = []
    for record in df.to_dicts():
        interaction_id = record.get("pxInteractionID")
        subject_id = record.get("pxSubjectID")
        common_inputs = parse_common_inputs(record)

        for me in record.get("pxModelExecutionResults", []) or []:
            raw_me = me.get("pxModelExecutionResults")
            if not raw_me:
                continue
            try:
                parsed_me = json.loads(raw_me)
            except Exception:
                continue

            partition = parsed_me.get("partition", {}) or {}
            if pyname is not None and partition.get("pyName") != pyname:
                continue
            if pychannel is not None and partition.get("pyChannel") != pychannel:
                continue
            if pydirection is not None and partition.get("pyDirection") != pydirection:
                continue
            if pygroup is not None and partition.get("pyGroup") != pygroup:
                continue
            if pyissue is not None and partition.get("pyIssue") != pyissue:
                continue
            if model_version is not None and parsed_me.get("modelVersion") != model_version:
                continue

            values = parsed_me.get("values", {}) or {}
            param_preds = ((parsed_me.get("parameterPredictors") or {}).get("values") or {})

            row = {
                "pxInteractionID": interaction_id,
                "pxSubjectID": subject_id,
                "modelVersion": parsed_me.get("modelVersion"),
                "modelExecutionID": parsed_me.get("modelExecutionID"),
                "pyName": partition.get("pyName"),
                "pyChannel": partition.get("pyChannel"),
                "pyDirection": partition.get("pyDirection"),
                "pyGroup": partition.get("pyGroup"),
                "pyIssue": partition.get("pyIssue"),
                "pyTreatment": partition.get("pyTreatment"),
                "TreatmentID": partition.get("TreatmentID"),
                "propensity": safe_float((values.get("pyPropensity") or {}).get("value")),
                "modelPerformance": safe_float((values.get("pyModelPerformance") or {}).get("value")),
                "modelEvidence": safe_float((values.get("pyModelEvidence") or {}).get("value")),
                "modelTechnique": (values.get("pyModelTechnique") or {}).get("value"),
                **common_inputs,
            }
            for k, v in param_preds.items():
                row[f"param::{k}"] = v.get("value")

            rows.append(row)

    return pd.DataFrame(rows)
