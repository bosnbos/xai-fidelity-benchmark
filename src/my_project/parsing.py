"""Shared parsing utilities for Pega ADM decision JSON exports."""
import json
import math

import pandas as pd


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


def extract_l5b15_rows(df, target_name: str = "L5B15") -> list[dict]:
    """Extract flat scoring-event rows from a raw Polars DataFrame of decision records."""
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
            if partition.get("pyName") != target_name:
                continue

            values = parsed_me.get("values", {}) or {}
            param_preds = ((parsed_me.get("parameterPredictors") or {}).get("values") or {})

            row = {
                "pxInteractionID": interaction_id,
                "pxSubjectID": subject_id,
                "modelExecutionID": parsed_me.get("modelExecutionID"),
                "modelVersion": parsed_me.get("modelVersion"),
                "pyName": partition.get("pyName"),
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
