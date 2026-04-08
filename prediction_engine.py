from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import pandas as pd

from services.prediction_resources import PredictionResources

TITLE_OUTPUT_COLUMNS = [
    "Model 1 OP",
    "Model 2 OP",
    "Model 3 OP",
    "LLM as a Judge OP",
    "LLM Decision Source",
    "Explanation",
]

PROGRESS_PERSIST_BATCH_SIZE = 10

REQUIRED_PIPELINE_COLUMNS = [
    "unspsc_code",
    "short_description_(english)",
    "long_description_(english)",
    "part_manufacturer_list_information",
    "hazardous_goods_(s.f.)",
    "is_related_to_fire&smoke?",
    "part_rohs",
    "perishable_part",
    "repairable_?",
    "unit_of_measure_information",
    "usability",
]

COLUMN_CANDIDATES = {
    "short_description_(english)": [
        "Short Description (English)",
        "short_description_(english)",
        "Short Description",
    ],
    "long_description_(english)": [
        "Long Description (English)",
        "long_description_(english)",
        "Long Description",
    ],
    "part_manufacturer_list_information": [
        "Part Manufacturer List information",
        "part_manufacturer_list_information",
    ],
    "hazardous_goods_(s.f.)": [
        "Hazardous Goods (S.F.)",
        "hazardous_goods_(s.f.)",
    ],
    "is_related_to_fire&smoke?": [
        "Is related to Fire&Smoke?",
        "is_related_to_fire&smoke?",
    ],
    "part_rohs": [
        "Part ROHS",
        "part_rohs",
    ],
    "perishable_part": [
        "Perishable Part",
        "perishable_part",
    ],
    "repairable_?": [
        "Repairable ?",
        "repairable_?",
    ],
    "unit_of_measure_information": [
        "Unit of Measure information",
        "unit_of_measure_information",
    ],
    "usability": [
        "Usability",
        "usability",
    ],
    "unspsc_code": [
        "unspsc_code",
        "UNSPSC Code",
    ],
}


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None




def _normalize_value(value: Any) -> str:
    if pd.isna(value) or value is None:
        return ""
    text = str(value).replace(" ", " ").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _build_prediction_key(row: pd.Series) -> str:
    payload = {col: _normalize_value(row.get(col)) for col in REQUIRED_PIPELINE_COLUMNS}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _prepare_pipeline_input(input_df: pd.DataFrame) -> pd.DataFrame:
    prepared = pd.DataFrame(index=input_df.index)
    for target_col in REQUIRED_PIPELINE_COLUMNS:
        source_col = _first_present(input_df, COLUMN_CANDIDATES.get(target_col, [target_col]))
        if source_col is None:
            prepared[target_col] = None
        else:
            prepared[target_col] = input_df[source_col]

    # unspsc_code is not required for inference inputs; keep it nullable.
    if "unspsc_code" not in prepared.columns:
        prepared["unspsc_code"] = None

    prepared["_prediction_key"] = prepared.apply(_build_prediction_key, axis=1)
    return prepared


def _is_content_filter_error(exc: Exception) -> bool:
    text = repr(exc).lower()
    markers = [
        "content_filter",
        "contentpolicyviolation",
        "responsibleaipolicyviolation",
        "content management policy",
    ]
    return any(marker in text for marker in markers)


def _build_ignored_record(row_position: int, row: pd.Series, exc: Exception) -> dict:
    payload = row.to_dict()
    payload["ignored_row_position"] = row_position
    payload["ignored_df_index"] = row.name
    payload["ignored_stage"] = "llm_judge"
    payload["ignored_error_type"] = type(exc).__name__
    payload["ignored_error"] = repr(exc)
    payload["ignored_reason"] = (
        "Azure content filter blocked the LLM judge prompt."
        if _is_content_filter_error(exc)
        else "LLM judge failed for this row."
    )
    return payload


def _judge_row(prediction_module, row_position: int, row: pd.Series) -> dict:
    query = row.get("short_description_(english)")
    m1 = row.get("Model 1 OP")
    m2 = row.get("Model 2 OP")
    m3 = row.get("Model 3 OP")
    try:
        code, explanation, source = prediction_module.predict_unspsc_with_rag(
            query,
            model1=m1,
            model2=m2,
            model3=m3,
        )
        return {
            "LLM as a Judge OP": code,
            "LLM Decision Source": source,
            "Explanation": explanation,
        }
    except Exception as exc:
        print("========== LLM JUDGE IGNORED RECORD ==========")
        print(f"row_position: {row_position}")
        print(f"df_index: {row.name}")
        print(f"short_description_(english): {row.get('short_description_(english)')}")
        print(f"long_description_(english): {row.get('long_description_(english)')}")
        print(f"Model 1 OP: {m1}")
        print(f"Model 2 OP: {m2}")
        print(f"Model 3 OP: {m3}")
        print(f"error_type: {type(exc).__name__}")
        print(f"error: {repr(exc)}")
        print("=============================================")
        raise
        
def _flush_progress_batch(
   progress_records: list[dict],
   key_to_rowhash_df: pd.DataFrame,
   partial_result_callback: Callable[[pd.DataFrame], None] | None,
) -> None:
   if partial_result_callback is None or not progress_records:
       return
   progress_df = pd.DataFrame(progress_records)
   if progress_df.empty:
       return
   expanded_df = key_to_rowhash_df.merge(
       progress_df,
       on="_prediction_key",
       how="inner",
   )
   keep_cols = ["row_hash"] + TITLE_OUTPUT_COLUMNS
   keep_cols = [c for c in keep_cols if c in expanded_df.columns]
   expanded_df = expanded_df[keep_cols].copy().drop_duplicates(subset=["row_hash"]).reset_index(drop=True)
   if expanded_df.empty:
       return
   partial_result_callback(expanded_df)

def _run_llm_judge(
   prediction_module,
   ml_df: pd.DataFrame,
   key_to_rowhash_df: pd.DataFrame,
   max_workers: int = 4,
   partial_result_callback: Callable[[pd.DataFrame], None] | None = None,
   progress_batch_size: int = PROGRESS_PERSIST_BATCH_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
   judged_df = ml_df.copy().reset_index(drop=True)
   judged_df["LLM as a Judge OP"] = None
   judged_df["LLM Decision Source"] = None
   judged_df["Explanation"] = None
   ignored_records: list[dict] = []
   progress_records: list[dict] = []
   with ThreadPoolExecutor(max_workers=max_workers) as executor:
       futures = {
           executor.submit(_judge_row, prediction_module, row_position, row.copy()): (row_position, row.copy())
           for row_position, (_, row) in enumerate(ml_df.iterrows())
       }
       for future in as_completed(futures):
           row_position, row = futures[future]
           progress_record = {
               "_prediction_key": row.get("_prediction_key"),
               "Model 1 OP": row.get("Model 1 OP"),
               "Model 2 OP": row.get("Model 2 OP"),
               "Model 3 OP": row.get("Model 3 OP"),
           }
           try:
               result = future.result()
               judged_df.at[row_position, "LLM as a Judge OP"] = result.get("LLM as a Judge OP")
               judged_df.at[row_position, "LLM Decision Source"] = result.get("LLM Decision Source")
               judged_df.at[row_position, "Explanation"] = result.get("Explanation")
               progress_record["LLM as a Judge OP"] = result.get("LLM as a Judge OP")
               progress_record["LLM Decision Source"] = result.get("LLM Decision Source")
               progress_record["Explanation"] = result.get("Explanation")
           except Exception as exc:
               ignored_records.append(_build_ignored_record(row_position, row, exc))
               llm_value = "--"
               source_value = "Ignored(ContentFilter)" if _is_content_filter_error(exc) else "Ignored(Error)"
               explanation_value = (
                   "Row ignored because the LLM judge prompt was blocked by Azure content filtering."
                   if _is_content_filter_error(exc)
                   else "Row ignored because the LLM judge failed for this record."
               )
               judged_df.at[row_position, "LLM as a Judge OP"] = llm_value
               judged_df.at[row_position, "LLM Decision Source"] = source_value
               judged_df.at[row_position, "Explanation"] = explanation_value
               progress_record["LLM as a Judge OP"] = llm_value
               progress_record["LLM Decision Source"] = source_value
               progress_record["Explanation"] = explanation_value
           progress_records.append(progress_record)
           if len(progress_records) >= progress_batch_size:
               _flush_progress_batch(
                   progress_records=progress_records,
                   key_to_rowhash_df=key_to_rowhash_df,
                   partial_result_callback=partial_result_callback,
               )
               progress_records = []
   if progress_records:
       _flush_progress_batch(
           progress_records=progress_records,
           key_to_rowhash_df=key_to_rowhash_df,
           partial_result_callback=partial_result_callback,
       )
   ignored_df = pd.DataFrame(ignored_records)
   if not ignored_df.empty and "ignored_row_position" in ignored_df.columns:
       ignored_df = ignored_df.sort_values("ignored_row_position").reset_index(drop=True)
   return judged_df, ignored_df


def predict_dataframe(
   input_df: pd.DataFrame,
   resources: PredictionResources,
   max_workers: int = 4,
   partial_result_callback: Callable[[pd.DataFrame], None] | None = None,
   progress_batch_size: int = PROGRESS_PERSIST_BATCH_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
   prediction_module = resources.prediction_module
   input_work_df = input_df.reset_index(drop=True).copy()
   if "row_hash" not in input_work_df.columns:
       raise ValueError(
           f"predict_dataframe expected 'row_hash' in input_df, but it was missing. "
           f"Available columns: {list(input_work_df.columns)}"
       )
   if input_work_df["row_hash"].isna().any():
       raise ValueError("predict_dataframe found null values in 'row_hash'.")
   pipeline_input_df = _prepare_pipeline_input(input_work_df)
   key_to_rowhash_df = pd.DataFrame({
       "_prediction_key": pipeline_input_df["_prediction_key"].astype(str),
       "row_hash": input_work_df["row_hash"].astype(str),
   }).drop_duplicates().reset_index(drop=True)
   unique_pipeline_df = (
       pipeline_input_df
       .drop_duplicates(subset=["_prediction_key"])
       .reset_index(drop=True)
       .copy()
   )
   unique_key_df = unique_pipeline_df[["_prediction_key"]].reset_index(drop=True).copy()
   ml_unique_df = prediction_module.generate_ml_output(unique_pipeline_df.copy())
   ml_unique_df = ml_unique_df.reset_index(drop=True).copy()
   if len(ml_unique_df) != len(unique_key_df):
       raise ValueError(
           f"Prediction output row count mismatch before merge-back: "
           f"unique_input_rows={len(unique_key_df)}, predicted_rows={len(ml_unique_df)}"
       )
   ml_unique_df["_prediction_key"] = unique_key_df["_prediction_key"]
   ml_unique_df.rename(
       columns={
           "Model1": "Model 1 OP",
           "Model2": "Model 2 OP",
           "Model3": "Model 3 OP",
       },
       inplace=True,
   )
   judged_unique_df, unique_ignored_df = _run_llm_judge(
       prediction_module,
       ml_unique_df,
       key_to_rowhash_df=key_to_rowhash_df,
       max_workers=max_workers,
       partial_result_callback=partial_result_callback,
       progress_batch_size=progress_batch_size,
   )
   prediction_cols = ["_prediction_key"] + TITLE_OUTPUT_COLUMNS
   missing_cols = [c for c in prediction_cols if c not in judged_unique_df.columns]
   if missing_cols:
       raise ValueError(
           f"Missing expected columns after prediction/judge step: {missing_cols}. "
           f"Available columns: {list(judged_unique_df.columns)}"
       )
   judged_unique_df = judged_unique_df[prediction_cols].copy()
   merged_full_df = pipeline_input_df.merge(
       judged_unique_df,
       on="_prediction_key",
       how="left",
   )
   result_df = input_work_df.copy()
   result_df["row_hash"] = input_work_df["row_hash"]
   if "run_id" in input_work_df.columns:
       result_df["run_id"] = input_work_df["run_id"]
   for col in TITLE_OUTPUT_COLUMNS:
       result_df[col] = merged_full_df[col]
   print("========== ENGINE RETURN DEBUG ==========")
   print("result_df rows:", len(result_df))
   print("result_df columns:", list(result_df.columns))
   print("row_hash non-null count:", int(result_df["row_hash"].notna().sum()))
   for col in TITLE_OUTPUT_COLUMNS:
       if col in result_df.columns:
           print(f"{col} non-null count:", int(result_df[col].notna().sum()))
   print("========================================")
   ignored_df = pd.DataFrame()
   if not unique_ignored_df.empty:
       ignored_keys = set(unique_ignored_df.get("_prediction_key", pd.Series(dtype=str)).dropna().astype(str).tolist())
       if not ignored_keys and "_prediction_key" in judged_unique_df.columns:
           ignored_keys = set(
               judged_unique_df.loc[
                   judged_unique_df["LLM Decision Source"].astype(str).str.startswith("Ignored", na=False),
                   "_prediction_key",
               ].astype(str).tolist()
           )
       if ignored_keys:
           ignored_df = merged_full_df[merged_full_df["_prediction_key"].astype(str).isin(ignored_keys)].copy()
           ignored_df = ignored_df.merge(
               unique_ignored_df[["_prediction_key", "ignored_stage", "ignored_error_type", "ignored_error", "ignored_reason"]]
               .drop_duplicates(subset=["_prediction_key"]),
               on="_prediction_key",
               how="left",
           )
           ignored_df.reset_index(drop=True, inplace=True)
   return result_df, ignored_df
