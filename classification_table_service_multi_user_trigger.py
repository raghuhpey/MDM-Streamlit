from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from typing import Any
import dataiku
import pandas as pd
from dataiku import SQLExecutor2
from services.run_storage_service import generate_prediction_job_id, get_current_user_login
PROJECT_KEY = "MDM_PARTS_CLASSIFICATION"
CLASSIFICATION_DATASET_NAME = "MDM_classification_tbl"
CLASSIFICATION_METADATA_COLUMNS = [
  "run_id",
  "file_name",
  "uploaded_by",
  "uploaded_ts",
  "run_status",
  "worker_scenario_id",   # NEW
  "backend_run_id",
  "backend_trigger_id",
  "prediction_started_ts",
  "prediction_completed_ts",
  "created_ts",
  "updated_ts",
]
CLASSIFICATION_OUTPUT_COLUMNS = [
   "model_1_op",
   "model_2_op",
   "model_3_op",
   "llm_judge_op",
   "llm_decision_source",
   "explanation",
]
CLASSIFICATION_VALIDATION_COLUMNS = [
   "validation_status",
   "validated_unspsc",
   "validation_decision_source",
   "validated_by",
   "validated_at",
   "validation_comment",
   "is_golden_record",
]
CLASSIFICATION_TECHNICAL_COLUMNS = [
   "row_hash",
]
CLASSIFICATION_RESERVED_COLUMNS = set(
   CLASSIFICATION_METADATA_COLUMNS
   + CLASSIFICATION_OUTPUT_COLUMNS
   + CLASSIFICATION_VALIDATION_COLUMNS
   + CLASSIFICATION_TECHNICAL_COLUMNS
)
SQL_INSERT_BATCH_SIZE = 200

def get_classification_dataset():
   return dataiku.Dataset(CLASSIFICATION_DATASET_NAME)

def get_classification_schema_columns() -> list[str]:
   ds = get_classification_dataset()
   schema = ds.read_schema()
   return [col["name"] for col in schema]

def get_classification_table_df() -> pd.DataFrame:
   ds = get_classification_dataset()
   return ds.get_dataframe()

def _utc_now_string() -> str:
   return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def normalize_uploaded_dataframe_for_sql(upload_df: pd.DataFrame) -> pd.DataFrame:
   df = upload_df.copy()
   for col in df.columns:
       df[col] = (
           df[col]
           .astype("object")
           .where(pd.notna(df[col]), None)
       )
       df[col] = df[col].apply(
           lambda x: str(x).replace("\xa0", " ").strip() if x is not None else None
       )
       df[col] = df[col].apply(
           lambda x: x[:-2] if isinstance(x, str) and x.endswith(".0") else x
       )
   return df

def _normalize_value_for_hash(value: Any) -> str:
   if pd.isna(value) or value is None:
       return ""
   if isinstance(value, pd.Timestamp):
       return value.isoformat()
   if isinstance(value, float):
       if value.is_integer():
           return str(int(value))
       return str(value)
   return str(value).replace("\xa0", " ").strip()

def build_row_hash_series(input_df: pd.DataFrame) -> pd.Series:
   input_cols = sorted(list(input_df.columns))
   def _row_hash(row: pd.Series) -> str:
       payload = {
           col: _normalize_value_for_hash(row.get(col))
           for col in input_cols
       }
       raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
       return hashlib.sha256(raw.encode("utf-8")).hexdigest()
   return input_df.apply(_row_hash, axis=1)

def get_original_input_columns(schema_columns: list[str] | None = None) -> list[str]:
   if schema_columns is None:
       schema_columns = get_classification_schema_columns()
   return [
       col for col in schema_columns
       if col not in CLASSIFICATION_RESERVED_COLUMNS
   ]

def _split_uploaded_columns_against_table_schema(
   upload_df: pd.DataFrame,
   schema_columns: list[str],
) -> tuple[list[str], list[str]]:
   allowed_input_columns = set(get_original_input_columns(schema_columns))
   matched_columns = [col for col in upload_df.columns if col in allowed_input_columns]
   ignored_columns = [col for col in upload_df.columns if col not in allowed_input_columns]
   return matched_columns, ignored_columns

def _prepare_upload_dataframe(
   upload_df: pd.DataFrame,
   *,
   run_id: str,
   file_name: str,
   uploaded_by: str,
) -> pd.DataFrame:
   prepared_df = upload_df.copy()
   now_ts = _utc_now_string()
   prepared_df["run_id"] = run_id
   prepared_df["file_name"] = file_name
   prepared_df["uploaded_by"] = uploaded_by
   prepared_df["uploaded_ts"] = now_ts
   prepared_df["run_status"] = "uploaded"
   prepared_df["backend_run_id"] = None
   prepared_df["backend_trigger_id"] = None
   prepared_df["prediction_started_ts"] = None
   prepared_df["prediction_completed_ts"] = None
   prepared_df["created_ts"] = now_ts
   prepared_df["updated_ts"] = now_ts
   prepared_df["model_1_op"] = None
   prepared_df["model_2_op"] = None
   prepared_df["model_3_op"] = None
   prepared_df["llm_judge_op"] = None
   prepared_df["llm_decision_source"] = None
   prepared_df["explanation"] = None
   prepared_df["validation_status"] = None
   prepared_df["validated_unspsc"] = None
   prepared_df["validation_decision_source"] = None
   prepared_df["validated_by"] = None
   prepared_df["validated_at"] = None
   prepared_df["validation_comment"] = None
   prepared_df["is_golden_record"] = False
   prepared_df["worker_scenario_id"] = None
   prepared_df["row_hash"] = build_row_hash_series(upload_df)
   return prepared_df

def _align_to_table_schema(df: pd.DataFrame, schema_columns: list[str]) -> pd.DataFrame:
   aligned_df = df.copy()
   for col in schema_columns:
       if col not in aligned_df.columns:
           aligned_df[col] = None
   return aligned_df[schema_columns]

def _sql_escape(value: Any) -> str:
   if pd.isna(value) or value is None:
       return "NULL"
   if isinstance(value, bool):
       return "TRUE" if value else "FALSE"
   if isinstance(value, pd.Timestamp):
       value = value.isoformat()
   s = str(value).replace("'", "''")
   return f"'{s}'"

def _quote_identifier(identifier: str) -> str:
   return f'"{identifier}"'

def _get_sql_dataset_location():
   dataset = get_classification_dataset()
   loc = dataset.get_location_info()
   info = loc.get("info", {})
   connection_name = info.get("connectionName")
   quoted_table_name = info.get("quotedResolvedTableName")
   if not connection_name or not quoted_table_name:
       raise ValueError(
           "Could not resolve SQL dataset location for MDM_classification_tbl. "
           "Please confirm it is a SQL dataset."
       )
   return {
       "dataset": dataset,
       "connection_name": connection_name,
       "quoted_table_name": quoted_table_name,
       "project_key": dataset.project_key,
   }
   

def _run_sql(sql: str):
   info = _get_sql_dataset_location()
   client = dataiku.api_client()
   query = client.sql_query(
       sql,
       connection=info["connection_name"],
       type="sql",
       project_key=info["project_key"],
       post_queries=["COMMIT"],
   )
   query.verify()

def _query_to_df(sql: str) -> pd.DataFrame:
   ds = get_classification_dataset()
   executor = SQLExecutor2(dataset=ds)
   return executor.query_to_df(sql)

def get_existing_row_hashes() -> set[str]:
   info = _get_sql_dataset_location()
   query = f"SELECT row_hash FROM {info['quoted_table_name']}"
   df = _query_to_df(query)
   if df.empty or "row_hash" not in df.columns:
       return set()
   return set(
       df["row_hash"]
       .dropna()
       .astype(str)
       .str.strip()
       .tolist()
   )

def insert_rows_via_sql(df_to_insert: pd.DataFrame):
   if df_to_insert.empty:
       return
   info = _get_sql_dataset_location()
   columns = list(df_to_insert.columns)
   quoted_cols = ", ".join(_quote_identifier(c) for c in columns)
   for start in range(0, len(df_to_insert), SQL_INSERT_BATCH_SIZE):
       batch_df = df_to_insert.iloc[start:start + SQL_INSERT_BATCH_SIZE]
       value_rows = []
       for _, row in batch_df.iterrows():
           row_values = ", ".join(_sql_escape(row[c]) for c in columns)
           value_rows.append(f"({row_values})")
       values_sql = ",\n".join(value_rows)
       sql = f"""
       INSERT INTO {info['quoted_table_name']} ({quoted_cols})
       VALUES
       {values_sql}
       """
       _run_sql(sql)

def insert_uploaded_rows_into_classification_table(
   upload_df: pd.DataFrame,
   file_name: str,
) -> dict:
   schema_columns = get_classification_schema_columns()
   normalized_upload_df = normalize_uploaded_dataframe_for_sql(upload_df)
   matched_columns, ignored_columns = _split_uploaded_columns_against_table_schema(
       normalized_upload_df,
       schema_columns,
   )
   if not matched_columns:
       raise ValueError(
           "None of the uploaded columns match the supported input columns in MDM_classification_tbl."
       )
   aligned_input_df = normalized_upload_df[matched_columns].copy()
   run_id = generate_prediction_job_id(file_name)
   uploaded_by = get_current_user_login()
   prepared_df = _prepare_upload_dataframe(
       upload_df=aligned_input_df,
       run_id=run_id,
       file_name=file_name,
       uploaded_by=uploaded_by,
   )
   prepared_df = prepared_df.drop_duplicates(subset=["row_hash"]).copy()
   existing_hashes = get_existing_row_hashes()
   new_rows_df = prepared_df[
       ~prepared_df["row_hash"].isin(existing_hashes)
   ].copy()
   aligned_new_rows_df = _align_to_table_schema(new_rows_df, schema_columns)
   insert_rows_via_sql(aligned_new_rows_df)
   total_uploaded = int(len(upload_df))
   inserted_count = int(len(aligned_new_rows_df))
   duplicate_count = int(total_uploaded - inserted_count)
   return {
       "run_id": run_id,
       "file_name": file_name,
       "uploaded_by": uploaded_by,
       "total_uploaded": total_uploaded,
       "inserted_count": inserted_count,
       "duplicate_count": duplicate_count,
       "run_status": "uploaded",
       "input_columns": matched_columns,
       "ignored_columns": ignored_columns,
   }

def mark_run_queued(run_id: str):
  update_run_rows_by_run_id(
      run_id,
      {
          "run_status": "queued",
      }
  )

def fetch_queued_run_ids(limit: int = 10) -> list[str]:
  info = _get_sql_dataset_location()
  query = f"""
  SELECT run_id
  FROM {info['quoted_table_name']}
  WHERE LOWER(run_status) = 'queued'
  GROUP BY run_id
  ORDER BY MIN(created_ts) ASC
  LIMIT {int(limit)}
  """
  df = _query_to_df(query)
  if df.empty:
      return []
  return df["run_id"].astype(str).tolist()




def fetch_classification_run_preview_df(run_id: str) -> pd.DataFrame:
   info = _get_sql_dataset_location()
   safe_run_id = str(run_id).replace("'", "''")
   query = f"""
   SELECT *
   FROM {info['quoted_table_name']}
   WHERE run_id = '{safe_run_id}'
   """
   run_df = _query_to_df(query)
   if run_df.empty:
       return pd.DataFrame()
   schema_columns = get_classification_schema_columns()
   preview_columns = [c for c in schema_columns if c in run_df.columns]
   return run_df[preview_columns].reset_index(drop=True)

def fetch_classification_run_input_df(run_id: str) -> pd.DataFrame:
   run_df = fetch_classification_run_preview_df(run_id)
   if run_df.empty:
       return pd.DataFrame()
   input_cols = [
       c for c in get_original_input_columns(run_df.columns.tolist())
       if c in run_df.columns
   ]
   return run_df[input_cols].copy().reset_index(drop=True)

def fetch_classification_run_display_df(run_id: str) -> pd.DataFrame:
   run_df = fetch_classification_run_preview_df(run_id)
   if run_df.empty:
       return pd.DataFrame()
   status = (
       str(run_df["run_status"].iloc[0]).strip().lower()
       if "run_status" in run_df.columns else "uploaded"
   )
   input_cols = [
       c for c in get_original_input_columns(run_df.columns.tolist())
       if c in run_df.columns
   ]
   if status == "completed":
       display_cols = input_cols + [c for c in CLASSIFICATION_OUTPUT_COLUMNS if c in run_df.columns]
   else:
       display_cols = input_cols
   return run_df[display_cols].copy().reset_index(drop=True)

def fetch_classification_run_metadata(run_id: str) -> dict | None:
   run_df = fetch_classification_run_preview_df(run_id)
   if run_df.empty:
       return None
   first_row = run_df.iloc[0]
   return {
       "run_id": str(first_row.get("run_id", "")),
       "file_name": first_row.get("file_name"),
       "uploaded_by": first_row.get("uploaded_by"),
       "uploaded_ts": first_row.get("uploaded_ts"),
       "run_status": first_row.get("run_status"),
       "worker_scenario_id": first_row.get("worker_scenario_id"),
       "backend_run_id": first_row.get("backend_run_id"),
       "backend_trigger_id": first_row.get("backend_trigger_id"),
       "prediction_started_ts": first_row.get("prediction_started_ts"),
       "prediction_completed_ts": first_row.get("prediction_completed_ts"),
       "updated_ts": first_row.get("updated_ts"),
       "row_count": int(len(run_df)),
   }

def fetch_classification_runs_summary_df() -> pd.DataFrame:
   full_df = get_classification_table_df()
   if full_df.empty or "run_id" not in full_df.columns:
       return pd.DataFrame(
           columns=[
               "run_id",
               "file_name",
               "uploaded_by",
               "uploaded_ts",
               "run_status",
               "worker_scenario_id",
               "backend_run_id",
               "backend_trigger_id",
               "prediction_started_ts",
               "prediction_completed_ts",
               "row_count",
               "action_label",
           ]
       )
   summary_df = (
       full_df.groupby("run_id", as_index=False)
       .agg(
           file_name=("file_name", "first"),
           uploaded_by=("uploaded_by", "first"),
           uploaded_ts=("uploaded_ts", "first"),
           run_status=("run_status", "first"),
           worker_scenario_id=("worker_scenario_id", "first"),
           backend_run_id=("backend_run_id", "first"),
           backend_trigger_id=("backend_trigger_id", "first"),
           prediction_started_ts=("prediction_started_ts", "first"),
           prediction_completed_ts=("prediction_completed_ts", "first"),
           row_count=("run_id", "size"),
           updated_ts=("updated_ts", "max"),
       )
       .sort_values(by="updated_ts", ascending=False)
       .reset_index(drop=True)
   )
   def _map_action(status: str) -> str:
    status = str(status).strip().lower()
    if status in {"uploaded", "failed", "stopped"}:
        return "Continue"
    if status in {"queued", "submitted", "running"}:
        return "Stop"
    if status == "completed":
        return "View"
    return "View"
   summary_df["action_label"] = summary_df["run_status"].apply(_map_action)
   return summary_df

def update_run_rows_by_run_id(run_id: str, updates: dict):
   meta = fetch_classification_run_metadata(run_id)
   if meta is None:
       raise ValueError(f"Run not found in MDM_classification_tbl: {run_id}")
   safe_run_id = str(run_id).replace("'", "''")
   schema_columns = get_classification_schema_columns()
   set_parts = []
   for key, value in updates.items():
       if key not in schema_columns:
           raise ValueError(f"Column '{key}' does not exist in MDM_classification_tbl")
       set_parts.append(f"{_quote_identifier(key)} = {_sql_escape(value)}")
   if "updated_ts" in schema_columns:
       set_parts.append(f'{_quote_identifier("updated_ts")} = {_sql_escape(_utc_now_string())}')
   set_sql = ", ".join(set_parts)
   info = _get_sql_dataset_location()
   sql = f"""
   UPDATE {info['quoted_table_name']}
   SET {set_sql}
   WHERE run_id = '{safe_run_id}'
   """
   _run_sql(sql)

def mark_run_submitted(run_id: str, backend_trigger_id: str | None = None):
   update_run_rows_by_run_id(
       run_id,
       {
           "run_status": "submitted",
           "worker_scenario_id": None,
           "backend_run_id": None,
           "backend_trigger_id": backend_trigger_id,
           "prediction_started_ts": None,
           "prediction_completed_ts": None,
       }
   )

def mark_run_running(run_id: str, backend_run_id: str | None = None):
   updates = {
       "run_status": "running",
       "prediction_started_ts": _utc_now_string(),
   }
   if backend_run_id:
       updates["backend_run_id"] = backend_run_id
   update_run_rows_by_run_id(run_id, updates)

def mark_run_completed(run_id: str):
   update_run_rows_by_run_id(
       run_id,
       {
           "run_status": "completed",
           "prediction_completed_ts": _utc_now_string(),
       }
   )

def mark_run_failed(run_id: str):
   update_run_rows_by_run_id(
       run_id,
       {
           "run_status": "failed",
           "prediction_completed_ts": _utc_now_string(),
       }
   )

def mark_run_stopped(run_id: str):
   update_run_rows_by_run_id(
       run_id,
       {
           "run_status": "stopped",
           "prediction_completed_ts": _utc_now_string(),
       }
   )

def request_stop_run_in_classification_table(run_id: str):
  meta = fetch_classification_run_metadata(run_id)
  if meta is None:
      raise ValueError(f"Run not found: {run_id}")

  current_status = str(meta.get("run_status", "")).strip().lower()
  if current_status not in {"queued", "submitted", "running"}:
      raise ValueError(f"Run is not stoppable in current status: {current_status}")

  worker_scenario_id = str(meta.get("worker_scenario_id") or "").strip()
  backend_run_id = str(meta.get("backend_run_id") or "").strip()

  # Queued without any worker assignment can be safely marked stopped in-app.
  if current_status == "queued" and not worker_scenario_id:
      mark_run_stopped(run_id)
      return

  if not worker_scenario_id:
      mark_run_stopped(run_id)
      return

  client = dataiku.api_client()
  project = client.get_project(PROJECT_KEY)
  scenario = project.get_scenario(worker_scenario_id)

  if not backend_run_id:
      current_scenario_run = None
      try:
          current_scenario_run = scenario.get_current_run()
      except Exception:
          current_scenario_run = None
      if current_scenario_run is not None:
          try:
              scenario.abort()
          except Exception:
              pass
      mark_run_stopped(run_id)
      return

  scenario_run = None
  try:
      scenario_run = scenario.get_run(backend_run_id)
  except Exception:
      scenario_run = None

  if scenario_run is not None:
      try:
          scenario_run.refresh()
      except Exception:
          pass
      raw = {}
      try:
          raw = scenario_run.get_raw() or {}
      except Exception:
          raw = {}
      is_running = bool(getattr(scenario_run, "running", raw.get("running", False)))
      if is_running:
          current_scenario_run = None
          try:
              current_scenario_run = scenario.get_current_run()
          except Exception:
              current_scenario_run = None

          if current_scenario_run is not None and getattr(current_scenario_run, "id", None) == backend_run_id:
              scenario.abort()
          mark_run_stopped(run_id)
          return

  current_scenario_run = None
  try:
      current_scenario_run = scenario.get_current_run()
  except Exception:
      current_scenario_run = None

  if current_scenario_run is not None and getattr(current_scenario_run, "id", None) == backend_run_id:
      scenario.abort()

  mark_run_stopped(run_id)

def persist_prediction_results_to_classification_table(run_id: str, result_df: pd.DataFrame):
   run_df = fetch_classification_run_preview_df(run_id)
   if run_df.empty:
       raise ValueError(f"Run not found in MDM_classification_tbl: {run_id}")
   result_work_df = result_df.copy()
   rename_map = {
       "Model 1 OP": "model_1_op",
       "Model 2 OP": "model_2_op",
       "Model 3 OP": "model_3_op",
       "LLM as a Judge OP": "llm_judge_op",
       "LLM Decision Source": "llm_decision_source",
       "Explanation": "explanation",
   }
   result_work_df = result_work_df.rename(columns=rename_map)
   available_output_cols = [
       c for c in CLASSIFICATION_OUTPUT_COLUMNS
       if c in result_work_df.columns
   ]
   if not available_output_cols:
       raise ValueError(
           "Prediction result dataframe does not contain any expected output columns. "
           f"Expected one of {CLASSIFICATION_OUTPUT_COLUMNS}, "
           f"but got columns: {list(result_work_df.columns)}"
       )
   original_input_cols = get_original_input_columns(run_df.columns.tolist())
   overlapping_input_cols = [c for c in original_input_cols if c in result_work_df.columns]
   aligned_result_df = None
   if overlapping_input_cols:
       result_work_df["row_hash"] = build_row_hash_series(result_work_df[overlapping_input_cols])
       aligned_result_df = (
           result_work_df[["row_hash"] + available_output_cols]
           .drop_duplicates(subset=["row_hash"])
           .copy()
       )
   elif len(result_work_df) == len(run_df):
       aligned_result_df = run_df[["row_hash"]].copy().reset_index(drop=True)
       pred_only_df = result_work_df[available_output_cols].reset_index(drop=True)
       for col in available_output_cols:
           aligned_result_df[col] = pred_only_df[col]
   else:
       raise ValueError(
           "Could not align prediction results to table rows. "
           f"Run rows: {len(run_df)}, Result rows: {len(result_work_df)}, "
           f"Overlapping input cols: {overlapping_input_cols}, "
           f"Run columns: {list(run_df.columns)}, "
           f"Result columns: {list(result_work_df.columns)}"
       )
   info = _get_sql_dataset_location()
   for _, pred_row in aligned_result_df.iterrows():
       row_hash = str(pred_row["row_hash"]).replace("'", "''")
       set_parts = []
       for col in available_output_cols:
           set_parts.append(f"{_quote_identifier(col)} = {_sql_escape(pred_row[col])}")
       if "updated_ts" in get_classification_schema_columns():
           set_parts.append(f'{_quote_identifier("updated_ts")} = {_sql_escape(_utc_now_string())}')
       set_sql = ", ".join(set_parts)
       sql = f"""
       UPDATE {info['quoted_table_name']}
       SET {set_sql}
       WHERE run_id = '{str(run_id).replace("'", "''")}'
         AND row_hash = '{row_hash}'
       """
       _run_sql(sql)
