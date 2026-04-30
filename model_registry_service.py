import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import dataiku
import pandas as pd
from dataiku import SQLExecutor2

PROJECT_KEY = dataiku.default_project_key()
MODEL_REGISTRY_DATASET = "mdm_model_registry_tbl"
DEPLOYMENTS_DATASET = "mdm_model_deployments_tbl" 
DEPLOYMENT_SCENARIO_ID = "RUN_MODEL_DEPLOYMENT"  # if decide to use a scenario later
RETRAINING_JOBS_DATASET = "mdm_retraining_jobs_tbl"
# Managed folders (IDs)
MODEL_REPOSITORY_FOLDER_ID = "MODELS_REPOSITORY"
MODEL_METRICS_FOLDER_ID = "MODEL_METRICS"
ACTIVE_SUBFOLDER = "active"  # MODEL_REPOSITORY/active/<version> (your definition)
METRICS_CSV_REL_PATH = f"/{MODEL_METRICS_FOLDER_ID}/model_deploy_metrics.csv"  # inside MODEL_METRICS/


# --------------------------
# helpers
# --------------------------
def _utc_now_str() -> str:
   return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def _safe_sql(value: Any) -> str:
   if value is None:
       return ""
   return str(value).replace("'", "''")

def _dataset_sql_info(dataset_name: str) -> dict:
   ds = dataiku.Dataset(dataset_name)
   loc = ds.get_location_info()
   info = loc.get("info", {})
   connection_name = info.get("connectionName")
   quoted_table_name = info.get("quotedResolvedTableName")
   if not connection_name or not quoted_table_name:
       raise ValueError(f"Could not resolve SQL info for dataset: {dataset_name}")
   return {
       "dataset": ds,
       "connection_name": connection_name,
       "quoted_table_name": quoted_table_name,
       "project_key": ds.project_key,
   }

def _run_sql_on_dataset(dataset_name: str, sql: str):
   info = _dataset_sql_info(dataset_name)
   client = dataiku.api_client()
   query = client.sql_query(
       sql,
       connection=info["connection_name"],
       type="sql",
       project_key=info["project_key"],
       post_queries=["COMMIT"],
   )
   query.verify()

def _query_to_df(dataset_name: str, sql: str) -> pd.DataFrame:
   ds = dataiku.Dataset(dataset_name)
   executor = SQLExecutor2(dataset=ds)
   return executor.query_to_df(sql)

def _read_metrics_csv_df() -> pd.DataFrame:
   """
   Reads MODEL_METRICS/model_deploy_metrics.csv from managed folder MODEL_METRICS.
   """
   mf = dataiku.Folder(MODEL_REPOSITORY_FOLDER_ID)
   with mf.get_download_stream(METRICS_CSV_REL_PATH) as s:
       raw = s.read()
   # pandas can read bytes via BytesIO
   from io import BytesIO
   df = pd.read_csv(BytesIO(raw))
   return df

def _normalize_version(v: Any) -> str:
   if v is None:
       return ""
   s = str(v).strip()
   # allow numeric versions in CSV (1,2,3)
   if s.endswith(".0"):
       s = s[:-2]
   return s

def _is_truthy(v: Any) -> bool:
   return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _get_latest_retraining_job_for_version(version: str) -> dict:
   """
   Find the retraining job that produced this version.
   mdm_retraining_jobs_tbl.model_version should match the numeric version like '1', '2', etc.
   """
   info = _dataset_sql_info(RETRAINING_JOBS_DATASET)
   table = info["quoted_table_name"]
   v = _safe_sql(str(version).replace("v", "").strip())
   sql = f"""
   SELECT
       retraining_job_id,
       job_name
   FROM {table}
   WHERE TRIM(COALESCE(CAST(model_version AS TEXT), '')) = '{v}'
   ORDER BY
       completed_ts DESC NULLS LAST,
       updated_ts DESC NULLS LAST,
       created_ts DESC NULLS LAST
   LIMIT 1
   """
   dfj = _query_to_df(RETRAINING_JOBS_DATASET, sql)
   if dfj is None or dfj.empty:
       return {"retraining_job_id": "", "retraining_job_name": ""}
   row = dfj.iloc[0].to_dict()
   return {
       "retraining_job_id": str(row.get("retraining_job_id") or ""),
       "retraining_job_name": str(row.get("job_name") or ""),
   }


# --------------------------
# Registry reads
# --------------------------
def fetch_model_registry_df() -> pd.DataFrame:
   info = _dataset_sql_info(MODEL_REGISTRY_DATASET)
   sql = f"""
   SELECT *
   FROM {info['quoted_table_name']}
   ORDER BY trained_completed_ts DESC NULLS LAST, trained_started_ts DESC NULLS LAST
   """
   return _query_to_df(MODEL_REGISTRY_DATASET, sql)

def fetch_model_registry_row(version: str) -> Optional[Dict[str, Any]]:
   version = _normalize_version(version)
   info = _dataset_sql_info(MODEL_REGISTRY_DATASET)
   sql = f"""
   SELECT *
   FROM {info['quoted_table_name']}
   WHERE model_version = '{_safe_sql(version)}'
   LIMIT 1
   """
   df = _query_to_df(MODEL_REGISTRY_DATASET, sql)
   if df is None or df.empty:
       return None
   return df.iloc[0].to_dict()

def fetch_deployments_df(limit: int = 200) -> pd.DataFrame:
   info = _dataset_sql_info(DEPLOYMENTS_DATASET)
   sql = f"""
   SELECT *
   FROM {info['quoted_table_name']}
   ORDER BY deployed_ts DESC NULLS LAST
   LIMIT {int(limit)}
   """
   return _query_to_df(DEPLOYMENTS_DATASET, sql)

# --------------------------
# Upserts from metrics CSV
# --------------------------
def upsert_registry_from_metrics_row(
   *,
   version: str,
   trained_started_ts: Optional[str] = None,
   trained_completed_ts: Optional[str] = None,
   trained_by: str = "",
   overall_accuracy: Optional[float] = None,
   metrics_json: str = "",
   metrics_csv_path: str = "",
   retraining_job_id: str = "",
   retraining_job_name: str = "",
):
   version = _normalize_version(version)
   now = _utc_now_str()
   info = _dataset_sql_info(MODEL_REGISTRY_DATASET)
   table = info["quoted_table_name"]
   existing = fetch_model_registry_row(version)
   is_active = bool(existing.get("is_active_inference")) if existing else False
   active_since_ts = existing.get("active_since_ts") if existing else None
   if is_active and not active_since_ts:
       active_since_ts = now
   sql = f"""
   INSERT INTO {table} (
       model_version,
       trained_started_ts,
       trained_completed_ts,
       trained_by,
       overall_accuracy,
       metrics_json,
       metrics_csv_path,
       retraining_job_id,
       retraining_job_name,
       is_active_inference,
       active_since_ts,
       last_updated_ts
   )
   VALUES (
       '{_safe_sql(version)}',
       {("NULL" if not trained_started_ts else f"'{_safe_sql(trained_started_ts)}'")},
       {("NULL" if not trained_completed_ts else f"'{_safe_sql(trained_completed_ts)}'")},
       '{_safe_sql(trained_by)}',
       {("NULL" if overall_accuracy is None else str(float(overall_accuracy)))},
       '{_safe_sql(metrics_json)}',
       '{_safe_sql(metrics_csv_path)}',
       '{_safe_sql(retraining_job_id)}',
       '{_safe_sql(retraining_job_name)}',
       {("TRUE" if is_active else "FALSE")},
       {("NULL" if not active_since_ts else f"'{_safe_sql(active_since_ts)}'")},
       '{_safe_sql(now)}'
   )
   ON CONFLICT (model_version)
   DO UPDATE SET
       trained_started_ts = COALESCE(EXCLUDED.trained_started_ts, {table}.trained_started_ts),
       trained_completed_ts = COALESCE(EXCLUDED.trained_completed_ts, {table}.trained_completed_ts),
       trained_by = CASE WHEN EXCLUDED.trained_by <> '' THEN EXCLUDED.trained_by ELSE {table}.trained_by END,
       overall_accuracy = COALESCE(EXCLUDED.overall_accuracy, {table}.overall_accuracy),
       metrics_json = CASE WHEN EXCLUDED.metrics_json <> '' THEN EXCLUDED.metrics_json ELSE {table}.metrics_json END,
       metrics_csv_path = CASE WHEN EXCLUDED.metrics_csv_path <> '' THEN EXCLUDED.metrics_csv_path ELSE {table}.metrics_csv_path END,
       retraining_job_id = CASE WHEN EXCLUDED.retraining_job_id <> '' THEN EXCLUDED.retraining_job_id ELSE {table}.retraining_job_id END,
       retraining_job_name = CASE WHEN EXCLUDED.retraining_job_name <> '' THEN EXCLUDED.retraining_job_name ELSE {table}.retraining_job_name END,
       is_active_inference = {table}.is_active_inference,
       active_since_ts = {table}.active_since_ts,
       last_updated_ts = EXCLUDED.last_updated_ts
   ;
   """
   _run_sql_on_dataset(MODEL_REGISTRY_DATASET, sql)

def refresh_registry_from_metrics_csv(trained_by: str = "") -> Dict[str, Any]:
    """
    Reads the CSV and upserts registry rows for all versions found.
    Also back-fills retraining_job_id + retraining_job_name by looking up mdm_retraining_jobs_tbl.
    """
    df = _read_metrics_csv_df()
    if df is None or df.empty:
        return {"ok": False, "message": "Metrics CSV is empty or missing rows."}
    updated = 0
    errors: List[str] = []
    for _, r in df.iterrows():
        try:
            version = _normalize_version(r.get("version"))
            if not version:
                continue
            # full row json for UI comparisons
            row_dict = {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
            metrics_json = json.dumps(row_dict, default=str)
            overall_acc = r.get("overall_accuracy")
            try:
                overall_acc = float(overall_acc) if overall_acc is not None and not pd.isna(overall_acc) else None
            except Exception:
                overall_acc = None
            completed = r.get("datetime")
            completed_ts = str(completed) if completed is not None and not pd.isna(completed) else None

            job_info = _get_latest_retraining_job_for_version(version)
            retraining_job_id = job_info.get("retraining_job_id", "")
            retraining_job_name = job_info.get("retraining_job_name", "")
            upsert_registry_from_metrics_row(
                version=version,
                trained_started_ts=None,
                trained_completed_ts=completed_ts,
                trained_by=trained_by,
                overall_accuracy=overall_acc,
                metrics_json=metrics_json,
                metrics_csv_path=f"{MODEL_METRICS_FOLDER_ID}/{METRICS_CSV_REL_PATH}",
                retraining_job_id=retraining_job_id,
                retraining_job_name=retraining_job_name,
            )
            updated += 1
        except Exception as e:
            errors.append(str(e))
    return {"ok": True, "updated": updated, "errors": errors}


# --------------------------
# Deployment
# --------------------------
def _set_active_version(version: str):
   """
   Marks exactly one version active in mdm_model_registry_tbl.
   """
   version = _normalize_version(version)
   now = _utc_now_str()
   info = _dataset_sql_info(MODEL_REGISTRY_DATASET)
   table = info["quoted_table_name"]
   sql1 = f"""
   UPDATE {table}
   SET is_active_inference = FALSE, last_updated_ts = '{_safe_sql(now)}'
   WHERE is_active_inference = TRUE
   """
   sql2 = f"""
   UPDATE {table}
   SET is_active_inference = TRUE,
       active_since_ts = COALESCE(active_since_ts, '{_safe_sql(now)}'),
       last_updated_ts = '{_safe_sql(now)}'
   WHERE model_version = '{_safe_sql(version)}'
   """
   _run_sql_on_dataset(MODEL_REGISTRY_DATASET, sql1)
   _run_sql_on_dataset(MODEL_REGISTRY_DATASET, sql2)

def _copy_version_to_active_folder(version: str) -> Dict[str, Any]:
   """
   Copies MODEL_REPOSITORY/<version>/... to MODEL_REPOSITORY/active/<version>/...
   using managed folder APIs (works with Azure).
   """
   version = _normalize_version(version)
   repo = dataiku.Folder(MODEL_REPOSITORY_FOLDER_ID)
   src_prefix = f"{version}/"
   dst_prefix = f"{ACTIVE_SUBFOLDER}/{version}/"
   # list source objects
   paths = repo.list_paths_in_partition()
   src_paths = [p for p in paths if p.startswith(src_prefix)]
   if not src_paths:
       return {"ok": False, "message": f"No files found under MODEL_REPOSITORY/{src_prefix}"}
   copied = 0
   for p in src_paths:
       dst = dst_prefix + p[len(src_prefix):]
       with repo.get_download_stream(p) as s_in:
           repo.upload_stream(dst, s_in)
       copied += 1
   return {"ok": True, "copied_files": copied, "active_path": f"{MODEL_REPOSITORY_FOLDER_ID}/{dst_prefix}"}

def append_deployment_log(
   *,
   version: str,
   deployed_by: str,
   status: str,
   message: str = "",
   scenario_run_id: str = "",
   trigger_id: str = "",
):
   now = _utc_now_str()
   info = _dataset_sql_info(DEPLOYMENTS_DATASET)
   table = info["quoted_table_name"]
   deployment_id = f"deploy_{_safe_sql(version)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
   sql = f"""
   INSERT INTO {table} (
       deployment_id,
       model_version,
       deployed_ts,
       deployed_by,
       status,
       message,
       scenario_run_id,
       trigger_id
   )
   VALUES (
       '{_safe_sql(deployment_id)}',
       '{_safe_sql(_normalize_version(version))}',
       '{_safe_sql(now)}',
       '{_safe_sql(deployed_by)}',
       '{_safe_sql(status)}',
       '{_safe_sql(message)}',
       '{_safe_sql(scenario_run_id)}',
       '{_safe_sql(trigger_id)}'
   )
   """
   _run_sql_on_dataset(DEPLOYMENTS_DATASET, sql)

def deploy_model_version(version: str, deployed_by: str = "") -> Dict[str, Any]:
   """
   Deployment semantics you defined:
   - copy MODEL_REPOSITORY/<version> -> MODEL_REPOSITORY/active/<version>
   - mark version active in mdm_model_registry_tbl
   - write deployment log row in mdm_model_deployments_tbl
   """
   version = _normalize_version(version)
   if not version:
       return {"ok": False, "message": "Missing version."}
   try:
       # 1) copy artifacts
       res = _copy_version_to_active_folder(version)
       if not res.get("ok"):
           append_deployment_log(
               version=version, deployed_by=deployed_by, status="failed", message=res.get("message", "")
           )
           return res
       # 2) set active in registry
       _set_active_version(version)
       # 3) log success
       append_deployment_log(
           version=version,
           deployed_by=deployed_by,
           status="completed",
           message=f"Deployed {version}. Copied {res.get('copied_files')} files to active.",
       )
       return {"ok": True, "message": f"Deployed version {version}", "details": res}
   except Exception as e:
       append_deployment_log(version=version, deployed_by=deployed_by, status="failed", message=str(e))
       return {"ok": False, "message": f"Deployment failed: {e}"}

# --------------------------
# Compare helpers
# --------------------------
def get_metrics_rows_for_versions(versions: List[str]) -> pd.DataFrame:
   """
   Returns metrics rows (from CSV) filtered by versions.
   """
   df = _read_metrics_csv_df()
   if df is None or df.empty:
       return pd.DataFrame()
   wanted = {_normalize_version(v) for v in versions if _normalize_version(v)}
   df["version_norm"] = df["version"].apply(_normalize_version)
   out = df[df["version_norm"].isin(wanted)].copy()
   return out

def list_available_versions_from_csv() -> List[str]:
   df = _read_metrics_csv_df()
   if df is None or df.empty or "version" not in df.columns:
       return []
   versions = sorted({_normalize_version(v) for v in df["version"].tolist() if _normalize_version(v)}, key=lambda x: int(x) if x.isdigit() else x)
   return versions