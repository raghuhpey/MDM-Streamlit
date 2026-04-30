import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import dataiku
import pandas as pd
from dataiku import SQLExecutor2

PROJECT_KEY = dataiku.default_project_key()

RETRAINING_DATASET_NAME = "mdm_retraining_jobs_tbl"
RETRAINING_STEPS_DATASET_NAME = "mdm_retraining_job_steps_tbl"
RETRAINING_SCENARIO_ID = "RUN_MODEL_RETRAINING"

# User-facing business stages (shown in UI)
BUSINESS_STAGES = [
    (1, "Prepare Input Data", [2, 3, 4]),
    (2, "Train Model 1", [5, 6, 7]),
    (3, "Train Model 2", [8, 9, 10]),
    (4, "Train Model 3", [11, 12, 13]),
    (5, "Refresh Knowledge Base", [14, 15, 16]),
    (6, "Update Models Repository", [17, 18, 19]),
]

# Actual scenario steps (tracked technically)
TECHNICAL_STEPS = [
    (1, "Mark Job Running", "system"),
    (2, "Mark Prepare Input Data Running", "Prepare Input Data"),
    (3, "Build Input Data output", "Prepare Input Data"),
    (4, "Mark Prepare Input Data Completed", "Prepare Input Data"),
    (5, "Mark Model 1 Running", "Train Model 1"),
    (6, "Build Model 1 output", "Train Model 1"),
    (7, "Mark Model 1 Completed", "Train Model 1"),
    (8, "Mark Model 2 Running", "Train Model 2"),
    (9, "Build Model 2 output", "Train Model 2"),
    (10, "Mark Model 2 Completed", "Train Model 2"),
    (11, "Mark Model 3 Running", "Train Model 3"),
    (12, "Build Model 3 output", "Train Model 3"),
    (13, "Mark Model 3 Completed", "Train Model 3"),
    (14, "Mark KB Refresh Running", "Refresh Knowledge Base"),
    (15, "Build KB", "Refresh Knowledge Base"),
    (16, "Mark KB Refresh Completed", "Refresh Knowledge Base"),
    (17, "Mark Models Repository Running", "Update Models Repository"),
    (18, "Build Models Repository output", "Update Models Repository"),
    (19, "Mark Models Repository Completed", "Update Models Repository"),
    (20, "Mark Job Completed", "system"),
]

RUNNING_STATUSES = {"created", "submitted", "running"}
FAILED_OUTCOMES = {"failed", "aborted", "killed", "cancelled"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_str() -> str:
    return _utc_now().strftime("%Y-%m-%d %H:%M:%S UTC")


def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return default
    return text


def _clean_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (hasattr(pd, "isna") and pd.isna(value)):
            return default
        return int(float(value))
    except Exception:
        return default


def _clean_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (hasattr(pd, "isna") and pd.isna(value)):
            return default
        return float(value)
    except Exception:
        return default


def _safe_sql(value: Any) -> str:
    return _clean_text(value).replace("'", "''")


def _dataset(dataset_name: str):
    return dataiku.Dataset(dataset_name)


def _dataset_sql_info(dataset_name: str) -> Dict[str, Any]:
    ds = _dataset(dataset_name)
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


def _query_to_df(sql: str) -> pd.DataFrame:
   ds = _dataset(RETRAINING_DATASET_NAME)
   executor = SQLExecutor2(dataset=ds)
   return executor.query_to_df(sql)

def _query_to_df_on_dataset(dataset_name: str, sql: str) -> pd.DataFrame:
    executor = SQLExecutor2(dataset=_dataset(dataset_name))
    return executor.query_to_df(sql)




def _fetch_rows(dataset_name: str, where_sql: str = "", order_sql: str = "") -> pd.DataFrame:
    info = _dataset_sql_info(dataset_name)
    sql = f"SELECT * FROM {info['quoted_table_name']}"
    if where_sql:
        sql += f"\nWHERE {where_sql}"
    if order_sql:
        sql += f"\nORDER BY {order_sql}"
    df = _query_to_df_on_dataset(dataset_name, sql)
    if df is None:
        return pd.DataFrame()
    return df.where(pd.notna(df), None)


def generate_retraining_job_id(job_name: str = "") -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    base = "retrain"
    job_name_clean = ""
    if job_name:
        job_name_clean = "".join(
            c if c.isalnum() or c in ("_", "-") else "_" for c in str(job_name).strip()
        )
        job_name_clean = "_".join([x for x in job_name_clean.split("_") if x])[:40]
    return f"{base}_{job_name_clean + '_' if job_name_clean else ''}{ts}_{suffix}"


def create_retraining_job(job_name: str = "", remarks: str = "", created_by: str = "") -> Dict[str, Any]:
    retraining_job_id = generate_retraining_job_id(job_name)
    created_ts = _utc_now_str()
    info = _dataset_sql_info(RETRAINING_DATASET_NAME)
    sql = f"""
    INSERT INTO {info['quoted_table_name']} (
        retraining_job_id,
        job_name,
        status,
        current_stage,
        current_stage_no,
        total_stages,
        completed_stages,
        progress_pct,
        created_by,
        remarks,
        created_ts,
        started_ts,
        completed_ts,
        duration_seconds,
        scenario_id,
        scenario_run_id,
        trigger_id,
        model_version,
        output_artifact_path,
        error_message,
        updated_ts
    ) VALUES (
        '{_safe_sql(retraining_job_id)}',
        '{_safe_sql(job_name)}',
        'created',
        '',
        0,
        {len(BUSINESS_STAGES)},
        0,
        0,
        '{_safe_sql(created_by)}',
        '{_safe_sql(remarks)}',
        '{_safe_sql(created_ts)}',
        NULL,
        NULL,
        NULL,
        '{_safe_sql(RETRAINING_SCENARIO_ID)}',
        '',
        '',
        '',
        '',
        '',
        '{_safe_sql(created_ts)}'
    )
    """
    _run_sql_on_dataset(RETRAINING_DATASET_NAME, sql)
    initialize_retraining_job_steps(retraining_job_id)
    return fetch_retraining_job_metadata(retraining_job_id) or {
        "retraining_job_id": retraining_job_id,
        "job_name": job_name,
        "remarks": remarks,
        "created_by": created_by,
        "status": "created",
    }

def stop_retraining_job(retraining_job_id: str) -> dict:
   """
   Stops the Dataiku scenario run (if available) and marks job as stopped.
   Returns a dict with stop status.
   """
   meta = fetch_retraining_job_metadata(retraining_job_id)
   if not meta:
       raise ValueError(f"Retraining job not found: {retraining_job_id}")
   status = str(meta.get("status") or "").strip().lower()
   scenario_id = str(meta.get("scenario_id") or RETRAINING_SCENARIO_ID).strip()
   scenario_run_id = str(meta.get("scenario_run_id") or "").strip()
   # Allow stop only for active statuses
   if status not in {"running", "submitted", "queued"}:
       return {"ok": False, "message": f"Job is not active (status={status})."}
   # Try aborting the scenario run (best-effort)
   aborted = False
   abort_error = ""
   try:
       if scenario_run_id:
           client = dataiku.api_client()
           project = client.get_project(PROJECT_KEY)
           scenario = project.get_scenario(scenario_id)
           # Dataiku API supports aborting a running scenario run
           # (method name differs slightly across versions; handle both)
           run = scenario.get_run(scenario_run_id)
           if hasattr(run, "abort"):
               run.abort()
           elif hasattr(run, "stop"):
               run.stop()
           else:
               # Fall back: raise explicit message
               raise RuntimeError("Scenario run object has no abort/stop method in this DSS version.")
           aborted = True
       else:
           abort_error = "scenario_run_id is empty; cannot abort scenario run via API."
   except Exception as e:
       abort_error = str(e)
   # Mark job as stopped in SQL table regardless (UI should reflect stop request)
   now_str = _utc_now_str()
   update_retraining_job_status(
       retraining_job_id=retraining_job_id,
       status="stopped",
       completed_ts=now_str,
       error_message=(abort_error or "Stopped by user"),
       current_step="Stopped by user",
   )
   # OPTIONAL: also mark any currently running step as failed/stopped
   try:
       steps_df = fetch_retraining_job_steps_df(retraining_job_id)
       if steps_df is not None and not steps_df.empty:
           running = steps_df[steps_df["step_status"].astype(str).str.lower() == "running"]
           if not running.empty:
               step_no = int(running.sort_values("step_no").iloc[0]["step_no"])
               mark_retraining_step_failed(
                   retraining_job_id=retraining_job_id,
                   step_no=step_no,
                   error_message="Stopped by user",
               )
   except Exception:
       pass
   if aborted:
       return {"ok": True, "message": "Stop requested. Scenario run aborted."}
   return {"ok": True, "message": f"Stop requested. (Scenario abort warning: {abort_error})"}

   
def initialize_retraining_job_steps(retraining_job_id: str):
    info = _dataset_sql_info(RETRAINING_STEPS_DATASET_NAME)
    now_str = _utc_now_str()
    values = []
    for step_no, step_name, business_stage in TECHNICAL_STEPS:
        business_stage_no = next((no for no, name, step_nos in BUSINESS_STAGES if business_stage == name), None)
        values.append(
            f"""(
                '{_safe_sql(retraining_job_id)}',
                {step_no},
                '{_safe_sql(step_name)}',
                '{_safe_sql(business_stage)}',
                {business_stage_no if business_stage_no is not None else 'NULL'},
                'pending',
                NULL,
                NULL,
                NULL,
                '',
                '',
                '{_safe_sql(now_str)}'
            )"""
        )
    sql = f"""
    INSERT INTO {info['quoted_table_name']} (
        retraining_job_id,
        step_no,
        step_name,
        business_stage,
        business_stage_no,
        step_status,
        started_ts,
        completed_ts,
        duration_seconds,
        message,
        error_message,
        updated_ts
    ) VALUES {', '.join(values)}
    """
    _run_sql_on_dataset(RETRAINING_STEPS_DATASET_NAME, sql)


def _fetch_retraining_job_metadata_raw(retraining_job_id: str) -> Optional[Dict[str, Any]]:
    df = _fetch_rows(
        RETRAINING_DATASET_NAME,
        where_sql=f"retraining_job_id = '{_safe_sql(retraining_job_id)}'",
        order_sql="created_ts DESC",
    )
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def _fetch_retraining_job_steps_raw(retraining_job_id: str) -> pd.DataFrame:
    return _fetch_rows(
        RETRAINING_STEPS_DATASET_NAME,
        where_sql=f"retraining_job_id = '{_safe_sql(retraining_job_id)}'",
        order_sql="step_no ASC",
    )


def _fetch_retraining_job_metadata_raw(retraining_job_id: str):
   info = _dataset_sql_info(RETRAINING_DATASET_NAME)
   sql = f"""
   SELECT *
   FROM {info['quoted_table_name']}
   WHERE retraining_job_id = '{_safe_sql(retraining_job_id)}'
   LIMIT 1
   """
   df = _query_to_df(sql)
   if df is None or df.empty:
       return None
   df = df.where(pd.notna(df), None)
   return df.iloc[0].to_dict()

def update_retraining_job_status(
    retraining_job_id: str,
    status: str,
    started_ts: Optional[str] = None,
    completed_ts: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    model_version: Optional[str] = None,
    output_artifact_path: Optional[str] = None,
    error_message: Optional[str] = None,
    remarks: Optional[str] = None,
    current_stage: Optional[str] = None,
    current_stage_no: Optional[int] = None,
    total_stages: Optional[int] = None,
    completed_stages: Optional[int] = None,
    progress_pct: Optional[float] = None,
    scenario_run_id: Optional[str] = None,
    trigger_id: Optional[str] = None,
):
    info = _dataset_sql_info(RETRAINING_DATASET_NAME)
    updates = [
        f"status = '{_safe_sql(status)}'",
        f"updated_ts = '{_safe_sql(_utc_now_str())}'",
    ]
    if started_ts is not None:
        updates.append(f"started_ts = '{_safe_sql(started_ts)}'")
    if completed_ts is not None:
        updates.append(f"completed_ts = '{_safe_sql(completed_ts)}'")
    if duration_seconds is not None:
        updates.append(f"duration_seconds = {float(duration_seconds)}")
    if model_version is not None:
        updates.append(f"model_version = '{_safe_sql(model_version)}'")
    if output_artifact_path is not None:
        updates.append(f"output_artifact_path = '{_safe_sql(output_artifact_path)}'")
    if error_message is not None:
        updates.append(f"error_message = '{_safe_sql(error_message)}'")
    if remarks is not None:
        updates.append(f"remarks = '{_safe_sql(remarks)}'")
    if current_stage is not None:
        updates.append(f"current_stage = '{_safe_sql(current_stage)}'")
    if current_stage_no is not None:
        updates.append(f"current_stage_no = {int(current_stage_no)}")
    if total_stages is not None:
        updates.append(f"total_stages = {int(total_stages)}")
    if completed_stages is not None:
        updates.append(f"completed_stages = {int(completed_stages)}")
    if progress_pct is not None:
        updates.append(f"progress_pct = {float(progress_pct)}")
    if scenario_run_id is not None:
        updates.append(f"scenario_run_id = '{_safe_sql(scenario_run_id)}'")
    if trigger_id is not None:
        updates.append(f"trigger_id = '{_safe_sql(trigger_id)}'")

    sql = f"""
    UPDATE {info['quoted_table_name']}
    SET {', '.join(updates)}
    WHERE retraining_job_id = '{_safe_sql(retraining_job_id)}'
    """
    _run_sql_on_dataset(RETRAINING_DATASET_NAME, sql)


def update_retraining_job_after_submit(retraining_job_id: str, scenario_run_id: str = "", trigger_id: str = ""):
    update_retraining_job_status(
        retraining_job_id=retraining_job_id,
        status="submitted",
        scenario_run_id=scenario_run_id,
        trigger_id=trigger_id,
    )


def trigger_retraining_scenario(retraining_job_id: str):
   client = dataiku.api_client()
   project = client.get_project(PROJECT_KEY)
   scenario = project.get_scenario(RETRAINING_SCENARIO_ID)
   scenario_run = scenario.run(params={"retraining_job_id": retraining_job_id})
   scenario_run_id = (
       getattr(scenario_run, "id", None)
       or getattr(scenario_run, "run_id", None)
       or getattr(scenario_run, "runId", None)
       or ""
   )
   trigger_id = (
       getattr(scenario_run, "trigger_id", None)
       or getattr(scenario_run, "triggerId", None)
       or ""
   )
   scenario_run_id = str(scenario_run_id).strip() if scenario_run_id else ""
   trigger_id = str(trigger_id).strip() if trigger_id else ""
   print(f"[retraining] retraining_job_id={retraining_job_id}")
   print(f"[retraining] scenario_run_id={scenario_run_id}")
   print(f"[retraining] trigger_id={trigger_id}")
   update_retraining_job_after_submit(
       retraining_job_id=retraining_job_id,
       scenario_run_id=scenario_run_id,
       trigger_id=trigger_id,
   )
   return {
       "scenario_id": RETRAINING_SCENARIO_ID,
       "scenario_run_id": scenario_run_id,
       "trigger_id": trigger_id,
   }


def mark_retraining_step_running(retraining_job_id: str, step_no: int, message: str = ""):
    info = _dataset_sql_info(RETRAINING_STEPS_DATASET_NAME)
    now_str = _utc_now_str()
    sql = f"""
    UPDATE {info['quoted_table_name']}
    SET step_status = 'running',
        started_ts = COALESCE(started_ts, '{_safe_sql(now_str)}'),
        message = '{_safe_sql(message)}',
        error_message = '',
        updated_ts = '{_safe_sql(now_str)}'
    WHERE retraining_job_id = '{_safe_sql(retraining_job_id)}'
      AND step_no = {int(step_no)}
    """
    _run_sql_on_dataset(RETRAINING_STEPS_DATASET_NAME, sql)
    refresh_retraining_job_progress(retraining_job_id, forced_status="running")


def _calc_duration_seconds(started_ts: Optional[str], completed_ts: str) -> Optional[float]:
    started = _clean_text(started_ts)
    if not started:
        return None
    try:
        start_dt = datetime.strptime(started, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(completed_ts, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
        return (end_dt - start_dt).total_seconds()
    except Exception:
        return None


def mark_retraining_step_completed(retraining_job_id: str, step_no: int, message: str = ""):
    info = _dataset_sql_info(RETRAINING_STEPS_DATASET_NAME)
    now_str = _utc_now_str()
    steps_df = _fetch_retraining_job_steps_raw(retraining_job_id)
    row = steps_df[steps_df["step_no"] == int(step_no)]
    started_ts = row.iloc[0]["started_ts"] if not row.empty else None
    duration_seconds = _calc_duration_seconds(started_ts, now_str)
    duration_sql = "NULL" if duration_seconds is None else str(float(duration_seconds))
    sql = f"""
    UPDATE {info['quoted_table_name']}
    SET step_status = 'completed',
        completed_ts = '{_safe_sql(now_str)}',
        duration_seconds = {duration_sql},
        message = '{_safe_sql(message)}',
        error_message = '',
        updated_ts = '{_safe_sql(now_str)}'
    WHERE retraining_job_id = '{_safe_sql(retraining_job_id)}'
      AND step_no = {int(step_no)}
    """
    _run_sql_on_dataset(RETRAINING_STEPS_DATASET_NAME, sql)
    refresh_retraining_job_progress(retraining_job_id)


def mark_retraining_step_failed(retraining_job_id: str, step_no: int, error_message: str = ""):
    info = _dataset_sql_info(RETRAINING_STEPS_DATASET_NAME)
    now_str = _utc_now_str()
    steps_df = _fetch_retraining_job_steps_raw(retraining_job_id)
    row = steps_df[steps_df["step_no"] == int(step_no)]
    started_ts = row.iloc[0]["started_ts"] if not row.empty else None
    duration_seconds = _calc_duration_seconds(started_ts, now_str)
    duration_sql = "NULL" if duration_seconds is None else str(float(duration_seconds))
    sql = f"""
    UPDATE {info['quoted_table_name']}
    SET step_status = 'failed',
        completed_ts = '{_safe_sql(now_str)}',
        duration_seconds = {duration_sql},
        error_message = '{_safe_sql(error_message)}',
        updated_ts = '{_safe_sql(now_str)}'
    WHERE retraining_job_id = '{_safe_sql(retraining_job_id)}'
      AND step_no = {int(step_no)}
    """
    _run_sql_on_dataset(RETRAINING_STEPS_DATASET_NAME, sql)
    refresh_retraining_job_progress(retraining_job_id, forced_status="failed", forced_error=error_message)


def _extract_scenario_outcome(run_details: Any) -> str:
    if run_details is None:
        return ""
    if isinstance(run_details, dict):
        for key in ("result", "outcome", "status", "scenarioState"):
            value = run_details.get(key)
            if value:
                return _clean_text(value).lower()
        scenario_run = run_details.get("scenarioRun") or {}
        for key in ("result", "outcome", "status", "scenarioState"):
            value = scenario_run.get(key)
            if value:
                return _clean_text(value).lower()
    else:
        for attr in ("result", "outcome", "status", "scenarioState"):
            value = getattr(run_details, attr, None)
            if value:
                return _clean_text(value).lower()
    return ""


def reconcile_retraining_job_status_with_scenario(retraining_job_id: str):
   meta = _fetch_retraining_job_metadata_raw(retraining_job_id)
   if not meta:
       return
   current_status = str(meta.get("status") or "").strip().lower()
   scenario_run_id = str(meta.get("scenario_run_id") or "").strip()
   if current_status not in {"submitted", "running"}:
       return
   if scenario_run_id in {"", "nan", "none", "null"}:
       print(f"[retraining][reconcile] missing scenario_run_id for {retraining_job_id}")
       return
   try:
       client = dataiku.api_client()
       project = client.get_project(PROJECT_KEY)
       scenario = project.get_scenario(RETRAINING_SCENARIO_ID)
       run = scenario.get_run(scenario_run_id)
       run_details = run.get_details()
       outcome = str(
           run_details.get("result")
           or run_details.get("outcome")
           or run_details.get("status")
           or ""
       ).strip().lower()
       print(f"[retraining][reconcile] job={retraining_job_id}, run={scenario_run_id}, outcome={outcome}")
   except Exception as e:
       print(f"[retraining][reconcile] failed for {retraining_job_id}: {e}")
       return
   if outcome in {"failed", "aborted", "cancelled", "killed"}:
       update_retraining_job_status(
           retraining_job_id=retraining_job_id,
           status="failed",
           completed_ts=_utc_now_str(),
           error_message="Retraining scenario failed. Check Dataiku scenario logs.",
       )

def refresh_retraining_job_progress(
    retraining_job_id: str,
    forced_status: Optional[str] = None,
    forced_error: Optional[str] = None,
):
    steps_df = _fetch_retraining_job_steps_raw(retraining_job_id)
    if steps_df.empty:
        return

    # Normalize statuses
    steps_df = steps_df.copy()
    steps_df["step_status_norm"] = steps_df["step_status"].apply(lambda x: _clean_text(x).lower())

    business_rows: List[Dict[str, Any]] = []
    for stage_no, stage_name, technical_step_nos in BUSINESS_STAGES:
        stage_df = steps_df[steps_df["step_no"].isin(technical_step_nos)].sort_values("step_no")
        if stage_df.empty:
            business_rows.append({
                "stage_no": stage_no,
                "stage_name": stage_name,
                "status": "pending",
                "started_ts": None,
                "completed_ts": None,
                "duration_seconds": None,
            })
            continue
        statuses = set(stage_df["step_status_norm"].tolist())
        if "failed" in statuses:
            stage_status = "failed"
        elif "running" in statuses:
            stage_status = "running"
        elif all(s == "completed" for s in stage_df["step_status_norm"].tolist()):
            stage_status = "completed"
        else:
            stage_status = "pending"

        started_candidates = [_clean_text(x) for x in stage_df["started_ts"].tolist() if _clean_text(x)]
        completed_candidates = [_clean_text(x) for x in stage_df["completed_ts"].tolist() if _clean_text(x)]
        duration_values = [_clean_float(x, 0.0) for x in stage_df["duration_seconds"].tolist() if x is not None]
        business_rows.append({
            "stage_no": stage_no,
            "stage_name": stage_name,
            "status": stage_status,
            "started_ts": min(started_candidates) if started_candidates else None,
            "completed_ts": max(completed_candidates) if completed_candidates else None,
            "duration_seconds": round(sum(duration_values), 2) if duration_values else None,
        })

    completed_stages = sum(1 for row in business_rows if row["status"] == "completed")
    running_row = next((row for row in business_rows if row["status"] == "running"), None)
    failed_row = next((row for row in business_rows if row["status"] == "failed"), None)
    pending_row = next((row for row in business_rows if row["status"] == "pending"), None)

    if forced_status == "failed" and failed_row is None and running_row is not None:
        failed_row = running_row
        running_row = None

    if failed_row is not None:
        overall_status = "failed"
        current_stage = failed_row["stage_name"]
        current_stage_no = failed_row["stage_no"]
    elif running_row is not None:
        overall_status = "running"
        current_stage = running_row["stage_name"]
        current_stage_no = running_row["stage_no"]
    elif completed_stages == len(BUSINESS_STAGES):
        overall_status = "completed"
        current_stage = "Completed"
        current_stage_no = len(BUSINESS_STAGES)
    else:
        overall_status = forced_status or "submitted"
        current_stage = pending_row["stage_name"] if pending_row else ""
        current_stage_no = pending_row["stage_no"] if pending_row else 0

    progress_pct = round((completed_stages / len(BUSINESS_STAGES)) * 100.0, 2)

    update_retraining_job_status(
        retraining_job_id=retraining_job_id,
        status=overall_status,
        current_stage=current_stage,
        current_stage_no=current_stage_no,
        total_stages=len(BUSINESS_STAGES),
        completed_stages=completed_stages,
        progress_pct=progress_pct,
        completed_ts=_utc_now_str() if overall_status in {"completed", "failed"} else None,
        error_message=forced_error if forced_status == "failed" else None,
    )


def fetch_retraining_job_metadata(retraining_job_id: str):
   try:
       reconcile_retraining_job_status_with_scenario(retraining_job_id)
   except Exception as e:
       print(f"[retraining][meta] reconcile failed for {retraining_job_id}: {e}")
   return _fetch_retraining_job_metadata_raw(retraining_job_id)



def fetch_retraining_jobs_df():
   info = _dataset_sql_info(RETRAINING_DATASET_NAME)
   sql = f"""
   SELECT *
   FROM {info['quoted_table_name']}
   ORDER BY created_ts DESC
   """
   df = _query_to_df(sql)
   if df is None or df.empty:
       return df
   for _, row in df.iterrows():
       retraining_job_id = str(row.get("retraining_job_id") or "").strip()
       status = str(row.get("status") or "").strip().lower()
       if retraining_job_id and status in {"submitted", "running"}:
           try:
               reconcile_retraining_job_status_with_scenario(retraining_job_id)
           except Exception as e:
               print(f"[retraining][jobs] reconcile failed for {retraining_job_id}: {e}")
   df = _query_to_df(sql)
   return df if df is not None else None


def fetch_retraining_job_steps_df(retraining_job_id: str) -> pd.DataFrame:
    return _fetch_retraining_job_steps_raw(retraining_job_id)


def fetch_retraining_business_stages_df(retraining_job_id: str) -> pd.DataFrame:
    steps_df = fetch_retraining_job_steps_df(retraining_job_id)
    if steps_df.empty:
        return pd.DataFrame(columns=[
            "stage_no", "stage_name", "status", "started_ts", "completed_ts", "duration_seconds"
        ])
    refresh_retraining_job_progress(retraining_job_id)
    rows = []
    steps_df = fetch_retraining_job_steps_df(retraining_job_id)
    steps_df["step_status_norm"] = steps_df["step_status"].apply(lambda x: _clean_text(x).lower())
    for stage_no, stage_name, technical_step_nos in BUSINESS_STAGES:
        stage_df = steps_df[steps_df["step_no"].isin(technical_step_nos)].sort_values("step_no")
        statuses = set(stage_df["step_status_norm"].tolist()) if not stage_df.empty else {"pending"}
        if "failed" in statuses:
            stage_status = "failed"
        elif "running" in statuses:
            stage_status = "running"
        elif not stage_df.empty and all(s == "completed" for s in stage_df["step_status_norm"].tolist()):
            stage_status = "completed"
        else:
            stage_status = "pending"
        started_candidates = [_clean_text(x) for x in stage_df["started_ts"].tolist() if _clean_text(x)]
        completed_candidates = [_clean_text(x) for x in stage_df["completed_ts"].tolist() if _clean_text(x)]
        duration_values = [_clean_float(x, 0.0) for x in stage_df["duration_seconds"].tolist() if x is not None]
        message = ""
        error_message = ""
        if not stage_df.empty:
            last_row = stage_df.sort_values("step_no").iloc[-1].to_dict()
            message = _clean_text(last_row.get("message"))
            error_message = _clean_text(last_row.get("error_message"))
        rows.append({
            "stage_no": stage_no,
            "stage_name": stage_name,
            "status": stage_status,
            "started_ts": min(started_candidates) if started_candidates else None,
            "completed_ts": max(completed_candidates) if completed_candidates else None,
            "duration_seconds": round(sum(duration_values), 2) if duration_values else None,
            "message": message,
            "error_message": error_message,
        })
    return pd.DataFrame(rows)
