import dataiku

from services.classification_table_service import (
    fetch_classification_run_metadata,
    mark_run_completed,
    mark_run_failed,
    mark_run_running,
    mark_run_stopped,
    update_run_rows_by_run_id,
)

PROJECT_KEY = "MDM_PARTS_CLASSIFICATION"


def _safe_get_raw(obj):
    try:
        return obj.get_raw() or {}
    except Exception:
        return {}


def _safe_refresh(obj):
    try:
        obj.refresh()
    except Exception:
        pass


def _safe_get_current_run(scenario):
    try:
        return scenario.get_current_run()
    except Exception:
        return None


def _safe_get_run(scenario, backend_run_id: str):
    try:
        return scenario.get_run(backend_run_id)
    except Exception:
        return None


def _infer_running_and_outcome(scenario_run):
    raw = _safe_get_raw(scenario_run)
    running = bool(getattr(scenario_run, "running", raw.get("running", False)))
    outcome = getattr(scenario_run, "outcome", None)

    if not outcome:
        result_info = raw.get("result")
        if isinstance(result_info, dict):
            outcome = result_info.get("outcome") or result_info.get("status")
        elif isinstance(result_info, str):
            outcome = result_info

    if not outcome:
        outcome = raw.get("outcome") or raw.get("status")

    return running, str(outcome or "").upper().strip()


def refresh_backend_run_status(run_id: str):
    meta = fetch_classification_run_metadata(run_id)
    if meta is None:
        return None

    current_status = str(meta.get("run_status") or "").strip().lower()
    if current_status in {"completed", "failed", "stopped"}:
        return meta

    worker_scenario_id = str(meta.get("worker_scenario_id") or "").strip()
    backend_run_id = str(meta.get("backend_run_id") or "").strip()

    if not worker_scenario_id:
        return meta

    client = dataiku.api_client()
    project = client.get_project(PROJECT_KEY)
    scenario = project.get_scenario(worker_scenario_id)

    if not backend_run_id:
        current_run = _safe_get_current_run(scenario)
        if current_run is None:
            return meta
        update_run_rows_by_run_id(run_id, {"backend_run_id": current_run.id})
        mark_run_running(run_id, backend_run_id=current_run.id)
        return fetch_classification_run_metadata(run_id)

    scenario_run = _safe_get_run(scenario, backend_run_id)
    if scenario_run is None:
        current_run = _safe_get_current_run(scenario)
        if current_run is not None and getattr(current_run, "id", None) == backend_run_id:
            mark_run_running(run_id, backend_run_id=backend_run_id)
            return fetch_classification_run_metadata(run_id)
        return meta

    _safe_refresh(scenario_run)
    running, outcome = _infer_running_and_outcome(scenario_run)

    if running:
        if current_status != "running":
            mark_run_running(run_id, backend_run_id=backend_run_id)
        return fetch_classification_run_metadata(run_id)

    if outcome == "SUCCESS":
        mark_run_completed(run_id)
    elif outcome == "ABORTED":
        mark_run_stopped(run_id)
    elif outcome:
        mark_run_failed(run_id)

    return fetch_classification_run_metadata(run_id)
