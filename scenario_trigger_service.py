import dataiku

from services.classification_table_service import (
    fetch_queued_run_ids,
    mark_run_queued,
    mark_run_running,
    mark_run_submitted,
    update_run_rows_by_run_id,
)

PROJECT_KEY = "MDM_PARTS_CLASSIFICATION"
WORKER_SCENARIO_IDS = [
    "RUN_PREDICTION_JOB_1",
    "RUN_PREDICTION_JOB_2",
    "RUN_PREDICTION_JOB_3",
]


def _safe_get_current_run(scenario):
    try:
        return scenario.get_current_run()
    except Exception:
        return None


def _extract_trigger_fire_id(trigger_fire):
    try:
        raw = trigger_fire.get_raw() or {}
    except Exception:
        raw = {}
    return raw.get("runId") or raw.get("triggerRunId") or raw.get("id")


def _try_start_on_worker(project, run_id: str, scenario_id: str):
    scenario = project.get_scenario(scenario_id)

    # Worker can only handle one app run at a time.
    if _safe_get_current_run(scenario) is not None:
        return None

    trigger_fire = scenario.run(params={"run_id": run_id})
    trigger_fire_id = _extract_trigger_fire_id(trigger_fire)

    # Persist worker assignment immediately so refresh/stop logic knows where the run is headed.
    update_run_rows_by_run_id(
        run_id,
        {
            "run_status": "submitted",
            "worker_scenario_id": scenario_id,
            "backend_trigger_id": trigger_fire_id,
        },
    )

    scenario_run = None
    try:
        scenario_run = trigger_fire.wait_for_scenario_run(no_fail=True)
    except Exception:
        scenario_run = None

    if scenario_run is not None:
        mark_run_running(run_id, backend_run_id=scenario_run.id)
        return {
            "worker_scenario_id": scenario_id,
            "backend_trigger_id": trigger_fire_id,
            "backend_run_id": scenario_run.id,
            "queued": False,
            "status": "running",
        }

    # Trigger was accepted, but the scenario run handle is not ready yet.
    return {
        "worker_scenario_id": scenario_id,
        "backend_trigger_id": trigger_fire_id,
        "backend_run_id": None,
        "queued": False,
        "status": "submitted",
    }


def trigger_prediction_scenario(run_id: str):
    client = dataiku.api_client()
    project = client.get_project(PROJECT_KEY)

    # Reset the run to a fresh backend submission state before assigning a worker.
    mark_run_submitted(run_id)

    for scenario_id in WORKER_SCENARIO_IDS:
        result = _try_start_on_worker(project, run_id, scenario_id)
        if result is not None:
            return result

    mark_run_queued(run_id)
    return {
        "worker_scenario_id": None,
        "backend_trigger_id": None,
        "backend_run_id": None,
        "queued": True,
        "status": "queued",
    }


def dispatch_queued_runs(max_to_dispatch: int = 3):
    client = dataiku.api_client()
    project = client.get_project(PROJECT_KEY)

    queued_run_ids = fetch_queued_run_ids(limit=max_to_dispatch * 3)
    started = []

    for run_id in queued_run_ids:
        for scenario_id in WORKER_SCENARIO_IDS:
            result = _try_start_on_worker(project, run_id, scenario_id)
            if result is not None:
                started.append(
                    {
                        "run_id": run_id,
                        "worker_scenario_id": result.get("worker_scenario_id"),
                        "backend_run_id": result.get("backend_run_id"),
                        "status": result.get("status"),
                    }
                )
                break
        if len(started) >= max_to_dispatch:
            break

    return started
