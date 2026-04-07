from __future__ import annotations

import pandas as pd

from services.classification_table_service import (
   fetch_classification_run_input_df,
   mark_run_completed,
   mark_run_failed,
   mark_run_running,
   persist_prediction_results_to_classification_table,
)
from services.prediction_engine import predict_dataframe
from services.prediction_resources import load_prediction_resources
from services.run_storage_service import save_ignored_records_csv, save_prediction_output_df


def _persist_partial_prediction_batch(run_id: str, batch_df: pd.DataFrame):
   if batch_df is None or batch_df.empty:
       return
   try:
       print(f"[prediction_service] persisting partial batch for run {run_id}: rows={len(batch_df)}, columns={list(batch_df.columns)}")
       if "row_hash" in batch_df.columns:
           print(f"[prediction_service] partial batch row_hash non-null={int(batch_df['row_hash'].notna().sum())}")
       persist_prediction_results_to_classification_table(run_id, batch_df)
       print(f"[prediction_service] partial progress persist succeeded for run {run_id}: rows={len(batch_df)}")
   except Exception as exc:
       print(f"[prediction_service] partial progress persist failed for run {run_id}: {type(exc).__name__}: {exc}")

def run_prediction_job(run_id: str, max_workers: int = 4):
   input_df = fetch_classification_run_input_df(run_id)
   print("*************************************** Input DF ***************************************************")
   print(input_df)
   if input_df is None or input_df.empty:
       raise ValueError(f"No input rows found in MDM_classification_tbl for run: {run_id}")
   mark_run_running(run_id)
   try:
       resources = load_prediction_resources()
       result_df, ignored_df = predict_dataframe(
           input_df=input_df,
           resources=resources,
           max_workers=max_workers,
           partial_result_callback=lambda batch_df: _persist_partial_prediction_batch(run_id, batch_df),
           progress_batch_size=10,
        )
       print("*************************************** Predicted Output ******************************************")
       print(result_df)
       save_prediction_output_df(run_id, result_df)
       print("*************************************** save_prediction_output_df done ******************************************")
       save_ignored_records_csv(run_id, ignored_df)
       persist_prediction_results_to_classification_table(run_id, result_df)
       mark_run_completed(run_id)
       return result_df
   except Exception:
       mark_run_failed(run_id)
       raise