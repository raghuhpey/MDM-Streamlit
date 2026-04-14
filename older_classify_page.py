import base64
import hashlib
from io import BytesIO
import time
import pandas as pd
import streamlit as st
from state.session import set_navigation_state
from services.scenario_trigger_service import trigger_prediction_scenario
from services.run_storage_service import (
 load_input_preview_df,
 load_output_df,
 load_run_metadata,
)
from services.classification_table_service import (
 fetch_classification_run_display_df,
 fetch_classification_run_metadata,
 fetch_classification_run_preview_df,
 fetch_classification_runs_summary_df,
 insert_uploaded_rows_into_classification_table,
 request_stop_run_in_classification_table,
)
from utils.pagination import get_page_bounds, get_page_df, get_total_pages
from utils.role_ui import get_visible_top_nav_items
from ui.layout import render_sidebar, render_top_nav
from utils.styles import (
 get_app_shell_css,
 get_classify_page_css,
 get_global_css,
 get_home_page_css,
 load_css,
)
CLASSIFICATION_OUTPUT_COLUMNS = [
 "model_1_op",
 "model_2_op",
 "model_3_op",
 "llm_judge_op",
 "llm_decision_source",
 "explanation",
]
CLASSIFICATION_HIDDEN_COLUMNS = {
 "run_id",
 "file_name",
 "uploaded_by",
 "uploaded_ts",
 "run_status",
 "backend_run_id",
 "backend_trigger_id",
 "prediction_started_ts",
 "prediction_completed_ts",
 "created_ts",
 "updated_ts",
 "row_hash",
 "validation_status",
 "validated_unspsc",
 "validation_decision_source",
 "validated_by",
 "validated_at",
 "validation_comment",
 "is_golden_record",
}
RUN_SYSTEM_COLUMNS = CLASSIFICATION_HIDDEN_COLUMNS.copy()
ACTIVE_RUN_STATUSES = {"queued", "submitted", "running"}
RESTARTABLE_RUN_STATUSES = {"uploaded", "failed", "stopped"}
def get_base64_image(image_path):
 with open(image_path, "rb") as img_file:
     return base64.b64encode(img_file.read()).decode()
def nav_pill(label, slug, active=False):
 css_class = "nav-pill active" if active else "nav-pill"
 return f"""
<form method="get" style="margin:0;">
<input type="hidden" name="page" value="{slug}">
<button type="submit" class="{css_class}">{label}</button>
</form>
 """
def left_menu_btn(label, page_value, sub_value, active=False):
 css_class = "left-menu-btn active" if active else "left-menu-btn"
 return f"""
<form method="get" class="left-menu-form">
<input type="hidden" name="page" value="{page_value}">
<input type="hidden" name="sub" value="{sub_value}">
<button type="submit" class="{css_class}">{label}</button>
</form>
 """
def get_current_page_slug():
 page = st.query_params.get("page", st.session_state.get("nav_page", "classify_existing"))
 if isinstance(page, list):
     page = page[0]
 return page
def read_uploaded_file(file_name, file_bytes):
 lower_name = file_name.lower()
 if lower_name.endswith(".xlsx"):
     return pd.read_excel(BytesIO(file_bytes))
 if lower_name.endswith(".csv"):
     encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "iso-8859-1"]
     last_error = None
     for enc in encodings_to_try:
         try:
             return pd.read_csv(BytesIO(file_bytes), encoding=enc)
         except Exception as e:
             last_error = e
     raise ValueError(
         f"Could not read CSV with supported encodings. Last error: {last_error}"
     )
 raise ValueError("Unsupported file type. Please upload CSV or XLSX.")
def format_preview_dataframe(df):
 preview_df = df.copy()
 for col in preview_df.columns:
     col_series = preview_df[col]
     if str(col_series.dtype) in ["bool", "boolean"]:
         preview_df[col] = col_series.map(lambda x: "Yes" if x else "No")
 return preview_df
def get_selected_submenu():
 current_page = get_current_page_slug()
 if current_page == "create_new_parts":
     return "create_new_parts"
 selected = st.query_params.get("sub", st.session_state.get("nav_sub", "existing_parts"))
 if isinstance(selected, list):
     selected = selected[0]
 if selected not in ["existing_parts", "runs", "run_detail"]:
     selected = "existing_parts"
 return selected
def get_selected_run_id():
 run_id = st.query_params.get("run_id", st.session_state.get("nav_run_id"))
 if isinstance(run_id, list):
     run_id = run_id[0]
 return run_id
def get_menu_config(selected_menu):
 if selected_menu == "existing_parts":
     return {
         "page_title": "Input Preview",
         "upload_title": "Upload file to classify",
         "hint_text": "Upload a supported file to create a prediction run and preview the input data.",
         "run_type": "classify_existing_parts",
         "current_run_key": "current_existing_run_id",
         "signature_key": "existing_parts_upload_signature",
         "action_button_label": "Classify",
         "scenario_id": "RUN_PREDICTION_JOB",
     }
 return {
     "page_title": "Input Preview",
     "upload_title": "Upload file to create new parts",
     "hint_text": "Upload a supported file to create a run and preview the input data.",
     "run_type": "create_new_parts",
     "current_run_key": "current_new_parts_run_id",
     "signature_key": "new_parts_upload_signature",
     "action_button_label": "Create Parts",
     "scenario_id": None,
 }
def render_paginated_dataframe(df, pager_prefix, default_page_size=10):
 total_rows = len(df)
 if total_rows == 0:
     st.markdown(
         """
<div class="empty-state-box">
 No rows available.
</div>
         """,
         unsafe_allow_html=True
     )
     return
 page_key = f"{pager_prefix}_page"
 if page_key not in st.session_state:
     st.session_state[page_key] = 1
 page_size = default_page_size
 total_pages = get_total_pages(total_rows, page_size)
 current_page = st.session_state[page_key]
 current_page = max(1, min(current_page, total_pages))
 st.session_state[page_key] = current_page
 start_idx, end_idx = get_page_bounds(current_page, page_size, total_rows)
 left_col, center_col, right_col = st.columns([2, 5.9, 1.0])
 with left_col:
     prev_col, next_col = st.columns(2)
     with prev_col:
         if st.button(
             "Prev",
             key=f"{pager_prefix}_prev",
             disabled=(current_page == 1),
             use_container_width=True
         ):
             st.session_state[page_key] = current_page - 1
             st.rerun()
     with next_col:
         if st.button(
             "Next",
             key=f"{pager_prefix}_next",
             disabled=(current_page == total_pages),
             use_container_width=True
         ):
             st.session_state[page_key] = current_page + 1
             st.rerun()
 with center_col:
     st.markdown(
         f"""
<div class="pagination-summary">
 Page {current_page} / {total_pages} - Rows {start_idx + 1}-{end_idx} of {total_rows}
</div>
         """,
         unsafe_allow_html=True
     )
 with right_col:
   #   st.markdown(
   #       '<div class="pagination-right-label">Go to page</div>',
   #       unsafe_allow_html=True
   #   )
     selected_page = st.number_input(
         "Go to page",
         min_value=1,
         max_value=total_pages,
         value=current_page,
         step=1,
         key=f"{pager_prefix}_page_input",
         label_visibility="collapsed"
     )
     if selected_page != current_page:
         st.session_state[page_key] = int(selected_page)
         st.rerun()
 page_df = get_page_df(df, st.session_state[page_key], page_size)
 preview_df = format_preview_dataframe(page_df)
 st.markdown('<div class="pagination-table-wrap">', unsafe_allow_html=True)
 st.dataframe(preview_df, use_container_width=True, hide_index=True)
 st.markdown("</div>", unsafe_allow_html=True)
def create_table_backed_run(uploaded_file, selected_menu):
 config = get_menu_config(selected_menu)
 file_bytes = uploaded_file.getvalue()
 signature = f"{uploaded_file.name}:{hashlib.md5(file_bytes).hexdigest()}"
 if st.session_state.get(config["signature_key"]) == signature:
     return st.session_state.get(f"{selected_menu}_last_upload_stats")
 upload_df = read_uploaded_file(uploaded_file.name, file_bytes)
 result = insert_uploaded_rows_into_classification_table(
     upload_df=upload_df,
     file_name=uploaded_file.name,
 )
 st.session_state[config["current_run_key"]] = result["run_id"]
 st.session_state[f"{selected_menu}_last_upload_stats"] = result
 st.session_state[f"{selected_menu}_preview_columns"] = result["input_columns"]
 st.session_state[config["signature_key"]] = signature
 return result
def load_table_run_into_state(run_id: str, selected_menu: str = "existing_parts"):
 config = get_menu_config(selected_menu)
 run_meta = fetch_classification_run_metadata(run_id)
 if run_meta is None:
     raise ValueError(f"Run not found in MDM_classification_tbl: {run_id}")
 preview_df = fetch_classification_run_preview_df(run_id)
 preview_columns = [
     c for c in preview_df.columns
     if c not in {
         "run_id",
         "file_name",
         "uploaded_by",
         "uploaded_ts",
         "run_status",
         "backend_run_id",
         "backend_trigger_id",
         "prediction_started_ts",
         "prediction_completed_ts",
         "created_ts",
         "updated_ts",
         "row_hash",
         "validation_status",
     }
 ]
 st.session_state[config["current_run_key"]] = run_id
 st.session_state[f"{selected_menu}_last_upload_stats"] = {
     "run_id": run_meta["run_id"],
     "file_name": run_meta["file_name"],
     "uploaded_by": run_meta["uploaded_by"],
     "total_uploaded": run_meta["row_count"],
     "inserted_count": run_meta["row_count"],
     "duplicate_count": 0,
     "run_status": run_meta["run_status"],
     "input_columns": preview_columns,
 }
 st.session_state[f"{selected_menu}_preview_columns"] = preview_columns
def render_runs_filters(runs_df: pd.DataFrame) -> pd.DataFrame:
 st.markdown(
     """
<div class="section-card">
<div class="section-card-title" style="margin-bottom: 14px;">Filter Runs</div>
</div>
     """,
     unsafe_allow_html=True
 )
 st.markdown("")
 filter_cols_top = st.columns(4)
 with filter_cols_top[0]:
     run_id_filter = st.text_input(
         "Run ID",
         key="runs_filter_run_id",
         placeholder="Contains..."
     )
 with filter_cols_top[1]:
     file_name_filter = st.text_input(
         "File Name",
         key="runs_filter_file_name",
         placeholder="Contains..."
     )
 with filter_cols_top[2]:
     created_by_filter = st.text_input(
         "Created By",
         key="runs_filter_created_by",
         placeholder="Contains..."
     )
 with filter_cols_top[3]:
     status_options = sorted(
         [str(v) for v in runs_df["run_status"].dropna().astype(str).unique().tolist()]
     )
     selected_statuses = st.multiselect(
         "Status",
         options=status_options,
         default=st.session_state.get("runs_filter_status", []),
         key="runs_filter_status",
         placeholder="All"
     )
 filter_cols_bottom = st.columns(4)
 with filter_cols_bottom[0]:
     rows_min = st.number_input(
         "Rows Min",
         min_value=0,
         value=int(st.session_state.get("runs_filter_rows_min", 0)),
         step=1,
         key="runs_filter_rows_min"
     )
 with filter_cols_bottom[1]:
     rows_max_raw = st.text_input(
         "Rows Max",
         value=st.session_state.get("runs_filter_rows_max", ""),
         key="runs_filter_rows_max",
         placeholder="No max"
     )
 with filter_cols_bottom[2]:
     created_filter = st.text_input(
         "Created",
         key="runs_filter_created",
         placeholder="Contains date/time..."
     )
 with filter_cols_bottom[3]:
     action_options = sorted(
         [str(v) for v in runs_df["action_label"].dropna().astype(str).unique().tolist()]
     )
     selected_actions = st.multiselect(
         "Action",
         options=action_options,
         default=st.session_state.get("runs_filter_action", []),
         key="runs_filter_action",
         placeholder="All"
     )
 clear_col_left, clear_col_right = st.columns([2, 8])
 with clear_col_left:
     if st.button("Clear Filters", key="runs_filter_clear_btn", use_container_width=True):
         reset_values = {
             "runs_filter_run_id": "",
             "runs_filter_file_name": "",
             "runs_filter_created_by": "",
             "runs_filter_status": [],
             "runs_filter_rows_min": 0,
             "runs_filter_rows_max": "",
             "runs_filter_created": "",
             "runs_filter_action": [],
         }
         for reset_key, reset_value in reset_values.items():
             st.session_state[reset_key] = reset_value
         st.rerun()
 filtered_df = runs_df.copy()
 if run_id_filter:
     filtered_df = filtered_df[
         filtered_df["run_id"].astype(str).str.contains(run_id_filter, case=False, na=False)
     ]
 if file_name_filter:
     filtered_df = filtered_df[
         filtered_df["file_name"].astype(str).str.contains(file_name_filter, case=False, na=False)
     ]
 if created_by_filter:
     filtered_df = filtered_df[
         filtered_df["uploaded_by"].astype(str).str.contains(created_by_filter, case=False, na=False)
     ]
 if selected_statuses:
     filtered_df = filtered_df[
         filtered_df["run_status"].astype(str).isin(selected_statuses)
     ]
 filtered_df["row_count_numeric"] = pd.to_numeric(filtered_df["row_count"], errors="coerce")
 if rows_min:
     filtered_df = filtered_df[filtered_df["row_count_numeric"] >= rows_min]
 rows_max = None
 if str(rows_max_raw).strip():
     try:
         rows_max = int(str(rows_max_raw).strip())
     except ValueError:
         st.warning("Rows Max must be a whole number.")
 if rows_max is not None:
     filtered_df = filtered_df[filtered_df["row_count_numeric"] <= rows_max]
 if created_filter:
     filtered_df = filtered_df[
         filtered_df["uploaded_ts"].astype(str).str.contains(created_filter, case=False, na=False)
     ]
 if selected_actions:
     filtered_df = filtered_df[
         filtered_df["action_label"].astype(str).isin(selected_actions)
     ]
 filtered_df = filtered_df.drop(columns=["row_count_numeric"], errors="ignore")
 st.divider()
#   st.caption(f"Showing {len(filtered_df)} of {len(runs_df)} runs")
 return filtered_df
def render_upload_view(selected_menu):
 config = get_menu_config(selected_menu)
 st.markdown(
     f'<div class="panel-title">{config["upload_title"]}</div>',
     unsafe_allow_html=True
 )
 uploaded_file = st.file_uploader(
     "Upload file",
     type=["csv", "xlsx"],
     label_visibility="collapsed",
     key=f"uploader_{selected_menu}"
 )
 if uploaded_file is not None:
     try:
         if selected_menu == "existing_parts":
             create_table_backed_run(uploaded_file, selected_menu)
         else:
             st.info("Create New Parts upload-to-table flow is not wired yet.")
     except Exception as e:
         st.error(f"File upload failed: {e}")
 current_run_id = st.session_state.get(config["current_run_key"])
 upload_stats = st.session_state.get(f"{selected_menu}_last_upload_stats")
 preview_columns = st.session_state.get(f"{selected_menu}_preview_columns", [])
 if selected_menu != "existing_parts":
     st.markdown(
         """
<div class="section-card">
<div class="section-card-title">Create New Parts</div>
<div class="section-card-subtitle">
 Backend table flow for Create New Parts will be implemented next.
</div>
<div class="empty-state-box">
 No preview available yet.
</div>
</div>
         """,
         unsafe_allow_html=True
     )
     return
 if current_run_id and not upload_stats:
     run_meta = fetch_classification_run_metadata(current_run_id)
     if run_meta is not None:
         preview_df = fetch_classification_run_preview_df(current_run_id)
         preview_columns = [
             c for c in preview_df.columns
             if c not in {
                 "run_id",
                 "file_name",
                 "uploaded_by",
                 "uploaded_ts",
                 "run_status",
                 "backend_run_id",
                 "backend_trigger_id",
                 "prediction_started_ts",
                 "prediction_completed_ts",
                 "created_ts",
                 "updated_ts",
                 "row_hash",
             }
         ]
         upload_stats = {
             "run_id": run_meta["run_id"],
             "file_name": run_meta["file_name"],
             "uploaded_by": run_meta["uploaded_by"],
             "total_uploaded": run_meta["row_count"],
             "inserted_count": run_meta["row_count"],
             "duplicate_count": 0,
             "run_status": run_meta["run_status"],
             "input_columns": preview_columns,
         }
         st.session_state[f"{selected_menu}_last_upload_stats"] = upload_stats
         st.session_state[f"{selected_menu}_preview_columns"] = preview_columns
 if not current_run_id or not upload_stats:
     st.markdown(
         """
<div class="section-card">
<div class="section-card-title">Preview</div>
<div class="section-card-subtitle">
 Upload a file to insert unique rows into MDM_classification_tbl and preview the uploaded batch.
</div>
<div class="empty-state-box">
 No preview available yet.
</div>
</div>
         """,
         unsafe_allow_html=True
     )
     return
 current_run_meta = fetch_classification_run_metadata(current_run_id)
 current_status = current_run_meta["run_status"] if current_run_meta else upload_stats["run_status"]
 display_df = fetch_classification_run_display_df(current_run_id)
 if not display_df.empty:
  if current_status == "completed":
      display_columns = []
      # input columns first
      for col in preview_columns:
          if col in display_df.columns and col not in display_columns:
              display_columns.append(col)
      # prediction columns next
      for col in CLASSIFICATION_OUTPUT_COLUMNS:
          if col in display_df.columns and col not in display_columns:
              display_columns.append(col)
      if display_columns:
          display_df = display_df[display_columns].copy()
  else:
      display_columns = [c for c in preview_columns if c in display_df.columns]
      if display_columns:
          display_df = display_df[display_columns].copy()
  # final safety: remove workflow/system columns if they still exist
  display_df = display_df[
      [c for c in display_df.columns if c not in CLASSIFICATION_HIDDEN_COLUMNS]
  ].copy()
 st.markdown(
     f"""
<div class="section-card">
<div class="section-card-title">Preview</div>
<div class="section-card-subtitle">
<b>Run ID:</b> {current_run_id} &nbsp;&nbsp;|&nbsp;&nbsp;
<b>File:</b> {upload_stats["file_name"]} &nbsp;&nbsp;|&nbsp;&nbsp;
<b>Total Uploaded:</b> {upload_stats["total_uploaded"]} &nbsp;&nbsp;|&nbsp;&nbsp;
<b>New Rows Inserted:</b> {upload_stats["inserted_count"]} &nbsp;&nbsp;|&nbsp;&nbsp;
<b>Duplicates Skipped:</b> {upload_stats["duplicate_count"]} &nbsp;&nbsp;|&nbsp;&nbsp;
<b>Status:</b> {current_status}
</div>
</div>
     """,
     unsafe_allow_html=True
 )
 if upload_stats["inserted_count"] == 0 and current_status == "uploaded":
     st.warning(
         "All uploaded rows were duplicates of records already present in MDM_classification_tbl. "
         "No new rows were inserted for this upload."
     )
 if display_df.empty:
     st.markdown(
         """
<div class="section-card">
<div class="empty-state-box">
 No preview rows available for this run.
</div>
</div>
         """,
         unsafe_allow_html=True
     )
 else:
     st.divider()
     render_paginated_dataframe(
         display_df,
         pager_prefix=f"{current_run_id}_table_preview"
     )
 action_cols = st.columns([8, 2])
 clicked = False
 show_action_button = current_status in RESTARTABLE_RUN_STATUSES or current_status in ACTIVE_RUN_STATUSES
 if show_action_button:
   if current_status in ACTIVE_RUN_STATUSES:
       action_button_label = "Stop"
   else:
       action_button_label = config["action_button_label"]
   action_cols = st.columns([8, 2])
   with action_cols[1]:
       clicked = st.button(
           action_button_label,
           key=f"{selected_menu}_action_btn_{current_run_id}",
           use_container_width=True,
           )
 if clicked:
     if current_status in RESTARTABLE_RUN_STATUSES:
         try:
             with st.spinner("Submitting classification..."):
                 trigger_prediction_scenario(current_run_id)
             st.rerun()
         except Exception as e:
             st.error(f"Failed to submit classification: {e}")
             st.rerun()
     elif current_status in {"submitted", "running"}:
         st.warning("This run is already in progress.")
     elif current_status == "completed":
         st.info("This run is already completed.")
     else:
         st.info(f"Current run status is '{current_status}'.")
 if current_status in {"submitted", "running"}:
     with st.spinner("Running classification..."):
         st.info("Track classification progress anytime in the Runs tab.")
         time.sleep(3)
     st.rerun()
def render_runs_view():
 runs_df = fetch_classification_runs_summary_df()
 print("========== RUNS VIEW DEBUG ==========")
 print("runs_df columns:", list(runs_df.columns))
 if not runs_df.empty:
    cols_to_show = [c for c in ["run_id", "row_count", "predicted_row_count", "remaining_row_count", "run_status"] if c in runs_df.columns]
    print(runs_df[cols_to_show].head(20))
 print("=====================================")
 has_active_runs = False
 if not runs_df.empty and "run_status" in runs_df.columns:
     has_active_runs = runs_df["run_status"].astype(str).str.strip().str.lower().isin({"queued", "submitted", "running"}).any()
 st.markdown(
     """
<div class="runs-placeholder">
<div class="runs-placeholder-title">Runs</div>
<div class="runs-placeholder-text">
 Runs are now built directly from MDM_classification_tbl, including uploaded runs.
</div>
</div>
     """,
     unsafe_allow_html=True
 )
 if runs_df.empty:
     st.markdown(
         """
<div class="section-card">
<div class="empty-state-box">
 No runs available yet.
</div>
</div>
         """,
         unsafe_allow_html=True
     )
     return
 st.markdown("")
 runs_df = render_runs_filters(runs_df)
#  st.caption("Remaining shows rows without persisted prediction outputs yet. For running jobs, it updates when predictions are written back to the table.")
 if runs_df.empty:
     st.markdown(
         """
<div class="section-card">
<div class="empty-state-box">
 No runs match the selected filters.
</div>
</div>
         """,
         unsafe_allow_html=True
     )
     return
 header_cols = st.columns([2.5, 2.3, 1.4, 0.9, 1.1, 1.3, 1.8, 1.1])
 headers = ["Run ID", "File Name", "Created By", "Rows", "Predicted", "Remaining", "Status", "Action"]
 for col, header in zip(header_cols, headers):
     with col:
         st.markdown(
             f'<div class="runs-header-cell">{header}</div>',
             unsafe_allow_html=True
         )
 for _, run in runs_df.iterrows():
     run_id = str(run["run_id"])
     action_label = str(run["action_label"])
     status_class = str(run["run_status"]).strip().lower()
     row_cols = st.columns([2.5, 2.3, 1.4, 0.9, 1.1, 1.3, 1.8, 1.1])
     with row_cols[0]:
         st.markdown(f'<div class="runs-cell">{run_id}</div>', unsafe_allow_html=True)
     with row_cols[1]:
         st.markdown(f'<div class="runs-cell">{run["file_name"]}</div>', unsafe_allow_html=True)
     with row_cols[2]:
         st.markdown(f'<div class="runs-cell">{run["uploaded_by"]}</div>', unsafe_allow_html=True)
     with row_cols[3]:
         st.markdown(f'<div class="runs-cell">{int(run["row_count"])}</div>', unsafe_allow_html=True)
     with row_cols[4]:
        predicted_val = run["predicted_row_count"] if "predicted_row_count" in run.index else "MISSING"
        st.markdown(f'<div class="runs-cell">{predicted_val}</div>', unsafe_allow_html=True)
     with row_cols[5]:
        remaining_val = run["remaining_row_count"] if "remaining_row_count" in run.index else "MISSING"
        st.markdown(f'<div class="runs-cell">{remaining_val}</div>', unsafe_allow_html=True)
     with row_cols[6]:
         st.markdown(
             f'<div class="runs-cell"><span class="status-pill {status_class}">{run["run_status"]}</span></div>',
             unsafe_allow_html=True
         )
     with row_cols[7]:
         if action_label == "Continue":
             if st.button(
                 "Continue",
                 key=f"continue_run_{run_id}",
                 use_container_width=True
             ):
                 load_table_run_into_state(run_id, selected_menu="existing_parts")
                 set_navigation_state(
                  page="classify_existing",
                  sub="existing_parts",
                  run_id=run_id,
                  )
                 st.rerun()
         elif action_label == "Stop":
             if st.button(
                 "Stop",
                 key=f"stop_run_{run_id}",
                 use_container_width=True
             ):
                 try:
                     request_stop_run_in_classification_table(run_id)
                     st.success(f"Stop requested for run {run_id}.")
                     st.rerun()
                 except Exception as e:
                     st.error(f"Unable to stop run {run_id}: {e}")
         else:
             if st.button(
                 "View",
                 key=f"view_run_{run_id}",
                 use_container_width=True
             ):
                 load_table_run_into_state(run_id, selected_menu="existing_parts")
                 set_navigation_state(
                    page="classify_existing",
                    sub="existing_parts",
                    run_id=run_id,
                    )
                 st.rerun()
     st.divider()
 if has_active_runs:
     st.info("Active runs detected. This table auto-refreshes every 5 seconds.")
     time.sleep(5)
     st.rerun()
def render_run_detail_view():
 run_id = get_selected_run_id()
 run_record = load_run_metadata(run_id) if run_id else None
 if run_record is None:
     st.error("Run not found.")
     return
 top_cols = st.columns([1.5, 6])
 with top_cols[0]:
     if st.button("Ã¢â€ Â Back to Runs", key="back_to_runs_btn", use_container_width=True):
         set_navigation_state(
             page="classify_existing",
             sub="runs",
             run_id=None,
         )
         st.rerun()
 st.markdown(
     f"""
<div class="run-meta-card">
<div class="run-meta-title">Run Detail</div>
<div class="run-meta-grid">
<div class="run-meta-item">
<div class="run-meta-label">Run ID</div>
<div class="run-meta-value">{run_record["prediction_job_id"]}</div>
</div>
<div class="run-meta-item">
<div class="run-meta-label">File Name</div>
<div class="run-meta-value">{run_record["file_name"]}</div>
</div>
<div class="run-meta-item">
<div class="run-meta-label">Status</div>
<div class="run-meta-value">{run_record["status"]}</div>
</div>
<div class="run-meta-item">
<div class="run-meta-label">Created By</div>
<div class="run-meta-value">{run_record["created_by"]}</div>
</div>
<div class="run-meta-item">
<div class="run-meta-label">Rows</div>
<div class="run-meta-value">{run_record["row_count"]}</div>
</div>
<div class="run-meta-item">
<div class="run-meta-label">Created At</div>
<div class="run-meta-value">{run_record["created_at"]}</div>
</div>
</div>
</div>
     """,
     unsafe_allow_html=True
 )
 input_df = load_input_preview_df(run_record["prediction_job_id"])
 st.markdown(
     """
<div class="section-card">
<div class="section-card-title">Input Preview</div>
</div>
     """,
     unsafe_allow_html=True
 )
 if input_df is None:
     st.markdown(
         """
<div class="empty-state-box">
 Input preview file not found.
</div>
         """,
         unsafe_allow_html=True
     )
 else:
     render_paginated_dataframe(
         input_df,
         pager_prefix=f'{run_record["prediction_job_id"]}_detail_input'
     )
 output_df = load_output_df(run_record["prediction_job_id"])
 if output_df is not None:
     st.markdown(
         """
<div class="section-card">
<div class="section-card-title">Output Preview</div>
<div class="section-card-subtitle">
 Output loaded from the managed folder.
</div>
</div>
         """,
         unsafe_allow_html=True
     )
     render_paginated_dataframe(
         output_df,
         pager_prefix=f'{run_record["prediction_job_id"]}_detail_output'
     )
 else:
     st.markdown(
         """
<div class="section-card">
<div class="section-card-title">Output Preview</div>
<div class="section-card-subtitle">
 Output is not available yet.
</div>
<div class="empty-state-box">
 No output available yet.
</div>
</div>
         """,
         unsafe_allow_html=True
     )
def render():
 load_css(get_global_css() + get_app_shell_css() + get_home_page_css() + get_classify_page_css())
 selected_menu = get_selected_submenu()
 set_navigation_state(
      page=get_current_page_slug(),
      sub=selected_menu,
      run_id=get_selected_run_id(),
      )
 current_page = get_current_page_slug()
 if current_page == "create_new_parts":
     menu_items = [
         {
             "label": "Create New Parts",
             "page": "create_new_parts",
             "sub": "create_new_parts",
             "active": True,
         }
     ]
     sidebar_subtitle = "Create New Parts"
 elif selected_menu in {"runs", "run_detail"}:
     menu_items = [
         {
             "label": "RUNS",
             "page": "classify_existing",
             "sub": "runs",
             "active": True,
         }
     ]
     sidebar_subtitle = "Runs"
 else:
     menu_items = [
         {
             "label": "Classify Existing Parts",
             "page": "classify_existing",
             "sub": "existing_parts",
             "active": True,
         }
     ]
     sidebar_subtitle = "Classify Existing Parts"
 render_sidebar(
     subtitle=sidebar_subtitle,
     section_label="Choose",
     menu_items=menu_items,
 )
 render_top_nav(active_slug="home")
 if selected_menu in ["existing_parts", "create_new_parts"]:
     render_upload_view(selected_menu)
 elif selected_menu == "runs":
     render_runs_view()
 else:
     render_run_detail_view()