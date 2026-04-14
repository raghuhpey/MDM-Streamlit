import base64
import hashlib
from io import BytesIO
import time
import pandas as pd
import streamlit as st
from state.session import set_navigation_state, persist_persisted_ui_state
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode
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

def reset_existing_parts_view_state():
   keys_to_clear = [
       "current_existing_run_id",
       "nav_run_id",
       "prediction_job_id",
       "prediction_output_df",
       "uploaded_df",
       "existing_parts_upload_signature",
       "existing_parts_last_upload_stats",
       "existing_parts_preview_columns",
       "prediction_status",
       "error_message",
       "success_message",
   ]
   for key in keys_to_clear:
       st.session_state[key] = None
   for key, value in {
       "prediction_completed_message": False,
       "run_view_mode": False,
       "home_view": "upload",
       "mode": "Classify Existing Parts",
       "active_flow": "Classify Existing Parts",
   }.items():
       if key in st.session_state:
           st.session_state[key] = value
   st.session_state["existing_parts_uploader_version"] = (
       st.session_state.get("existing_parts_uploader_version", 0) + 1
   )
   # Important: force upload page to ignore any stale restored run
   st.session_state["force_fresh_existing_parts"] = True
   st.session_state["current_page"] = "classify_existing"
   st.session_state["current_flow"] = "existing_parts"
   set_navigation_state(
       page="classify_existing",
       sub="existing_parts",
       run_id=None,
   )
   persist_persisted_ui_state()

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
   if selected not in ["existing_parts", "existing_parts_new", "runs", "run_detail"]:
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
       st.session_state["force_fresh_existing_parts"] = False
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
   # IMPORTANT
   st.session_state["force_fresh_existing_parts"] = False
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
 st.session_state["force_fresh_existing_parts"] = False

# def render_runs_filters(runs_df: pd.DataFrame) -> pd.DataFrame:
#  st.markdown(
#      """
# <div class="section-card">
# <div class="section-card-title" style="margin-bottom: 14px;">Filter Runs</div>
# </div>
#      """,
#      unsafe_allow_html=True
#  )
#  st.markdown("")
#  filter_cols_top = st.columns(4)

# #  with filter_cols_top[0]:
# #      run_id_filter_option = sorted(
# #          [str(v) for v in runs_df["run_id"].dropna().astype(str).unique().tolist()]
# #      )
# #      selected_run_id = st.multiselect(
# #          "Run Id",
# #          options=run_id_filter_option,
# #          default=st.session_state.get("runs_filter_run_id", []),
# #          key="runs_filter_run_id",
# #          placeholder="All"
# #      )

#  with filter_cols_top[0]:
#      file_name_filter_option = sorted(
#          [str(v) for v in runs_df["file_name"].dropna().astype(str).unique().tolist()]
#      )
#      selected_file_name = st.multiselect(
#          "File Name",
#          options=file_name_filter_option,
#          default=st.session_state.get("runs_filter_file_name", []),
#          key="runs_filter_file_name",
#          placeholder="All"
#      )

#  with filter_cols_top[1]:
#      create_by_filter_option = sorted(
#          [str(v) for v in runs_df["uploaded_by"].dropna().astype(str).unique().tolist()]
#      )
#      selected_created_by = st.multiselect(
#          "Created By",
#          options=create_by_filter_option,
#          default=st.session_state.get("runs_filter_created_by", []),
#          key="runs_filter_created_by",
#          placeholder="All"
#      )
#  with filter_cols_top[2]:
#      status_options = sorted(
#          [str(v) for v in runs_df["run_status"].dropna().astype(str).unique().tolist()]
#      )
#      selected_statuses = st.multiselect(
#          "Status",
#          options=status_options,
#          default=st.session_state.get("runs_filter_status", []),
#          key="runs_filter_status",
#          placeholder="All"
#      )
#  filter_cols_bottom = st.columns(4)
# #  with filter_cols_bottom[0]:
# #      rows_min = st.number_input(
# #          "Rows Min",
# #          min_value=0,
# #          value=int(st.session_state.get("runs_filter_rows_min", 0)),
# #          step=1,
# #          key="runs_filter_rows_min"
# #      )
# #  with filter_cols_bottom[1]:
# #      rows_max_raw = st.text_input(
# #          "Rows Max",
# #          value=st.session_state.get("runs_filter_rows_max", ""),
# #          key="runs_filter_rows_max",
# #          placeholder="No max"
# #      )
# #  with filter_cols_bottom[0]:
# #      created_filter = st.text_input(
# #          "Created",
# #          key="runs_filter_created",
# #          placeholder="Contains date/time..."
# #      )
# #  with filter_cols_bottom[1]:
# #      action_options = sorted(
# #          [str(v) for v in runs_df["action_label"].dropna().astype(str).unique().tolist()]
# #      )
# #      selected_actions = st.multiselect(
# #          "Action",
# #          options=action_options,
# #          default=st.session_state.get("runs_filter_action", []),
# #          key="runs_filter_action",
# #          placeholder="All"
# #      )
#  clear_col_left, clear_col_right = st.columns([2, 8])
#  with clear_col_left:
#      if st.button("Clear Filters", key="runs_filter_clear_btn", use_container_width=True):
#          reset_values = {
#              "runs_filter_run_id": [],
#              "runs_filter_file_name": [],
#              "runs_filter_created_by": [],
#              "runs_filter_status": [],
#              "runs_filter_rows_min": 0,
#              "runs_filter_rows_max": "",
#              "runs_filter_created": "",
#              "runs_filter_action": [],
#          }
#          for reset_key, reset_value in reset_values.items():
#              st.session_state[reset_key] = reset_value
#          st.rerun()
#  filtered_df = runs_df.copy()

# #  if selected_run_id:
# #      filtered_df = filtered_df[
# #          filtered_df["run_id"].astype(str).str.contains(selected_run_id, case=False, na=False)
# #      ]
#  if selected_file_name:
#      filtered_df = filtered_df[
#          filtered_df["file_name"].astype(str).isin(selected_file_name)
#      ]
#  if selected_created_by:
#      filtered_df = filtered_df[
#          filtered_df["uploaded_by"].astype(str).isin(selected_created_by)
#      ]
#  if selected_statuses:
#      filtered_df = filtered_df[
#          filtered_df["run_status"].astype(str).isin(selected_statuses)
#      ]
#  filtered_df["row_count_numeric"] = pd.to_numeric(filtered_df["row_count"], errors="coerce")
# #  if rows_min:
# #      filtered_df = filtered_df[filtered_df["row_count_numeric"] >= rows_min]
# #  rows_max = None
# #  if str(rows_max_raw).strip():
# #      try:
# #          rows_max = int(str(rows_max_raw).strip())
# #      except ValueError:
# #          st.warning("Rows Max must be a whole number.")
# #  if rows_max is not None:
# #      filtered_df = filtered_df[filtered_df["row_count_numeric"] <= rows_max]
# #  if created_filter:
# #      filtered_df = filtered_df[
# #          filtered_df["uploaded_ts"].astype(str).str.contains(created_filter, case=False, na=False)
# #      ]
# #  if selected_actions:
# #      filtered_df = filtered_df[
# #          filtered_df["action_label"].astype(str).isin(selected_actions)
# #      ]
#  filtered_df = filtered_df.drop(columns=["row_count_numeric"], errors="ignore")
#  st.divider()
# #   st.caption(f"Showing {len(filtered_df)} of {len(runs_df)} runs")
#  return filtered_df


def render_upload_view(selected_menu):
   config = get_menu_config(selected_menu)
   st.markdown(
       f'<div class="panel-title">{config["upload_title"]}</div>',
       unsafe_allow_html=True
   )
   if selected_menu == "existing_parts":
       uploader_version = st.session_state.get("existing_parts_uploader_version", 0)
       uploader_key = f"uploader_existing_parts_{uploader_version}"
   else:
       uploader_key = f"uploader_{selected_menu}"
   uploaded_file = st.file_uploader(
       "Upload file",
       type=["csv", "xlsx"],
       label_visibility="collapsed",
       key=uploader_key
   )
   if uploaded_file is not None:
    try:
        if selected_menu == "existing_parts":
            create_table_backed_run(uploaded_file, selected_menu)
            st.rerun()
        else:
            st.info("Create New Parts upload-to-table flow is not wired yet.")
    except Exception as e:
        st.error(f"File upload failed: {e}")
   fresh_mode = (
       selected_menu == "existing_parts"
       and st.session_state.get("force_fresh_existing_parts", False)
   )
   if fresh_mode:
       current_run_id = None
       upload_stats = None
       preview_columns = []
   else:
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
       cols_to_show = [
           c for c in
           ["run_id", "row_count", "predicted_row_count", "remaining_row_count", "run_status"]
           if c in runs_df.columns
       ]
       print(runs_df[cols_to_show].head(20))
   print("=====================================")
   has_active_runs = False
   if not runs_df.empty and "run_status" in runs_df.columns:
       has_active_runs = runs_df["run_status"].astype(str).str.strip().str.lower().isin(
           {"queued", "submitted", "running"}
       ).any()
   st.markdown(
       """
<div class="runs-placeholder">
<div class="runs-placeholder-title">Runs</div>
<div class="runs-placeholder-text">
Table shows all prediction runs.
</div>
</div>
       """,
       unsafe_allow_html=True
   )
   st.markdown("")
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
   grid_df = runs_df.copy()
   # Parse timestamps
   grid_df["created_on_dt"] = pd.to_datetime(grid_df["uploaded_ts"], errors="coerce", utc=True)
   grid_df["created_on_display"] = grid_df["created_on_dt"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
   grid_df["created_on_display"] = grid_df["created_on_display"].fillna("")
   # Data shown in grid
   display_df = pd.DataFrame({
       "Run ID": grid_df["run_id"].astype(str),
       "File Name": grid_df["file_name"].fillna("").astype(str),
       "Created By": grid_df["uploaded_by"].fillna("").astype(str),
       "Created on": grid_df["created_on_display"].astype(str),
       "Rows": pd.to_numeric(grid_df["row_count"], errors="coerce").fillna(0).astype(int),
       "Predicted": pd.to_numeric(grid_df["predicted_row_count"], errors="coerce").fillna(0).astype(int),
       "Remaining": pd.to_numeric(grid_df["remaining_row_count"], errors="coerce").fillna(0).astype(int),
       "Status": grid_df["run_status"].fillna("").astype(str),
    #    "Action": grid_df["action_label"].fillna("").astype(str),
       "__run_id": grid_df["run_id"].astype(str),
       "__action_label": grid_df["action_label"].fillna("").astype(str),
   })
   centered_cell_style = {
       "display": "flex",
       "alignItems": "center",
       "justifyContent": "center",
       "textAlign": "center",
   }
   left_cell_style = {
       "display": "flex",
       "alignItems": "center",
       "justifyContent": "center",
       "textAlign": "center",
   }
   status_style = JsCode(
       """
       function(params) {
           const value = String(params.value || "").trim().toLowerCase();
           let bg = "#6b7280";
           let color = "white";
           if (value === "completed") bg = "#16a34a";
           else if (["queued", "submitted", "running"].includes(value)) bg = "#f59e0b";
           else if (["failed", "stopped"].includes(value)) bg = "#dc2626";
           else if (value === "uploaded") bg = "#2563eb";
           return {
               backgroundColor: bg,
               color: color,
               fontWeight: "700",
               textAlign: "center",
               display: "flex",
               alignItems: "center",
               justifyContent: "center",
               borderRadius: "5px",
               marginTop: "2px",
               marginBottom: "2px",
               paddingTop: "2px",
               paddingBottom: "2px"
           };
       }
       """
   )
   gb = GridOptionsBuilder.from_dataframe(display_df)
   gb.configure_default_column(
       sortable=True,
       filter=True,
       floatingFilter=True,
       resizable=True,
       suppressMenu=False,
   )
   gb.configure_selection(
       selection_mode="single",
       use_checkbox=False,
   )
   gb.configure_column("Run ID", width=280, cellStyle=left_cell_style, headerClass="center-header")
   gb.configure_column("File Name", width=220, cellStyle=left_cell_style, headerClass="center-header")
   gb.configure_column("Created By", width=150, cellStyle=centered_cell_style, headerClass="center-header")
   gb.configure_column("Created on", width=220, cellStyle=left_cell_style, headerClass="center-header")
   gb.configure_column("Rows", width=90, cellStyle=centered_cell_style, headerClass="center-header")
   gb.configure_column("Predicted", width=110, cellStyle=centered_cell_style, headerClass="center-header")
   gb.configure_column("Remaining", width=120, cellStyle=centered_cell_style, headerClass="center-header")
   gb.configure_column("Status", width=120, cellStyle=status_style, filter=True, headerClass="center-header")
#    gb.configure_column("Action", width=120, cellStyle=centered_cell_style, filter=True)
   gb.configure_column("__run_id", hide=True)
   gb.configure_column("__action_label", hide=True)
   gb.configure_grid_options(
       rowHeight=54,
       headerHeight=46,
       suppressRowClickSelection=False,
       animateRows=True,
       ensureDomOrder=True,
   )
   grid_options = gb.build()

   custom_css = {
        ".ag-header": {
            "background-color": "#9ca3af !important",
            "border-bottom": "1px solid #d1d5db !important",
        },
        ".ag-header-cell": {
            "background-color": "#9ca3af !important",
            "color": "#1F2937 !important",
            "font-weight": "700 !important",
        },
        ".ag-header-cell-label": {
            "color": "#1F2937 !important",
            "font-weight": "700 !important",
            "justify-content": "center !important",
            "text-align": "center !important",
            "width": "100% !important",
        },
        ".ag-header-cell-text": {
            "width": "100% !important",
            "text-align": "center !important",
        },
        ".ag-floating-filter": {
            "background-color": "#f3f4f6 !important",
        },
        }

   grid_response = AgGrid(
       display_df,
       gridOptions=grid_options,
       data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
       update_mode=GridUpdateMode.SELECTION_CHANGED,
       allow_unsafe_jscode=True,
       fit_columns_on_grid_load=False,
       theme="streamlit",
       height=380,
       reload_data=False,
       custom_css = custom_css, 
   )
   selected_rows = grid_response.get("selected_rows", [])
   selected_row = None
   if isinstance(selected_rows, list) and selected_rows:
       selected_row = selected_rows[0]
   elif hasattr(selected_rows, "empty") and not selected_rows.empty:
       selected_row = selected_rows.iloc[0].to_dict()
#    st.markdown("")
   if selected_row:
       selected_run_id = str(selected_row.get("__run_id") or selected_row.get("Run ID") or "")
       selected_action = str(selected_row.get("__action_label") or selected_row.get("Action") or "").strip()
       st.markdown(
           f"""
<div style="
padding:10px 14px;
border:1px solid #e5e7eb;
border-radius:10px;
background:#f9fafb;
margin-bottom:12px;
color:#374151;
font-size:14px;
">
<b>Selected Run ID:</b> {selected_run_id}
&nbsp;&nbsp;|&nbsp;&nbsp;
<b>Available Action:</b> {selected_action}
</div>
           """,
           unsafe_allow_html=True,
       )
       btn_cols = st.columns([1.2, 1.2, 1.2, 5])
       with btn_cols[0]:
           if selected_action == "Continue":
               if st.button("Continue", key="continue_selected_run_btn", use_container_width=True):
                   load_table_run_into_state(selected_run_id, selected_menu="existing_parts")
                   set_navigation_state(
                       page="classify_existing",
                       sub="existing_parts",
                       run_id=selected_run_id,
                   )
                   st.rerun()
           elif selected_action == "View":
               if st.button("View", key="view_selected_run_btn", use_container_width=True):
                   load_table_run_into_state(selected_run_id, selected_menu="existing_parts")
                   set_navigation_state(
                       page="classify_existing",
                       sub="run_detail",
                       run_id=selected_run_id,
                   )
                   st.rerun()
           elif selected_action == "Stop":
               if st.button("Stop", key="stop_selected_run_btn", use_container_width=True):
                   try:
                       request_stop_run_in_classification_table(selected_run_id)
                       st.success(f"Stop requested for run {selected_run_id}.")
                       st.rerun()
                   except Exception as e:
                       st.error(f"Unable to stop run {selected_run_id}: {e}")
       with btn_cols[1]:
           if st.button("Clear Selection", key="clear_selected_run_btn", use_container_width=True):
               st.rerun()
   else:
       st.info("Select a row, then use the action button below the grid.")
   if has_active_runs:
       st.info("Active runs detected. This table auto-refreshes every 5 seconds.")
       time.sleep(5)
       st.rerun()

# def render_run_detail_view():
#  run_id = get_selected_run_id()
#  run_record = load_run_metadata(run_id) if run_id else None
#  if run_record is None:
#      st.error("Run not found.")
#      return
#  top_cols = st.columns([1.5, 6])
#  with top_cols[0]:
#      if st.button("Ã¢â€ Â Back to Runs", key="back_to_runs_btn", use_container_width=True):
#          set_navigation_state(
#              page="classify_existing",
#              sub="runs",
#              run_id=None,
#          )
#          st.rerun()
#  st.markdown(
#      f"""
# <div class="run-meta-card">
# <div class="run-meta-title">Run Detail</div>
# <div class="run-meta-grid">
# <div class="run-meta-item">
# <div class="run-meta-label">Run ID</div>
# <div class="run-meta-value">{run_record["prediction_job_id"]}</div>
# </div>
# <div class="run-meta-item">
# <div class="run-meta-label">File Name</div>
# <div class="run-meta-value">{run_record["file_name"]}</div>
# </div>
# <div class="run-meta-item">
# <div class="run-meta-label">Status</div>
# <div class="run-meta-value">{run_record["status"]}</div>
# </div>
# <div class="run-meta-item">
# <div class="run-meta-label">Created By</div>
# <div class="run-meta-value">{run_record["created_by"]}</div>
# </div>
# <div class="run-meta-item">
# <div class="run-meta-label">Rows</div>
# <div class="run-meta-value">{run_record["row_count"]}</div>
# </div>
# <div class="run-meta-item">
# <div class="run-meta-label">Created At</div>
# <div class="run-meta-value">{run_record["created_at"]}</div>
# </div>
# </div>
# </div>
#      """,
#      unsafe_allow_html=True
#  )
#  input_df = load_input_preview_df(run_record["prediction_job_id"])
#  st.markdown(
#      """
# <div class="section-card">
# <div class="section-card-title">Input Preview</div>
# </div>
#      """,
#      unsafe_allow_html=True
#  )
#  if input_df is None:
#      st.markdown(
#          """
# <div class="empty-state-box">
#  Input preview file not found.
# </div>
#          """,
#          unsafe_allow_html=True
#      )
#  else:
#      render_paginated_dataframe(
#          input_df,
#          pager_prefix=f'{run_record["prediction_job_id"]}_detail_input'
#      )
#  output_df = load_output_df(run_record["prediction_job_id"])
#  if output_df is not None:
#      st.markdown(
#          """
# <div class="section-card">
# <div class="section-card-title">Output Preview</div>
# <div class="section-card-subtitle">
#  Output loaded from the managed folder.
# </div>
# </div>
#          """,
#          unsafe_allow_html=True
#      )
#      render_paginated_dataframe(
#          output_df,
#          pager_prefix=f'{run_record["prediction_job_id"]}_detail_output'
#      )
#  else:
#      st.markdown(
#          """
# <div class="section-card">
# <div class="section-card-title">Output Preview</div>
# <div class="section-card-subtitle">
#  Output is not available yet.
# </div>
# <div class="empty-state-box">
#  No output available yet.
# </div>
# </div>
#          """,
#          unsafe_allow_html=True
#      )

def render_run_detail_view():
   run_id = get_selected_run_id()
   run_record = fetch_classification_run_metadata(run_id) if run_id else None
   if run_record is None:
       st.error("Run not found.")
       return
   def _fmt(value, default="-"):
       if value is None:
           return default
       value = str(value).strip()
       return value if value else default
   top_cols = st.columns([1.5, 6])
   with top_cols[0]:
       if st.button("← Back to Runs", key="back_to_runs_btn", use_container_width=True):
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
<div class="run-meta-value">{_fmt(run_record.get("run_id"))}</div>
</div>
<div class="run-meta-item">
<div class="run-meta-label">File Name</div>
<div class="run-meta-value">{_fmt(run_record.get("file_name"))}</div>
</div>
<div class="run-meta-item">
<div class="run-meta-label">Status</div>
<div class="run-meta-value">{_fmt(run_record.get("run_status"))}</div>
</div>
<div class="run-meta-item">
<div class="run-meta-label">Uploaded By</div>
<div class="run-meta-value">{_fmt(run_record.get("uploaded_by"))}</div>
</div>
<div class="run-meta-item">
<div class="run-meta-label">Rows</div>
<div class="run-meta-value">{_fmt(run_record.get("row_count"))}</div>
</div>
<div class="run-meta-item">
<div class="run-meta-label">Uploaded At</div>
<div class="run-meta-value">{_fmt(run_record.get("uploaded_ts"))}</div>
</div>
<div class="run-meta-item">
<div class="run-meta-label">Prediction Started</div>
<div class="run-meta-value">{_fmt(run_record.get("prediction_started_ts"))}</div>
</div>
<div class="run-meta-item">
<div class="run-meta-label">Prediction Completed</div>
<div class="run-meta-value">{_fmt(run_record.get("prediction_completed_ts"))}</div>
</div>
</div>
</div>
       """,
       unsafe_allow_html=True
   )
   display_df = fetch_classification_run_display_df(run_id)
   st.markdown(
       """
<div class="section-card">
<div class="section-card-title">Run Preview</div>
<dic class="empty-state"</div>
</div>
       """,
       unsafe_allow_html=True
   )
   if display_df is None or display_df.empty:
       st.markdown(
           """
<div class="empty-state-box">
No rows available for this run.
</div>
           """,
           unsafe_allow_html=True
       )
       return
   visible_cols = [c for c in display_df.columns if c not in CLASSIFICATION_HIDDEN_COLUMNS]
   if visible_cols:
       display_df = display_df[visible_cols].copy()
   render_paginated_dataframe(
       display_df,
       pager_prefix=f"{run_id}_detail_table"
   )

     
def render():
 load_css(get_global_css() + get_app_shell_css() + get_home_page_css() + get_classify_page_css())
 selected_menu = get_selected_submenu()
 if selected_menu == "existing_parts_new":
   reset_existing_parts_view_state()
   st.rerun()
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