import base64
import html
import re
import io
import dataiku
import pandas as pd
import streamlit as st
from state.session import set_navigation_state
FOLDER_ID = "MDM_application"
UNSPSC_LOOKUP_PATH = "reference/unspsc_code_description.csv"
UNSPSC_CODE_COL = "unspsc_code"
UNSPSC_DESC_COL = "unspsc_description"
folder = dataiku.Folder(FOLDER_ID)
# from services.validation_storage_service import (
#     approve_validation_row,
#     ensure_validation_run,
#     list_runs_to_validate,
#     load_golden_dataset_df,
#     load_validation_queue_df,
# )
from services.validation_table_service import (
  approve_validation_row,
  ensure_validation_run,
  list_runs_to_validate,
  load_golden_dataset_df,
  load_validation_queue_df,
)
from utils.pagination import get_page_bounds, get_page_df, get_total_pages
from ui.layout import render_sidebar, render_top_nav
from utils.styles import (
   get_app_shell_css,
   get_classify_page_css,
   get_global_css,
   get_home_page_css,
   get_validation_page_css,
   load_css,
)
TOP_REVIEW_COLUMNS = {
   "short_descrition_(english)",
   "short_description_(english)",
   "Short Description (English)",
   "Short Description",
   "long_descritpion_(english)",
   "long_description_(english)",
   "Long Description (English)",
   "Long Description",
   "Explanation",
   "Model 1 OP",
   "Model 2 OP",
   "Model 3 OP",
   "LLM as a Judge OP",
   "LLM Decision Source",
   "Validated UNSPSC",
   "Validation Decision Source",
   "Validation Comment",
   "Validation Status",
   "Validated By",
   "Validated At",
   "_validation_row_id",
   "Prediction Job ID",
   "Prediction Created By",
}

def get_base64_image(image_path: str) -> str:
   with open(image_path, "rb") as img_file:
       return base64.b64encode(img_file.read()).decode()

def nav_pill(label, slug, active=False):
   css_class = "nav-pill active" if active else "nav-pill"
   return f'''
<form method="get" style="margin:0;">
<input type="hidden" name="page" value="{slug}">
<button type="submit" class="{css_class}">{label}</button>
</form>
'''

def left_menu_btn(label, sub_value, active=False):
   css_class = "left-menu-btn active" if active else "left-menu-btn"
   return f'''
<form method="get" class="left-menu-form">
<input type="hidden" name="page" value="validation">
<input type="hidden" name="sub" value="{sub_value}">
<button type="submit" class="{css_class}">{label}</button>
</form>
'''

def get_selected_submenu():
  selected = st.query_params.get("sub", st.session_state.get("nav_sub", "to_validate"))
  if isinstance(selected, list):
      selected = selected[0]
  if selected not in {"to_validate", "golden_dataset", "validate_detail"}:
      selected = "to_validate"
  return selected

def get_selected_run_id():
  run_id = st.query_params.get("run_id", st.session_state.get("nav_run_id"))
  if isinstance(run_id, list):
      run_id = run_id[0]
  return run_id
def safe_text(value, default="-"):
   if pd.isna(value) or value is None:
       return default
   value = str(value).strip()
   return value if value else default

@st.cache_data(show_spinner=False)
def load_unspsc_description_map():
  lookup_df = load_unspsc_lookup_df(UNSPSC_LOOKUP_PATH)
  return build_unspsc_description_map(lookup_df, UNSPSC_CODE_COL, UNSPSC_DESC_COL)

@st.cache_data(show_spinner=False)
def load_unspsc_dropdown_items():
  lookup_df = load_unspsc_lookup_df(UNSPSC_LOOKUP_PATH)
  temp = lookup_df[[UNSPSC_CODE_COL, UNSPSC_DESC_COL]].copy()
  temp[UNSPSC_CODE_COL] = (
      temp[UNSPSC_CODE_COL]
      .fillna("")
      .astype(str)
      .str.replace("\xa0", " ", regex=False)
      .str.strip()
      .str.replace(".0", "", regex=False)
  )
  temp[UNSPSC_DESC_COL] = (
      temp[UNSPSC_DESC_COL]
      .fillna("")
      .astype(str)
      .str.replace("\xa0", " ", regex=False)
      .str.strip()
  )
  temp = temp[temp[UNSPSC_CODE_COL] != ""].drop_duplicates(subset=[UNSPSC_CODE_COL]).sort_values(by=[UNSPSC_CODE_COL])
  items = []
  for _, rec in temp.iterrows():
      code = str(rec[UNSPSC_CODE_COL]).strip()
      desc = str(rec[UNSPSC_DESC_COL]).strip()
      label = f"{code} - {desc}" if desc else code
      items.append({"code": code, "desc": desc, "label": label})
  return items
def load_unspsc_lookup_df(path_in_folder: str) -> pd.DataFrame:
  data = folder.get_download_stream(path_in_folder).read()
  lower_path = path_in_folder.lower()
  if lower_path.endswith(".csv"):
      encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "iso-8859-1"]
      last_error = None
      for enc in encodings_to_try:
          try:
              return pd.read_csv(io.BytesIO(data), dtype=str, encoding=enc)
          except UnicodeDecodeError as e:
              last_error = e
      raise ValueError(
          f"Unable to decode CSV lookup file '{path_in_folder}' with supported encodings. "
          f"Last error: {last_error}"
      )
  elif lower_path.endswith(".xlsx") or lower_path.endswith(".xls"):
      return pd.read_excel(io.BytesIO(data), dtype=str)
  elif lower_path.endswith(".json"):
      return pd.read_json(io.BytesIO(data), dtype=str)
  else:
      raise ValueError(f"Unsupported lookup file type: {path_in_folder}")
def normalize_unspsc_display(value):
  if pd.isna(value) or value is None:
      return ""
  return str(value).strip().replace(".0", "")
def build_unspsc_description_map(df: pd.DataFrame, code_col: str, desc_col: str) -> dict:
  temp = df[[code_col, desc_col]].copy()
  temp[code_col] = (
      temp[code_col]
      .fillna("")
      .astype(str)
      .str.replace("\xa0", " ", regex=False)
      .str.strip()
      .str.replace(".0", "", regex=False)
  )
  temp[desc_col] = (
      temp[desc_col]
      .fillna("")
      .astype(str)
      .str.replace("\xa0", " ", regex=False)
      .str.strip()
  )
  temp = temp[temp[code_col] != ""]
  return dict(zip(temp[code_col], temp[desc_col]))
def get_unspsc_description(code, desc_map):
  code = normalize_unspsc_display(code)
  if not code:
      return ""
# direct match
  if code in desc_map:
      return desc_map[code]
# fallback: try zero-padded 8-digit form
  if code.isdigit():
      padded = code.zfill(8)
      if padded in desc_map:
          return desc_map[padded]
  return "Description not found"

def normalize_unspsc_display(value):
   if pd.isna(value) or value is None:
       return ""
   return str(value).strip().replace(".0", "")

def format_preview_dataframe(df: pd.DataFrame) -> pd.DataFrame:
   preview_df = df.copy()
   for col in preview_df.columns:
       series = preview_df[col]
       if str(series.dtype) in ["bool", "boolean"]:
           preview_df[col] = series.map(lambda x: "Yes" if x else "No")
   return preview_df

def _normalized_col_name(col: str) -> str:
   return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")

def render_paginated_dataframe(df, pager_prefix, default_page_size=10):
   total_rows = len(df)
   if total_rows == 0:
       st.markdown('<div class="empty-state-box">No rows available.</div>', unsafe_allow_html=True)
       return
   page_key = f"{pager_prefix}_page"
   if page_key not in st.session_state:
       st.session_state[page_key] = 1
   page_size = default_page_size
   total_pages = get_total_pages(total_rows, page_size)
   current_page = max(1, min(st.session_state[page_key], total_pages))
   st.session_state[page_key] = current_page
   start_idx, end_idx = get_page_bounds(current_page, page_size, total_rows)
   left_col, center_col, right_col = st.columns([2.2, 4.8, 2.0])
   with left_col:
       prev_col, next_col = st.columns(2)
       with prev_col:
           if st.button("Prev", key=f"{pager_prefix}_prev", disabled=(current_page == 1), use_container_width=True):
               st.session_state[page_key] = current_page - 1
               st.rerun()
       with next_col:
           if st.button("Next", key=f"{pager_prefix}_next", disabled=(current_page == total_pages), use_container_width=True):
               st.session_state[page_key] = current_page + 1
               st.rerun()
   with center_col:
       st.markdown(
           f'<div class="pagination-summary">Page {current_page} / {total_pages} â€¢ Rows {start_idx + 1}-{end_idx} of {total_rows}</div>',
           unsafe_allow_html=True,
       )
   with right_col:
       st.markdown('<div class="pagination-right-label">Go to page</div>', unsafe_allow_html=True)
       selected_page = st.number_input(
           "Go to page",
           min_value=1,
           max_value=total_pages,
           value=current_page,
           step=1,
           key=f"{pager_prefix}_page_input",
           label_visibility="collapsed",
       )
       if selected_page != current_page:
           st.session_state[page_key] = int(selected_page)
           st.rerun()
   page_df = format_preview_dataframe(get_page_df(df, st.session_state[page_key], page_size))
   st.dataframe(page_df, use_container_width=True, hide_index=True)

def _clear_to_validate_filter_state():
   reset_values = {
       "val_filter_prediction_job": "",
       "val_filter_prediction_created_by": "",
       "val_filter_status": [],
       "val_filter_updated_by": "",
       "val_sort_by": "Last Updated",
       "val_sort_order": "Descending",
   }
   for k, v in reset_values.items():
       st.session_state[k] = v

def _prepare_to_validate_df(runs: list[dict]) -> pd.DataFrame:
   df = pd.DataFrame(runs).copy()
   if df.empty:
       return df
   preferred_cols = [
       "prediction_job_id",
       "Prediction Job",
       "Prediction Created By",
       "Approved Rows",
       "Status",
       "Last Updated",
       "Updated By",
   ]
   for col in preferred_cols:
       if col not in df.columns:
           df[col] = ""
   df["Prediction Job"] = df["Prediction Job"].fillna("").astype(str)
   df["Prediction Created By"] = df["Prediction Created By"].fillna("").astype(str)
   df["Status"] = df["Status"].fillna("").astype(str)
   df["Updated By"] = df["Updated By"].fillna("").astype(str)
   df["Last Updated"] = df["Last Updated"].fillna("").astype(str)
   df["Approved Rows"] = df["Approved Rows"].fillna("").astype(str)
   df["_approved_rows_num"] = pd.to_numeric(
       df["Approved Rows"].astype(str).str.extract(r"^(\d+)", expand=False),
       errors="coerce",
   ).fillna(0)
   df["_last_updated_dt"] = pd.to_datetime(df["Last Updated"], errors="coerce", utc=True)
   return df

def render_to_validate_filters(runs_df: pd.DataFrame) -> pd.DataFrame:
   st.markdown(
       """
<div class="section-card">
<div class="section-card-title" style="margin-bottom: 14px;">Filter & Sort</div>
</div>
       """,
       unsafe_allow_html=True,
   )
   st.markdown("")
   top_cols = st.columns(4)
   with top_cols[0]:
       prediction_job_filter = st.text_input(
           "Prediction Job",
           key="val_filter_prediction_job",
           placeholder="Contains...",
       )
   with top_cols[1]:
       created_by_filter = st.text_input(
           "Prediction Created By",
           key="val_filter_prediction_created_by",
           placeholder="Contains...",
       )
   with top_cols[2]:
       status_options = sorted([str(v) for v in runs_df["Status"].dropna().astype(str).unique().tolist() if str(v).strip()])
       selected_statuses = st.multiselect(
           "Status",
           options=status_options,
           default=st.session_state.get("val_filter_status", []),
           key="val_filter_status",
           placeholder="All",
       )
   with top_cols[3]:
       updated_by_filter = st.text_input(
           "Updated By",
           key="val_filter_updated_by",
           placeholder="Contains...",
       )
   sort_cols = st.columns([2.2, 1.6, 1.2, 3])
   with sort_cols[0]:
       sort_by = st.selectbox(
           "Sort By",
           options=["Prediction Job", "Prediction Created By", "Approved Rows", "Status", "Last Updated", "Updated By"],
        #    options=["Approved Rows", "Status", "Last Updated"],
           key="val_sort_by",
       )
   with sort_cols[1]:
       sort_order = st.selectbox(
           "Order",
           options=["Descending", "Ascending"],
           key="val_sort_order",
       )
   with sort_cols[2]:
       st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
       if st.button("Clear", key="val_filter_clear_btn", use_container_width=True):
           _clear_to_validate_filter_state()
           st.rerun()
   filtered_df = runs_df.copy()
   if prediction_job_filter:
       filtered_df = filtered_df[
           filtered_df["Prediction Job"].astype(str).str.contains(prediction_job_filter, case=False, na=False)
       ]
   if created_by_filter:
       filtered_df = filtered_df[
           filtered_df["Prediction Created By"].astype(str).str.contains(created_by_filter, case=False, na=False)
       ]
   if selected_statuses:
       filtered_df = filtered_df[
           filtered_df["Status"].astype(str).isin(selected_statuses)
       ]
   if updated_by_filter:
       filtered_df = filtered_df[
           filtered_df["Updated By"].astype(str).str.contains(updated_by_filter, case=False, na=False)
       ]
   ascending = sort_order == "Ascending"
   if sort_by == "Approved Rows":
       filtered_df = filtered_df.sort_values(by=["_approved_rows_num", "Prediction Job"], ascending=[ascending, True], na_position="last")
   elif sort_by == "Last Updated":
       filtered_df = filtered_df.sort_values(by=["_last_updated_dt", "Prediction Job"], ascending=[ascending, True], na_position="last")
   else:
       filtered_df = filtered_df.sort_values(by=[sort_by], ascending=ascending, na_position="last")
   st.caption(f"Showing {len(filtered_df)} of {len(runs_df)} runs")
   st.divider()
   return filtered_df
def render_to_validate_view():
   runs = list_runs_to_validate()
   st.markdown(
       """
<div class="section-card">
<div class="section-card-title">To Validate</div>
<div class="section-card-subtitle">
                   Review and validate the records in the runs listed below.
</div>
</div>
""",
       unsafe_allow_html=True,
   )
   if not runs:
       st.markdown(
           """
<div class="section-card">
<div class="empty-state-box">No runs pending validation.</div>
</div>
""",
           unsafe_allow_html=True,
       )
       return
   runs_df = _prepare_to_validate_df(runs)
   runs_df = render_to_validate_filters(runs_df)
   if runs_df.empty:
       st.markdown(
           """
<div class="section-card">
<div class="empty-state-box">No runs match the selected filters.</div>
</div>
""",
           unsafe_allow_html=True,
       )
       return
   widths = [0.8, 2.5, 1.8, 1.4, 1.3, 2.0, 1.6, 1.2]
   header_cols = st.columns(widths)
   headers = [
       "S. No",
       "Prediction Job",
       "Prediction Created By",
       "Approved Rows",
       "Status",
       "Last Updated",
       "Updated By",
       "Action",
   ]
   for col, header in zip(header_cols, headers):
       with col:
           st.markdown("***")
           st.markdown(f'<div class="runs-header-cell">{header}</div>', unsafe_allow_html=True)
   display_df = runs_df.reset_index(drop=True)
   for idx, (_, run) in enumerate(display_df.iterrows(), start=1):
       row_cols = st.columns(widths)
       with row_cols[0]:
           st.markdown(f'<div class="runs-cell">{idx}</div>', unsafe_allow_html=True)
       with row_cols[1]:
           st.markdown(f'<div class="runs-cell">{html.escape(str(run["Prediction Job"]))}</div>', unsafe_allow_html=True)
       with row_cols[2]:
           st.markdown(f'<div class="runs-cell">{html.escape(str(run["Prediction Created By"]))}</div>', unsafe_allow_html=True)
       with row_cols[3]:
           st.markdown(f'<div class="runs-cell">{html.escape(str(run["Approved Rows"]))}</div>', unsafe_allow_html=True)
       with row_cols[4]:
           status_class = str(run["Status"]).lower()
           st.markdown(
               f'<div class="runs-cell"><span class="status-pill {status_class}">{html.escape(str(run["Status"]))}</span></div>',
               unsafe_allow_html=True,
           )
       with row_cols[5]:
           st.markdown(f'<div class="runs-cell">{html.escape(str(run["Last Updated"]))}</div>', unsafe_allow_html=True)
       with row_cols[6]:
           st.markdown(f'<div class="runs-cell">{html.escape(str(run["Updated By"]))}</div>', unsafe_allow_html=True)
       with row_cols[7]:
           if st.button("Continue", key=f'continue_validation_{run["prediction_job_id"]}', use_container_width=True):
               set_navigation_state(
                       page="validation",
                       sub="validate_detail",
                       run_id=run["prediction_job_id"],
                       )
               st.rerun()
       st.divider()

def _get_description_block(row: pd.Series) -> str:
   short_candidates = [
       "short_descrition_(english)",
       "short_description_(english)",
       "Short Description (English)",
       "Short Description",
   ]
   long_candidates = [
       "long_descritpion_(english)",
       "long_description_(english)",
       "Long Description (English)",
       "Long Description",
   ]
   short_value = ""
   long_value = ""
   for col in short_candidates:
       if col in row.index and pd.notna(row.get(col)) and str(row.get(col)).strip():
           short_value = str(row.get(col)).strip()
           break
   for col in long_candidates:
       if col in row.index and pd.notna(row.get(col)) and str(row.get(col)).strip():
           long_value = str(row.get(col)).strip()
           break
   return (
       '<div style="font-size:15px; font-weight:600; color:#111827; line-height:1.8;">'
       f'<div><span style="font-weight:800;">Short description:</span> {html.escape(safe_text(short_value))}</div>'
       f'<div style="margin-top:14px;"><span style="font-weight:800;">Long description:</span> {html.escape(safe_text(long_value))}</div>'
       '</div>'
   )

def _render_box(title: str, body_html: str, min_height: int = 180):
   st.markdown(f"**{title}**")
   st.markdown(
       f'''
<div style="
   background:#ffffff;
   border:1px solid #dfe3ea;
   border-radius:16px;
   padding:16px 18px;
   min-height:{min_height}px;
   color:#111827;
   line-height:1.6;
   font-size:15px;
   box-sizing:border-box;
   overflow-wrap:anywhere;
">
{body_html}
</div>
''',
       unsafe_allow_html=True,
   )

def _render_pred_desc_block(row: pd.Series, desc_map: dict) -> str:
   entries = []
   for label, col in [
       ("Model 1", "Model 1 OP"),
       ("Model 2", "Model 2 OP"),
       ("Model 3", "Model 3 OP"),
       ("LLM Judge", "LLM as a Judge OP"),
   ]:
       code = normalize_unspsc_display(row.get(col))
       if not code:
           continue
       desc = safe_text(get_unspsc_description(code, desc_map), default="Description not found")
       entries.append(
           f'''
<div style="margin-bottom:16px;">
<div style="font-weight:800; color:#111827; margin-bottom:4px;">{html.escape(label)}: {html.escape(code)}</div>
<div style="color:#4b5563; line-height:1.65;">{html.escape(desc)}</div>
</div>
'''
       )
   if not entries:
       return "<div style='color:#9ca3af; font-style:italic;'>No descriptions found.</div>"
   return "".join(entries)

def _render_predicted_code_description_box(height=185):
  entries = []
  for label, code in [
      ("Model 1", model1_code),
      ("Model 2", model2_code),
      ("Model 3", model3_code),
      ("LLM Judge", llm_code),
  ]:
      if not code:
          continue
      desc = _safe_text(desc_map.get(code, ""), default="Description not found")
      entries.append(
          f"""
<div style="margin-bottom:18px;">
<div style="font-size:16px; font-weight:800; color:#111827; margin-bottom:6px;">
                  {html.escape(label)}: {html.escape(code)}
</div>
<div style="font-size:15px; color:#4b5563; line-height:1.7;">
                  {html.escape(desc)}
</div>
</div>
          """
      )
  if not entries:
      entries = ["<div style='color:#9ca3af; font-style:italic;'>No descriptions found.</div>"]
  st.markdown(
      f"""
<div style="
          background:#ffffff;
          border:1px solid #dfe3ea;
          border-radius:16px;
          padding:16px 18px;
          min-height:{height}px;
          color:#111827;
          line-height:1.6;
          font-size:15px;
          box-sizing:border-box;
          overflow-wrap:anywhere;
      ">
          {''.join(entries)}
</div>
      """,
      unsafe_allow_html=True
  )

def render_validation_review_header():
  st.markdown(
      """
<div class="validation-review-header-bar">
<div class="validation-review-header-cell">Description</div>
<div class="validation-review-header-cell">Explanation</div>
<div class="validation-review-header-cell">Model Prediction</div>
<div class="validation-review-header-cell">Predicted Code Description</div>
<div class="validation-review-header-cell">Revised Code</div>
<div class="validation-review-header-cell">Approve</div>
</div>
      """,
      unsafe_allow_html=True
  )

def render_validation_record_card(run_id, row, desc_map):
   import html
   row_id = int(row["_validation_row_id"])
   def _safe_text(value, default="-"):
       if pd.isna(value) or value is None:
           return default
       value = str(value).strip()
       return value if value else default
   def _normalize_code(value):
       if pd.isna(value) or value is None:
           return ""
       return str(value).strip().replace(".0", "")
   def _first_non_empty(series, candidates):
       for col in candidates:
           if col in series.index:
               val = series.get(col)
               if pd.notna(val) and str(val).strip():
                   return str(val).strip()
       return ""
   def _render_info_box(lines, height=185, bold=False):
       if isinstance(lines, str):
           lines = [lines]
       rendered_lines = []
       for line in lines:
           safe_line = html.escape(_safe_text(line))
           if bold:
               rendered_lines.append(
                   f"<div style='font-weight:700; margin-bottom:14px; line-height:1.7;'>{safe_line}</div>"
               )
           else:
               rendered_lines.append(
                   f"<div style='margin-bottom:12px; line-height:1.8;'>{safe_line}</div>"
               )
       body_html = "".join(rendered_lines)
       st.markdown(
           f"""
<div style="
   background:#ffffff;
   border:1px solid #dfe3ea;
   border-radius:16px;
   padding:16px 18px;
   min-height:{height}px;
   color:#111827;
   font-size:15px;
   box-sizing:border-box;
   overflow-wrap:anywhere;
">
   {body_html}
</div>
           """,
           unsafe_allow_html=True,
       )
   def _render_code_box(code, desc):
       code = _normalize_code(code)
       desc = _safe_text(desc, default="Description not found") if code else ""
       st.markdown(
           f"""
<div style="
   background:#ffffff;
   border:1px solid #dfe3ea;
   border-radius:14px;
   padding:12px 14px;
   min-height:52px;
   box-sizing:border-box;
">
<div style="font-size:15px; font-weight:700; color:#111827;">
       {html.escape(code) if code else "-"}
</div>
<div style="font-size:13px; color:#6b7280; margin-top:4px;">
       {html.escape(desc) if desc else ""}
</div>
</div>
           """,
           unsafe_allow_html=True,
       )
   model1_code = _normalize_code(row.get("Model 1 OP"))
   model2_code = _normalize_code(row.get("Model 2 OP"))
   model3_code = _normalize_code(row.get("Model 3 OP"))
   llm_code = _normalize_code(row.get("LLM as a Judge OP"))
   def _render_predicted_code_description_box(height=185):
       entries = []
       for label, code in [
           ("Model 1", model1_code),
           ("Model 2", model2_code),
           ("Model 3", model3_code),
           ("LLM Judge", llm_code),
       ]:
           if not code:
               continue
           desc = _safe_text(desc_map.get(code, ""), default="Description not found")
           entries.append(
               f"""
<div style="margin-bottom:18px;">
<div style="font-size:16px; font-weight:800; color:#111827; margin-bottom:6px;">
       {html.escape(label)}: {html.escape(code)}
</div>
<div style="font-size:15px; color:#4b5563; line-height:1.7;">
       {html.escape(desc)}
</div>
</div>
               """
           )
       if not entries:
           entries = ["<div style='color:#9ca3af; font-style:italic;'>No descriptions found.</div>"]
       st.markdown(
           f"""
<div style="
   background:#ffffff;
   border:1px solid #dfe3ea;
   border-radius:16px;
   padding:16px 18px;
   min-height:{height}px;
   color:#111827;
   font-size:15px;
   box-sizing:border-box;
   overflow-wrap:anywhere;
">
   {''.join(entries)}
</div>
           """,
           unsafe_allow_html=True,
       )
   short_value = _first_non_empty(
       row,
       [
           "short_descrition_(english)",
           "short_description_(english)",
           "Short Description (English)",
           "Short Description",
       ],
   )
   long_value = _first_non_empty(
       row,
       [
           "long_descritpion_(english)",
           "long_description_(english)",
           "Long Description (English)",
           "Long Description",
       ],
   )
   description_lines = [
       f"Short description: {_safe_text(short_value)}",
       f"Long description: {_safe_text(long_value)}",
   ]
   explanation_text = _safe_text(row.get("Explanation", ""))
   option_items = [
       ("Edit", ""),
       ("Model 1", model1_code),
       ("Model 2", model2_code),
       ("Model 3", model3_code),
       ("LLM Judge", llm_code),
   ]
   option_labels = []
   option_value_map = {}
   for source, code in option_items:
       if source == "Edit":
           label = "Edit"
           option_labels.append(label)
           option_value_map[label] = {"source": "Manual Edit", "code": ""}
       else:
           label = f"{source}: {code}" if code else f"{source}:"
           option_labels.append(label)
           option_value_map[label] = {"source": source, "code": code}
   default_label = None
   for label in option_labels:
       if label.startswith("LLM Judge") and option_value_map[label]["code"]:
           default_label = label
           break
   if default_label is None:
       for label in option_labels:
           if option_value_map[label]["code"]:
               default_label = label
               break
   if default_label is None:
       default_label = "Edit"
   st.markdown(
       """
<div style="
   display:grid;
   grid-template-columns: 1.8fr 2fr 2fr 2.8fr 1.4fr 1.1fr;
   gap:18px;
   background:#dce9fb;
   border:1px solid #c9daf4;
   border-radius:16px;
   padding:14px 18px;
   margin-bottom:14px;
   align-items:center;
">
<div style="font-size:14px; font-weight:800; color:#334155; text-align:center;">Description</div>
<div style="font-size:14px; font-weight:800; color:#334155; text-align:center;">Explanation</div>
<div style="font-size:14px; font-weight:800; color:#334155; text-align:center;">Model Prediction</div>
<div style="font-size:14px; font-weight:800; color:#334155; text-align:center;">Predicted Code Description</div>
<div style="font-size:14px; font-weight:800; color:#334155; text-align:center;">Revised Code</div>
<div style="font-size:14px; font-weight:800; color:#334155; text-align:center;">Approve</div>
</div>
       """,
       unsafe_allow_html=True,
   )
   review_cols = st.columns([1.8, 2.8, 1.8, 2.8, 1.4, 1.1])
   with review_cols[0]:
       _render_info_box(description_lines, height=185, bold=True)
   with review_cols[1]:
       _render_info_box(explanation_text, height=185, bold=False)
   with review_cols[2]:
       selected_label = st.radio(
           "Select Output",
           options=option_labels,
           index=option_labels.index(default_label),
           key=f"validation_choice_{run_id}_{row_id}",
           label_visibility="collapsed",
       )
   selected_meta = option_value_map[selected_label]
   revised_key = f"revised_code_{run_id}_{row_id}"
   manual_backup_key = f"manual_revised_code_{run_id}_{row_id}"
   last_selected_key = f"last_selected_option_{run_id}_{row_id}"
   if revised_key not in st.session_state:
       st.session_state[revised_key] = selected_meta["code"] if selected_meta["source"] != "Manual Edit" else ""
   if manual_backup_key not in st.session_state:
       st.session_state[manual_backup_key] = ""
   previous_selected = st.session_state.get(last_selected_key)
   if previous_selected != selected_label:
       if selected_meta["source"] == "Manual Edit":
           st.session_state[revised_key] = st.session_state.get(manual_backup_key, "")
       else:
           st.session_state[revised_key] = selected_meta["code"]
       st.session_state[last_selected_key] = selected_label
   with review_cols[3]:
       _render_predicted_code_description_box(height=185)
   final_value = ""
   final_source = selected_meta["source"]
   with review_cols[4]:
       if selected_meta["source"] == "Manual Edit":
           dropdown_items = load_unspsc_dropdown_items()
           prompt_label = "Select UNSPSC code"
           label_to_code = {item["label"]: item["code"] for item in dropdown_items}
           code_to_label = {item["code"]: item["label"] for item in dropdown_items}
           code_to_desc = {item["code"]: item["desc"] for item in dropdown_items}
           picker_open_key = f"manual_picker_open_{run_id}_{row_id}"
           picker_select_key = f"manual_picker_select_{run_id}_{row_id}"
           picker_widget_key = f"{picker_select_key}_widget"
           stored_manual_code = st.session_state.get(manual_backup_key, "").strip()
           stored_manual_desc = code_to_desc.get(stored_manual_code, "")
           if picker_open_key not in st.session_state:
               st.session_state[picker_open_key] = not bool(stored_manual_code)
           if picker_select_key not in st.session_state:
               st.session_state[picker_select_key] = (
                   code_to_label.get(stored_manual_code, prompt_label) if stored_manual_code else prompt_label
               )
           if st.session_state[picker_open_key]:
               select_options = [prompt_label] + [item["label"] for item in dropdown_items]
               current_value = st.session_state.get(picker_select_key, prompt_label)
               current_index = select_options.index(current_value) if current_value in select_options else 0
               selected_dropdown_label = st.selectbox(
                   "Revised Code",
                   options=select_options,
                   index=current_index,
                   key=picker_widget_key,
                   label_visibility="collapsed",
               )
               st.session_state[picker_select_key] = selected_dropdown_label
               save_col, cancel_col = st.columns(2)
               with save_col:
                   if st.button("Apply", key=f"apply_manual_code_{run_id}_{row_id}", use_container_width=True):
                       if selected_dropdown_label == prompt_label:
                           st.warning("Please select a UNSPSC code.")
                       else:
                           selected_manual_code = label_to_code[selected_dropdown_label]
                           st.session_state[manual_backup_key] = selected_manual_code
                           st.session_state[revised_key] = selected_manual_code
                           st.session_state[picker_open_key] = False
                           st.rerun()
               with cancel_col:
                   if st.button("Cancel", key=f"cancel_manual_code_{run_id}_{row_id}", use_container_width=True):
                       st.session_state[picker_open_key] = False
                       st.session_state[picker_select_key] = (
                           code_to_label.get(stored_manual_code, prompt_label) if stored_manual_code else prompt_label
                       )
                       st.rerun()
               preview_code = ""
               preview_desc = ""
               if selected_dropdown_label != prompt_label:
                   preview_code = label_to_code[selected_dropdown_label]
                   preview_desc = code_to_desc.get(preview_code, "")
               if preview_code:
                   _render_code_box(preview_code, preview_desc)
               final_value = st.session_state.get(manual_backup_key, "").strip()
           else:
               manual_code = st.session_state.get(manual_backup_key, "").strip()
               manual_desc = code_to_desc.get(manual_code, "")
               _render_code_box(manual_code, manual_desc)
               if st.button("Change", key=f"change_manual_code_{run_id}_{row_id}", use_container_width=True):
                   st.session_state[picker_open_key] = True
                   st.session_state[picker_select_key] = (
                       code_to_label.get(manual_code, prompt_label) if manual_code else prompt_label
                   )
                   st.rerun()
               final_value = manual_code
       else:
           display_code = selected_meta["code"].strip()
           display_desc = desc_map.get(display_code, "")
           _render_code_box(display_code, display_desc)
           final_value = display_code
   with review_cols[5]:
       approve_clicked = st.button(
           "Approve",
           key=f"approve_row_{run_id}_{row_id}",
           use_container_width=True,
       )
   if approve_clicked:
       if not final_value:
           st.error("Please choose a predicted value or enter a revised code before approving.")
       else:
           approve_validation_row(
               run_id=run_id,
               validation_row_id=row_id,
               validated_unspsc=final_value,
               validation_source=final_source,
               validation_comment="",
           )
           st.success(f"Row {row_id} approved and moved to Golden Dataset.")
           st.rerun()
   st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
   with st.expander("Input Parameters", expanded=False):
       extra_exclude = {
           "short_descrition_(english)",
           "short_description_(english)",
           "Short Description (English)",
           "Short Description",
           "long_descritpion_(english)",
           "long_description_(english)",
           "Long Description (English)",
           "Long Description",
       }
       input_param_columns = [
           c for c in row.index
           if c not in TOP_REVIEW_COLUMNS and c not in extra_exclude
       ]
       input_params_df = pd.DataFrame([row[input_param_columns].to_dict()])
       input_params_df = format_preview_dataframe(input_params_df)
       st.dataframe(input_params_df, use_container_width=True, hide_index=True)

def render_validation_detail_view():
  run_id = get_selected_run_id()
  if not run_id:
      st.error("Missing run_id.")
      return
  vmeta = ensure_validation_run(run_id)
  queue_df = load_validation_queue_df(run_id)
  top_cols = st.columns([1.6, 6.4])
  with top_cols[0]:
      if st.button("â† Back to To Validate", key="back_to_to_validate", use_container_width=True):
          set_navigation_state(
           page="validation",
           sub="to_validate",
           run_id=None,
           )
          st.rerun()
  st.markdown(
      f"""
<div class="run-meta-card">
<div class="run-meta-title">Validation Detail</div>
<div class="run-meta-grid">
<div class="run-meta-item">
<div class="run-meta-label">Prediction Job</div>
<div class="run-meta-value">{run_id}</div>
</div>
<div class="run-meta-item">
<div class="run-meta-label">Approved Rows</div>
<div class="run-meta-value">{vmeta.get("approved_rows", 0)}/{vmeta.get("total_rows", 0)}</div>
</div>
<div class="run-meta-item">
<div class="run-meta-label">Status</div>
<div class="run-meta-value">{vmeta.get("status", "pending")}</div>
</div>
<div class="run-meta-item">
<div class="run-meta-label">Updated At</div>
<div class="run-meta-value">{vmeta.get("updated_at", "")}</div>
</div>
<div class="run-meta-item">
<div class="run-meta-label">Updated By</div>
<div class="run-meta-value">{vmeta.get("updated_by", "")}</div>
</div>
<div class="run-meta-item">
<div class="run-meta-label">Remaining Rows</div>
<div class="run-meta-value">{vmeta.get("remaining_rows", 0)}</div>
</div>
</div>
</div>
      """,
      unsafe_allow_html=True
  )
  if queue_df is None or queue_df.empty:
      st.markdown(
          """
<div class="section-card">
<div class="section-card-title">Validation Queue</div>
<div class="section-card-subtitle">
                  All rows for this prediction run have been approved.
</div>
<div class="empty-state-box">
                  No rows remaining in the validation queue.
</div>
</div>
          """,
          unsafe_allow_html=True
      )
      return
  desc_map = load_unspsc_description_map()
  page_size = 5
  page_key = f"validation_page_{run_id}"
  if page_key not in st.session_state:
      st.session_state[page_key] = 1
  total_rows = len(queue_df)
  total_pages = max(1, (total_rows + page_size - 1) // page_size)
  current_page = st.session_state[page_key]
  current_page = max(1, min(current_page, total_pages))
  st.session_state[page_key] = current_page
  start_idx = (current_page - 1) * page_size
  end_idx = min(start_idx + page_size, total_rows)
  st.markdown(
       """
<hr style="border: none; height: 2px; background-color: #6b7280; margin: 16px 0;">
       """,
       unsafe_allow_html=True
   )
  nav_cols = st.columns([1.2, 1.2, 4.8, 1.6])
  with nav_cols[0]:
      if st.button(
          "Prev Page",
          key=f"prev_validation_page_{run_id}",
          disabled=(current_page == 1),
          use_container_width=True
      ):
          st.session_state[page_key] = current_page - 1
          st.rerun()
  with nav_cols[1]:
      if st.button(
          "Next Page",
          key=f"next_validation_page_{run_id}",
          disabled=(current_page == total_pages),
          use_container_width=True
      ):
          st.session_state[page_key] = current_page + 1
          st.rerun()
  with nav_cols[3]:
      st.markdown(
          f'<div class="validation-summary-chip">Records {start_idx + 1}-{end_idx} / {total_rows}</div>',
          unsafe_allow_html=True
      )
  st.markdown(
       """
<hr style="double: none; height: 2px; background-color: #93B5E1; margin: 16px 0;">
       """,
       unsafe_allow_html=True
   )
  page_df = queue_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
  for idx, (_, row) in enumerate(page_df.iterrows(), start=1):
      render_validation_record_card(run_id, row, desc_map)
      st.markdown(
       """
<hr style="double: none; height: 2px; background-color: #93B5E1; margin: 16px 0;">
       """,
       unsafe_allow_html=True
   )

def render_golden_dataset_view():
   golden_df = load_golden_dataset_df()
   st.markdown(
       """
<div class="section-card">
<div class="section-card-title">Validated Parts</div>
</div>
""",
       unsafe_allow_html=True,
   )
   st.markdown("***")
   if golden_df is None or golden_df.empty:
       st.markdown(
           """
<div class="section-card">
<div class="empty-state-box">Validated data is empty.</div>
</div>
""",
           unsafe_allow_html=True,
       )
       return
   display_df = golden_df.copy()
   if "_validation_row_id" in display_df.columns:
       display_df = display_df.drop(columns=["_validation_row_id"])
   render_paginated_dataframe(display_df, pager_prefix="golden_dataset", default_page_size=10)

def render():
   load_css(
       get_global_css()
       + get_app_shell_css()
       + get_home_page_css()
       + get_classify_page_css()
       + get_validation_page_css()
   )
   selected_menu = get_selected_submenu()
   set_navigation_state(
       page="validation",
       sub=selected_menu,
       run_id=get_selected_run_id(),
       )
   render_sidebar(
       subtitle="Validate Classified Parts",
       section_label="Section",
       menu_items=[
           {
               "label": "To Validate",
               "page": "validation",
               "sub": "to_validate",
               "active": selected_menu == "to_validate",
           },
           {
               "label": "Validated Parts Data",
               "page": "validation",
               "sub": "golden_dataset",
               "active": selected_menu == "golden_dataset",
           },
       ],
   )
   render_top_nav(active_slug="home")
   if selected_menu == "to_validate":
       render_to_validate_view()
   elif selected_menu == "golden_dataset":
       render_golden_dataset_view()
   else:
       render_validation_detail_view()