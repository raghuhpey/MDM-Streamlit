# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
from dataiku import SQLExecutor2
import io
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import json
from services.model_registry_service import upsert_registry_from_metrics_row
from credentials import *

tz_ist = ZoneInfo('Asia/Kolkata')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
input_data = "INPUT_DATA"

model1_folder = "MODEL_1_OUTPUT"
model2_folder = "MODEL_2_OUTPUT"
model3_folder = "MODEL_3_OUTPUT"
llm_folder = "LLM_OUTPUT"

model_repo = "MODELS_REPOSITORY"
model_versions_csv = "Model_Versions.csv"
model_deploy_csv = "/MODEL_METRICS/model_deploy_metrics.csv"
version_col = "version"

model_metric_keys = ["accuracy", "macro_f1", "macro_recall", "weighted_f1", "weighted_recall"]
model_deploy_metric_keys = ["accuracy", "macro_f1", "macro_recall"]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
timestamp = datetime.now(tz=tz_ist).strftime("%Y_%m_%d_%H_%M")
print(timestamp)
time= datetime.now(tz=tz_ist)
time

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pd.DataFrame({"time":[time]})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Get data, check file exist in folder and get version
def get_data(folder_name: str = model_repo,
             filename: str = model_deploy_csv) -> pd.DataFrame:
    """
    Reads a CSV from a Dataiku managed folder and normalizes columns:
    lower-case + spaces -> underscores.
    """
    folder = dataiku.Folder(folder_name)

    with folder.get_download_stream(filename) as stream:
        df = pd.read_csv(stream)
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]

    return df


def _file_exists_in_managed_folder(folder: dataiku.Folder, filename: str) -> bool:
    """
    Returns True if filename exists in managed folder, else False.
    Works regardless of Dataiku returning paths with leading '/'.
    """
    try:
        paths = folder.list_paths_in_partition()
        normalized = filename.lstrip("/")
        return normalized in {p.lstrip("/") for p in paths}
    except Exception:
        # In rare cases list_paths_in_partition can fail depending on permissions/backends.
        # Fallback: attempt opening a download stream.
        try:
            with folder.get_download_stream(filename) as _:
                return True
        except Exception:
            return False


def _extract_numeric_version(series: pd.Series) -> pd.Series:
    """
    Converts a 'version' column to numeric versions safely.
    Handles:
      - ints: 1, 2, 3
      - strings: '1', 'v2', 'version_3', 'V10'
    Non-parsable values become <NA>.
    """
    if series is None:
        return pd.Series([], dtype="Int64")

    s = series.astype(str).str.strip()
    # extract the first group of digits
    nums = s.str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(nums, errors="coerce").astype("Int64")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_next_version(folder_name: str = model_repo,
                     filename: str = model_deploy_csv,
                     version_col: str = version_col) -> int:
    """
    If filename exists, reads it and returns (max(version)+1).
    If not exists (or no valid versions), returns 1.
    """
    folder = dataiku.Folder(folder_name)

    if not _file_exists_in_managed_folder(folder, filename):
        return 1

    df = get_data(folder_name, filename)

    if version_col not in df.columns:
        # file exists but doesn't have version column -> start at 1
        return 1

    v = _extract_numeric_version(df[version_col])
    if v.dropna().empty:
        return 1

    return int(v.max()) + 1

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Copy contents from one folder to another
def copy_managed_folder_contents(source_folder_id: str,
                                 target_folder_id: str,
                                 target_subfolder_name: str = "",
                                 filename: str = None,
                                 overwrite: bool = True):

    source = dataiku.Folder(source_folder_id)
    destination = dataiku.Folder(target_folder_id)

    # Normalize prefix
    prefix = (target_subfolder_name or "").strip().strip("/")
    if prefix:
        prefix += "/"

    # List all source paths once
    all_paths = source.list_paths_in_partition()

    # Decide which paths to copy
    if filename:
        requested_key = filename.strip().lstrip("/")  # compare without leading '/'

        # Map "no-leading-slash" -> "actual Dataiku path"
        lookup = {p.lstrip("/"): p for p in all_paths}

        if requested_key not in lookup:
            raise FileNotFoundError(
                f"File '{filename}' not found in source folder '{source_folder_id}'. "
                f"Available examples: {list(sorted(lookup.keys()))[:10]}"
            )

        paths_to_copy = [lookup[requested_key]]
        print(f"Copying a single file from '{source_folder_id}' to '{target_folder_id}/{prefix}': {lookup[requested_key]}")
    else:
        paths_to_copy = all_paths
        print(f"Found {len(paths_to_copy)} files to copy from '{source_folder_id}' to '{target_folder_id}/{prefix}'")

    copied = 0
    for i, path in enumerate(paths_to_copy, start=1):
        dest_path = f"{prefix}{path.lstrip('/')}"
#         print(f"{i}/{len(paths_to_copy)} Copying: {path} -> {dest_path}")

        with source.get_download_stream(path) as reader:
            with destination.get_writer(dest_path) as writer:
                writer.write(reader.read())

        copied += 1

    print("Copy completed successfully")
    return copied

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# read individual jsons ans create a df for model metrics


def _read_json_from_managed_folder(folder: dataiku.Folder, path: str) -> dict:
    """Read a JSON file from a Dataiku managed folder."""
    normalized = path if path.startswith("/") else "/" + path
    with folder.get_download_stream(normalized) as stream:
        return json.loads(stream.read().decode("utf-8"))

def _flatten_metrics(meta: dict, model_prefix: str, metric_keys: list[str], include_train: bool = True) -> dict:
    row = {}

    train = meta.get("training_metrics", {}) or {}
    test  = meta.get("test_metrics", {}) or {}

    # Train/Test metrics
    for k in metric_keys:
        if include_train:
            row[f"{model_prefix}_train_{k}"] = train.get(k)
        row[f"{model_prefix}_test_{k}"]  = test.get(k)

    # Optional extra metrics at top-level in your metadata
#     row[f"{model_prefix}_top_3_accuracy"] = meta.get("top_3_accuracy")
#     row[f"{model_prefix}_top_5_accuracy"] = meta.get("top_5_accuracy")

    # row[f"{model_prefix}_model_name"] = meta.get("model_name")
    # row[f"{model_prefix}_num_classes"] = meta.get("number_of_classes")
    # row[f"{model_prefix}_train_samples"] = meta.get("training_samples")
    # row[f"{model_prefix}_num_features"] = meta.get("number_of_features")
    # row[f"{model_prefix}_train_duration_min"] = meta.get("model_train_duration_in_minutes")

    return row

def build_models_metrics_df_from_repo(model_repo_folder_id: str, version: str, metric_keys: list[str],include_train: bool = True) -> pd.DataFrame:
    """
    Reads:
      <version>/model1/metadata.json
      <version>/model2/metadata.json
      <version>/model3/metadata.json
    from the MODELS_REPOSITORY folder and builds a single-row DF.
    """
    repo = dataiku.Folder(model_repo_folder_id)

    meta1 = _read_json_from_managed_folder(repo, f"v{version}/model1/metadata.json")
    meta2 = _read_json_from_managed_folder(repo, f"v{version}/model2/metadata.json")
    meta3 = _read_json_from_managed_folder(repo, f"v{version}/model3/metadata.json")

    row = {"version": version,"datetime": time}

    row.update(_flatten_metrics(meta1, "model1",metric_keys,include_train))
    row.update(_flatten_metrics(meta2, "model2",metric_keys,include_train))
    row.update(_flatten_metrics(meta3, "model3",metric_keys,include_train))

    df = pd.DataFrame([row])
    return df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def write_csv_to_folder(df: pd.DataFrame,
                        folder_name: str,
                        filename: str):
    filename_without_ext = filename[:-4] if filename.lower().endswith(".csv") else filename
    models_folder = dataiku.Folder(folder_name)
    with models_folder.get_writer(f"{filename_without_ext}.csv") as writer:
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        writer.write(buffer.getvalue())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# concatenate new model metrics with earlier ones
def update_model_versions(df_metrics: pd.DataFrame,
                          folder_name: str = model_repo,
                          versions_filename: str = model_deploy_csv,
                          version_col: str = version_col,
                          version: int = None) -> pd.DataFrame:

    folder = dataiku.Folder(folder_name)
    exists = _file_exists_in_managed_folder(folder, versions_filename)
    print(version)
    # Decide version to use
    if version is None:
        version = get_next_version(folder_name, versions_filename, version_col=version_col)
        print(f"No versions found: new version{version}")

    # Ensure df_metrics is a dataframe and add version column
    df_new = df_metrics.copy()
    df_new.columns = [c.lower().replace(" ", "_") for c in df_new.columns]
    df_new[version_col] = version

    if exists:
        df_old = get_data(folder_name, versions_filename)
        # normalize old columns too (in case the file was not normalized previously)
        df_old.columns = [c.lower().replace(" ", "_") for c in df_old.columns]

        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        # Create new file starting with df_new (version will be 1 by default)
        df_all = df_new

    # Save back (strip ".csv" because writer adds it)
    base_name = versions_filename[:-4] if versions_filename.lower().endswith(".csv") else versions_filename
    write_csv_to_folder(df_all, folder_name, base_name)

    return df_all

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
version = get_next_version()
print(f"next version: {version}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -------------------------------------------------------------------------- Update version to sql table
FOLDER_ID = "RETRAINING_CONTEXT"
RETRAINING_DATASET_NAME = "mdm_retraining_jobs_tbl"

mf = dataiku.Folder(FOLDER_ID)

with mf.get_download_stream("retraining_context.json") as s:
   ctx = json.loads(s.read().decode("utf-8"))

retraining_job_id = ctx.get("retraining_job_id")

if not retraining_job_id:
   raise ValueError("retraining_job_id missing in retraining_context.json")

def _safe_sql(v):
   if v is None:
       return ""
   return str(v).replace("'", "''")

def update_job_model_version(retraining_job_id: str, model_version: str):
   ds = dataiku.Dataset(RETRAINING_DATASET_NAME)
   loc = ds.get_location_info()
   info = (loc or {}).get("info", {})
   conn = info.get("connectionName")
   table = info.get("quotedResolvedTableName")
   if not conn or not table:
       raise ValueError(f"Could not resolve SQL connection/table for dataset {RETRAINING_DATASET_NAME}")
   sql = f"""
   UPDATE {table}
   SET model_version = '{_safe_sql(model_version)}'
   WHERE retraining_job_id = '{_safe_sql(retraining_job_id)}'
   """
   client = dataiku.api_client()
   q = client.sql_query(
       sql,
       connection=conn,
       type="sql",
       project_key=ds.project_key,
       post_queries=["COMMIT"],
   )
   q.verify()



update_job_model_version(retraining_job_id, version)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# copy files
copy_managed_folder_contents(model1_folder,model_repo,f"v{version}/model1")
copy_managed_folder_contents(model2_folder,model_repo,f"v{version}/model2")
copy_managed_folder_contents(model3_folder,model_repo,f"v{version}/model3")
copy_managed_folder_contents(llm_folder,model_repo,f"v{version}/llm")
copy_managed_folder_contents(input_data,model_repo,f"v{version}")
copy_managed_folder_contents(model1_folder,model_repo,f"v{version}","features.json")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_deploy_metrics = build_models_metrics_df_from_repo(model_repo, version,model_deploy_metric_keys,include_train=False)
df_deploy_metrics

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_metrics = build_models_metrics_df_from_repo(model_repo, version,model_metric_keys,include_train=True)
df_metrics

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_all_versions = update_model_versions(
    df_metrics=df_metrics,
    folder_name=model_repo,
    versions_filename=model_versions_csv,
    version = version
)
df_deploy_all_versions = update_model_versions(
    df_metrics=df_deploy_metrics,
    folder_name=model_repo,
    versions_filename=model_deploy_csv,
    version = version
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
ver2 = get_next_version(model_repo, model_versions_csv, version_col)
ver2

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
ver = get_next_version(model_repo, model_deploy_csv, version_col)
ver

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_all_versions.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Model Metrics

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import pandas as pd
import numpy as np
import json
import io
import os
import warnings
import re
import joblib
import dataiku
import faiss
import tempfile
import tensorflow as tf
from tensorflow import keras
from openai import AzureOpenAI
from sklearn.preprocessing import OneHotEncoder
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd
import os
import re
from openai import AzureOpenAI
import numpy as np
import faiss
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
tz_ist = ZoneInfo('Asia/Kolkata')

warnings.filterwarnings('ignore')


# ==================== ARTIFACT PATHS ====================
# DATA_FOLDER = "Data_Folder"
MODEL_FOLDER = "MODELS_REPOSITORY"


# MODEL_1_PATH = "/Hari/2026_04_03_model_xgboost_new_wo_clamp/"
# MODEL_2_PATH = "/Hari/2026_04_03_Model_Catboost_new_wo_clamp/"
# MODEL_3_PATH = "/Hari/2026_04_03_model_mlp_new_wo_clamp/"

# MODEL_1_PATH = "/Hari/2026_04_16_all_unspsc/2026_04_12_13_58_all_unspsc_XGboost_HPO/"
# MODEL_2_PATH = "/Hari/2026_04_16_all_unspsc/2026_04_13_16_45_all_unspsc_Catboost_HPO/"
# MODEL_3_PATH = "/Hari/2026_04_16_all_unspsc/2026_04_10_14_37_all_unspsc_MLP_HPO/"


# ARTIFACT_NAMES = {
#     'model1_pipeline': f'{MODEL_1_PATH}model.pkl',
#     'model2_pipeline': f'{MODEL_2_PATH}model.pkl',
#     'model3_pipeline': f'{MODEL_3_PATH}model.keras',
#     'pca_model': f'{MODEL_3_PATH}pca.pkl',
#     'label_encoder_y': f'{MODEL_3_PATH}label_encoder_y_variable.pkl',
#     'metadata': f'{MODEL_3_PATH}metadata.json',
#     'features': f'{MODEL_3_PATH}features.json',
# }

USE_CACHED_EMBEDDINGS = True
TARGET_COL = "unspsc_code"
TEXT_COL = "combined_description"
BINARY_COLS = []
CATEGORICAL_COLS = [
    "is_related_to_fire&smoke?",
    "repairable_?",
    "perishable_part",
    "hazardous_goods_(s.f.)",
    "unit_of_measure_information",
    "usability",
    "part_rohs",

]

MODEL_NAME = embedding_model
# ---- Limits ----
MAX_CHUNK_TOKENS = 7000
MAX_TPM = 320_000
EMBED_BATCH_SIZE = 32
MAX_RETRIES = 3
SAVE_EVERY = 5000


client_aoai = AzureOpenAI(
    api_key=embedding_api_key,
    azure_endpoint=embedding_azure_endpoint,
    api_version=embedding_api_version
)
# data_folder = dataiku.Folder(DATA_FOLDER)
model_folder = dataiku.Folder(MODEL_FOLDER)



MODEL_1_PATH = f"/v{version}/model3/"
MODEL_2_PATH = f"/v{version}/model2/"
MODEL_3_PATH = f"/v{version}/model1/"
COMMON_PATH = f"/v{version}/"

ARTIFACT_NAMES = {
    'model1_pipeline': f'{MODEL_1_PATH}model.pkl',
    'model2_pipeline': f'{MODEL_2_PATH}model.pkl',
    'model3_pipeline': f'{MODEL_3_PATH}model.keras',
    'pca_model': f'{COMMON_PATH}pca.pkl',
    'label_encoder_y': f'{COMMON_PATH}label_encoder_y_variable.pkl',
    'metadata': f'{MODEL_3_PATH}metadata.json',
    'features': f'{COMMON_PATH}features.json',
}

# ==================== Load Artifacts ====================
def load_artifact(models_folder, path):
    _, ext = os.path.splitext(path.lower())
    if ext in [".pkl", ".joblib", ".json"]:
        with models_folder.get_download_stream(path) as stream:
            data = stream.read()
        if ext in [".pkl", ".joblib"]:
            return joblib.load(io.BytesIO(data))
        elif ext == ".json":
            return json.loads(data.decode("utf-8"))
    elif ext in [".keras",]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = os.path.join(tmp_dir, os.path.basename(path) or "model.keras")
            with models_folder.get_download_stream(path) as stream, open(local_path, "wb") as f:
                f.write(stream.read())
            model = keras.models.load_model(local_path)
            return model

    else:
        raise ValueError(f"Unsupported file type: {ext}")

def load_artifacts_from_dataiku():
    try:
        folder = model_folder
        model1_pipeline = load_artifact(folder, ARTIFACT_NAMES['model1_pipeline'])
        model2_pipeline = load_artifact(folder, ARTIFACT_NAMES['model2_pipeline'])
        model3_pipeline = load_artifact(folder, ARTIFACT_NAMES['model3_pipeline'])
        pca_model = load_artifact(folder, ARTIFACT_NAMES['pca_model'])
        label_encoder_y = load_artifact(folder, ARTIFACT_NAMES['label_encoder_y'])
        metadata = load_artifact(folder, ARTIFACT_NAMES['metadata'])
        feature_trained_on = load_artifact(folder, ARTIFACT_NAMES['features'])
        return model1_pipeline, model2_pipeline, model3_pipeline, pca_model, label_encoder_y, metadata, feature_trained_on
    except Exception as e:
        return None

result = load_artifacts_from_dataiku()
if result is None:
    raise Exception("Failed to load artifacts from Dataiku and local storage!")

model1_pipeline, model2_pipeline, model3_pipeline, pca_model, label_encoder_y, metadata, feature_trained_on = result

# ==================== Data Cleaning & Feature Engineering ====================
def compose_description_upper(obj: str, doc: str) -> str:
    obj = obj if isinstance(obj, str) else ""
    doc = doc if isinstance(doc, str) else ""
    if obj and doc and re.search(re.escape(obj), doc, flags=re.IGNORECASE):
        head = doc
    else:
        head = f"{doc} FOR {obj}" if doc and obj else doc or obj
    return (head.strip().upper() + ".") if head.strip() else ""

def data_cleaning_pipeline(df):
    initial_rows = df.shape[0]
#     df = df.apply(
#         lambda col: col.str.strip().str.replace(r"\s+", " ", regex=True)
#         if col.dtype == "object" else col
#     )
    df = df.apply(
        lambda col: col.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        if col.dtype == "object" else col
    )
    df = df[~df['short_description_(english)'].isna()]
    df = df[['short_description_(english)','long_description_(english)',
             'part_manufacturer_list_information','hazardous_goods_(s.f.)',
             'is_related_to_fire&smoke?','part_rohs','perishable_part','repairable_?',
             'unit_of_measure_information','usability']] #,'governance','parent_information','branch']]

    df['part_manufacturer_list_information'].fillna('unknown_manufacturer', inplace=True)
    df['hazardous_goods_(s.f.)'].fillna('unknown', inplace=True)
    df['is_related_to_fire&smoke?'].fillna('unknown', inplace=True)
    df['part_rohs'].fillna('not_defined', inplace=True)
    df['perishable_part'].fillna('unknown', inplace=True)
    df['repairable_?'].fillna('unknown', inplace=True)
    df['unit_of_measure_information'].fillna('other_measure', inplace=True)
    df['usability'].fillna('other_usability', inplace=True)
    df["combined_description"] = df.apply(
        lambda r: compose_description_upper(r["short_description_(english)"], r["long_description_(english)"]),
        axis=1
    )

    df['combined_description'] += ", manufacturer is " + df['part_manufacturer_list_information']
    df.drop(['short_description_(english)','long_description_(english)','part_manufacturer_list_information'], axis=1, inplace=True)
#     df = df.drop_duplicates().reset_index(drop=True)
    return df

def count_tokens(text):
    return len(text) // 4

def embed_batch_with_retry(texts):
    for attempt in range(MAX_RETRIES):
        try:
            response = client_aoai.embeddings.create(
                model=MODEL_NAME,
                input=texts
            )
            return [r.embedding for r in response.data]

        except Exception as e:
            retryable = any(
                word in str(e).lower()
                for word in ["rate limit", "timeout", "internal", "429"]
            )

            if not retryable or attempt == MAX_RETRIES - 1:
                raise

            sleep_time = 2 ** attempt
            print(f"Retry {attempt+1}, sleeping {sleep_time}s due to {e}")
            time.sleep(sleep_time)

# ==================== Binary & Categorical Processing ====================
def process_binary(df):
    return df[BINARY_COLS].replace({"True": 1, "False": 0, True: 1, False: 0}).fillna(0).astype(int).values

# ==================== Prediction Function ====================
def predict_via_neural_network(model, X_test_aligned):
#     X_test_aligned=[]
    X_np = X_test_aligned.values.astype("float32")
    logits = model.predict(X_np, verbose=0)
    y_proba = tf.nn.softmax(logits, axis=1).numpy()
    y_pred = np.argmax(y_proba, axis=1)

    return y_pred


def predict(df_input):
    df_cleaned = data_cleaning_pipeline(df_input.copy())
#     avg_tokens = average_rough_tokens(df_cleaned, TEXT_COL)
#     print(f"Average tokens per row: {avg_tokens:.2f}")


    EMBED_BATCH_SIZE = calculate_embed_batch_size(
        df_cleaned,
        text_col=TEXT_COL,
        max_chunk_tokens=MAX_CHUNK_TOKENS
    )

    print("Recommended EMBED_BATCH_SIZE:", EMBED_BATCH_SIZE)


    start_embedding = datetime.now(tz=tz_ist)
    embedding_df = get_embeddings(df_cleaned, TEXT_COL)
    end_embedding = datetime.now(tz=tz_ist)
    duration = (end_embedding - start_embedding).total_seconds()
    print("Embedding Duration: ",duration)
    df_merged = df_cleaned.merge(embedding_df, on="combined_description", how="left").drop(columns=[TEXT_COL])
    df_final = df_merged.copy()
    X_test_df = df_final.drop(columns=[TARGET_COL],errors ="ignore")

    # PCA
    X_test_emb = np.vstack(X_test_df['embedding'].values)
    X_test_pca = pca_model.transform(X_test_emb)
    df_X_test_pca = X_test_df.join(pd.DataFrame(X_test_pca, index=X_test_df.index).add_prefix("pca_"))
    df_X_test_pca.drop(columns=["embedding"], inplace=True)
    X_test_df = df_X_test_pca.copy()

    # Binary & Categorical
    X_test_text = X_test_df.drop(columns=[*BINARY_COLS,*CATEGORICAL_COLS], axis=1)
    X_test_bin = process_binary(X_test_df)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_test_cat = ohe.fit_transform(X_test_df[CATEGORICAL_COLS])
    X_test_np = np.hstack([X_test_text, X_test_bin, X_test_cat])

    text_feature_names = [f"text_emb_{i}" for i in range(X_test_text.shape[1])]
    binary_feature_names = BINARY_COLS
    categorical_feature_names = ohe.get_feature_names_out(CATEGORICAL_COLS)
    feature_names = text_feature_names + binary_feature_names + list(categorical_feature_names)
    X_test_aligned = pd.DataFrame(X_test_np, columns=feature_names, index=X_test_df.index)

    # Align with training features
    X_test_aligned = X_test_aligned.reindex(columns=feature_trained_on, fill_value=0)

    # Predictions
    model1_pred = model1_pipeline.predict(X_test_aligned)
    model2_pred = model2_pipeline.predict(X_test_aligned)
    model3_pred = predict_via_neural_network(model3_pipeline,X_test_aligned)

    model1_pred_labelled = label_encoder_y.inverse_transform(model1_pred)
    model2_pred_labelled = label_encoder_y.inverse_transform(model2_pred)
    model3_pred_labelled = label_encoder_y.inverse_transform(model3_pred)

    return model1_pred_labelled, model2_pred_labelled, model3_pred_labelled


def generate_ml_output(incoming_df):

    df=incoming_df.copy()
    model1_pred_labelled, model2_pred_labelled, model3_pred_labelled = predict(df)

    # Add to final df
    df["Model1"] = model1_pred_labelled
    df["Model2"] = model2_pred_labelled
    df["Model3"] = model3_pred_labelled
    print("ML pipeline done")

    return df

import tiktoken
import math

def calculate_embed_batch_size(
    df,
    text_col,
    max_chunk_tokens=7000,
    model="text-embedding-3-large",
    safety_margin=1
):
    encoding = tiktoken.encoding_for_model(model)

    token_counts = df[text_col].astype(str).apply(
        lambda x: len(encoding.encode(x))
    )

    p95_tokens = token_counts.quantile(0.9)
#     print(p95_tokens)
    usable_tokens = max_chunk_tokens * safety_margin

    embed_batch_size = math.floor(usable_tokens / p95_tokens)

    return max(1, embed_batch_size)

from concurrent.futures import ThreadPoolExecutor, as_completed

def get_embeddings(df, TEXT_COL, max_workers=4):
    texts = df[TEXT_COL].astype(str).drop_duplicates().tolist()

    batches = []
    current_batch = []
    token_budget = 0
    start_time = time.time()

    for text in texts:
        tokens = count_tokens(text)

        if tokens > MAX_CHUNK_TOKENS:
            continue

        if token_budget + tokens > MAX_TPM:
            elapsed = time.time() - start_time
            if elapsed < 60:
                time.sleep(60 - elapsed)
            token_budget = 0
            start_time = time.time()

        current_batch.append(text)
        token_budget += tokens

        if len(current_batch) == EMBED_BATCH_SIZE:
            batches.append(current_batch)
            current_batch = []

    if current_batch:
        batches.append(current_batch)

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(embed_batch_with_retry, batch): batch
            for batch in batches
        }

        for future in as_completed(futures):
            batch = futures[future]
            embeddings = future.result()
            results.extend(zip(batch, embeddings))

    final_df = pd.DataFrame(results, columns=[TEXT_COL, "embedding"])

    return final_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# GEN AI

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
client = dataiku.api_client()
project = client.get_default_project()
folders = project.list_managed_folders()
folder = dataiku.Folder("MODELS_REPOSITORY")
# folder_path = folder.get_path()



client_gpt = AzureOpenAI(
    api_key=gpt_api_key,
    azure_endpoint=gpt_azure_endpoint,
    api_version=gpt_api_version
)

def normalize_unspsc_code(x):

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None

    if isinstance(x, (int,)):
        s = str(x)
        return s.zfill(8) if len(s) <= 8 else None

    if isinstance(x, float):
        if x.is_integer():
            s = str(int(x))
            return s.zfill(8) if len(s) <= 8 else None
        return None

    s = str(x).strip()

    m = re.match(r"^(\d{8})\.0+$", s)
    if m:
        return m.group(1)

    s = re.sub(r"\D", "", s)
    return s if len(s) == 8 else None

# def gov_index_vector():

#     gov_index = faiss.read_index(os.path.join(folder_path, f"v{version}/llm/governance_faiss.index"))
#     faiss.write_index(gov_index, os.path.join(folder_path, f"v{version}/llm/governance_faiss.index"))

#     return gov_index

def gov_index_vector():
    folder = dataiku.Folder("MODELS_REPOSITORY")
    filename = f"v{version}/llm/governance_faiss.index"

    # --- Read index bytes from managed folder ---
    with folder.get_download_stream(filename) as f:
        index_bytes = f.read()

    # Convert bytes -> numpy uint8 -> deserialize
    index_arr = np.frombuffer(index_bytes, dtype='uint8')
    gov_index = faiss.deserialize_index(index_arr)

    # --- Serialize and write back to the managed folder ---
    buf = faiss.serialize_index(gov_index)

    # Ensure we upload bytes (handle different FAISS return types)
    try:
        data = bytes(buf)  # works if buf is a Python buffer-like
    except TypeError:
        data = np.array(buf, dtype='uint8').tobytes()  # works if buf is a numpy array-like

    with folder.get_writer(filename) as w:
        w.write(data)

    return gov_index

# def gov_metadata_vector():
#     with open(os.path.join(folder_path, "governance_metadata_new.json"), "r") as f:
#         gov_metadata = json.load(f)

#     with open(os.path.join(folder_path, "governance_metadata_new.json"), "w") as f:
#         json.dump(gov_metadata, f)

#     return gov_metadata

# def gov_metadata_vector():
#     # Read JSON
#     with folder.get_download_stream(f"v{version}/llm/governance_metadata.json") as f:
#         gov_metadata = json.load(f)

#     # Write JSON
# #     with folder.get_writer("governance_metadata_new.json") as w:
# #         w.write(json.dumps(gov_metadata).encode("utf-8"))

#     return gov_metadata

def gov_metadata_vector():
    # Read JSON
    with folder.get_download_stream(f"v{version}/llm/governance_metadata.json") as f:
        gov_metadata = json.load(f)

    # Write JSON
    with folder.get_writer(f"v{version}/llm/governance_metadata.json") as w:
        w.write(json.dumps(gov_metadata).encode("utf-8"))

    return gov_metadata

def retrieve_governance_context(query_text, k=5):
    query_emb = client_aoai.embeddings.create(
        model=embedding_model,
        input=[query_text.lower()]
    )

    query_vec = np.array(
        query_emb.data[0].embedding,
        dtype="float32"
    ).reshape(1, -1)

    gov_index = gov_index_vector()
    gov_metadata = gov_metadata_vector()

    D, I = gov_index.search(query_vec, k)

    retrieved = []
    for idx, dist in zip(I[0], D[0]):
        item = gov_metadata[idx].copy()
        item["distance"] = float(dist)
        retrieved.append(item)

    return retrieved
def extract_unspsc_candidates(retrieved_items):
    candidates = {}
    for item in retrieved_items:
        code = normalize_unspsc_code(item.get("UNSPSC code"))
        if not code:
            continue

        family = item["Family Name English information"]
        part = item["Short Description (English)"]

        if code not in candidates:
            candidates[code] = {
                "family": family,
                "parent": item["Parent information"],
                "description": item["Description"],
                "part" : part

            }

    return candidates

def build_rag_prompt(part_description, retrieved_items, unspsc_candidates, model_preds=None):

    governance_lines = []

    for item in retrieved_items:

        examples = []
        for i in range(1, 11):
            col_name = "Part Description Example " + str(i)
            if col_name in item and item[col_name] is not None:
                examples.append(str(item[col_name]))

        example_text = ", ".join(examples)

        line = (
            f"- Family: {item['Family Name English information']}, "
            f"Parent: {item['Parent information']}, "
            f"Description: {item['Description']}, "
            f"UNSPSC: {item['UNSPSC code']}, "
            f"Part: {item['Short Description (English)']},"
            f"Examples: {example_text}"

        )

        governance_lines.append(line)

    governance_context = "\n".join(governance_lines)

    candidate_context = "\n".join(
        [
            f"- UNSPSC {code}: Family {info['family']}, "
            f"Parent {info['parent']}, Description {info['description']}, Part{info['part']}"
            for code, info in unspsc_candidates.items()
        ]
    )


    model_block = ""
    if model_preds:
        model_block = f"""
    Model predictions (only consider these if they are supported by the Knowledge Base Context):
    - Model 1: {model_preds.get("model1")}
    - Model 2: {model_preds.get("model2")}
    - Model 3: {model_preds.get("model3")}
    """

    prompt = f"""

You are a UNSPSC classification expert working with Alstom governance rules and also an LLM that will act as a judge to select the rightly classified UNSPSC code.
You are an SME in Alstom who understands the entered part descriptions and then can match it to the
family and parent details that it is most likely to fall under and generate the UNSPSC code accordingly.
Also look at the examples of part descriptions already matched correctly to their descriptions and use the sam approach for the new parts.
Look at all the candidates in the candidate_context and governance_context and choose the most appropriate UNSPSC code.


Part description:
"{part_description}"
{model_block}

Knowledge Base Context:
{governance_context}

Allowed UNSPSC candidates:
{candidate_context}

Judge rule:
1) First validate the Model 1 / Model 2 / Model 3 outputs against the Knowledge Base Context and check whether any model-predicted UNSPSC is a correct and supported match, using Short Description (English) where present; if Short Description is not available for a candidate, fall back to validating via Family and Parent context.  // UPDATED
    1a) If multiple model outputs are partially supported, rank them and select the best-supported one instead of rejecting all by default.
2) If one of the model codes is correct, select that exact code and treat it as accepted by the LLM, with reasoning supported by the Knowledge Base.  // UPDATED
3) If none of the model codes is correct or supported by the Knowledge Base, explicitly reject the model outputs and select the best UNSPSC ONLY from the Allowed UNSPSC candidates using Knowledge Base reasoning (prioritizing Short Description (English) match where present; otherwise Family/Parent match).  // UPDATED
4) Do NOT output a blank or empty GenAI result. If an exact match is not available, select the closest possible Family and Parent from the Knowledge Base.  // UPDATED
5) If no reasonable match exists even at the closest part description match or Family level, output UNSPSC: NONE.
6) If two UNSPSC candidates are equally valid, select the one with the closest functional use over material or form factor.

Rule:
Review the information entered and identify the closest part description match, Short Description (English) match (where available), and then Family and Parent match for the entered Part description.
Cross-reference the entries in the UNSPSC column.
Only map UNSPSC codes from the given data and not from any external source outside the Alstom master data used to create the knowledge base.
Prefer UNSPSC codes that have an explicit and exact match in the Knowledge Base over semantically similar but higher-level matches.

Notes:
Provide any pertinent but concise explanations for your classification choices.
Avoid selecting overly generic or catch-all families if a more specific Family or Parent is available in the Knowledge Base.

Output rules (mandatory):
- The first line MUST be exactly: UNSPSC: <8-digit code>  OR  UNSPSC: NONE
- The second line MUST be: Explanation: <one sentence>
- Do not add any extra lines or text.

Explanation Requirements:
The explanation must clearly describe the decision flow:

- First state whether any ML Model–predicted UNSPSC was validated and accepted by the LLM, using Knowledge Base support via Short Description (English) where present, otherwise Family/Parent context.  // UPDATED
- If a Model output is accepted, clearly mention that the final UNSPSC is selected from the Model output and justify it using Knowledge Base support.  // UPDATED
- If Model outputs are rejected, explain why they are not supported by the Knowledge Base.  // UPDATED
- If the final UNSPSC is selected by the LLM, clearly state that it is chosen from the Knowledge Base (not ML) and explain why it is the closest match (prioritizing Short Description (English) where present; otherwise Family/Parent match).  // UPDATED
- The explanation must always make it explicit whether the final UNSPSC came from ML acceptance or LLM (KB-based) judgment.
- The explanation must be exactly ONE sentence and must not contain multiple decision paths.

Respond in the following format:
UNSPSC: <8-digit code>
Explanation: <as per explanation requirements>

"""

    return prompt

# New
def parse_genai_response(text: str):
    """
    Robustly extract an 8-digit UNSPSC from the LLM output.
    1) Prefer 'UNSPSC: ########'
    2) If missing, fallback to any standalone 8-digit number in the text
    Also extract explanation if present.
    """
    m = re.search(r"UNSPSC\s*:\s*(\d{8}|NONE)", text, re.IGNORECASE)
    predicted = m.group(1) if m else None
    if predicted and predicted.upper() == "NONE":
        predicted = None

    if predicted is None:
        m2 = re.search(r"\b(\d{8})\b", text)
        predicted = m2.group(1) if m2 else None

    em = re.search(r"Explanation\s*:\s*(.*?)(?:Confidence\s*:|$)", text, re.DOTALL | re.IGNORECASE)
    explanation = em.group(1).strip() if em else ""

    return predicted, explanation


def predict_unspsc_with_rag(part_description, model1=None, model2=None, model3=None):

    gov_metadata = gov_metadata_vector()
    KB_CODES = set()
    for item in gov_metadata:
        code = normalize_unspsc_code(item.get("UNSPSC code"))
        if code:
            KB_CODES.add(code)

    retrieved = retrieve_governance_context(part_description, k=5)
    unspsc_candidates = extract_unspsc_candidates(retrieved)
    m1 = normalize_unspsc_code(model1)
    m2 = normalize_unspsc_code(model2)
    m3 = normalize_unspsc_code(model3)

    m1 = m1 if m1 in KB_CODES else None
    m2 = m2 if m2 in KB_CODES else None
    m3 = m3 if m3 in KB_CODES else None

    model_preds = {"model1": m1, "model2": m2, "model3": m3}
    prompt = build_rag_prompt(part_description, retrieved, unspsc_candidates, model_preds=model_preds)

    response = client_gpt.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": "You are a UNSPSC part classification subject matter expert and an LLM as a judge."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw_text = response.choices[0].message.content
    predicted_unspsc, explanation = parse_genai_response(raw_text)
    if predicted_unspsc is None:
        for c in [m1, m2, m3]:
            if c and c in KB_CODES:
                predicted_unspsc = c
                if not explanation:
                    explanation = "Chosen from ML model outputs because LLM did not return a valid UNSPSC line."
                break

    if predicted_unspsc is None and len(unspsc_candidates) > 0:
        predicted_unspsc = next(iter(unspsc_candidates.keys()))
        if not explanation:
            explanation = "Chosen from top retrieved KB candidate because no valid model code was accepted."

    predicted_unspsc = normalize_unspsc_code(predicted_unspsc)
    if predicted_unspsc and predicted_unspsc not in KB_CODES:
        predicted_unspsc = None

    if predicted_unspsc is None:
        source = "NONE"
    elif predicted_unspsc == m1:
        source = "ML(Model1)"
    elif predicted_unspsc == m2:
        source = "ML(Model2)"
    elif predicted_unspsc == m3:
        source = "ML(Model3)"
    else:
        source = "GenAI"

    return predicted_unspsc, explanation, source


def run_unspsc_predictions(df):
    results = []

    total = len(df)

#     for _, row in df.iterrows():

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        print(f"{idx:04d}/{total:04d}", end="\r", flush=True)
        part_query = row["short_description_(english)"]
#         print(row["short_description_(english)"])
        m1 = row.get("Model1")
        m2 = row.get("Model2")
        m3 = row.get("Model3")

        pred_code, exp, src = predict_unspsc_with_rag(
            part_query,
            model1=m1,
            model2=m2,
            model3=m3
        )

        results.append({
            "Part Query": part_query,
            "Model 1 OP": m1,
            "Model 2 OP": m2,
            "Model 3 OP": m3,
            "LLM as a Judge OP": pred_code
        })
    df_pred = pd.DataFrame(results)
    df["LLM as a Judge OP"]=df_pred["LLM as a Judge OP"]
    df.rename(columns={"Model1":"Model 1 OP","Model2":"Model 2 OP","Model3":"Model 3 OP"},inplace=True)
    return df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # LLM Metrics Calculation

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# incoming_df = get_data("MODELS_REPOSITORY",f"{COMMON_PATH}test_dataset.csv").sample(10)
incoming_df = get_data("MODELS_REPOSITORY",f"{COMMON_PATH}test_dataset.csv")

print(f"llm prediction: {incoming_df.shape}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
start_time = datetime.now(tz=tz_ist)

df_ml = generate_ml_output(incoming_df)

end = datetime.now(tz=tz_ist)
duration = (end - start_time).total_seconds()
print("ML Prediction Duration: ",duration)

start = datetime.now(tz=tz_ist)

df_ml = df_ml.reset_index(drop=True).copy()
df_ml = run_unspsc_predictions(df_ml)

end = datetime.now(tz=tz_ist)
duration = (end - start).total_seconds()
print("LLM Prediction Duration: ",duration)

df_ml.head(1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def oracle_metrics(
    df: pd.DataFrame,
    prediction_cols,
    actual_col: str
):

    # ---------- Oracle accuracy ----------
    any_correct = df[prediction_cols].eq(df[actual_col], axis=0).any(axis=1)
    oracle_accuracy = any_correct.mean()

    # ---------- Per-class oracle recall ----------
    classes = df[actual_col].unique()
    recalls = {}

    for cls in classes:
        actual_mask = df[actual_col] == cls
        hit_mask = df.loc[actual_mask, prediction_cols].eq(cls).any(axis=1)

        tp = hit_mask.sum()
        fn = actual_mask.sum() - tp

        recalls[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    per_class_recall = pd.Series(recalls, name="oracle_recall")
    macro_recall = per_class_recall.mean()

    return per_class_recall, macro_recall, oracle_accuracy

prediction_cols = ["Model 1 OP", "Model 2 OP", "Model 3 OP", "LLM as a Judge OP"]


per_class, overall_macro_recall, overall_accuracy = oracle_metrics(
    df_ml,
    prediction_cols=prediction_cols,
    actual_col="unspsc_code"
)

# print("Per-class recall:")
# print(per_class)

print("\nmacro recall:", overall_macro_recall)
print("accuracy:", overall_accuracy)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


def print_macro_metrics(y_true, y_pred):

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    print("Model Evaluation Metrics")
    print("-" * 30)
    print(f"Macro F1 Score   : {macro_f1:.4f}")
    print(f"Macro Recall    : {macro_recall:.4f}")
    print(f"Accuracy        : {accuracy:.4f}")

#     return {
#         "macro_f1": macro_f1,
#         "macro_recall": macro_recall,
#         "accuracy": accuracy
#     }
    return macro_f1, macro_recall, accuracy

df_ml["LLM as a Judge OP"]=df_ml["LLM as a Judge OP"].astype("int64")
temp_encoder = LabelEncoder()
temp_encoder.fit(
    pd.concat([df_ml["LLM as a Judge OP"], df_ml["unspsc_code"]]).unique()
)

y_true = temp_encoder.transform(df_ml["unspsc_code"])
y_pred = temp_encoder.transform(df_ml["LLM as a Judge OP"])


llm_macro_f1, llm_macro_recall, llm_accuracy = print_macro_metrics(
    y_true=y_true,
    y_pred=y_pred
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Saving File

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_model_results = get_data(model_repo, model_versions_csv)
df_model_results.loc[df_model_results["version"] == version, "llm_test_accuracy"] = llm_accuracy
df_model_results.loc[df_model_results["version"] == version, "llm_test_macro_f1"] = llm_macro_f1
df_model_results.loc[df_model_results["version"] == version, "llm_test_macro_recall"] = llm_macro_recall
df_model_results.loc[df_model_results["version"] == version, "overall_accuracy"] = overall_accuracy

write_csv_to_folder(df_model_results,model_repo, model_versions_csv)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_model_results = get_data(model_repo, model_deploy_csv)
df_model_results.loc[df_model_results["version"] == version, "llm_test_accuracy"] = llm_accuracy
df_model_results.loc[df_model_results["version"] == version, "llm_test_macro_f1"] = llm_macro_f1
df_model_results.loc[df_model_results["version"] == version, "llm_test_macro_recall"] = llm_macro_recall
df_model_results.loc[df_model_results["version"] == version, "overall_accuracy"] = overall_accuracy

write_csv_to_folder(df_model_results,model_repo, model_deploy_csv)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# --------------------------------------------------------------------------------
# Update SQL Model Registry (mdm_model_registry_tbl) from latest metrics row
# Place this AFTER you update/write model_deploy_metrics.csv
# --------------------------------------------------------------------------------

MODEL_REPO_FOLDER = "MODELS_REPOSITORY"
DEPLOY_METRICS_CSV_PATH = "MODEL_METRICS/model_deploy_metrics.csv"  # no leading "/"

def _display_value(v, default=""):
   if v is None:
       return default
   try:
       if pd.isna(v):
           return default
   except Exception:
       pass
   s = str(v).strip()
   return s if s and s.lower() not in {"nan", "none", "null"} else default

# Read the updated deploy metrics CSV from managed folder
repo_folder = dataiku.Folder(MODEL_REPO_FOLDER)

with repo_folder.get_download_stream(DEPLOY_METRICS_CSV_PATH) as s:
   df_deploy = pd.read_csv(s)

df_deploy.columns = [c.strip().lower().replace(" ", "_") for c in df_deploy.columns]
# Find the row for current version
# (your CSV stores numeric versions like 1,2,3)
df_v = df_deploy[df_deploy["version"].astype(str).str.extract(r"(\d+)")[0].astype(float).astype(int) == int(version)]

if df_v.empty:
   raise ValueError(f"Could not find metrics row for version={version} in {DEPLOY_METRICS_CSV_PATH}")

# Take the latest row for that version (safe if duplicates)
row = df_v.tail(1).iloc[0].to_dict()
# trained_completed_ts from CSV datetime
trained_completed_ts = _display_value(row.get("datetime"), None)
# trained_by: if you can derive user/email put it here, else keep blank
trained_by = ""  # optional
# Store full row JSON so UI can compare/plot any metric later
metrics_json = json.dumps(row, default=str)
# Upsert registry

upsert_registry_from_metrics_row(
   version=str(version).replace("v", "").strip(),
   trained_started_ts=None,  # optional if you capture it earlier
   trained_completed_ts=trained_completed_ts,
   trained_by=trained_by,
   overall_accuracy=float(row.get("overall_accuracy")) if row.get("overall_accuracy") is not None else None,
   metrics_json=metrics_json,
   metrics_csv_path=DEPLOY_METRICS_CSV_PATH,
)

print(f"[MODEL REGISTRY] Upserted mdm_model_registry_tbl for version={version}")