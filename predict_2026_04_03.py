import io
import json
import os
import re
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
import dataiku
import faiss
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from dataiku import pandasutils as pdu
from openai import AzureOpenAI
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
import threading

from services.prediction_resources import (
   ARTIFACT_NAMES,
   DATA_FOLDER,
   EMBED_MODEL_NAME,
   FAISS_FOLDER,
   JUDGE_MODEL_NAME,
   MODEL_1_PATH,
   MODEL_2_PATH,
   MODEL_3_PATH,
   MODEL_FOLDER,
   PredictionResources,
   load_prediction_resources,
)

tz_ist = ZoneInfo("Asia/Kolkata")
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# All managed-folder ids, artifact paths, clients, and loaded models
# are owned by prediction_resources.py. This file only consumes them.
# ------------------------------------------------------------------

_RESOURCES: PredictionResources = load_prediction_resources()
data_folder = _RESOURCES.data_folder
model_folder = _RESOURCES.model_folder
folder = _RESOURCES.faiss_folder
client_aoai = _RESOURCES.client_aoai
client_gpt = _RESOURCES.client_gpt
model1_pipeline = _RESOURCES.model1_pipeline
model2_pipeline = _RESOURCES.model2_pipeline
model3_pipeline = _RESOURCES.model3_pipeline
pca_model = _RESOURCES.pca_model
label_encoder_y = _RESOURCES.label_encoder_y
metadata = _RESOURCES.metadata
feature_trained_on = _RESOURCES.feature_trained_on
EMBEDDING_MODEL_NAME = _RESOURCES.embed_model_name
GPT_JUDGE_MODEL_NAME = _RESOURCES.judge_model_name



# Legacy aliases retained so the rest of the pipeline can stay unchanged.
DATA_FOLDER_ID = DATA_FOLDER
MODEL_FOLDER_ID = MODEL_FOLDER
FAISS_FOLDER_ID = FAISS_FOLDER

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

MAX_CHUNK_TOKENS = 7000
MAX_TPM = 320_000
EMBED_BATCH_SIZE = 32
MAX_RETRIES = 3
SAVE_EVERY = 5000


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
   df = df.apply(
       lambda col: col.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
       if col.dtype == "object" else col
   )
   df = df[~df['short_description_(english)'].isna()]
   df = df[['unspsc_code','short_description_(english)','long_description_(english)',
            'part_manufacturer_list_information','hazardous_goods_(s.f.)',
            'is_related_to_fire&smoke?','part_rohs','perishable_part','repairable_?',
            'unit_of_measure_information','usability',]]
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
   df = df.reset_index(drop=True)
   return df

# ==================== Embedding & PCA ====================

# Function to get the latest embedding file number for loading/saving
def get_highest_embedding_number(data_folder):
   folder_path = "2026_02_11_Embedding"
   pattern = re.compile(
       r"combined_data_embedding_new(\d+)\.parquet"
   )
   existing_files = data_folder.list_paths_in_partition()
   max_number = 0
   for file_path in existing_files:
       # Only check files inside your folder
       if folder_path in file_path:
           match = pattern.search(file_path)  # â† use search instead of match
           if match:
               num = int(match.group(1))
               max_number = max(max_number, num)
   return max_number
# Function to get saved embedding
def get_existing_embedding(folder_id="Data_Folder"):
   folder = dataiku.Folder(folder_id)
   emb_no = get_highest_embedding_number(folder)
   filename=f"2026_02_11_Embedding/combined_data_embedding_new{emb_no:04d}.parquet"
   buffer = io.BytesIO()
   with folder.get_download_stream(filename) as stream:
       buffer.write(stream.read())
   buffer.seek(0)
   return pd.read_parquet(buffer)

# -----------------------------
# Token Counter
# -----------------------------
def count_tokens(text):
   return len(text) // 4


# -----------------------------
# Safe embedding with TPM control
# -----------------------------
def embed_batch_with_retry(texts):
   for attempt in range(MAX_RETRIES):
       try:
           response = client_aoai.embeddings.create(
               model=EMBEDDING_MODEL_NAME,
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

# -----------------------------
# Main function
# -----------------------------
def get_embeddings(df, TEXT_COL):
   existing_df = pd.DataFrame(columns=[TEXT_COL, "embedding"])
   missing_texts = set(df[TEXT_COL].astype(str))
   # ---- TPM control ----
   token_budget = 0
   start_time = time.time()
   new_rows = []
   processed_count = 0
   total_processed = 0
   emb_no = get_highest_embedding_number(data_folder)

   batch = []
   batch_tokens = 0
   for text in missing_texts:
       text = str(text)
       tokens = count_tokens(text)
       # ---- Skip oversized chunks ----
       if tokens > MAX_CHUNK_TOKENS:
           print(f"Skipping text >8191 tokens ({tokens})")
           continue
       # ---- TPM throttling ----
       if token_budget + tokens > MAX_TPM:
           elapsed = time.time() - start_time
           sleep_time = max(0, 60 - elapsed)
           if sleep_time > 0:
               print(f"TPM limit reached. Sleeping {sleep_time:.2f}s")
               time.sleep(sleep_time)
           token_budget = 0
           start_time = time.time()
       batch.append(text)
       batch_tokens += tokens
       token_budget += tokens
       # ---- Send batch ----
       if len(batch) >= EMBED_BATCH_SIZE:
           embeddings = embed_batch_with_retry(batch)
           batch_df = pd.DataFrame({
               TEXT_COL: batch,
               "embedding": embeddings
           })
           new_rows.append(batch_df)
           processed_count += len(batch)
           total_processed += len(batch)

           batch = []
           batch_tokens = 0
           remaining_count = len(missing_texts) - total_processed
       # ---- Checkpoint save ----
       if processed_count >= SAVE_EVERY:
           remaining_count = len(missing_texts) - total_processed
           existing_df = pd.concat([existing_df] + new_rows, ignore_index=True)
           new_rows = []
           processed_count = 0
   # ---- Process final batch ----
   if batch:
       embeddings = embed_batch_with_retry(batch)
       batch_df = pd.DataFrame({
           TEXT_COL: batch,
           "embedding": embeddings
       })
       new_rows.append(batch_df)
   # ---- Final save ----
   if new_rows:
       existing_df = pd.concat([existing_df] + new_rows, ignore_index=True)

       return existing_df
   return existing_df



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
   embedding_df = get_embeddings(df_cleaned, TEXT_COL)
   df_merged = df_cleaned.merge(embedding_df, on="combined_description", how="left").drop(columns=[TEXT_COL])
   df_final = df_merged.copy()
   X_test_df = df_final.drop(columns=[TARGET_COL])
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
#     y_proba = model1_pipeline.predict_proba(X_test_aligned)
   model2_pred = model2_pipeline.predict(X_test_aligned)
   model3_pred = predict_via_neural_network(model3_pipeline,X_test_aligned)
   model1_pred_labelled = label_encoder_y.inverse_transform(model1_pred)
   model2_pred_labelled = label_encoder_y.inverse_transform(model2_pred)
   model3_pred_labelled = label_encoder_y.inverse_transform(model3_pred)

   return model1_pred_labelled, model2_pred_labelled, model3_pred_labelled

def get_data(folder_name: str = "Data", filename: str = "consolidated_data_v2.csv") -> pd.DataFrame:
   folder = dataiku.Folder(folder_name)
   with folder.get_download_stream(filename) as stream:
       df = pd.read_csv(stream)
       df.columns = [col.lower().replace(" ", "_") for col in df.columns]
   return df

def merge_family_data(incoming_df):
   # Load and clean family dataset
   df_family = get_data(filename="Family-Classification_Mapping-UNSPSC-branch 1 (1).csv")
   df_family = df_family.dropna(subset=["unspsc_-_unspsc"])
   df_family["unspsc_-_unspsc"] = df_family["unspsc_-_unspsc"].astype(int)
   # Keep only Leaf Family rows
   df_leaf = df_family[df_family["type"] == "Leaf Family"]
   # Aggregate everything (including governance)
   df_family_grouped = (
       df_leaf
       .groupby("unspsc_-_unspsc")
       .agg({
           "governance": lambda x: "/".join(pd.unique(x.dropna().astype(str))),
           "parent_information": lambda x: "/".join(pd.unique(x.dropna().astype(str))),
           "branch": lambda x: "/".join(pd.unique(x.dropna().astype(str)))
           })
       .reset_index()
   )
   # Merge once
   df = (
       incoming_df
       .merge(df_family_grouped,
              left_on="unspsc_code",
              right_on="unspsc_-_unspsc",
              how="left")
       .drop(columns=["unspsc_-_unspsc"])
   )
   return df

# def generate_ml_output(incoming_df):
#     df = merge_family_data(incoming_df)
#     model1_pred_labelled, model2_pred_labelled, model3_pred_labelled = predict(df)
#     # Add to final df
#     df["Model1"] = model1_pred_labelled
#     df["Model2"] = model2_pred_labelled
#     df["Model3"] = model3_pred_labelled
#     return df

def generate_ml_output(incoming_df):
   df=incoming_df.copy()    
   model1_pred_labelled, model2_pred_labelled, model3_pred_labelled = predict(df)
   # Add to final df
   df["Model1"] = model1_pred_labelled
   df["Model2"] = model2_pred_labelled
   df["Model3"] = model3_pred_labelled
   return df

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


# # new one
# def gov_index_vector():
#     folder = dataiku.Folder("FAISS_KB")
#     filename = "governance_faiss_new.index"

#     # --- Read index bytes from managed folder ---
#     with folder.get_download_stream(filename) as f:
#         index_bytes = f.read()

#     # Convert bytes -> numpy uint8 -> deserialize
#     index_arr = np.frombuffer(index_bytes, dtype='uint8')
#     gov_index = faiss.deserialize_index(index_arr)

#     # --- Serialize and write back to the managed folder ---
#     buf = faiss.serialize_index(gov_index)

#     # Ensure we upload bytes (handle different FAISS return types)
#     try:
#         data = bytes(buf)  # works if buf is a Python buffer-like
#     except TypeError:
#         data = np.array(buf, dtype='uint8').tobytes()  # works if buf is a numpy array-like

#     with folder.get_writer(filename) as w:
#         w.write(data)

#     return gov_index


_GOV_INDEX_CACHE = None
_GOV_METADATA_CACHE = None
_GOV_CACHE_LOCK = threading.Lock()

def gov_index_vector(force_reload: bool = False):
   global _GOV_INDEX_CACHE
   if _GOV_INDEX_CACHE is not None and not force_reload:
       return _GOV_INDEX_CACHE
   with _GOV_CACHE_LOCK:
       if _GOV_INDEX_CACHE is not None and not force_reload:
           return _GOV_INDEX_CACHE
       folder = dataiku.Folder("FAISS_KB")
       filename = "governance_faiss_new.index"
       # Read only
       with folder.get_download_stream(filename) as f:
           index_bytes = f.read()
       index_arr = np.frombuffer(index_bytes, dtype="uint8")
       gov_index = faiss.deserialize_index(index_arr)
       _GOV_INDEX_CACHE = gov_index
       return _GOV_INDEX_CACHE


# new one
# def gov_metadata_vector():    
#     # Read JSON
#     with folder.get_download_stream("governance_metadata_new.json") as f:
#         gov_metadata = json.load(f)

#     # Write JSON
#     with folder.get_writer("governance_metadata_new.json") as w:
#         w.write(json.dumps(gov_metadata).encode("utf-8"))

#     return gov_metadata

def gov_metadata_vector(force_reload: bool = False):
   global _GOV_METADATA_CACHE
   if _GOV_METADATA_CACHE is not None and not force_reload:
       return _GOV_METADATA_CACHE
   with _GOV_CACHE_LOCK:
       if _GOV_METADATA_CACHE is not None and not force_reload:
           return _GOV_METADATA_CACHE
       folder = dataiku.Folder("FAISS_KB")
       # Read only
       with folder.get_download_stream("governance_metadata_new.json") as f:
           gov_metadata = json.load(f)
       _GOV_METADATA_CACHE = gov_metadata
       return _GOV_METADATA_CACHE
    

# def retrieve_governance_context(query_text, k=5):
#     query_emb = client_aoai.embeddings.create(
#         model="text-embedding-3-large",
#         input=[query_text.lower()]
#     )

#     query_vec = np.array(
#         query_emb.data[0].embedding,
#         dtype="float32"
#     ).reshape(1, -1)

#     gov_index = gov_index_vector()
#     gov_metadata = gov_metadata_vector()
    
#     D, I = gov_index.search(query_vec, k)

#     retrieved = []
#     for idx, dist in zip(I[0], D[0]):
#         item = gov_metadata[idx].copy()
#         item["distance"] = float(dist)
#         retrieved.append(item)

#     return retrieved


def retrieve_governance_context(query_text, k=5):
   query_emb = client_aoai.embeddings.create(
       model="text-embedding-3-large",
       input=[str(query_text).lower()]
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
       if idx < 0 or idx >= len(gov_metadata):
           continue
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
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a UNSPSC part classification subject matter expert."},
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
        source = "LLM as a Judge"
        
    return predicted_unspsc, explanation, source

# def run_unspsc_predictions(incoming_df):
#    df =generate_ml_output(incoming_df)
#    results = []
#    for _, row in df.iterrows():
#        part_query = row["short_description_(english)"]
#        m1 = row.get("Model1")
#        m2 = row.get("Model2")
#        m3 = row.get("Model3")
#        pred_code, exp, src = predict_unspsc_with_rag(
#            part_query,
#            model1=m1,
#            model2=m2,
#            model3=m3
#        )
#        results.append({
#            "Short Description (English)": part_query,
#            "Model 1 OP": m1,
#            "Model 2 OP": m2,
#            "Model 3 OP": m3,
#            "LLM as a Judge OP": pred_code,
#            "LLM Decision Source": src,
#           "Explanation": exp
#       })
#   df_pred = pd.DataFrame(results)
#   df["LLM as a Judge OP"]=df_pred["LLM as a Judge OP"]
#   df["LLM Decision Source"]=df_pred["LLM Decision Source"]
#   df["Explanation"]=df_pred["Explanation"]
#   df.rename(columns={"Model1":"Model 1 OP","Model2":"Model 2 OP","Model3":"Model 3 OP"},inplace=True)
#   print(df)
#   return df