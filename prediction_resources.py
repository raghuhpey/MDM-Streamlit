from __future__ import annotations
import importlib
import importlib.util
import io
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import dataiku
import faiss
import joblib
import numpy as np
from openai import AzureOpenAI
from tensorflow import keras

DATA_FOLDER = "Data_Folder"
MODEL_FOLDER = "Pipeline_Models"
FAISS_FOLDER = "FAISS_KB"
DATA_FOLDER_ID = DATA_FOLDER
MODEL_FOLDER_ID = MODEL_FOLDER
FAISS_FOLDER_ID = FAISS_FOLDER
MODEL_1_PATH = "/Hari/2026_04_03_model_xgboost_new_wo_clamp/"
MODEL_2_PATH = "/Hari/2026_04_03_Model_Catboost_new_wo_clamp/"
MODEL_3_PATH = "/Hari/2026_04_03_model_mlp_new_wo_clamp/"
ARTIFACT_NAMES = {
   "model1_pipeline": f"{MODEL_1_PATH}model.pkl",
   "model2_pipeline": f"{MODEL_2_PATH}model.pkl",
   "model3_pipeline": f"{MODEL_3_PATH}model.keras",
   "pca_model": f"{MODEL_3_PATH}pca.pkl",
   "label_encoder_y": f"{MODEL_3_PATH}label_encoder_y_variable.pkl",
   "metadata": f"{MODEL_3_PATH}metadata.json",
   "features": f"{MODEL_3_PATH}features.json",
}
EMBED_MODEL_NAME = "text-embedding-3-large"
JUDGE_MODEL_NAME = "gpt-4o"



def _get_env(name: str, default: str | None = None) -> str:
   value = os.getenv(name, default)
   if value is None:
       raise ValueError(f"Missing required environment variable: {name}")
   return value

def build_aoai_client() -> AzureOpenAI:
   return AzureOpenAI(
       api_key=_get_env("AOAI_API_KEY", "sk-WSYwqK94KaLosJStD88Mdg"),
       azure_endpoint=_get_env("AOAI_AZURE_ENDPOINT", "https://dev-emea-cloudapi.alstom.com/10222/N4/V1/"),
       api_version=_get_env("AOAI_API_VERSION", "2024-02-01"),
   )

def build_gpt_client() -> AzureOpenAI:
   return AzureOpenAI(
       api_key=_get_env("GPT_API_KEY", os.getenv("AOAI_API_KEY", "sk-l9moESKkdVd1g2IAHsbAGg")),
       azure_endpoint=_get_env("GPT_AZURE_ENDPOINT", os.getenv("AOAI_AZURE_ENDPOINT", "https://dev-emea-cloudapi.alstom.com/10222/N4/V1/")),
       api_version=_get_env("GPT_API_VERSION", "2024-02-01"),
   )

@dataclass
class PredictionResources:
   data_folder: Any
   model_folder: Any
   faiss_folder: Any
   model1_pipeline: Any
   model2_pipeline: Any
   model3_pipeline: Any
   pca_model: Any
   label_encoder_y: Any
   metadata: dict
   feature_trained_on: list[str]
   client_aoai: AzureOpenAI
   client_gpt: AzureOpenAI
   governance_index: Any
   governance_metadata: list[dict]
   prediction_module: Any | None = None
   embed_model_name: str = EMBED_MODEL_NAME
   judge_model_name: str = JUDGE_MODEL_NAME
   artifact_names: dict[str, str] | None = None

def load_artifact(models_folder, path: str):
   _, ext = os.path.splitext(path.lower())
   if ext in [".pkl", ".joblib", ".json"]:
       with models_folder.get_download_stream(path) as stream:
           data = stream.read()
       if ext in [".pkl", ".joblib"]:
           return joblib.load(io.BytesIO(data))
       return json.loads(data.decode("utf-8"))
   if ext == ".keras":
       with tempfile.TemporaryDirectory() as tmp_dir:
           local_path = os.path.join(tmp_dir, os.path.basename(path) or "model.keras")
           with models_folder.get_download_stream(path) as stream, open(local_path, "wb") as f:
               f.write(stream.read())
           return keras.models.load_model(local_path)
   raise ValueError(f"Unsupported file type: {ext}")

def load_governance_index(resources: PredictionResources):
   with resources.faiss_folder.get_download_stream("governance_faiss.index") as stream:
       index_bytes = stream.read()
   index_arr = np.frombuffer(index_bytes, dtype=np.uint8).copy()
   return faiss.deserialize_index(index_arr)

def load_governance_metadata(resources: PredictionResources) -> list[dict]:
   with resources.faiss_folder.get_download_stream("governance_metadata.json") as stream:
       raw = stream.read()
   return json.loads(raw.decode("utf-8"))

def _import_prediction_module():
   module_candidates = ["services.predict_2026_04_03", "predict_2026_04_03"]
   for module_name in module_candidates:
       try:
           return importlib.import_module(module_name)
       except Exception:
           pass
   # fall back to direct file import from same folder
   this_dir = Path(__file__).resolve().parent
   module_path = this_dir / "predict_2026_04_03.py"
   if not module_path.exists():
       raise ImportError(
           f"Unable to import prediction module. Expected file at: {module_path}"
       )
   spec = importlib.util.spec_from_file_location("predict_2026_04_03", str(module_path))
   if spec is None or spec.loader is None:
       raise ImportError(f"Unable to build import spec for: {module_path}")
   module = importlib.util.module_from_spec(spec)
   sys.modules.setdefault("predict_2026_04_03", module)
   spec.loader.exec_module(module)
   return module

@lru_cache(maxsize=1)
def load_prediction_resources(load_prediction_module: bool = True) -> PredictionResources:
   data_folder = dataiku.Folder(DATA_FOLDER)
   model_folder = dataiku.Folder(MODEL_FOLDER)
   faiss_folder = dataiku.Folder(FAISS_FOLDER)
   model1_pipeline = load_artifact(model_folder, ARTIFACT_NAMES["model1_pipeline"])
   model2_pipeline = load_artifact(model_folder, ARTIFACT_NAMES["model2_pipeline"])
   model3_pipeline = load_artifact(model_folder, ARTIFACT_NAMES["model3_pipeline"])
   pca_model = load_artifact(model_folder, ARTIFACT_NAMES["pca_model"])
   label_encoder_y = load_artifact(model_folder, ARTIFACT_NAMES["label_encoder_y"])
   metadata = load_artifact(model_folder, ARTIFACT_NAMES["metadata"])
   feature_trained_on = load_artifact(model_folder, ARTIFACT_NAMES["features"])
   resources = PredictionResources(
       data_folder=data_folder,
       model_folder=model_folder,
       faiss_folder=faiss_folder,
       model1_pipeline=model1_pipeline,
       model2_pipeline=model2_pipeline,
       model3_pipeline=model3_pipeline,
       pca_model=pca_model,
       label_encoder_y=label_encoder_y,
       metadata=metadata,
       feature_trained_on=feature_trained_on,
       client_aoai=build_aoai_client(),
       client_gpt=build_gpt_client(),
       governance_index=None,
       governance_metadata=[],
       prediction_module=None,
       artifact_names=ARTIFACT_NAMES.copy(),
   )
   resources.governance_index = load_governance_index(resources)
   resources.governance_metadata = load_governance_metadata(resources)
   if load_prediction_module:
       resources.prediction_module = _import_prediction_module()
   return resources