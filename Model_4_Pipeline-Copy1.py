
print("Kernel Working")
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report, accuracy_score
import dataiku
import warnings
import tempfile
import os
import shutil
import joblib
import io
import json

import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import EarlyStopping


from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
# from sentence_transformers import SentenceTransformer
# Set random seeds for reproducibility
np.random.seed(42)
import time
from datetime import datetime
from zoneinfo import ZoneInfo
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader

tz_ist = ZoneInfo('Asia/Kolkata')

print("✓ All libraries imported successfully") 

Starting_Timestamp = datetime.now(tz=tz_ist)
print(Starting_Timestamp)

timestamp = datetime.now(tz=tz_ist).strftime("%Y_%m_%d_%H_%M")

models_folder = dataiku.Folder("model2_output")
model_name ="XGBoost_HPO"

folder_name_path=f"{timestamp}"

user_path = f"{folder_name_path}/"
print(f"Folder path: {folder_name_path}")

preprocessed_data_folder = dataiku.Folder("preprocessed_data")

with preprocessed_data_folder.get_download_stream("label_encoder_y_variable.pkl") as stream:
    data = stream.read()
    label_encoder = joblib.load(io.BytesIO(data))

def get_parquet(filename):

    buffer = io.BytesIO()
    with preprocessed_data_folder.get_download_stream(f"{filename}.parquet") as stream:
        buffer.write(stream.read())

    buffer.seek(0)
    return pd.read_parquet(buffer)

X_train = get_parquet("X_train")
X_test = get_parquet("X_test")
y_train = get_parquet("y_train")["target"].to_numpy()
y_test = get_parquet("y_test")["target"].to_numpy()


X_train =X_train.astype("float32")


counts = np.bincount(y_train)
weights = np.log1p(np.max(counts) / counts)

weights = np.minimum(weights, 10.0)

class_weights = {
    class_idx: weight
    for class_idx, weight in enumerate(weights)
    if counts[class_idx] > 0   # keep only existing classes
}

sample_weights = np.array([class_weights[y] for y in y_train])



tz_ist = ZoneInfo('Asia/Kolkata')
start_hpo = datetime.now(tz=tz_ist)
start_hpo
print("Start Time: ",start_hpo)



X_train_fit, X_val, y_train_fit, y_val, sw_train_fit, sw_val = train_test_split(
    X_train, y_train, sample_weights, test_size=0.1, random_state=42, stratify=y_train
)
eval_set = [(X_train_fit, y_train_fit), (X_val, y_val)]

# num_classes = len(label_encoder.classes_)
num_classes = len(np.unique(y_train))


# pip install cuml


from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.datasets.classification import make_classification

from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.datasets.classification import make_classification

# Generate sample data
# X, y = make_classification(n_samples=1000, n_features=10, n_classes=2)

# Train on GPU
model = cuRF(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# preds = model.predict(X)
# print(preds[:10])


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd



model = CatBoostClassifier(
    iterations=425,
    learning_rate=0.05,
    depth=6,
    loss_function="MultiClass",
    verbose=1,
    task_type="GPU", 
    devices="0"
)

# Fit the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print("Accuracy:", acc)
print("Macro F1:", f1)



get_ipython().system('nvidia-smi')


#CatBoost HPO with results for each iteration

import io
import optuna
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold



cat_features = None 


X_train_np = X_train if isinstance(X_train, (np.ndarray,)) else X_train.copy()
y_train_np = np.array(y_train)
sample_weights_np = np.array(sample_weights) if 'sample_weights' in globals() and sample_weights is not None else None

n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

results_df = pd.DataFrame()

def objective(trial):
    # --- Hyperparameter search space (CatBoost) ---
    params = {
        "iterations": trial.suggest_int("iterations", 200, 800),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "bootstrap_type": "Bernoulli",
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1:average=Macro",
        "od_type": "Iter",
        "od_wait": trial.suggest_int("od_wait", 30, 100),
        "task_type": "GPU",  
        "devices": "0",   
        "thread_count": 1,
#         "thread_count": -1,
        "allow_writing_files": False,
        "random_seed": 42,
        "verbose": 1,
    }

    fold_scores = []

    train_accs, val_accs = [], []
    train_f1s,  val_f1s  = [], []
    train_recs, val_recs = [], []

    print(f"\n============  Trial: {trial.number} ============")
    print("Trial iterations:", params["iterations"])
    start_trial = datetime.now(tz=tz_ist)
    print(start_trial)

    for train_idx, val_idx in skf.split(X_train_np, y_train_np):
        # Slice data - supports both numpy arrays and pandas
        if hasattr(X_train_np, "iloc"):
            X_tr = X_train_np.iloc[train_idx]
            X_val_fold = X_train_np.iloc[val_idx]
        else:
            X_tr = X_train_np[train_idx]
            X_val_fold = X_train_np[val_idx]

        y_tr = y_train_np[train_idx]
        y_val_fold = y_train_np[val_idx]

        sw_tr = sample_weights_np[train_idx] if sample_weights_np is not None else None

        print("Trial iterations:", params["iterations"])

        model = CatBoostClassifier(**params)

        model.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=(X_val_fold, y_val_fold),
            verbose=False,
            use_best_model=True,    # uses best iteration on eval_set
            # If you have categorical features:
            cat_features=cat_features
        )

        # Predictions (classes)
        preds = model.predict(X_val_fold, prediction_type="Class")
        preds = np.array(preds).reshape(-1)

        print("y_val_fold shape:", y_val_fold.shape)
        print("preds shape:", preds.shape)

        # Macro F1 for this fold (what Optuna maximizes)
        fold_scores.append(f1_score(y_val_fold, preds, average="macro"))

        # Train predictions for logging
        train_preds = model.predict(X_tr, prediction_type="Class")
        train_preds = np.array(train_preds).reshape(-1)

        train_accs.append(accuracy_score(y_tr, train_preds))
        train_f1s.append(f1_score(y_tr, train_preds, average="macro"))
        train_recs.append(recall_score(y_tr, train_preds, average="macro"))

        val_accs.append(accuracy_score(y_val_fold, preds))
        val_f1s.append(f1_score(y_val_fold, preds, average="macro"))
        val_recs.append(recall_score(y_val_fold, preds, average="macro"))

    # Aggregate metrics across folds
    train_acc_mean = float(np.mean(train_accs))
    val_acc_mean   = float(np.mean(val_accs))
    train_f1_mean  = float(np.mean(train_f1s))
    val_f1_mean    = float(np.mean(val_f1s))
    train_rec_mean = float(np.mean(train_recs))
    val_rec_mean   = float(np.mean(val_recs))

    end_trial = datetime.now(tz=tz_ist)
    duration = (end_trial - start_trial).total_seconds() / 60
    print(end_trial)
    print(duration)

    # Record a row for this trial
    row = {
        "trial_number": trial.number,
        "iterations": params["iterations"],
        "depth": params["depth"],
        "learning_rate": params["learning_rate"],
        "subsample": params["subsample"],
#         "rsm": params["rsm"],
        "l2_leaf_reg": params["l2_leaf_reg"],
        "random_strength": params["random_strength"],
        "min_data_in_leaf": params["min_data_in_leaf"],
        "od_wait": params["od_wait"],
        "train_accuracy": train_acc_mean,
        "val_accuracy": val_acc_mean,
        "train_f1_macro": train_f1_mean,
        "val_f1_macro": val_f1_mean,
        "train_recall_macro": train_rec_mean,
        "val_recall_macro": val_rec_mean,
        "trial_start_time": start_trial,
        "trial_end_time": end_trial,
        "trial_Duration_min": duration
    }

    trial_id_no = int(trial.number)

    global results_df
    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

    # Persist rolling CSV checkpoint (same logic as your original)
    buffer = io.BytesIO()
    results_df.to_csv(buffer, index=False)
    buffer.seek(0)
    models_folder.upload_stream(f"{user_path}hpo_{trial_id_no:04d}_results.csv", buffer)

    # Delete older checkpoint (keep last 3)
    no_not_req = trial_id_no - 3
    if trial_id_no == 3 or no_not_req > 0:
        old_file = f"/{user_path}hpo_{no_not_req:04d}_results.csv"
        existing_files = models_folder.list_paths_in_partition()
        if old_file in existing_files:
            models_folder.delete_path(old_file)
            print(f"  Deleted old checkpoint: {no_not_req}")
        else:
            print("  file not found")

    # Return mean macro-F1 across folds
    return float(np.mean(fold_scores))


# === Run the Optuna study ===
n_trials = 80  # adjust as needed
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=n_trials, n_jobs=1, timeout=60*60*72)

print("Best trial:")
print(study.best_trial.params)
print(f'Best value (macro_f1): {study.best_value:.4f}')

# Get best parameters from Optuna
best_params = study.best_trial.params

# Add required fixed parameters for LightGBM
best_params.update({
    "task_type": "GPU",  
    "devices": "0",   
    "thread_count": 1,
    "allow_writing_files": False,
    "random_seed": 42,
    "verbose": 1
})

# Initialize model
model = CatBoostClassifier(**best_params)

# Train on full training set

model.fit(
    X_train,
    y_train,
    cat_features=cat_features
    verbose=False
)


print("Best Catboost model trained successfully")


print("Completed Model Building") 


end_hpo = datetime.now(tz=tz_ist)
print("End Time: ",end_hpo)
duration = (end_hpo - start_hpo).total_seconds()/60
print("Tot Duration: ",duration)



# Saving Model, y variable Label Encoder and Parent Information Label Encoder

def save_object(obj, path):
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    buffer.seek(0)
    models_folder.upload_stream(path, buffer)

# Save model
save_object(model, f"{user_path}model.pkl")



print("Saved:")
print(f"  {user_path}model.pkl")
# print(f"  {user_path}label_encoder_y_variable.pkl")
# print(f"  {user_path}label_encoder_parent_info.pkl")
# print(f"  {user_path}pca_{n_components}.pkl")



# Save feature names in X_train
feature_names = X_train.columns.tolist()
buffer = io.BytesIO(json.dumps(feature_names).encode("utf-8"))
models_folder.upload_stream(f"{user_path}features.json", buffer)
print("Saved:")
print(f"  {user_path}features.json")


#Predict top3 and 5 accuracy 

# Predict class labels
y_pred = model.predict(X_test)

# Predict class probabilities
y_proba = model.predict_proba(X_test)

# ---- F1 Scores ----
macro_f1 = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")

print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

# ---- Top-K Accuracy ----
def top_k_accuracy(y_true, y_proba, k):
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    return np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])

top_3_acc = top_k_accuracy(y_test, y_proba, k=3)
top_5_acc = top_k_accuracy(y_test, y_proba, k=5)

print(f"Top-3 Accuracy: {top_3_acc:.4f}")
print(f"Top-5 Accuracy: {top_5_acc:.4f}")


# Get train and test accuracy
def evaluate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_recall": recall_score(y_true, y_pred, average='macro'),
        "weighted_f1" : f1_score(y_true, y_pred, average='weighted'),
        "weighted_recall": recall_score(y_true, y_pred, average='weighted')        
    }

train_preds = model.predict(X_train)
trn_metrics = evaluate_metrics(y_train, train_preds)
print('Training metrics', trn_metrics)

test_preds = model.predict(X_test)
test_metrics = evaluate_metrics(y_test, test_preds)
print('Test metrics', test_metrics)


# classification report
class_names = label_encoder.classes_
report_df = pd.DataFrame(
    classification_report(
        y_test,
        test_preds,
        target_names=label_encoder.classes_,
        output_dict=True
    )
).transpose()
report_df = report_df.reset_index().rename(columns={"index": "class_name"})
# full data value counts
# full_counts = df["unspsc_code"].value_counts()
# full_counts_df = full_counts.rename_axis("class_name").reset_index(name="full_dataset_count")

# report_df = report_df.merge(
#     full_counts_df,
#     how="left",
#     on="class_name"
# )
# save to dataiku
buffer = io.BytesIO()
report_df.to_csv(buffer, index=False)
buffer.seek(0)

models_folder.upload_stream(f"{user_path}classification_report.csv",buffer)

print("Classification Report Saved successfully.")



end_evaluation = datetime.now(tz=tz_ist)
print("End Time: ",end_evaluation)
duration_last = (end_evaluation - start_hpo).total_seconds()/60
print("Total Duration: ",duration_last)


Ending_Timestamp = datetime.now(tz=tz_ist)
print(Ending_Timestamp)
Total_scenario_duration = (Ending_Timestamp - Starting_Timestamp).total_seconds()/60
print("Total_scenario_duration: ",Total_scenario_duration)



metadata = {

    "model_name": model_name,
    "number_of_classes": num_classes,
#     "number_of_omitted_classes": len(omitted_unspc_code),
#     "class_filtering_with_valuecounts_gte": omitted_number,
#     "target_col": TARGET_COL,
    "training_samples": len(X_train),
    "number_of_features": int(X_train.shape[1]),
#     "number_of_onehot_encoded_columns" : len(ohe.get_feature_names_out()),
#     "pca": n_components,
    
    "folder_timestamp": str(timestamp),
    "starting_time": str(Starting_Timestamp),
    "ending_time": str(Ending_Timestamp),
    "total_execution_duration": Total_scenario_duration,
    "model_starting_time": str(start_hpo),
    "model_ending_time": str(end_hpo),
    "model_train_duration_in_minutes": duration,

    "training_metrics": trn_metrics,
    "test_metrics": test_metrics,
    "top_3_accuracy": round(top_3_acc, 4),
    "top_5_accuracy": round(top_5_acc, 4),
    "weighted_f1": round(weighted_f1, 4),
    
#     "cat_cols": CATEGORICAL_COLS
}
buffer = io.BytesIO(json.dumps(metadata).encode("utf-8"))
models_folder.upload_stream(f"{user_path}metadata.json", buffer)

print("Saved:")
print(f"  {user_path}metadata.json")



print(f"Run Successfully Completed at {datetime.now(tz=tz_ist)}")





