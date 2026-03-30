


print("Kernel Working")
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import dataiku
import warnings
import tempfile
import os
import io
import json

from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

np.random.seed(42)
import time
from datetime import datetime
from zoneinfo import ZoneInfo

tz_ist = ZoneInfo('Asia/Kolkata')


def get_parquet(filename):
    preprocessed_data_folder = dataiku.Folder("preprocessed_data")

    buffer = io.BytesIO()
    with preprocessed_data_folder.get_download_stream(f"{filename}.parquet") as stream:
        buffer.write(stream.read())

    buffer.seek(0)
    return pd.read_parquet(buffer)

X_train = get_parquet("X_train")
y_train = get_parquet("y_train")["target"].to_numpy()



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


num_classes = len(np.unique(y_train))


# catboost

model = CatBoostClassifier(
    iterations=425,
    learning_rate=0.08960785365368121,
    depth=10,
    loss_function="MultiClass",
    verbose=1,
    task_type="GPU", 
    devices="0"
)

model.fit(X_train, y_train)





