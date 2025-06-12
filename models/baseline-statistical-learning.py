import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor,
    RANSACRegressor, TheilSenRegressor, BayesianRidge, SGDRegressor
)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

from utils import create_data_loaders, train_model, test_model

RANDOM_STATE = 42
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
PATIENCE = 10
MIN_DELTA = 1e-4

root_path = os.path.abspath(os.path.dirname(os.getcwd()))
processed_path = os.path.join(root_path, "data", "processed")
os.makedirs(processed_path, exist_ok=True)

run_matrices = load(os.path.join(processed_path, 'run_matrices.joblib'))
incoming_run_matrices = load(os.path.join(processed_path, 'incoming_run_matrices.joblib'))
metrology_matrix = load(os.path.join(processed_path, 'metrology_matrix.joblib'))

print(run_matrices.shape, incoming_run_matrices.shape, metrology_matrix.shape)
print(type(run_matrices), type(incoming_run_matrices), type(metrology_matrix))

X = np.concatenate([run_matrices, incoming_run_matrices], axis=2)
y = metrology_matrix

# Convert to torch tensors
X = torch.FloatTensor(X)
y = torch.FloatTensor(y)

# Define dataset sizes
total_size = len(X)
train_size = int(0.7 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

# Create TensorDataset to keep X and y aligned
dataset = torch.utils.data.TensorDataset(X, y)

# Split dataset
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(RANDOM_STATE)
)

# Extract X and y from datasets
X_train, y_train = next(iter(torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))))
X_val, y_val = next(iter(torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))))
X_test, y_test = next(iter(torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))))

X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_val_flattened = X_val.reshape(X_val.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)
print(X_train_flattened.shape)

models = {
    "LinearRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    # "SGDRegressor": Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("model", MultiOutputRegressor(SGDRegressor(max_iter=1000, tol=1e-3)))
    # ]),
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),  # RF doesn't need scaling but keeping for consistency
        ("model", MultiOutputRegressor(RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),  # SVM definitely needs scaling
        ("model", MultiOutputRegressor(SVR(
            kernel='rbf',
            C=1.0,
            gamma='scale'
        )))
    ])
}

for model_name, pipeline in models.items():
    pipeline.fit(X_train_flattened, y_train)
    # scaled_sample = pipeline.named_steps['scaler'].transform(X_test_flattened)[0]
    # print("First scaled sample:", scaled_sample)
    y_test_pred = pipeline.predict(X_test_flattened)
    print(f"{model_name}:")
    print(f"mse: {mean_squared_error(y_test_pred, y_test)}")
    print(f"r2_score: {r2_score(y_test_pred, y_test)}")

