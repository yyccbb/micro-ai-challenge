import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from joblib import load
from tqdm import tqdm

from utils import train_and_predict_with_validation
from models.BiLSTM import BidirectionalDualLSTMModel

model = BidirectionalDualLSTMModel(
    run_hidden_size=128,
    incoming_run_hidden_size=128,
    num_layers=5,
    lstm_dropout=0.3,
    ff_dropout=0,
    ff_hidden_sizes=[512, 256]
)

# Load data
directory_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

run_matrices = load(os.path.join(directory_path, 'data/processed/run_matrices.joblib'))
incoming_run_matrices = load(os.path.join(directory_path, 'data/processed/incoming_run_matrices.joblib'))
metrology_matrix = load(os.path.join(directory_path, 'data/processed/metrology_matrix.joblib'))

test_run_matrices = load(os.path.join(directory_path, 'data/processed/test_run_matrices.joblib'))
test_incoming_run_matrices = load(os.path.join(directory_path, 'data/processed/test_incoming_run_matrices.joblib'))

X_run = torch.from_numpy(run_matrices).float()
X_incoming_run = torch.from_numpy(incoming_run_matrices).float()
y = torch.from_numpy(metrology_matrix).float()
X_run_test = torch.from_numpy(test_run_matrices).float()
X_incoming_run_test = torch.from_numpy(test_incoming_run_matrices).float()

print(X_run.shape, X_incoming_run.shape, y.shape, X_run_test.shape, X_incoming_run_test.shape)

print(train_and_predict_with_validation(model, X_run, X_incoming_run, y, X_run_test, X_incoming_run_test, 200, 1e-3))
