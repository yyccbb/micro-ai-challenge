import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from joblib import load
from torchinfo import summary
from tqdm import tqdm

from utils import create_data_loaders, train_model, test_model

RANDOM_STATE = 42
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 20
MIN_DELTA = 1e-4

def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class BidirectionalDualLSTMModel(nn.Module):
    def __init__(self,
                 run_size=20,
                 incoming_run_size=45,
                 run_hidden_size=128,
                 incoming_run_hidden_size=128,
                 num_layers=1,
                 lstm_dropout=0.5,
                 ff_dropout=0.2,
                 ff_hidden_sizes=None,
                 ff_output_size=49):
        super().__init__()
        self.run_size = run_size
        self.incoming_run_size = incoming_run_size
        self.run_hidden_size = run_hidden_size
        self.incoming_run_hidden_size = incoming_run_hidden_size

        if ff_hidden_sizes is None:
            ff_hidden_sizes = [128, 64]

        # Bidirectional LSTM for run data
        self.lstm_run = nn.LSTM(
            input_size=run_size,
            hidden_size=run_hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Added bidirectional
        )

        # Bidirectional LSTM for incoming run data
        self.lstm_incoming_run = nn.LSTM(
            input_size=incoming_run_size,
            hidden_size=incoming_run_hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Added bidirectional
        )

        # Update the input size for feedforward network
        # Bidirectional LSTMs have 2x the hidden size
        last_output_size = (run_hidden_size * 2) + (incoming_run_hidden_size * 2)
        ff_layers = []
        prev_hidden_size = last_output_size

        for hidden_size in ff_hidden_sizes:
            ff_layers.extend([
                nn.Linear(prev_hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(ff_dropout)
            ])
            prev_hidden_size = hidden_size

        ff_layers.append(nn.Linear(prev_hidden_size, ff_output_size))
        self.feed_forward = nn.Sequential(*ff_layers)

    def forward(self, x1, x2, lengths1, lengths2):
        # Process first input with bidirectional LSTM
        if lengths1 is not None:
            x1_packed = nn.utils.rnn.pack_padded_sequence(
                x1, lengths1.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm1_out_packed, (h1_n, c1_n) = self.lstm_run(x1_packed)
            lstm1_out, _ = nn.utils.rnn.pad_packed_sequence(lstm1_out_packed, batch_first=True)

            lengths1 = lengths1.unsqueeze(1)
            out_run = lstm1_out.sum(dim=1) / lengths1
        else:
            lstm1_out, (h1_n, c1_n) = self.lstm_run(x1)
            out_run = lstm1_out.mean(dim=1)

        # Process second input with bidirectional LSTM
        if lengths2 is not None:
            x2_packed = nn.utils.rnn.pack_padded_sequence(
                x2, lengths2.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm2_out_packed, (h2_n, c2_n) = self.lstm_incoming_run(x2_packed)
            lstm2_out, _ = nn.utils.rnn.pad_packed_sequence(lstm2_out_packed, batch_first=True)

            lengths2 = lengths2.unsqueeze(1)
            out_incoming_run = lstm2_out.sum(dim=1) / lengths2
        else:
            lstm2_out, (h2_n, c2_n) = self.lstm_incoming_run(x2)
            out_incoming_run = lstm2_out.mean(dim=1)

        # Concatenate and pass through feedforward network
        combined_features = torch.concat([out_run, out_incoming_run], dim=1)
        return self.feed_forward(combined_features)


if __name__ == "__main__":
    set_seed(RANDOM_STATE)
    # Load data
    directory_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    run_matrices = load(os.path.join(directory_path, 'data/processed/run_matrices.joblib'))
    incoming_run_matrices = load(os.path.join(directory_path, 'data/processed/incoming_run_matrices.joblib'))
    metrology_matrix = load(os.path.join(directory_path, 'data/processed/metrology_matrix.joblib'))

    X_run = torch.from_numpy(run_matrices).float()
    X_incoming_run = torch.from_numpy(incoming_run_matrices).float()
    y = torch.from_numpy(metrology_matrix).float()
    print(X_run.shape, X_incoming_run.shape, y.shape)

    # Initialize bidirectional model
    bidirectional_model = BidirectionalDualLSTMModel(
        run_hidden_size=128,
        incoming_run_hidden_size=128,
        num_layers=1,
        lstm_dropout=0.5,
        ff_dropout=0.2,
        ff_hidden_sizes=[256, 128]
    )

    summary(
        bidirectional_model,
        input_data=(
            torch.randn(32, 755, 20),  # x1: batch_size=32, seq_len=755, feature_dim=20
            torch.randn(32, 755, 45),  # x2: batch_size=32, seq_len=755, feature_dim=45
            torch.full((32,), 700),  # lengths1
            torch.full((32,), 700)  # lengths2
        )
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_run, X_incoming_run, y,
        train_ratio=0.7,
        val_ratio=0.1,
        batch_size=BATCH_SIZE,
        standardize=True,
        random_state=RANDOM_STATE
    )

    # Train the model
    train_losses, val_losses = train_model(
        bidirectional_model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        model_save_path='bidirectional-best-model.pth'
    )

    # Test the model
    test_results = test_model(bidirectional_model, test_loader)

    print({k: test_results[k] for k in ['test_loss', 'mse', 'mae', 'r2_score']})