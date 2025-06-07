import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from joblib import load
from tqdm import tqdm

from utils import create_data_loaders, train_model, test_model

RANDOM_STATE = 42
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 20
MIN_DELTA = 1e-4

N_LAYERS = 1
LSTM_HIDDEN_SIZE = 128
FF_HIDDEN_SIZE = [256, 128]
ATTENTION_SIZE = 128

class LSTMAttention(nn.Module):
    def __init__(self, hidden_size, attention_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size

        # Attention layers
        self.attention_linear = nn.Linear(hidden_size, attention_size)
        self.context_vector = nn.Linear(attention_size, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, lstm_outputs, lengths=None):
        batch_size, seq_len, hidden_size = lstm_outputs.shape

        # Calculate attention scores
        # (batch_size, seq_len, attention_size)
        attention_hidden = self.tanh(self.attention_linear(lstm_outputs))

        # (batch_size, seq_len, 1)
        attention_scores = self.context_vector(attention_hidden)

        # (batch_size, seq_len)
        attention_scores = attention_scores.squeeze(-1)

        if lengths is not None:
            # Create mask: (batch_size, seq_len)
            mask = torch.arange(seq_len, device=lstm_outputs.device).expand(
                batch_size, seq_len
            ) < lengths.unsqueeze(1)

            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=1)

        # Apply attention weights to get context vector
        # (batch_size, seq_len, 1) * (batch_size, seq_len, hidden_size)
        context = torch.sum(attention_weights.unsqueeze(-1) * lstm_outputs, dim=1)

        return context, attention_weights


class BidirectionalDualLSTMWithAttention(nn.Module):
    def __init__(self,
                 run_size=20,
                 incoming_run_size=45,
                 run_hidden_size=128,
                 incoming_run_hidden_size=128,
                 num_layers=1,
                 dropout=0.2,
                 attention_size=128,
                 ff_hidden_sizes=None,
                 ff_output_size=49):
        super().__init__()
        self.run_size = run_size
        self.incoming_run_size = incoming_run_size
        self.run_hidden_size = run_hidden_size
        self.incoming_run_hidden_size = incoming_run_hidden_size

        if ff_hidden_sizes is None:
            ff_hidden_sizes = [256, 128]

        self.lstm_run = nn.LSTM(
            input_size=run_size,
            hidden_size=run_hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.lstm_incoming_run = nn.LSTM(
            input_size=incoming_run_size,
            hidden_size=incoming_run_hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.attention_run = LSTMAttention(
            hidden_size=run_hidden_size * 2,
            attention_size=attention_size
        )

        self.attention_incoming_run = LSTMAttention(
            hidden_size=incoming_run_hidden_size * 2,
            attention_size=attention_size
        )

        last_output_size = (run_hidden_size * 2) + (incoming_run_hidden_size * 2)
        ff_layers = []
        prev_hidden_size = last_output_size

        for hidden_size in ff_hidden_sizes:
            ff_layers.extend([
                nn.Linear(prev_hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
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

            out_run, attention_weights_1 = self.attention_run(lstm1_out, lengths1)
        else:
            lstm1_out, (h1_n, c1_n) = self.lstm_run(x1)
            out_run, attention_weights_1 = self.attention_run(lstm1_out, None)

        # Process second input with bidirectional LSTM
        if lengths2 is not None:
            x2_packed = nn.utils.rnn.pack_padded_sequence(
                x2, lengths2.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm2_out_packed, (h2_n, c2_n) = self.lstm_incoming_run(x2_packed)
            lstm2_out, _ = nn.utils.rnn.pad_packed_sequence(lstm2_out_packed, batch_first=True)

            out_incoming_run, attention_weights_2 = self.attention_incoming_run(lstm2_out, lengths2)
        else:
            lstm2_out, (h2_n, c2_n) = self.lstm_incoming_run(x2)
            out_incoming_run, attention_weights_2 = self.attention_incoming_run(lstm2_out, None)

        combined_features = torch.concat([out_run, out_incoming_run], dim=1)
        output = self.feed_forward(combined_features)

        self.last_attention_weights = {
            'run_attention': attention_weights_1,
            'incoming_run_attention': attention_weights_2
        }

        return output

    def get_attention_weights(self):
        # Returns None if the attribute does not exist
        return getattr(self, 'last_attention_weights', None)


# Model summary
from torchinfo import summary

model = BidirectionalDualLSTMWithAttention()

summary(
    model,
    input_data=(
        torch.randn(32, 755, 20),  # x1: batch_size=32, seq_len=755, feature_dim=20
        torch.randn(32, 755, 45),  # x2: batch_size=32, seq_len=755, feature_dim=45
        torch.full((32,), 700),  # lengths1
        torch.full((32,), 700)  # lengths2
    )
)

# Load data
directory_path = os.path.dirname(os.path.abspath(__file__))

run_matrices = load(os.path.join(directory_path, 'data/processed/run_matrices.joblib'))
incoming_run_matrices = load(os.path.join(directory_path, 'data/processed/incoming_run_matrices.joblib'))
metrology_matrix = load(os.path.join(directory_path, 'data/processed/metrology_matrix.joblib'))

X_run = torch.from_numpy(run_matrices).float()
X_incoming_run = torch.from_numpy(incoming_run_matrices).float()
y = torch.from_numpy(metrology_matrix).float()
print(X_run.shape, X_incoming_run.shape, y.shape)

# Initialize bidirectional model with attention
attention_model = BidirectionalDualLSTMWithAttention(
    run_hidden_size=LSTM_HIDDEN_SIZE,
    incoming_run_hidden_size=LSTM_HIDDEN_SIZE,
    num_layers=N_LAYERS,
    dropout=0.2,
    attention_size=ATTENTION_SIZE,
    ff_hidden_sizes=FF_HIDDEN_SIZE
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
    attention_model,
    train_loader,
    val_loader,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    patience=PATIENCE,
    min_delta=MIN_DELTA,
    model_save_path='bidirectional-lstm-attention-best-model.pth'
)

# Test the model
test_results = test_model(attention_model, test_loader)

print({k: test_results[k] for k in ['test_loss', 'mse', 'mae', 'r2_score']})

# # Example: Visualize attention weights for a sample
# print("\nGetting attention weights for interpretation...")
# attention_model.eval()
# with torch.no_grad():
#     sample_batch = next(iter(test_loader))
#     x1, x2, y, lengths1, lengths2 = sample_batch
#
#     # Get predictions and attention weights
#     predictions = attention_model(x1, x2, lengths1, lengths2)
#     attention_weights = attention_model.get_attention_weights()
#
#     print(f"Run attention shape: {attention_weights['run_attention'].shape}")
#     print(f"Incoming run attention shape: {attention_weights['incoming_run_attention'].shape}")
#     print(f"Sample attention weights (first sequence): {attention_weights['run_attention'][0][:10]}")