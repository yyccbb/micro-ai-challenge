import os
import torch
import torch.nn as nn
import math

from joblib import load
from torchinfo import summary

from utils import create_data_loaders, train_model, test_model

RANDOM_STATE = 42
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 20
MIN_DELTA = 1e-4

N_ENCODER_DECODER_LAYERS = 2
D_MODEL = 128
N_HEAD = 4
AGGREGATION = 'mean' # mean, max, last, first
USE_TEMPORAL_ENCODING = True

class TemporalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        # Learnable embedding for each timestamp position
        self.time_embedding = nn.Embedding(max_len, d_model)

        # Alternative
        # self.time_projection = nn.Linear(1, d_model)

    def forward(self, x, timestamps):
        """
        Args:
            x: Input embeddings (batch, seq_len, d_model)
            timestamps: Actual timestamps (batch, seq_len) - seconds since start
        """
        time_emb = self.time_embedding(timestamps.long())

        # Alternative
        # time_emb = self.time_projection(timestamps.unsqueeze(-1) / 100.0)  # Normalize

        x = x + time_emb
        x = self.dropout(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=800, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class FullTransformer(nn.Module):
    def __init__(self,
                 incoming_feature_size=45,
                 run_feature_size=20,
                 d_model=128,
                 nhead=4,
                 num_encoder_layers=2,
                 num_decoder_layers=2,
                 dim_feedforward=512,
                 dropout=0.3,
                 output_size=49,
                 aggregation='mean',
                 use_temporal_encoding=True):
        super().__init__()

        self.d_model = d_model
        self.aggregation = aggregation
        self.use_temporal_encoding = use_temporal_encoding

        self.incoming_projection = nn.Linear(incoming_feature_size, d_model)
        self.run_projection = nn.Linear(run_feature_size, d_model)

        if use_temporal_encoding:
            self.temporal_encoder = TemporalEncoding(d_model, max_len=800, dropout=dropout)
        else:
            self.pos_encoder = PositionalEncoding(d_model, max_len=800, dropout=dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # PyTorch transformer expects (seq_len, batch, features)
        )

        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )

    def extract_timestamps(self, data):
        timestamps = data[:, :, 3] - data[:, :, 2]  # (Time Stamp - Run Start Time)
        return timestamps

    def create_padding_mask(self, lengths, max_len, device):
        batch_size = lengths.shape[0]
        # Create mask where True indicates positions to ignore (padding)
        mask = torch.arange(max_len, device=device).expand(
            batch_size, max_len
        ) >= lengths.unsqueeze(1)
        return mask

    def forward(self, incoming_data, run_data, incoming_lengths, run_lengths):
        batch_size = incoming_data.size(0)
        device = incoming_data.device

        if self.use_temporal_encoding:
            # Extract timestamps before projecting features
            incoming_timestamps = self.extract_timestamps(incoming_data)
            run_timestamps = self.extract_timestamps(run_data)

            # Project ALL features to d_model dimension
            incoming_emb = self.incoming_projection(incoming_data)
            run_emb = self.run_projection(run_data)

            # Add temporal encoding using actual timestamps
            incoming_emb = self.temporal_encoder(incoming_emb, incoming_timestamps)
            run_emb = self.temporal_encoder(run_emb, run_timestamps)

            # Transpose for transformer (seq_len, batch, d_model)
            incoming_emb = incoming_emb.transpose(0, 1)
            run_emb = run_emb.transpose(0, 1)
        else:
            # Standard approach without temporal encoding
            incoming_emb = self.incoming_projection(incoming_data)
            run_emb = self.run_projection(run_data)

            incoming_emb = incoming_emb.transpose(0, 1)
            run_emb = run_emb.transpose(0, 1)

            incoming_emb = self.pos_encoder(incoming_emb)
            run_emb = self.pos_encoder(run_emb)

        # Create padding masks
        src_key_padding_mask = self.create_padding_mask(
            incoming_lengths, incoming_data.size(1), device
        )
        tgt_key_padding_mask = self.create_padding_mask(
            run_lengths, run_data.size(1), device
        )
        memory_key_padding_mask = src_key_padding_mask

        output = self.transformer(
            src=incoming_emb,
            tgt=run_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # Transpose back to (batch, seq, d_model)
        output = output.transpose(0, 1)

        # Aggregate sequence into single representation
        if self.aggregation == 'mean':
            # Masked mean pooling
            mask = ~tgt_key_padding_mask  # Invert to get valid positions
            mask = mask.unsqueeze(-1).float()
            output_sum = (output * mask).sum(dim=1)
            output_lengths = mask.sum(dim=1)
            aggregated = output_sum / output_lengths.clamp(min=1)

        elif self.aggregation == 'max':
            # Masked max pooling
            mask = tgt_key_padding_mask.unsqueeze(-1)
            output_masked = output.masked_fill(mask, float('-inf'))
            aggregated, _ = output_masked.max(dim=1)

        elif self.aggregation == 'last':
            # Take the last valid position
            batch_indices = torch.arange(batch_size, device=device)
            last_indices = (run_lengths - 1).clamp(min=0)
            aggregated = output[batch_indices, last_indices]

        elif self.aggregation == 'first':
            # Take the first position
            aggregated = output[:, 0]

        # Final projection to output size
        output = self.output_projection(aggregated)

        return output




if __name__ == "__main__":
    model = FullTransformer(
        incoming_feature_size=45,
        run_feature_size=20,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=N_ENCODER_DECODER_LAYERS,
        num_decoder_layers=N_ENCODER_DECODER_LAYERS,
        dim_feedforward=4 * D_MODEL,
        dropout=0.2,
        output_size=49,
        aggregation=AGGREGATION,
        use_temporal_encoding=USE_TEMPORAL_ENCODING
    )

    # Test forward pass
    batch_size = 4
    seq_len = 755

    # Create dummy data with proper timestamp structure
    incoming = torch.randn(batch_size, seq_len, 45)
    run = torch.randn(batch_size, seq_len, 20)

    # Run Start Time (constant per sample)
    incoming[:, :, 2] = 1000.0
    run[:, :, 2] = 1000.0

    for i in range(batch_size):
        incoming[i, :, 3] = 1000.0 + torch.arange(seq_len)
        run[i, :, 3] = 1000.0 + torch.arange(seq_len)

    incoming_lengths = torch.randint(100, 700, (batch_size,))
    run_lengths = torch.randint(100, 700, (batch_size,))

    summary(model,
            input_data=(incoming, run, incoming_lengths, run_lengths),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            row_settings=["var_names"],
            verbose=1)

    # Load data
    directory_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    run_matrices = load(os.path.join(directory_path, 'data/processed/run_matrices.joblib'))
    incoming_run_matrices = load(os.path.join(directory_path, 'data/processed/incoming_run_matrices.joblib'))
    metrology_matrix = load(os.path.join(directory_path, 'data/processed/metrology_matrix.joblib'))

    X_run = torch.from_numpy(run_matrices).float()
    X_incoming_run = torch.from_numpy(incoming_run_matrices).float()
    y = torch.from_numpy(metrology_matrix).float()
    print(X_run.shape, X_incoming_run.shape, y.shape)

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
        model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        model_save_path='transformer-full-best-model.pth'
    )

    # Test the model
    test_results = test_model(model, test_loader)

    print({k: test_results[k] for k in ['test_loss', 'mse', 'mae', 'r2_score']})
