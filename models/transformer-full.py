import torch
import torch.nn as nn
import math

from torchinfo import summary

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
        timestamps = data[:, :, 4] - data[:, :, 3]  # (Time Stamp - Run Start Time)
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
    # Test the model
    model = FullTransformer(
        incoming_feature_size=45,
        run_feature_size=20,
        d_model=128,  # Reduced from 256 - more appropriate for input size
        nhead=4,  # Reduced from 8 - each head gets 32 dimensions
        num_encoder_layers=2,  # Reduced from 3 - helps prevent overfitting
        num_decoder_layers=2,  # Reduced from 3
        dim_feedforward=512,  # Reduced from 1024 - still 4x d_model
        dropout=0.3,  # Increased from 0.2 for more regularization
        output_size=49,
        aggregation='mean',
        use_temporal_encoding=True
    )

    # Test forward pass
    batch_size = 4
    seq_len = 755

    # Create dummy data with proper timestamp structure
    incoming = torch.randn(batch_size, seq_len, 45)
    run = torch.randn(batch_size, seq_len, 20)

    # Set up timestamps (features 3 and 4)
    # Run Start Time (constant per sample)
    incoming[:, :, 3] = 1000.0
    run[:, :, 3] = 1000.0

    # Time Stamp (incrementing by 1 second each step)
    for i in range(batch_size):
        incoming[i, :, 4] = 1000.0 + torch.arange(seq_len)
        run[i, :, 4] = 1000.0 + torch.arange(seq_len)

    incoming_lengths = torch.randint(100, 700, (batch_size,))
    run_lengths = torch.randint(100, 700, (batch_size,))

    output = model(incoming, run, incoming_lengths, run_lengths)

    # Use torchinfo with your multi-input setup
    summary(model,
            input_data=(incoming, run, incoming_lengths, run_lengths),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            row_settings=["var_names"],
            verbose=1)

