import torch
import torch.nn as nn
import math


class TemporalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        # Learnable embedding for each timestamp position
        # More efficient than linear projection for integer timestamps
        self.time_embedding = nn.Embedding(max_len, d_model)

        # Alternative: simple linear projection (uncomment if preferred)
        # self.time_projection = nn.Linear(1, d_model)

    def forward(self, x, timestamps):
        """
        Args:
            x: Input embeddings (batch, seq_len, d_model)
            timestamps: Actual timestamps (batch, seq_len) - seconds since start
        """
        # Since timestamps are integers (0, 1, 2, ...), use them directly
        time_emb = self.time_embedding(timestamps.long())

        # Alternative with linear projection:
        # time_emb = self.time_projection(timestamps.unsqueeze(-1) / 100.0)  # Normalize

        # Add temporal encoding to input
        x = x + time_emb
        return self.dropout(x)


class WaferTransformer(nn.Module):
    def __init__(self,
                 incoming_feature_size=45,
                 run_feature_size=20,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dim_feedforward=1024,
                 dropout=0.2,
                 output_size=49,
                 aggregation='mean',
                 use_temporal_encoding=True):
        super().__init__()

        self.d_model = d_model
        self.aggregation = aggregation
        self.use_temporal_encoding = use_temporal_encoding

        # Input projections (excluding timestamp features if we use them separately)
        # We'll process timestamps separately, so reduce feature size by 2
        # (removing indices 3 and 4: Run Start Time and Time Stamp)
        if use_temporal_encoding:
            self.incoming_projection = nn.Linear(incoming_feature_size - 2, d_model)
            self.run_projection = nn.Linear(run_feature_size - 2, d_model)
        else:
            self.incoming_projection = nn.Linear(incoming_feature_size, d_model)
            self.run_projection = nn.Linear(run_feature_size, d_model)

        # Temporal encoding using actual timestamps
        if use_temporal_encoding:
            self.temporal_encoder = TemporalEncoding(d_model, dropout=dropout)
        else:
            # Fallback to standard positional encoding
            self.pos_encoder = PositionalEncoding(d_model, max_len=1000, dropout=dropout)

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
        """
        Extract timestamp information from the data.
        Timestamp = data[:, :, 4] - data[:, :, 3]  (Time Stamp - Run Start Time)
        """
        timestamps = data[:, :, 4] - data[:, :, 3]  # Seconds since start
        return timestamps

    def create_padding_mask(self, lengths, max_len, device):
        """Create padding mask from lengths"""
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

            # Remove the timestamp features from the data
            # Keep all features except the time-related ones we'll encode separately
            incoming_features = torch.cat([
                incoming_data[:, :, :3],  # Features before Run Start Time
                incoming_data[:, :, 5:]  # Features after Time Stamp
            ], dim=-1)

            run_features = torch.cat([
                run_data[:, :, :3],  # Features before Run Start Time
                run_data[:, :, 5:]  # Features after Time Stamp
            ], dim=-1)

            # Project inputs to d_model dimension
            incoming_emb = self.incoming_projection(incoming_features)
            run_emb = self.run_projection(run_features)

            # Transpose for transformer (seq_len, batch, d_model)
            incoming_emb = incoming_emb.transpose(0, 1)
            run_emb = run_emb.transpose(0, 1)

            # Add temporal encoding using actual timestamps
            incoming_emb = incoming_emb.transpose(0, 1)  # Back to (batch, seq, d_model)
            run_emb = run_emb.transpose(0, 1)

            incoming_emb = self.temporal_encoder(incoming_emb, incoming_timestamps)
            run_emb = self.temporal_encoder(run_emb, run_timestamps)

            # Transpose back for transformer
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

        # Pass through transformer
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


class PositionalEncoding(nn.Module):
    """Standard positional encoding as fallback"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
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


# Integration with your existing code
def create_transformer_model():
    """Create transformer model with optimal settings for wafer data"""
    model = WaferTransformer(
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
    return model


if __name__ == "__main__":
    # Test the model
    model = create_transformer_model()

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
    print(f"Output shape: {output.shape}")  # Should be (4, 49)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Compare with baseline LSTM model size
    print(f"\nFor comparison, your DualLSTMModel2 likely has ~500K-1M parameters")
    print(f"This transformer has {total_params:,} parameters")

    # Alternative configurations to try:
    print("\nAlternative configurations to experiment with:")
    print("1. Tiny: d_model=64, nhead=4, layers=2, feedforward=256")
    print("2. Small: d_model=128, nhead=4, layers=2, feedforward=512 (current)")
    print("3. Medium: d_model=192, nhead=6, layers=3, feedforward=768")
    print("4. Large: d_model=256, nhead=8, layers=3, feedforward=1024")