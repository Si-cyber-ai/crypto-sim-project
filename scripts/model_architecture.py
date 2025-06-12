import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for the Transformer model."""
    def __init__(self, d_model, dropout=0.1, max_seq_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Transformer model for volatility prediction."""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, max_seq_length=100):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Add positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_length, input_dim]
        x = self.embedding(x)
        
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Use the last token for prediction
        x = x[:, -1, :]
        
        return self.fc_out(x).squeeze(-1)

