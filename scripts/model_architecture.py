import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    """Transformer model for volatility prediction."""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
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
        x = self.embedding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        # Take the average of the sequence
        x = torch.mean(x, dim=1)
        x = self.fc_out(x)
        return x.squeeze(-1)  # Ensure proper shape for loss calculation

