import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pywt
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('../models', exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Wavelet feature extraction function
def wavelet_features(data, wavelet='db4', level=3):
    """Extract wavelet features from time series data."""
    # Pad the data if needed
    n = len(data)
    pad_size = 2**level - (n % 2**level) if n % 2**level != 0 else 0
    padded_data = np.pad(data, (0, pad_size), 'constant', constant_values=(0))
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(padded_data, wavelet, level=level)
    
    # Extract features from coefficients
    features = []
    for i, coeff in enumerate(coeffs):
        features.append(np.mean(coeff))
        features.append(np.std(coeff))
        features.append(np.max(coeff))
        features.append(np.min(coeff))
    
    return features

# Transformer model for volatility prediction
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                  dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        # Take the average of the sequence
        x = torch.mean(x, dim=1)
        return self.fc_out(x).squeeze()

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('../data/crypto_volatility_fraud_dataset.csv')

# Basic preprocessing
print("Preprocessing data...")
# Convert date to datetime
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    # Extract time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    # Sort by date
    df = df.sort_values('date')

# Fill missing values if any
df = df.fillna(0)

# Create additional features
df['price_change'] = df['price'].pct_change().fillna(0)
df['volume_change'] = df['on_chain_volume'].pct_change().fillna(0)
df['volume_to_price_ratio'] = df['on_chain_volume'] / df['price']
df['inflow_to_volume_ratio'] = df['exchange_inflow'] / df['on_chain_volume'].replace(0, 1)

# Extract wavelet features from price and volume
print("Extracting wavelet features...")
price_wavelets = np.array([wavelet_features(df['price'].values, level=3)])
volume_wavelets = np.array([wavelet_features(df['on_chain_volume'].values, level=3)])

# Create wavelet feature columns
for i in range(price_wavelets.shape[1]):
    df[f'price_wavelet_{i}'] = price_wavelets[0, i]
for i in range(volume_wavelets.shape[1]):
    df[f'volume_wavelet_{i}'] = volume_wavelets[0, i]

# Prepare data for modeling
X = df.drop(columns=['fraud_label', 'volatility'])
if 'date' in X.columns:
    X = X.drop(columns=['date'])
y = df['volatility'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences for time series prediction
def create_sequences(data, target, seq_length=5):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = target[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Create sequences
seq_length = 5  # Use 5 days of data to predict the next day
X_seq, y_seq = create_sequences(X_scaled, y, seq_length)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.FloatTensor(y_train).to(device)
y_test = torch.FloatTensor(y_test).to(device)

# Create data loaders
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model
input_dim = X_train.shape[2]  # Number of features
model = TransformerModel(input_dim).to(device)

# Training parameters
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
epochs = 100
early_stopping_patience = 10
best_loss = float('inf')
early_stopping_counter = 0

# Training loop
print("\nTraining Transformer model for volatility prediction...")
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Average training loss
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(test_loader)
    val_losses.append(avg_val_loss)
    
    # Update learning rate
    scheduler.step(avg_val_loss)
    
    # Print progress
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    # Early stopping
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        early_stopping_counter = 0
        # Save best model
        torch.save(model.state_dict(), '../models/best_transformer_model.pt')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('../models/volatility_training_loss.png')

# Load best model for evaluation
model.load_state_dict(torch.load('../models/best_transformer_model.pt'))
model.eval()

# Evaluate on test set
with torch.no_grad():
    y_pred = model(X_test).cpu().numpy()
    y_true = y_test.cpu().numpy()

# Calculate metrics
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\nModel Evaluation:")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R²: {r2:.6f}")

# Plot predictions vs actual
plt.figure(figsize=(12, 6))
plt.plot(y_true[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Volatility')
plt.title('Volatility Prediction: Actual vs Predicted')
plt.legend()
plt.grid(True)
plt.savefig('../models/volatility_predictions.png')

# Save model and scaler
torch.save(model.state_dict(), '../models/transformer_model.pt')
joblib.dump(scaler, '../models/volatility_scaler.pkl')
joblib.dump(seq_length, '../models/sequence_length.pkl')
print("✅ Transformer model and scaler saved successfully")

print("\nVolatility prediction model training complete!")