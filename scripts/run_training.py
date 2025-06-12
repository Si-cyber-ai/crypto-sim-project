import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE
import math
import logging
import warnings
warnings.filterwarnings('ignore')

# Import the TransformerModel from model_architecture.py
from model_architecture import TransformerModel, PositionalEncoding

# Create necessary directories
os.makedirs('../models', exist_ok=True)
os.makedirs('../data', exist_ok=True)
os.makedirs('../models/plots', exist_ok=True)

# Check if data exists, if not generate it
if not os.path.exists('../data/crypto_volatility_fraud_dataset.csv'):
    print("Dataset not found. Generating synthetic data...")
    import generate_crypto_data
    generate_crypto_data.main()

def preprocess_data(df):
    """Preprocess the dataset."""
    # Convert date to datetime if it exists
    if 'date' in df.columns:
        # Use errors='coerce' to handle any parsing errors and format='mixed' to infer format
        df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed', dayfirst=True)
        
        # Check for NaT values after conversion and print a warning
        nat_count = df['date'].isna().sum()
        if nat_count > 0:
            print(f"Warning: {nat_count} date values couldn't be parsed and were set to NaT")
            
        # Extract time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
    
    # Fill missing values
    df = df.fillna(0)
    
    # Create additional features
    df['price_change'] = df['price'].pct_change().fillna(0)
    df['volume_change'] = df['on_chain_volume'].pct_change().fillna(0)
    df['volume_to_price_ratio'] = df['on_chain_volume'] / df['price']
    df['inflow_to_volume_ratio'] = df['exchange_inflow'] / df['on_chain_volume'].replace(0, 1)
    
    # Drop date column for modeling
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    return df

def create_sequences(data, seq_length=5):
    """Create sequences for time series prediction."""
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, -1]  # Volatility is the last column
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_fraud_model(df):
    """Train Random Forest for fraud detection with SMOTE."""
    print("\n--- Training Fraud Detection Model ---")
    
    # Prepare data
    X = df.drop(columns=['fraud_label', 'volatility'])
    y = df['fraud_label']
    
    # Check class distribution before SMOTE
    print("Class distribution before SMOTE:")
    class_counts = y.value_counts()
    print(class_counts)
    print(f"Fraud percentage: {y.mean() * 100:.2f}%")
    
    # Visualize class distribution before SMOTE
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=y)
    plt.title('Class Distribution Before SMOTE')
    plt.xlabel('Fraud Label (1 = Fraud)')
    plt.ylabel('Count')
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom')
    plt.savefig('../models/plots/class_distribution_before_smote.png')
    plt.close()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE to balance the training data
    print("Applying SMOTE to balance the dataset...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Check class distribution after SMOTE
    print("Class distribution after SMOTE:")
    smote_counts = pd.Series(y_train_smote).value_counts()
    print(smote_counts)
    print(f"Fraud percentage after SMOTE: {pd.Series(y_train_smote).mean() * 100:.2f}%")
    
    # Visualize class distribution after SMOTE
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=pd.Series(y_train_smote))
    plt.title('Class Distribution After SMOTE')
    plt.xlabel('Fraud Label (1 = Fraud)')
    plt.ylabel('Count')
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom')
    plt.savefig('../models/plots/class_distribution_after_smote.png')
    plt.close()
    
    # Train model with SMOTE-balanced data
    print("Training Random Forest model with SMOTE-balanced data...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_smote, y_train_smote)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    # Handle potential KeyError in classification report
    try:
        print("Fraud Detection Model Evaluation:")
        report = classification_report(y_test, y_pred)
        print(report)
    except KeyError as e:
        print(f"Error in classification report: {e}")
        # Calculate metrics manually
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('../models/plots/fraud_confusion_matrix.png')
    plt.close()
    
    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.savefig('../models/plots/fraud_feature_importance.png')
    plt.close()
    
    # Save model
    joblib.dump(model, '../models/fraud_detection_model.pkl')
    print("✅ Random Forest model saved to models/fraud_detection_model.pkl")
    
    return model

def train_volatility_model(df):
    """Train Transformer model for volatility prediction with improved training."""
    print("\n--- Training Volatility Prediction Model ---")
    
    # Prepare data
    X = df.drop(columns=['fraud_label'])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences
    seq_length = 10  # Increased from 5 for better temporal context
    X_seq, y_seq = create_sequences(X_scaled, seq_length)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # Initialize model
    input_dim = X_train.shape[2]
    model = TransformerModel(input_dim=input_dim, d_model=128, nhead=4, num_layers=3, dropout=0.2)
    
    # Use learning rate scheduler - remove verbose parameter
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Train for more epochs
    num_epochs = 100  # Increased from default
    batch_size = 64
    criterion = nn.MSELoss()
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Training loop with early stopping
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test).item()
        
        # Update learning rate
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(X_train):.6f}, Val Loss: {val_loss:.6f}')
        
        # Print learning rate changes
        if current_lr != prev_lr:
            print(f'Learning rate changed from {prev_lr:.6f} to {current_lr:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '../models/best_transformer_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Save model and metadata
    joblib.dump(scaler, '../models/volatility_scaler.pkl')
    joblib.dump(seq_length, '../models/sequence_length.pkl')
    print("✅ Transformer model saved to models/best_transformer_model.pt")
    print("✅ Scaler saved to models/volatility_scaler.pkl")
    print("✅ Sequence length saved to models/sequence_length.pkl")
    
    return model, scaler

if __name__ == "__main__":
    # Load dataset
    print("Loading dataset...")
    try:
        df = pd.read_csv('../data/crypto_volatility_fraud_dataset.csv')
        print(f"Dataset loaded with shape: {df.shape}")
    except FileNotFoundError:
        print("Dataset not found. Generating synthetic data...")
        import generate_crypto_data
        generate_crypto_data.main()
        df = pd.read_csv('../data/crypto_volatility_fraud_dataset.csv')
    
    # Preprocess data
    processed_df = preprocess_data(df)
    
    # Train fraud detection model
    fraud_model = train_fraud_model(processed_df)
    
    # Train volatility prediction model
    volatility_model, scaler = train_volatility_model(processed_df)
    
    print("\n✅ Training complete! Models saved to the 'models' directory.")
