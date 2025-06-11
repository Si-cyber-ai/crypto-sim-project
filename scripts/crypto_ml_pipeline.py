import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import logging
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, 
                            precision_recall_curve, roc_curve, auc,
                            mean_squared_error, mean_absolute_error, r2_score)
import xgboost as xgb
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import pywt

# Set up argument parser
parser = argparse.ArgumentParser(description='Cryptocurrency ML Pipeline')
parser.add_argument('--mode', type=str, default='production', choices=['debug', 'production'],
                    help='Running mode: debug or production')
parser.add_argument('--data_path', type=str, default='../data/crypto_volatility_fraud_dataset.csv',
                    help='Path to the dataset')
parser.add_argument('--output_dir', type=str, default='../models',
                    help='Directory to save models and results')
parser.add_argument('--log_dir', type=str, default='../logs',
                    help='Directory to save logs')
parser.add_argument('--balancing', type=str, default='smote', choices=['smote', 'ros', 'none'],
                    help='Class balancing method: smote, ros, or none')
parser.add_argument('--fraud_model', type=str, default='random_forest', 
                    choices=['random_forest', 'xgboost', 'lightgbm', 'all'],
                    help='Fraud detection model to use')
parser.add_argument('--volatility_model', type=str, default='transformer',
                    choices=['transformer', 'lstm', 'none'],
                    help='Volatility prediction model to use')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs for deep learning models')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for deep learning models')
parser.add_argument('--seq_length', type=int, default=5,
                    help='Sequence length for time series models')
parser.add_argument('--wavelet', type=str, default='db4',
                    help='Wavelet type for feature extraction')
parser.add_argument('--wavelet_level', type=int, default=3,
                    help='Wavelet decomposition level')

args = parser.parse_args()

# Create output directories
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)

# Set up logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(args.log_dir, f'crypto_ml_{timestamp}.log')
logging.basicConfig(
    level=logging.DEBUG if args.mode == 'debug' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Wavelet feature extraction function
def wavelet_features(data, wavelet=None, level=None):
    """Extract wavelet features from time series data."""
    if wavelet is None:
        wavelet = args.wavelet
    if level is None:
        level = args.wavelet_level
        
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
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1, max_seq_length=100):
        super().__init__()
        self.d_model = d_model
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                  dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc_out = nn.Linear(d_model, 1)
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_length, input_dim]
        x = self.embedding(x)  # [batch_size, seq_length, d_model]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        # Take the average of the sequence for prediction
        x = torch.mean(x, dim=1)
        return self.fc_out(x).squeeze()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

# LSTM model for volatility prediction
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)
        # Take the last time step
        out = self.fc_out(lstm_out[:, -1, :])
        return out.squeeze()

# Create sequences for time series prediction
def create_sequences(data, target, seq_length=None):
    if seq_length is None:
        seq_length = args.seq_length
        
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = target[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Preprocess data
def preprocess_data(df):
    """Preprocess the dataset for both models."""
    logger.info("Preprocessing data...")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert date to datetime if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        # Extract time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        # Sort by date for time series
        df = df.sort_values('date')
    
    # Fill missing values
    df = df.fillna(0)
    
    # Create additional features
    if 'price' in df.columns and 'on_chain_volume' in df.columns:
        df['price_change'] = df['price'].pct_change().fillna(0)
        df['volume_change'] = df['on_chain_volume'].pct_change().fillna(0)
        df['volume_to_price_ratio'] = df['on_chain_volume'] / df['price'].replace(0, 1)
        
        if 'exchange_inflow' in df.columns:
            df['inflow_to_volume_ratio'] = df['exchange_inflow'] / df['on_chain_volume'].replace(0, 1)
    
        # Extract wavelet features from price and volume
        logger.info("Extracting wavelet features...")
        try:
            price_wavelets = np.array([wavelet_features(df['price'].values)])
            volume_wavelets = np.array([wavelet_features(df['on_chain_volume'].values)])
            
            # Create wavelet feature columns
            for i in range(price_wavelets.shape[1]):
                df[f'price_wavelet_{i}'] = price_wavelets[0, i]
            for i in range(volume_wavelets.shape[1]):
                df[f'volume_wavelet_{i}'] = volume_wavelets[0, i]
        except Exception as e:
            logger.error(f"Error extracting wavelet features: {e}")
    
    # Drop date column for modeling
    if 'date' in df.columns:
        df_model = df.drop(columns=['date'])
    else:
        df_model = df.copy()
    
    # Ensure fraud_label and volatility columns exist
    if 'fraud_label' not in df_model.columns:
        logger.warning("fraud_label column not found, creating dummy column")
        df_model['fraud_label'] = 0
    
    if 'volatility' not in df_model.columns:
        logger.warning("volatility column not found, creating dummy column")
        df_model['volatility'] = 0
    
    return df_model

# Train fraud detection model
def train_fraud_model(df):
    """Train and evaluate fraud detection models."""
    logger.info("Training fraud detection model...")
    
    # Prepare data
    X = df.drop(columns=['fraud_label', 'volatility'])
    y = df['fraud_label'].astype(int)  # Ensure labels are integers
    
    # Check class distribution
    class_counts = y.value_counts()
    logger.info(f"Class distribution before resampling:\n{class_counts}")
    logger.info(f"Fraud percentage: {y.mean() * 100:.2f}%")
    
    # Visualize class distribution before resampling
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y)
    plt.title('Class Distribution Before Resampling')
    plt.xlabel('Fraud Label')
    plt.ylabel('Count')
    plt.savefig(os.path.join(args.output_dir, 'plots', 'class_distribution_before.png'))
    plt.close()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models to try
    if args.fraud_model == 'all':
        models = {
            "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
            "XGBoost": xgb.XGBClassifier(random_state=42, scale_pos_weight=len(y_train[y_train==0])/max(1, len(y_train[y_train==1]))),
            "LightGBM": LGBMClassifier(random_state=42, class_weight='balanced')
        }
    else:
        if args.fraud_model == 'random_forest':
            models = {"Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)}
        elif args.fraud_model == 'xgboost':
            models = {"XGBoost": xgb.XGBClassifier(random_state=42, scale_pos_weight=len(y_train[y_train==0])/max(1, len(y_train[y_train==1])))}
        elif args.fraud_model == 'lightgbm':
            models = {"LightGBM": LGBMClassifier(random_state=42, class_weight='balanced')}
    
    # Define parameters for grid search
    param_grids = {
        "Random Forest": {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        "XGBoost": {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 200]
        },
        "LightGBM": {
            'learning_rate': [0.01, 0.1],
            'num_leaves': [31, 63, 127],
            'n_estimators': [100, 200]
        }
    }
    
    # Choose resampling method
    if args.balancing == 'smote':
        resampler = SMOTE(random_state=42)
        logger.info("Using SMOTE for class balancing")
    elif args.balancing == 'ros':
        resampler = RandomOverSampler(random_state=42)
        logger.info("Using RandomOverSampler for class balancing")
    else:
        resampler = None
        logger.info("No class balancing applied")
    
    # Train and evaluate models
    best_models = {}
    best_f1 = 0
    best_model_name = None
    
    for name, model in models.items():
        logger.info(f"\nTraining {name} model...")
        
        # Create pipeline with resampling if needed
        if resampler:
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('resampler', resampler),
                ('classifier', model)
            ])
        else:
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
        
        # Grid search with cross-validation
        if args.mode == 'production' and name in param_grids:
            param_grid = {f'classifier__{key}': value for key, value in param_grids[name].items()}
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
            
            pipeline = grid_search.best_estimator_
        else:
            pipeline.fit(X_train, y_train)
        
        # If using resampler, check class distribution after resampling
        if resampler and args.mode == 'debug':
            # Get the resampled data
            X_resampled, y_resampled = pipeline.named_steps['resampler'].fit_resample(
                pipeline.named_steps['scaler'].transform(X_train), y_train
            )
            resampled_class_counts = pd.Series(y_resampled).value_counts()
            logger.info(f"Class distribution after resampling:\n{resampled_class_counts}")
            
            # Visualize class distribution after resampling
            plt.figure(figsize=(10, 6))
            sns.countplot(x=pd.Series(y_resampled))
            plt.title('Class Distribution After Resampling')
            plt.xlabel('Fraud Label')
            plt.ylabel('Count')
            plt.savefig(os.path.join(args.output_dir, 'plots', f'class_distribution_after_{name}.png'))
            plt.close()
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        
        # Get probabilities if available
        y_prob = None
        if hasattr(pipeline, 'predict_proba'):
            try:
                proba = pipeline.predict_proba(X_test)
                if proba.shape[1] > 1:  # Ensure we have probabilities for multiple classes
                    y_prob = proba[:, 1]
            except Exception as e:
                logger.warning(f"Error getting prediction probabilities: {e}")
        
        # Ensure y_test and y_pred are integers
        y_test_int = y_test.astype(int)
        y_pred_int = np.array(y_pred).astype(int)
        
        # Get unique classes
        unique_test = np.unique(y_test_int)
        unique_pred = np.unique(y_pred_int)
        logger.info(f"Unique classes in test set: {unique_test}")
        logger.info(f"Unique classes in predictions: {unique_pred}")
        
        # Classification report with zero_division=0 to handle missing classes
        try:
            # Ensure both classes are represented in the report
            labels = sorted(np.unique(np.concatenate([unique_test, unique_pred])))
            report = classification_report(y_test_int, y_pred_int, 
                                          output_dict=True, 
                                          zero_division=0,
                                          labels=labels)
            logger.info(f"\n{name} Classification Report:")
            logger.info(classification_report(y_test_int, y_pred_int, 
                                             zero_division=0,
                                             labels=labels))
            
            # Save report to CSV
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(args.output_dir, f'{name.lower().replace(" ", "_")}_report.csv'))
        except Exception as e:
            logger.error(f"Error generating classification report: {e}")
            # Fallback to manual calculation
            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
            precision = precision_score(y_test_int, y_pred_int, average='weighted', zero_division=0)
            recall = recall_score(y_test_int, y_pred_int, average='weighted', zero_division=0)
            f1 = f1_score(y_test_int, y_pred_int, average='weighted', zero_division=0)
            accuracy = accuracy_score(y_test_int, y_pred_int)
            logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test_int, y_pred_int)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(os.path.join(args.output_dir, 'plots', f'{name.lower().replace(" ", "_")}_confusion_matrix.png'))
        plt.close()
        
        # ROC Curve if probabilities are available
        if y_prob is not None:
            try:
                fpr, tpr, _ = roc_curve(y_test_int, y_prob)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{name} ROC Curve')
                plt.legend(loc="lower right")
                plt.savefig(os.path.join(args.output_dir, 'plots', f'{name.lower().replace(" ", "_")}_roc_curve.png'))
                plt.close()
                
                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_test_int, y_prob)
                
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, color='blue', lw=2)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'{name} Precision-Recall Curve')
                plt.savefig(os.path.join(args.output_dir, 'plots', f'{name.lower().replace(" ", "_")}_pr_curve.png'))
                plt.close()
            except Exception as e:
                logger.error(f"Error generating ROC curve: {e}")
        
        # Save model
        joblib.dump(pipeline, os.path.join(args.output_dir, f'{name.lower().replace(" ", "_")}_pipeline.pkl'))
        best_models[name] = pipeline
        
        # Calculate F1 score for fraud class
        try:
            # Check if '1' exists in the report
            if '1' in report:
                f1 = report['1']['f1-score']
                if f1 > best_f1:
                    best_f1 = f1
                    best_model_name = name
            else:
                logger.warning(f"No fraud instances (class 1) predicted correctly by {name}")
        except (KeyError, NameError) as e:
            logger.warning(f"Error when accessing report: {e}")
    
    # Save the best model as the default fraud detection model
    if best_model_name:
        logger.info(f"\nBest model: {best_model_name} with fraud F1-score: {best_f1:.4f}")
        joblib.dump(best_models[best_model_name], os.path.join(args.output_dir, 'fraud_detection_model.pkl'))
        logger.info(f"✅ Best model ({best_model_name}) saved as fraud_detection_model.pkl")
    else:
        logger.warning("No model performed well on fraud detection. Saving the first model as default.")
        first_model_name = list(models.keys())[0]
        joblib.dump(best_models[first_model_name], os.path.join(args.output_dir, 'fraud_detection_model.pkl'))
        logger.info(f"✅ {first_model_name} saved as fraud_detection_model.pkl (default)")
    
    return best_models

# Train volatility prediction model
def train_volatility_model(df):
    """Train and evaluate volatility prediction model."""
    logger.info("Training volatility prediction model...")
    
    # Prepare data
    X = df.drop(columns=['fraud_label', 'volatility'])
    y = df['volatility'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences for time series prediction
    X_seq, y_seq = create_sequences(X_scaled, y, seq_length=args.seq_length)
    logger.info(f"Created {len(X_seq)} sequences with length {args.seq_length}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize model based on choice
    input_dim = X_train.shape[2]  # Number of features
    if args.volatility_model == 'transformer':
        model = TransformerModel(input_dim).to(device)
        logger.info(f"Initialized Transformer model with input dimension {input_dim}")
    elif args.volatility_model == 'lstm':
        model = LSTMModel(input_dim).to(device)
        logger.info(f"Initialized LSTM model with input dimension {input_dim}")
    else:
        logger.info("Skipping volatility model training as requested")
        return None, None
    
    # Training parameters
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Early stopping parameters
    early_stopping_patience = 10
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    # Training loop
    train_losses = []
    val_losses = []
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
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
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_volatility_model.pt'))
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                logger.info(f'Early stopping at epoch {epoch+1}')
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
    plt.savefig(os.path.join(args.output_dir, 'plots', 'volatility_training_loss.png'))
    plt.close()
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_volatility_model.pt')))
    model.eval()
    
    # Evaluate on test set
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    logger.info("\nVolatility Model Evaluation:")
    logger.info(f"MSE: {mse:.6f}")
    logger.info(f"RMSE: {rmse:.6f}")
    logger.info(f"MAE: {mae:.6f}")
    logger.info(f"R²: {r2:.6f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:100], label='Actual')
    plt.plot(y_pred[:100], label='Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Volatility')
    plt.title('Volatility Prediction: Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'plots', 'volatility_predictions.png'))
    plt.close()
    
    # Save model and scaler
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'volatility_model.pt'))
    joblib.dump(scaler, os.path.join(args.output_dir, 'volatility_scaler.pkl'))
    joblib.dump(args.seq_length, os.path.join(args.output_dir, 'sequence_length.pkl'))
    logger.info("✅ Volatility model and scaler saved successfully")
    
    return model, scaler

# Main function
def main():
    """Run the complete ML pipeline."""
    logger.info(f"Starting cryptocurrency ML pipeline in {args.mode} mode")
    logger.info(f"Arguments: {args}")
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    try:
        df = pd.read_csv(args.data_path)
        logger.info(f"Loaded dataset with shape {df.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Preprocess data
    df_processed = preprocess_data(df)
    logger.info(f"Preprocessed data shape: {df_processed.shape}")
    
    # Train fraud detection model
    fraud_models = train_fraud_model(df_processed)
    
    # Train volatility prediction model
    if args.volatility_model != 'none':
        volatility_model, scaler = train_volatility_model(df_processed)
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()







