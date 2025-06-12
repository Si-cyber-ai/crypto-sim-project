import os
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import classification_report, confusion_matrix
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("crypto-inference")

# Create output directories if they don't exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Import the model architecture
from model_architecture import TransformerModel

class TransformerModel(torch.nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1, max_seq_length=100):
        super().__init__()
        self.input_linear = torch.nn.Linear(input_dim, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = torch.nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        x = self.input_linear(x)
        x = self.transformer_encoder(x)
        # Use the last token for prediction
        x = x[:, -1, :]
        return self.output_linear(x).squeeze(-1)

def load_models():
    """Load all required models and scalers"""
    logger.info("Loading models and scalers...")
    
    # Load volatility prediction model
    try:
        # Load model architecture
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load sequence length
        seq_length = joblib.load("models/sequence_length.pkl")
        logger.info(f"Loaded sequence length: {seq_length}")
        
        # Load scaler
        volatility_scaler = joblib.load("models/volatility_scaler.pkl")
        logger.info(f"Loaded volatility scaler with {volatility_scaler.n_features_in_} features")
        
        # Determine input dimension from scaler
        input_dim = volatility_scaler.n_features_in_
        
        # Initialize model (without max_seq_length parameter)
        volatility_model = TransformerModel(input_dim=input_dim).to(device)
        
        # Load model weights
        volatility_model.load_state_dict(torch.load("models/best_transformer_model.pt", map_location=device))
        volatility_model.eval()
        logger.info("Volatility model loaded successfully")
        
        # Load fraud detection model
        try:
            fraud_model = joblib.load("models/improved_fraud_detection_model.pkl")
            # Load optimal threshold
            try:
                fraud_threshold = joblib.load("models/fraud_detection_threshold.pkl")
                logger.info(f"Loaded fraud detection threshold: {fraud_threshold}")
            except:
                fraud_threshold = 0.5
                logger.info("Using default threshold of 0.5")
                
            logger.info("Improved fraud detection model loaded successfully")
        except:
            fraud_model = joblib.load("models/fraud_detection_model.pkl")
            fraud_threshold = 0.5
            logger.info("Original fraud detection model loaded successfully")
        
        return volatility_model, volatility_scaler, fraud_model, fraud_threshold, seq_length, device
    
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, None, None, None

def create_sequences(data, seq_length):
    """Create sequences for time series prediction"""
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    
    return np.array(sequences)

def forecast_volatility(model, scaler, data, seq_length, forecast_horizon=30, device="cpu"):
    """
    Generate recursive forecasts using the transformer model
    
    Args:
        model: Trained transformer model
        scaler: Fitted scaler for the features
        data: DataFrame with features
        seq_length: Length of input sequences
        forecast_horizon: Number of steps to forecast
        device: Device to run model on
        
    Returns:
        DataFrame with forecasted values
    """
    logger.info(f"Generating {forecast_horizon} day forecast...")
    
    # Prepare data for prediction
    numeric_data = data.select_dtypes(include=['number'])
    
    # Ensure we have the right features for the scaler
    if hasattr(scaler, 'feature_names_in_'):
        feature_names = scaler.feature_names_in_
        logger.info(f"Required features: {feature_names}")
        
        # Check if all required features exist
        missing_features = [f for f in feature_names if f not in numeric_data.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features with zeros
            for feature in missing_features:
                numeric_data[feature] = 0
        
        # Reorder columns to match training features
        numeric_data = numeric_data[feature_names]
    
    # Scale the data
    scaled_data = scaler.transform(numeric_data)
    logger.info(f"Scaled data shape: {scaled_data.shape}")
    
    # Create sequences
    sequences = create_sequences(scaled_data, seq_length)
    logger.info(f"Created {len(sequences)} sequences with shape {sequences.shape}")
    
    if len(sequences) == 0:
        logger.error("No sequences could be created. Data may be too short.")
        return None
    
    # Get the last sequence for forecasting
    last_sequence = sequences[-1]
    logger.info(f"Last sequence shape: {last_sequence.shape}")
    
    # Make initial prediction
    model.eval()
    current_sequence = last_sequence.copy()
    forecasted_values = []
    dates = []
    
    # Get the last date from the data
    last_date = data['date'].iloc[-1]
    
    with torch.no_grad():
        for i in range(forecast_horizon):
            # Convert sequence to tensor
            sequence_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
            
            # Debug log
            logger.debug(f"Input sequence shape: {sequence_tensor.shape}")
            
            # Make prediction
            prediction = model(sequence_tensor).cpu().numpy()
            
            # Debug log
            logger.debug(f"Output prediction shape: {prediction.shape}")
            
            # Get the predicted value
            predicted_value = prediction.item() if hasattr(prediction, 'item') else prediction[0]
            
            # Store the prediction
            forecasted_values.append(predicted_value)
            
            # Calculate the next date
            next_date = last_date + timedelta(days=i+1)
            dates.append(next_date)
            
            # Update the sequence for the next prediction
            # Create a new row with the same feature values as the last row
            new_row = current_sequence[-1].copy()
            
            # Update the volatility value (assuming it's the target feature)
            # This might need adjustment based on your specific feature ordering
            volatility_index = list(feature_names).index('volatility') if 'volatility' in feature_names else 0
            new_row[volatility_index] = predicted_value
            
            # Remove the oldest entry and add the new prediction
            current_sequence = np.vstack([current_sequence[1:], new_row])
    
    # Create a DataFrame with the forecasted values
    forecast_df = pd.DataFrame({
        'date': dates,
        'forecasted_volatility': forecasted_values
    })
    
    # Log unique values to check for flat forecasts
    unique_values = np.unique(forecasted_values)
    logger.info(f"Unique forecast values: {unique_values[:10]}{'...' if len(unique_values) > 10 else ''}")
    logger.info(f"Number of unique values: {len(unique_values)}")
    
    return forecast_df

def predict_fraud(model, data, threshold=0.5):
    """
    Predict fraud using the trained model
    
    Args:
        model: Trained fraud detection model
        data: DataFrame with features
        threshold: Probability threshold for fraud classification
        
    Returns:
        DataFrame with fraud predictions
    """
    logger.info(f"Predicting fraud with threshold {threshold}...")
    
    # Prepare data for prediction
    X = data.drop(columns=['fraud_label', 'volatility', 'date'], errors='ignore')
    
    # Make predictions
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    # Add predictions to the data
    result_df = data.copy()
    result_df['fraud_prediction'] = predictions
    result_df['fraud_probability'] = probabilities
    
    # Log prediction statistics
    fraud_count = predictions.sum()
    total_count = len(predictions)
    logger.info(f"Predicted {fraud_count} fraudulent transactions out of {total_count} ({fraud_count/total_count:.2%})")
    
    # If actual fraud labels exist, calculate metrics
    if 'fraud_label' in data.columns:
        y_true = data['fraud_label']
        logger.info("\nFraud Detection Performance:")
        logger.info(classification_report(y_true, predictions))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Fraud Detection Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('outputs/fraud_confusion_matrix.png')
        plt.close()
    
    return result_df

def main():
    """Main function to run the inference pipeline"""
    logger.info("Starting inference pipeline...")
    
    # Load data
    try:
        data_path = "data/crypto_volatility_fraud_dataset.csv"
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Convert date to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Load models
    volatility_model, volatility_scaler, fraud_model, fraud_threshold, seq_length, device = load_models()
    
    if volatility_model is None or volatility_scaler is None:
        logger.error("Failed to load models. Exiting.")
        return
    
    # Generate volatility forecast
    forecast_df = forecast_volatility(
        model=volatility_model,
        scaler=volatility_scaler,
        data=df,
        seq_length=seq_length,
        forecast_horizon=30,
        device=device
    )
    
    if forecast_df is not None:
        logger.info(f"Generated forecast for {len(forecast_df)} days")
        forecast_df.to_csv("outputs/volatility_forecast.csv", index=False)
        logger.info("Forecast saved to outputs/volatility_forecast.csv")
        
        # Plot forecast
        plt.figure(figsize=(12, 6))
        
        # Plot historical volatility
        plt.plot(df['date'], df['volatility'], label='Historical Volatility', color='blue')
        
        # Plot forecasted volatility
        plt.plot(forecast_df['date'], forecast_df['forecasted_volatility'], 
                label='Forecasted Volatility', color='red', linestyle='--')
        
        # Add vertical line to separate historical and forecasted data
        plt.axvline(x=df['date'].iloc[-1], color='gray', linestyle='-', alpha=0.5)
        
        plt.title('Cryptocurrency Volatility Forecast')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("outputs/volatility_forecast.png")
        plt.close()
        logger.info("Forecast plot saved to outputs/volatility_forecast.png")
    
    # Predict fraud
    if fraud_model is not None:
        fraud_results = predict_fraud(fraud_model, df)
        
        if fraud_results is not None:
            logger.info("Fraud prediction completed")
            fraud_results.to_csv("outputs/fraud_predictions.csv", index=False)
            logger.info("Fraud predictions saved to outputs/fraud_predictions.csv")
            
            # Plot fraud probability distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(fraud_results['fraud_probability'], bins=30, kde=True)
            plt.title('Fraud Probability Distribution')
            plt.xlabel('Fraud Probability')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("outputs/fraud_probability_distribution.png")
            plt.close()
            logger.info("Fraud probability distribution plot saved to outputs/fraud_probability_distribution.png")
    
    logger.info("Inference pipeline completed successfully!")

if __name__ == "__main__":
    main()



