import os
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve, 
    roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score
)
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/model_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model-evaluation")

# Create output directories if they don't exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def load_data(data_path="data/crypto_volatility_fraud_dataset.csv"):
    """Load and preprocess the dataset."""
    logger.info(f"Loading data from {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Convert date to datetime if it exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def evaluate_fraud_model(model, data):
    """Evaluate the fraud detection model."""
    logger.info("Evaluating fraud detection model...")
    
    if 'fraud_label' not in data.columns:
        logger.error("No fraud_label column found in the dataset")
        return None
    
    # Prepare data for evaluation
    X = data.drop(columns=['fraud_label', 'volatility', 'date'], errors='ignore')
    y_true = data['fraud_label']
    
    # Make predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Extract key metrics
    accuracy = report['accuracy']
    precision = report['1']['precision']
    recall = report['1']['recall']
    f1 = report['1']['f1-score']
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Fraud Detection Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('outputs/fraud_model_evaluation_cm.png')
    plt.close()
    
    # Create ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Fraud Detection ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('outputs/fraud_model_evaluation_roc.png')
    plt.close()
    
    # Create precision-recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Fraud Detection Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('outputs/fraud_model_evaluation_pr.png')
    plt.close()
    
    # Return metrics
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

def create_sequences(data, target, seq_length):
    """Create sequences for time series prediction."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

def evaluate_volatility_model(model, scaler, data, seq_length, device):
    """Evaluate the volatility prediction model."""
    logger.info("Evaluating volatility prediction model...")
    
    if 'volatility' not in data.columns:
        logger.error("No volatility column found in the dataset")
        return None
    
    # Prepare data for evaluation
    numeric_data = data.select_dtypes(include=['number'])
    
    # Ensure we have the right features for the scaler
    if hasattr(scaler, 'feature_names_in_'):
        feature_names = scaler.feature_names_in_
        
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
    
    # Create sequences
    X, y_true = create_sequences(scaled_data, data['volatility'].values, seq_length)
    
    if len(X) == 0:
        logger.error("No sequences could be created. Data may be too short.")
        return None
    
    # Make predictions
    model.eval()
    y_pred = []
    
    with torch.no_grad():
        for i in range(0, len(X), 64):  # Process in batches to avoid memory issues
            batch_X = X[i:i+64]
            batch_tensor = torch.FloatTensor(batch_X).to(device)
            batch_pred = model(batch_tensor).cpu().numpy()
            y_pred.extend(batch_pred)
    
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    logger.info(f"Mean Squared Error: {mse:.6f}")
    logger.info(f"Root Mean Squared Error: {rmse:.6f}")
    logger.info(f"Mean Absolute Error: {mae:.6f}")
    logger.info(f"R² Score: {r2:.6f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual Volatility', alpha=0.7)
    plt.plot(y_pred, label='Predicted Volatility', alpha=0.7)
    plt.title('Volatility Prediction: Actual vs Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/volatility_model_evaluation_prediction.png')
    plt.close()
    
    # Plot scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.title('Volatility Prediction: Actual vs Predicted')
    plt.xlabel('Actual Volatility')
    plt.ylabel('Predicted Volatility')
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/volatility_model_evaluation_scatter.png')
    plt.close()
    
    # Return metrics
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def main():
    """Main function to evaluate models."""
    logger.info("Starting model evaluation...")
    
    # Load data
    df = load_data()
    if df is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Load models
    try:
        # Load fraud detection model
        try:
            fraud_model = joblib.load("models/improved_fraud_detection_model.pkl")
            logger.info("Loaded improved fraud detection model")
        except:
            fraud_model = joblib.load("models/fraud_detection_model.pkl")
            logger.info("Loaded original fraud detection model")
        
        # Load volatility prediction model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_length = joblib.load("models/sequence_length.pkl")
        volatility_scaler = joblib.load("models/volatility_scaler.pkl")
        
        # Determine input dimension from scaler
        input_dim = volatility_scaler.n_features_in_
        
        # Load TransformerModel class from inference.py
        from inference import TransformerModel
        
        # Initialize model
        volatility_model = TransformerModel(input_dim=input_dim).to(device)
        
        # Load model weights
        volatility_model.load_state_dict(torch.load("models/best_transformer_model.pt", map_location=device))
        volatility_model.eval()
        logger.info("Loaded volatility prediction model")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Evaluate fraud detection model
    fraud_metrics = evaluate_fraud_model(fraud_model, df)
    
    # Evaluate volatility prediction model
    volatility_metrics = evaluate_volatility_model(volatility_model, volatility_scaler, df, seq_length, device)
    
    # Save evaluation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"outputs/model_evaluation_{timestamp}.txt", "w") as f:
        f.write("=== Fraud Detection Model Evaluation ===\n")
        if fraud_metrics:
            f.write(f"Accuracy: {fraud_metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {fraud_metrics['precision']:.4f}\n")
            f.write(f"Recall: {fraud_metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {fraud_metrics['f1']:.4f}\n")
            f.write(f"ROC AUC: {fraud_metrics['roc_auc']:.4f}\n")
            f.write(f"PR AUC: {fraud_metrics['pr_auc']:.4f}\n")
        else:
            f.write("Evaluation failed\n")
        
        f.write("\n=== Volatility Prediction Model Evaluation ===\n")
        if volatility_metrics:
            f.write(f"Mean Squared Error: {volatility_metrics['mse']:.6f}\n")
            f.write(f"Root Mean Squared Error: {volatility_metrics['rmse']:.6f}\n")
            f.write(f"Mean Absolute Error: {volatility_metrics['mae']:.6f}\n")
            f.write(f"R² Score: {volatility_metrics['r2']:.6f}\n")
        else:
            f.write("Evaluation failed\n")
    
    logger.info(f"Evaluation results saved to outputs/model_evaluation_{timestamp}.txt")
    logger.info("Model evaluation completed successfully!")

if __name__ == "__main__":
    main()