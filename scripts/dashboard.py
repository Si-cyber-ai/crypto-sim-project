import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import torch
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, classification_report

# Set page config
st.set_page_config(
    page_title="Crypto Fraud & Volatility Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load models and data
@st.cache_resource
def load_models():
    """Load all required models and scalers."""
    st.sidebar.subheader("Model Status")
    
    # Load fraud detection model
    try:
        # First try to load improved model, then fall back to original
        try:
            fraud_model = joblib.load("models/improved_fraud_detection_model.pkl")
            st.sidebar.success("✅ Using improved fraud detection model")
        except:
            fraud_model = joblib.load("models/fraud_detection_model.pkl")
            st.sidebar.info("ℹ️ Using original fraud detection model")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading fraud model: {e}")
        fraud_model = None
    
    # Load volatility prediction model
    try:
        # Load sequence length
        seq_length = joblib.load("models/sequence_length.pkl")
        
        # Load scaler
        volatility_scaler = joblib.load("models/volatility_scaler.pkl")
        
        # Determine input dimension from scaler
        input_dim = volatility_scaler.n_features_in_
        
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        volatility_model = TransformerModel(input_dim=input_dim).to(device)
        
        # Load model weights
        volatility_model.load_state_dict(torch.load("models/best_transformer_model.pt", map_location=device))
        volatility_model.eval()
        
        st.sidebar.success("✅ Volatility model loaded successfully")
        
        # Log model details
        if st.sidebar.checkbox("Show model details"):
            st.sidebar.code(f"Device: {device}\nInput dim: {input_dim}\nSeq length: {seq_length}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading volatility model: {e}")
        volatility_model = None
        volatility_scaler = None
        seq_length = None
    
    return fraud_model, volatility_model, volatility_scaler, seq_length

@st.cache_data
def load_data(data_path="../data/crypto_volatility_fraud_dataset.csv"):
    try:
        df = pd.read_csv(data_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Extract date features consistently
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['date_day'] = df['date'].dt.day
        
        # Ensure all required features exist
        required_features = [
            'price', 'volatility', 'on_chain_volume', 'network_hashrate', 
            'active_addresses', 'exchange_inflow', 'exchange_outflow', 
            'wallet_age', 'transaction_size', 'transaction_fee', 
            'day_of_week', 'month'
        ]
        
        # Add missing features with default values
        for feature in required_features:
            if feature not in df.columns:
                st.warning(f"Adding missing feature: {feature}")
                df[feature] = 0
        
        # Create derived features
        df['price_change'] = df['price'].pct_change().fillna(0)
        df['volume_change'] = df['on_chain_volume'].pct_change().fillna(0)
        df['volume_to_price_ratio'] = df['on_chain_volume'] / df['price'].replace(0, 1)
        df['inflow_to_volume_ratio'] = df['exchange_inflow'] / df['on_chain_volume'].replace(0, 1)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_sequences(data, seq_length):
    """Create sequences for time series prediction."""
    xs = []
    for i in range(len(data) - seq_length + 1):
        x = data[i:i+seq_length]
        xs.append(x)
    return np.array(xs)

def forecast_volatility(model, scaler, data, seq_length, forecast_horizon=30):
    """Generate volatility forecast using the transformer model."""
    try:
        # Prepare data for prediction
        numeric_data = data.select_dtypes(include=['number'])
        
        # Ensure we have the right features for the scaler
        if hasattr(scaler, 'feature_names_in_'):
            feature_names = scaler.feature_names_in_
            
            # Check if all required features exist
            missing_features = [f for f in feature_names if f not in numeric_data.columns]
            if missing_features:
                st.warning(f"Missing features: {', '.join(missing_features)}")
                # Add missing features with zeros
                for feature in missing_features:
                    numeric_data[feature] = 0
            
            # Reorder columns to match training features
            numeric_data = numeric_data[feature_names]
        
        # Scale the data
        scaled_data = scaler.transform(numeric_data)
        st.write(f"Scaled data shape: {scaled_data.shape}")
        
        # Create sequences
        sequences = []
        for i in range(len(scaled_data) - seq_length + 1):
            seq = scaled_data[i:i+seq_length]
            sequences.append(seq)
        
        sequences = np.array(sequences)
        st.write(f"Created {len(sequences)} sequences with shape {sequences.shape}")
        
        if len(sequences) == 0:
            st.error("No sequences could be created. Data may be too short.")
            return None
        
        # Get the last sequence for forecasting
        last_sequence = sequences[-1]
        
        # Make initial prediction
        device = next(model.parameters()).device
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
                
                # Make prediction
                prediction = model(sequence_tensor).cpu().numpy()
                
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
        st.write(f"Number of unique forecast values: {len(unique_values)}")
        
        return forecast_df
    
    except Exception as e:
        st.error(f"Error in forecast_volatility: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def predict_fraud(model, data):
    """Predict fraud using the trained model."""
    try:
        # Prepare data for prediction
        X = data.drop(columns=['fraud_label', 'volatility', 'date'], errors='ignore')
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        # Add predictions to the data
        result_df = data.copy()
        result_df['fraud_prediction'] = predictions
        result_df['fraud_probability'] = probabilities
        
        # Log prediction statistics
        fraud_count = predictions.sum()
        total_count = len(predictions)
        st.write(f"Predicted {fraud_count} fraudulent transactions out of {total_count} ({fraud_count/total_count:.2%})")
        
        # If actual fraud labels exist, calculate metrics
        if 'fraud_label' in data.columns:
            y_true = data['fraud_label']
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, predictions)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.title('Fraud Detection Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            st.pyplot(fig)
            
            # Display classification report
            report = classification_report(y_true, predictions, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.write("Classification Report:")
            st.dataframe(report_df)
        
        return result_df
    
    except Exception as e:
        st.error(f"Error in predict_fraud: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def main():
    st.title("Cryptocurrency Fraud & Volatility Dashboard")
    
    # Load models
    fraud_model, volatility_model, volatility_scaler, seq_length = load_models()
    
    # Sidebar
    st.sidebar.title("Controls")
    data_path = st.sidebar.text_input("Data Path", "data/crypto_volatility_fraud_dataset.csv")
    
    # Load data
    df = load_data(data_path)
    
    if df is None:
        st.warning("No data found. Please generate data first using the generate_crypto_data.py script.")
        st.code("python scripts/generate_crypto_data.py --samples 5000 --output data/crypto_volatility_fraud_dataset.csv")
        return
    
    # Dashboard tabs
    tab1, tab2, tab3 = st.tabs(["Overview", "Fraud Detection", "Volatility Prediction"])
    
    with tab1:
        st.header("Dataset Overview")
        
        # Display basic statistics
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        st.subheader("Dataset Statistics")
        st.write(f"Total records: {len(df)}")
        st.write(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        if 'fraud_label' in df.columns:
            fraud_count = df['fraud_label'].sum()
            st.write(f"Fraud transactions: {fraud_count} ({fraud_count/len(df):.2%})")
        
        # Plot price and volatility
        st.subheader("Price and Volatility")
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color='tab:blue')
        ax1.plot(df['date'], df['price'], color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Volatility', color='tab:red')
        ax2.plot(df['date'], df['volatility'], color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        fig.tight_layout()
        st.pyplot(fig)
        
        # Plot on-chain volume
        st.subheader("On-Chain Volume")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['date'], df['on_chain_volume'])
        ax.set_xlabel('Date')
        ax.set_ylabel('Volume')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with tab2:
        st.header("Fraud Detection")
        
        if fraud_model is None:
            st.error("Fraud detection model not loaded. Please check the model files.")
            return
        
        # Predict fraud
        fraud_results = predict_fraud(fraud_model, df)
        
        if fraud_results is not None:
            # Display fraud predictions
            st.subheader("Fraud Predictions")
            
            # Filter to show only fraudulent transactions
            show_fraudulent = st.checkbox("Show only fraudulent transactions")
            if show_fraudulent:
                filtered_df = fraud_results[fraud_results['fraud_prediction'] == 1]
            else:
                filtered_df = fraud_results
            
            st.dataframe(filtered_df[['date', 'price', 'transaction_size', 'fraud_prediction', 'fraud_probability']])
            
            # Plot fraud probability distribution
            st.subheader("Fraud Probability Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(fraud_results['fraud_probability'], bins=30, kde=True, ax=ax)
            ax.set_xlabel('Fraud Probability')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Save predictions
            if st.button("Save Fraud Predictions"):
                fraud_results.to_csv("outputs/fraud_predictions.csv", index=False)
                st.success("Fraud predictions saved to outputs/fraud_predictions.csv")
    
    with tab3:
        st.header("Volatility Prediction")
        
        if volatility_model is None or volatility_scaler is None:
            st.error("Volatility prediction model not loaded. Please check the model files.")
            return
        
        # Forecast settings
        st.subheader("Forecast Settings")
        forecast_horizon = st.slider("Forecast Horizon (Days)", 1, 60, 30)
        
        # Generate forecast
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                forecast_df = forecast_volatility(
                    model=volatility_model,
                    scaler=volatility_scaler,
                    data=df,
                    seq_length=seq_length,
                    forecast_horizon=forecast_horizon
                )
                
                if forecast_df is not None:
                    # Display forecast
                    st.subheader("Volatility Forecast")
                    st.dataframe(forecast_df)
                    
                    # Plot forecast
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot historical volatility
                    ax.plot(df['date'], df['volatility'], label='Historical Volatility', color='blue')
                    
                    # Plot forecasted volatility
                    ax.plot(forecast_df['date'], forecast_df['forecasted_volatility'], 
                            label='Forecasted Volatility', color='red', linestyle='--')
                    
                    # Add vertical line to separate historical and forecasted data
                    ax.axvline(x=df['date'].iloc[-1], color='gray', linestyle='-', alpha=0.5)
                    
                    ax.set_title('Cryptocurrency Volatility Forecast')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Volatility')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    st.pyplot(fig)
                    
                    # Save forecast
                    if st.button("Save Forecast"):
                        forecast_df.to_csv("outputs/forecast_results.csv", index=False)
                        st.success("Forecast saved to outputs/forecast_results.csv")
                else:
                    st.error("Failed to generate forecast. Please check the logs for details.")

if __name__ == "__main__":
    main()


