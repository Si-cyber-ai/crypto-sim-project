import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path to import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.crypto_ml_pipeline import TransformerModel, wavelet_features

# Set page config
st.set_page_config(
    page_title="Crypto Fraud & Volatility Prediction",
    page_icon="📊",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    
    # Import the model architecture
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from model_architecture import TransformerModel
    
    # Load fraud detection model
    try:
        rf_model = joblib.load(os.path.join(models_dir, "fraud_detection_model.pkl"))
    except Exception as e:
        st.error(f"Error loading fraud model: {e}")
        rf_model = None
    
    # Load volatility model
    try:
        # Load model architecture
        volatility_model_path = os.path.join(models_dir, "volatility_model.pt")
        volatility_scaler_path = os.path.join(models_dir, "volatility_scaler.pkl")
        seq_length_path = os.path.join(models_dir, "sequence_length.pkl")
        
        # Load scaler and sequence length
        scaler = joblib.load(volatility_scaler_path)
        seq_length = joblib.load(seq_length_path)
        
        # Determine input dimension from scaler
        input_dim = len(scaler.mean_)
        
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transformer_model = TransformerModel(input_dim=input_dim).to(device)
        transformer_model.load_state_dict(torch.load(volatility_model_path, map_location=device))
        transformer_model.eval()
    except Exception as e:
        st.error(f"Error loading volatility model: {e}")
        transformer_model = None
        scaler = None
        seq_length = None
    
    return rf_model, transformer_model, scaler, seq_length

def preprocess_data(data):
    """Preprocess input data for prediction."""
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Fill missing values
    df = df.fillna(0)
    
    # Create additional features if needed
    if 'price' in df.columns and 'on_chain_volume' in df.columns:
        df['price_change'] = df['price'].pct_change().fillna(0)
        df['volume_change'] = df['on_chain_volume'].pct_change().fillna(0)
        df['volume_to_price_ratio'] = df['on_chain_volume'] / df['price'].replace(0, 1)
        
        if 'exchange_inflow' in df.columns:
            df['inflow_to_volume_ratio'] = df['exchange_inflow'] / df['on_chain_volume'].replace(0, 1)
    
        # Extract wavelet features from price and volume
        try:
            price_wavelets = np.array([wavelet_features(df['price'].values)])
            volume_wavelets = np.array([wavelet_features(df['on_chain_volume'].values)])
            
            # Create wavelet feature columns
            for i in range(price_wavelets.shape[1]):
                df[f'price_wavelet_{i}'] = price_wavelets[0, i]
            for i in range(volume_wavelets.shape[1]):
                df[f'volume_wavelet_{i}'] = volume_wavelets[0, i]
        except Exception as e:
            st.warning(f"Error extracting wavelet features: {e}")
    
    return df

def create_sequences(data, seq_length=5):
    """Create sequences for time series prediction."""
    xs = []
    for i in range(len(data) - seq_length + 1):
        x = data[i:i+seq_length]
        xs.append(x)
    return np.array(xs)

def predict_fraud(model, data):
    """Predict fraud using the trained model."""
    if model is None:
        return None
    
    try:
        # Make predictions
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]
        return predictions, probabilities
    except Exception as e:
        st.error(f"Error predicting fraud: {e}")
        return None, None

def predict_volatility(model, scaler, data, seq_length=5):
    """Predict volatility using the trained model."""
    if model is None or scaler is None:
        return None
    
    try:
        # Scale data
        data_scaled = scaler.transform(data)
        
        # Create sequences
        sequences = create_sequences(data_scaled, seq_length)
        
        # Convert to tensor
        device = next(model.parameters()).device
        sequences_tensor = torch.FloatTensor(sequences).to(device)
        
        # Make predictions
        with torch.no_grad():
            predictions = model(sequences_tensor).cpu().numpy()
        
        return predictions
    except Exception as e:
        st.error(f"Error predicting volatility: {e}")
        return None

def main():
    st.title("Cryptocurrency Fraud Detection & Volatility Prediction")
    
    # Load models
    rf_model, transformer_model, scaler, seq_length = load_models()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload Data", "Sample Prediction", "About"])
    
    if page == "Upload Data":
        st.header("Upload Transaction Data")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(data.head())
            
            # Preprocess data
            processed_data = preprocess_data(data)
            
            # Make predictions
            if rf_model is not None:
                # Prepare data for fraud detection
                X_fraud = processed_data.drop(columns=['fraud_label', 'volatility'], errors='ignore')
                
                # Predict fraud
                fraud_predictions, fraud_probabilities = predict_fraud(rf_model, X_fraud)
                
                if fraud_predictions is not None:
                    # Add predictions to dataframe
                    data['fraud_prediction'] = fraud_predictions
                    data['fraud_probability'] = fraud_probabilities
                    
                    # Display results
                    st.subheader("Fraud Detection Results")
                    st.write(f"Detected {fraud_predictions.sum()} potential fraudulent transactions out of {len(data)}")
                    
                    # Plot fraud probability distribution
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(fraud_probabilities, bins=50, kde=True, ax=ax)
                    ax.set_title('Fraud Probability Distribution')
                    ax.set_xlabel('Fraud Probability')
                    ax.set_ylabel('Count')
                    st.pyplot(fig)
                    
                    # Show transactions with high fraud probability
                    st.subheader("High-Risk Transactions")
                    high_risk = data[data['fraud_probability'] > 0.7].sort_values('fraud_probability', ascending=False)
                    if len(high_risk) > 0:
                        st.dataframe(high_risk)
                    else:
                        st.write("No high-risk transactions detected.")
            
            # Predict volatility if model is available
            if transformer_model is not None and scaler is not None:
                # Prepare data for volatility prediction
                X_volatility = processed_data.drop(columns=['fraud_label', 'volatility'], errors='ignore')
                
                # Predict volatility
                volatility_predictions = predict_volatility(transformer_model, scaler, X_volatility, seq_length)
                
                if volatility_predictions is not None:
                    # Add predictions to dataframe
                    volatility_df = data.iloc[seq_length-1:].copy()  # Adjust based on sequence length
                    volatility_df['predicted_volatility'] = volatility_predictions
                    
                    # Display results
                    st.subheader("Volatility Prediction Results")
                    
                    # Plot actual vs predicted volatility if actual is available
                    if 'volatility' in volatility_df.columns:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(volatility_df.index, volatility_df['volatility'], label='Actual')
                        ax.plot(volatility_df.index, volatility_df['predicted_volatility'], label='Predicted')
                        ax.set_title('Actual vs Predicted Volatility')
                        ax.set_xlabel('Sample Index')
                        ax.set_ylabel('Volatility')
                        ax.legend()
                        st.pyplot(fig)
                    else:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(volatility_df.index, volatility_df['predicted_volatility'])
                        ax.set_title('Predicted Volatility')
                        ax.set_xlabel('Sample Index')
                        ax.set_ylabel('Volatility')
                        st.pyplot(fig)
    
    elif page == "Sample Prediction":
        st.header("Sample Transaction Prediction")
        
        # Create form for user input
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                price = st.number_input("Price (USD)", min_value=100.0, max_value=100000.0, value=10000.0)
                volume = st.number_input("On-chain Volume", min_value=1000.0, value=1000000.0)
                wallet_age = st.number_input("Wallet Age (days)", min_value=0.0, max_value=1000.0, value=100.0)
                tx_size = st.number_input("Transaction Size", min_value=0.0, max_value=10.0, value=0.5)
                tx_fee = st.number_input("Transaction Fee", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
            
            with col2:
                hashrate = st.number_input("Network Hashrate", min_value=1000.0, value=1000000.0)
                active_addr = st.number_input("Active Addresses", min_value=100.0, value=10000.0)
                exchange_inflow = st.number_input("Exchange Inflow", min_value=0.0, value=400000.0)
                exchange_outflow = st.number_input("Exchange Outflow", min_value=0.0, value=380000.0)
            
            submit_button = st.form_submit_button("Predict")
        
        if submit_button:
            # Create sample data
            sample_data = pd.DataFrame({
                'price': [price],
                'on_chain_volume': [volume],
                'wallet_age': [wallet_age],
                'transaction_size': [tx_size],
                'transaction_fee': [tx_fee],
                'network_hashrate': [hashrate],
                'active_addresses': [active_addr],
                'exchange_inflow': [exchange_inflow],
                'exchange_outflow': [exchange_outflow]
            })
            
            # Add historical data for sequence-based prediction
            # In a real app, you would use actual historical data
            historical_data = pd.DataFrame({
                'price': [price * 0.98, price * 0.99, price * 0.995, price],
                'on_chain_volume': [volume * 0.97, volume * 0.98, volume * 0.99, volume],
                'wallet_age': [wallet_age] * 4,
                'transaction_size': [tx_size] * 4,
                'transaction_fee': [tx_fee] * 4,
                'network_hashrate': [hashrate] * 4,
                'active_addresses': [active_addr] * 4,
                'exchange_inflow': [exchange_inflow] * 4,
                'exchange_outflow': [exchange_outflow] * 4
            })
            
            # Combine for preprocessing
            combined_data = pd.concat([historical_data, sample_data]).reset_index(drop=True)
            
            # Preprocess
            processed_data = preprocess_data(combined_data)
            
            # Make predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Fraud Detection")
                if rf_model is not None:
                    # Predict fraud for the last row (current transaction)
                    X_fraud = processed_data.iloc[-1:].drop(columns=['fraud_label', 'volatility'], errors='ignore')
                    fraud_pred, fraud_prob = predict_fraud(rf_model, X_fraud)
                    
                    if fraud_pred is not None:
                        # Display result
                        fraud_status = "Fraudulent" if fraud_pred[0] == 1 else "Legitimate"
                        fraud_color = "red" if fraud_pred[0] == 1 else "green"
                        
                        st.markdown(f"<h3 style='color: {fraud_color};'>Transaction is: {fraud_status}</h3>", unsafe_allow_html=True)
                        st.metric("Fraud Probability", f"{fraud_prob[0]:.2%}")
                        
                        # Gauge chart for fraud probability
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = fraud_prob[0] * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Fraud Risk"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "green"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ]
                            }
                        ))
                        st.plotly_chart(fig)
                else:
                    st.warning("Fraud detection model not loaded.")
            
            with col2:
                st.subheader("Volatility Prediction")
                if transformer_model is not None and scaler is not None:
                    # Predict volatility
                    X_volatility = processed_data.drop(columns=['fraud_label', 'volatility'], errors='ignore')
                    volatility_pred = predict_volatility(transformer_model, scaler, X_volatility, seq_length)
                    
                    if volatility_pred is not None:
                        # Display result
                        predicted_volatility = volatility_pred[-1][0]
                        
                        # Categorize volatility
                        if predicted_volatility < 0.01:
                            volatility_category = "Low"
                            volatility_color = "green"
                        elif predicted_volatility < 0.03:
                            volatility_category = "Medium"
                            volatility_color = "orange"
                        else:
                            volatility_category = "High"
                            volatility_color = "red"
                        
                        st.markdown(f"<h3 style='color: {volatility_color};'>Predicted Volatility: {volatility_category}</h3>", unsafe_allow_html=True)
                        st.metric("Volatility Value", f"{predicted_volatility:.4f}")
                        
                        # Gauge chart for volatility
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = predicted_volatility * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Volatility (%)"},
                            gauge = {
                                'axis': {'range': [0, 5]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 1], 'color': "green"},
                                    {'range': [1, 3], 'color': "yellow"},
                                    {'range': [3, 5], 'color': "red"}
                                ]
                            }
                        ))
                        st.plotly_chart(fig)
                else:
                    st.warning("Volatility prediction model not loaded.")
    
    else:  # About page
        st.header("About This Project")
        st.write("""
        ## Integrating Wavelet Decomposition, Non-Local Graph Neural Networks, and Transformer Models for On-Chain Cryptocurrency Volatility Prediction and Fraud Detection
        
        This project combines advanced techniques to analyze cryptocurrency transaction data:
        
        1. **Wavelet Decomposition**: Extracts time-frequency features from price and volume data
        2. **Random Forest**: Detects fraudulent transactions with high accuracy
        3. **Transformer Model**: Predicts cryptocurrency volatility using sequential data
        
        ### How It Works
        
        - The system processes transaction data to extract meaningful features
        - Wavelet decomposition captures multi-scale patterns in time series data
        - The Random Forest model identifies potential fraud based on transaction characteristics
        - The Transformer model forecasts volatility using attention mechanisms
        
        ### Applications
        
        - Risk management for cryptocurrency exchanges
        - Regulatory compliance and fraud prevention
        - Investment strategy optimization
        - Market monitoring and early warning systems
        """)
        
        st.info("This is a demonstration project showcasing the integration of machine learning and deep learning techniques for cryptocurrency analysis.")

if __name__ == "__main__":
    main()



