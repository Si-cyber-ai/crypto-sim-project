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

# Set page config
st.set_page_config(
    page_title="Crypto Fraud & Volatility Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load models and data
@st.cache_resource
def load_models():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    
    # Load fraud detection model
    try:
        fraud_model = joblib.load(os.path.join(models_dir, "fraud_detection_model.pkl"))
    except Exception as e:
        st.error(f"Error loading fraud model: {e}")
        fraud_model = None
    
    # Load volatility model
    try:
        # Import the model architecture
        from model_architecture import TransformerModel
        
        # Load model architecture
        volatility_model_path = os.path.join(models_dir, "volatility_model.pt")
        volatility_scaler_path = os.path.join(models_dir, "volatility_scaler.pkl")
        seq_length_path = os.path.join(models_dir, "sequence_length.pkl")
        
        # Load sequence length
        seq_length = joblib.load(seq_length_path)
        
        # Load scaler
        volatility_scaler = joblib.load(volatility_scaler_path)
        
        # Determine input dimension from scaler
        input_dim = len(volatility_scaler.mean_)
        
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        volatility_model = TransformerModel(input_dim=input_dim).to(device)
        
        # Try to load state dict directly
        try:
            volatility_model.load_state_dict(torch.load(volatility_model_path, map_location=device))
        except Exception as e:
            # If direct loading fails, try to adapt the old state dict
            st.warning("Using model adapter to load old model format")
            from model_adapter import adapt_old_state_dict
            old_state_dict = torch.load(volatility_model_path, map_location=device)
            new_state_dict = adapt_old_state_dict(old_state_dict, volatility_model)
            volatility_model.load_state_dict(new_state_dict, strict=False)
            
        volatility_model.eval()
        
        return fraud_model, volatility_model, volatility_scaler, seq_length
    except Exception as e:
        st.error(f"Error loading volatility model: {e}")
        return fraud_model, None, None, None

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

def forecast_volatility(model, scaler, last_sequence, forecast_horizon, feature_names=None):
    """
    Generate recursive forecasts using the transformer model.
    
    Args:
        model: Trained transformer model
        scaler: Fitted scaler for the features
        last_sequence: Last known sequence of data (scaled)
        forecast_horizon: Number of steps to forecast
        feature_names: Names of features (for debugging)
        
    Returns:
        Array of forecasted volatility values
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Make a copy of the last sequence to avoid modifying the original
    current_sequence = last_sequence.copy()
    forecasts = []
    
    # Convert to tensor for initial prediction
    with torch.no_grad():
        for _ in range(forecast_horizon):
            # Convert current sequence to tensor
            sequence_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
            
            # Make prediction for next step
            next_pred = model(sequence_tensor).cpu().numpy()
            
            # Store the prediction
            if isinstance(next_pred, np.ndarray) and next_pred.size > 0:
                next_value = next_pred.item() if next_pred.size == 1 else next_pred[0]
            else:
                next_value = float(next_pred)
            
            forecasts.append(next_value)
            
            # Update sequence for next prediction by removing oldest entry and adding new prediction
            # Create a new row with the same feature values as the last row, but update the target
            new_row = current_sequence[-1].copy()
            
            # Assuming the target (volatility) is the first feature in the scaled data
            # This might need adjustment based on your specific feature ordering
            new_row[0] = next_value
            
            # Remove the oldest entry and add the new prediction
            current_sequence = np.vstack([current_sequence[1:], new_row])
    
    return np.array(forecasts)

def predict_volatility(model, scaler, data, seq_length):
    """Make volatility predictions using the trained model."""
    try:
        # Get the feature names used during training
        if hasattr(scaler, 'feature_names_in_'):
            train_features = scaler.feature_names_in_
            
            # Ensure all required features exist in the data
            if isinstance(data, pd.DataFrame):
                for feature in train_features:
                    if feature not in data.columns:
                        st.warning(f"Adding missing feature: {feature}")
                        data[feature] = 0
                
                # Reorder columns to match training features
                data = data[train_features]
        
        # Scale the data
        try:
            data_scaled = scaler.transform(data)
        except ValueError as e:
            st.error(f"Scaling error: {e}")
            
            # As a last resort, create a dummy array with the right number of features
            if hasattr(scaler, 'n_features_in_'):
                st.warning(f"Creating dummy data with {scaler.n_features_in_} features")
                dummy_data = np.zeros((len(data), scaler.n_features_in_))
                # Copy available data
                if isinstance(data, pd.DataFrame):
                    for i, col in enumerate(data.columns):
                        if i < scaler.n_features_in_:
                            dummy_data[:, i] = data[col].values
                else:
                    dummy_data[:, :min(data.shape[1], scaler.n_features_in_)] = data[:, :min(data.shape[1], scaler.n_features_in_)]
                
                data_scaled = scaler.transform(dummy_data)
            else:
                raise e
        
        # Create sequences
        sequences = []
        for i in range(len(data_scaled) - seq_length + 1):
            seq = data_scaled[i:i+seq_length]
            sequences.append(seq)
        
        # Convert to tensor
        if len(sequences) == 0:
            st.warning("No sequences could be created. Data may be too short for the sequence length.")
            # Return a small array with a default prediction instead of empty array
            return np.array([[0.02]]), []  # Default volatility value of 2% and empty sequences
            
        X = torch.FloatTensor(np.array(sequences))
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            predictions = model(X).cpu().numpy()
        
        return predictions, sequences
    except Exception as e:
        st.error(f"Error in volatility prediction: {e}")
        import traceback
        st.error(traceback.format_exc())
        # Return a small array with a default prediction instead of empty array
        return np.array([[0.02]]), []  # Default volatility value of 2% and empty sequences

def main():
    st.title("Cryptocurrency Fraud & Volatility Dashboard")
    
    # Load models
    fraud_model, volatility_model, volatility_scaler, seq_length = load_models()
    
    # Sidebar
    st.sidebar.title("Controls")
    data_path = st.sidebar.text_input("Data Path", "../data/crypto_volatility_fraud_dataset.csv")
    
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
        st.subheader("Dataset Information")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Time Period", f"{df['date'].min().date()} to {df['date'].max().date()}")
        col3.metric("Fraud Ratio", f"{df['fraud_label'].mean():.2%}")
        
        # Price and volume chart
        st.subheader("Price and Volume Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'], 
            y=df['price'],
            name='Price',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'], 
            y=df['on_chain_volume'] / df['on_chain_volume'].max() * df['price'].max() * 0.8,
            name='Volume (Scaled)',
            line=dict(color='gray', dash='dot')
        ))
        
        # Add fraud markers
        fraud_df = df[df['fraud_label'] == 1]
        fig.add_trace(go.Scatter(
            x=fraud_df['date'], 
            y=fraud_df['price'],
            mode='markers',
            name='Fraud',
            marker=dict(color='red', size=8, symbol='x')
        ))
        
        fig.update_layout(
            title='Price and Volume with Fraud Markers',
            xaxis_title='Date',
            yaxis_title='Price',
            legend=dict(x=0, y=1),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        corr = numeric_df.corr()
        
        fig = px.imshow(
            corr,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu_r'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Fraud Detection Analysis")
        
        if fraud_model is None:
            st.warning("Fraud detection model not loaded. Please train the model first.")
        else:
            # Feature importance
            if hasattr(fraud_model, 'named_steps') and hasattr(fraud_model.named_steps['classifier'], 'feature_importances_'):
                st.subheader("Feature Importance")
                
                # Get feature names and importances
                feature_names = df.drop(columns=['date', 'fraud_label', 'volatility']).columns
                importances = fraud_model.named_steps['classifier'].feature_importances_
                
                # Sort by importance
                indices = np.argsort(importances)[::-1]
                top_n = min(15, len(feature_names))
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(top_n), importances[indices][:top_n], align='center')
                ax.set_yticks(range(top_n))
                ax.set_yticklabels([feature_names[i] for i in indices[:top_n]])
                ax.set_xlabel('Feature Importance')
                ax.set_title('Top Features for Fraud Detection')
                st.pyplot(fig)
            
            # Fraud distribution over time
            st.subheader("Fraud Distribution Over Time")
            
            # Group by day and count frauds
            df['date_day'] = df['date'].dt.date
            fraud_by_day = df.groupby('date_day')['fraud_label'].mean().reset_index()
            fraud_by_day['fraud_percentage'] = fraud_by_day['fraud_label'] * 100
            
            fig = px.line(
                fraud_by_day, 
                x='date_day', 
                y='fraud_percentage',
                labels={'date_day': 'Date', 'fraud_percentage': 'Fraud Percentage (%)'},
                title='Daily Fraud Percentage'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Fraud vs. legitimate transaction characteristics
            st.subheader("Fraud vs. Legitimate Transaction Characteristics")
            
            # Select features to compare
            compare_features = ['transaction_size', 'wallet_age', 'transaction_fee']
            
            fig = go.Figure()
            for feature in compare_features:
                fig.add_trace(go.Box(
                    y=df[df['fraud_label'] == 0][feature],
                    name=f'Legitimate - {feature}',
                    boxmean=True
                ))
                fig.add_trace(go.Box(
                    y=df[df['fraud_label'] == 1][feature],
                    name=f'Fraud - {feature}',
                    boxmean=True
                ))
            
            fig.update_layout(
                title='Transaction Characteristics Comparison',
                yaxis_title='Value',
                boxmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Volatility Prediction")
        
        if volatility_model is None or volatility_scaler is None:
            st.warning("Volatility prediction model not loaded. Please train the model first.")
        else:
            # Actual vs. Predicted Volatility
            st.subheader("Actual vs. Predicted Volatility")
            
            # Check what features the scaler was trained on
            if hasattr(volatility_scaler, 'feature_names_in_'):
                st.info(f"Model was trained with these features: {', '.join(volatility_scaler.feature_names_in_)}")
            
            # Prepare data for prediction - ensure we're using the right columns
            try:
                # First try: Use all numeric columns except target variables
                X = df.select_dtypes(include=['number']).drop(columns=['fraud_label'], errors='ignore')
                
                # Make predictions
                predictions, sequences = predict_volatility(volatility_model, volatility_scaler, X, seq_length)
                
                # Create a dataframe with predictions
                if len(predictions) > 0:
                    pred_df = pd.DataFrame({
                        'date': df['date'][seq_length-1:seq_length-1+len(predictions)],
                        'actual': df['volatility'][seq_length-1:seq_length-1+len(predictions)],
                        'predicted': predictions.flatten()
                    })
                    
                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=pred_df['date'], 
                        y=pred_df['actual'],
                        name='Actual Volatility',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=pred_df['date'], 
                        y=pred_df['predicted'],
                        name='Predicted Volatility',
                        line=dict(color='red')
                    ))
                    
                    fig.update_layout(
                        title='Actual vs. Predicted Volatility',
                        xaxis_title='Date',
                        yaxis_title='Volatility',
                        legend=dict(x=0, y=1),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No predictions could be generated. The data may be too short for the sequence length.")
            
            except Exception as e:
                st.error(f"Error in volatility prediction: {e}")
                import traceback
                st.error(traceback.format_exc())
            
            # Volatility vs. Price
            st.subheader("Volatility vs. Price")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['date'], 
                y=df['price'],
                name='Price',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=df['date'], 
                y=df['volatility'] * df['price'].max() * 0.2,  # Scale for visibility
                name='Volatility (Scaled)',
                line=dict(color='orange')
            ))
            
            fig.update_layout(
                title='Price and Volatility Over Time',
                xaxis_title='Date',
                yaxis_title='Value',
                legend=dict(x=0, y=1),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volatility Forecast
            st.subheader("Volatility Forecast")

            # Get the latest data for forecasting
            forecast_days = st.slider("Forecast Days", 1, 30, 7)

            st.info("Note: This forecast uses recursive prediction where each prediction becomes input for the next step.")

            # Create a simple forecast by repeating the last prediction
            last_date = df['date'].max()
            forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days+1)]

            # Check if we have predictions before accessing them
            if 'X_volatility' in locals() and volatility_model is not None and volatility_scaler is not None:
                try:
                    # Make predictions and get sequences
                    predictions, sequences = predict_volatility(volatility_model, volatility_scaler, X_volatility, seq_length)
                    
                    # Check if we have predictions and sequences
                    if len(predictions) > 0 and len(sequences) > 0:
                        # Get the last sequence for forecasting
                        last_sequence = sequences[-1]
                        
                        # Generate recursive forecasts
                        forecast_values = forecast_volatility(
                            volatility_model, 
                            volatility_scaler, 
                            last_sequence, 
                            forecast_days,
                            feature_names=X_volatility.columns if isinstance(X_volatility, pd.DataFrame) else None
                        )
                        
                        # Create historical dataframe
                        pred_df = pd.DataFrame({
                            'date': df['date'][seq_length-1:seq_length-1+len(predictions)],
                            'actual': df['volatility'][seq_length-1:seq_length-1+len(predictions)],
                            'predicted': predictions.flatten()
                        })
                        
                        # Combine historical and forecast data
                        historical_df = pred_df[['date', 'predicted']].rename(columns={'predicted': 'volatility'})
                        historical_df['type'] = 'Historical'
                        
                        forecast_df = pd.DataFrame({
                            'date': forecast_dates,
                            'volatility': forecast_values,
                            'type': 'Forecast'
                        })
                        
                        combined_df = pd.concat([historical_df.tail(30), forecast_df])
                        
                        # Plot
                        fig = px.line(
                            combined_df, 
                            x='date', 
                            y='volatility',
                            color='type',
                            title='Volatility Forecast',
                            color_discrete_map={'Historical': 'blue', 'Forecast': 'red'}
                        )
                        
                        fig.add_vrect(
                            x0=last_date,
                            x1=forecast_dates[-1],
                            fillcolor="lightgray",
                            opacity=0.3,
                            layer="below",
                            line_width=0
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        raise ValueError("No predictions or sequences available")
                except Exception as e:
                    st.error(f"Error generating forecast: {e}")
                    import traceback
                    st.error(traceback.format_exc())
                    # Fallback to simple forecast with some randomness
                    st.warning("Using fallback forecast due to error")
                    
                    # Create a placeholder forecast with random values
                    import random
                    placeholder_volatility = 0.02  # A reasonable default volatility value
                    random_volatility = [placeholder_volatility * (1 + 0.05 * (i/forecast_days) + (random.random() - 0.5) * 0.1) for i in range(forecast_days)]
                    
                    # Create a dataframe for the placeholder forecast
                    forecast_df = pd.DataFrame({
                        'date': forecast_dates,
                        'volatility': random_volatility,
                        'type': 'Placeholder Forecast'
                    })
                    
                    # Plot the placeholder forecast
                    fig = px.line(
                        forecast_df, 
                        x='date', 
                        y='volatility',
                        title='Placeholder Volatility Forecast',
                        color_discrete_map={'Placeholder Forecast': 'gray'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # If no predictions are available, show a placeholder forecast
                st.warning("No predictions available for forecasting. Showing placeholder forecast.")
                
                # Create a placeholder forecast with random values
                import random
                placeholder_volatility = 0.02  # A reasonable default volatility value
                random_volatility = [placeholder_volatility * (1 + 0.05 * (i/forecast_days) + (random.random() - 0.5) * 0.1) for i in range(forecast_days)]
                
                # Create a dataframe for the placeholder forecast
                forecast_df = pd.DataFrame({
                    'date': forecast_dates,
                    'volatility': random_volatility,
                    'type': 'Placeholder Forecast'
                })
                
                # Plot the placeholder forecast
                fig = px.line(
                    forecast_df, 
                    x='date', 
                    y='volatility',
                    title='Placeholder Volatility Forecast (No model predictions available)',
                    color_discrete_map={'Placeholder Forecast': 'gray'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show a message explaining how to fix this
                st.info("To get actual forecasts, please ensure your data has all required features and is long enough for sequence-based prediction.")

if __name__ == "__main__":
    main()







