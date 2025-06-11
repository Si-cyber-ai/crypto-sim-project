import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def fix_missing_model_files():
    """Create missing model files with default values."""
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Check for volatility_scaler.pkl
    scaler_path = os.path.join(models_dir, 'volatility_scaler.pkl')
    if not os.path.exists(scaler_path):
        print("Missing volatility_scaler.pkl - Creating a default scaler...")
        # Create a default scaler with reasonable values
        # This is just a placeholder - retraining is recommended
        scaler = StandardScaler()
        # Fit with some dummy data
        dummy_data = np.random.randn(100, 16)  # 16 features to match expected count
        scaler.fit(dummy_data)
        # Set feature names to match expected features
        feature_names = [
            'price', 'volatility', 'on_chain_volume', 'network_hashrate', 
            'active_addresses', 'exchange_inflow', 'exchange_outflow', 
            'wallet_age', 'transaction_size', 'transaction_fee', 
            'day_of_week', 'month', 'price_change', 'volume_change', 
            'volume_to_price_ratio', 'inflow_to_volume_ratio'
        ]
        scaler.feature_names_in_ = np.array(feature_names)
        joblib.dump(scaler, scaler_path)
        print("✅ Created a default volatility_scaler.pkl")
        print("⚠️ Warning: This is a placeholder. Retraining is recommended.")
    
    # Check for sequence_length.pkl
    seq_length_path = os.path.join(models_dir, 'sequence_length.pkl')
    if not os.path.exists(seq_length_path):
        print("Missing sequence_length.pkl - Creating with default value...")
        joblib.dump(5, seq_length_path)
        print("✅ Created sequence_length.pkl with default value of 5")

if __name__ == "__main__":
    fix_missing_model_files()
    print("✅ All missing files have been fixed!")
