import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(n_samples=1000, start_date='2022-01-01'):
    """Generate synthetic cryptocurrency transaction data with fraud labels."""
    np.random.seed(42)
    
    # Generate dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    dates = [start + timedelta(days=i) for i in range(n_samples)]
    dates = [d.strftime('%Y-%m-%d') for d in dates]
    
    # Generate features
    price = np.cumsum(np.random.normal(0, 1, n_samples)) + 30000
    volatility = np.abs(np.random.normal(0, 0.02, n_samples))
    exchange_inflow = np.random.uniform(2000, 7000, n_samples)
    whale_activity = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    active_addresses = np.random.randint(100000, 200000, n_samples)
    miner_inflows = np.random.uniform(300, 700, n_samples)
    on_chain_volume = price * np.random.uniform(0.9, 1.1, n_samples)
    
    # Generate graph embeddings (simplified)
    graph_embs = np.random.normal(0, 1, (n_samples, 5))
    
    # Generate fraud labels (rare event)
    fraud_probs = 0.01 + 0.1 * (volatility > 0.03) + 0.15 * (whale_activity == 1)
    fraud_label = np.random.binomial(1, fraud_probs)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'price': price,
        'volatility': volatility,
        'exchange_inflow': exchange_inflow,
        'whale_activity': whale_activity,
        'active_addresses': active_addresses,
        'miner_inflows': miner_inflows,
        'on_chain_volume': on_chain_volume,
        'fraud_label': fraud_label,
    })
    
    # Add graph embeddings
    for i in range(5):
        df[f'graph_emb_{i+1}'] = graph_embs[:, i]
    
    return df

def augment_existing_data(filepath, n_new_samples=500):
    """Augment existing dataset with new synthetic samples."""
    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)
        last_date = existing_df['date'].iloc[-1]
        start_date = (datetime.strptime(last_date, '%Y-%m-%d') + 
                      timedelta(days=1)).strftime('%Y-%m-%d')
        
        new_df = generate_synthetic_data(n_new_samples, start_date)
        augmented_df = pd.concat([existing_df, new_df], ignore_index=True)
        return augmented_df
    else:
        return generate_synthetic_data(n_new_samples)

if __name__ == "__main__":
    output_path = '../data/crypto_volatility_fraud_dataset.csv'
    
    # Check if file exists and augment or create new
    if os.path.exists(output_path):
        print(f"Augmenting existing dataset at {output_path}")
        df = augment_existing_data(output_path, n_new_samples=500)
    else:
        print(f"Creating new synthetic dataset at {output_path}")
        df = generate_synthetic_data(n_samples=2000)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df)} records and saved to {output_path}")