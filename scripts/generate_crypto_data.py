import numpy as np
import pandas as pd
import argparse
import os
from datetime import datetime, timedelta
import random

def generate_crypto_data(samples=5000, fraud_ratio=0.05, start_date=None):
    """
    Generate synthetic cryptocurrency transaction data.
    
    Parameters:
    -----------
    samples : int
        Number of data points to generate
    fraud_ratio : float
        Ratio of fraudulent transactions
    start_date : datetime
        Starting date for the time series
        
    Returns:
    --------
    pandas.DataFrame
        Synthetic cryptocurrency dataset
    """
    if start_date is None:
        start_date = datetime(2022, 1, 1)
    
    # Generate dates
    dates = [start_date + timedelta(hours=i*4) for i in range(samples)]
    
    # Base price and volatility
    base_price = 10000
    base_volatility = 0.02
    
    # Generate price with random walk and some seasonality
    prices = [base_price]
    volatilities = [base_volatility]
    
    # Add some market regimes
    regime_changes = np.random.choice(range(1, samples-1), size=5, replace=False)
    regimes = np.zeros(samples)
    
    for change in regime_changes:
        regimes[change:] += np.random.choice([-1, 1]) * np.random.uniform(0.5, 2.0)
    
    # Generate price and volatility
    for i in range(1, samples):
        # Volatility follows a mean-reverting process
        volatility_change = 0.001 * np.random.randn() + 0.0002 * (base_volatility - volatilities[i-1])
        
        # Add regime effect to volatility
        volatility_change += 0.001 * regimes[i]
        
        # Ensure volatility is positive
        new_volatility = max(0.001, volatilities[i-1] + volatility_change)
        volatilities.append(new_volatility)
        
        # Price follows a random walk with the current volatility
        price_change = prices[i-1] * new_volatility * np.random.randn()
        
        # Add trend and seasonality
        trend = 0.0001 * prices[i-1]  # Small upward trend
        seasonality = 50 * np.sin(2 * np.pi * i / (24*7))  # Weekly seasonality
        
        # Add regime effect to price
        regime_effect = prices[i-1] * 0.01 * regimes[i]
        
        new_price = prices[i-1] + price_change + trend + seasonality + regime_effect
        prices.append(max(100, new_price))  # Ensure price doesn't go too low
    
    # Generate on-chain volume (correlated with volatility)
    volumes = []
    for i in range(samples):
        base_volume = prices[i] * 100  # Base volume proportional to price
        volatility_effect = base_volume * volatilities[i] * 10  # Higher volatility -> higher volume
        random_factor = np.random.lognormal(0, 0.5)  # Random variation
        volumes.append(base_volume + volatility_effect * random_factor)
    
    # Generate network metrics
    hashrates = []
    active_addresses = []
    
    for i in range(samples):
        # Hashrate grows over time with some random variation
        hashrate_base = 1000000 * (1 + 0.0001 * i)
        hashrate_random = hashrate_base * np.random.lognormal(0, 0.01)
        hashrates.append(hashrate_random)
        
        # Active addresses correlated with price and volume
        address_base = 10000 * (1 + 0.1 * (prices[i] / base_price - 1))
        address_volume = 0.01 * (volumes[i] / (base_price * 100) - 1)
        address_random = np.random.lognormal(0, 0.1)
        active_addresses.append(max(1000, address_base * (1 + address_volume) * address_random))
    
    # Generate exchange flows
    exchange_inflows = []
    exchange_outflows = []
    
    for i in range(samples):
        # Exchange inflows correlated with volume and volatility
        inflow_base = volumes[i] * 0.4  # 40% of volume goes to exchanges
        inflow_volatility = inflow_base * volatilities[i] * 5  # Higher during volatile periods
        inflow_random = np.random.lognormal(0, 0.2)
        exchange_inflows.append(inflow_base + inflow_volatility * inflow_random)
        
        # Exchange outflows slightly less than inflows on average
        outflow_ratio = np.random.normal(0.95, 0.1)  # Outflow is about 95% of inflow on average
        outflow_random = np.random.lognormal(0, 0.2)
        exchange_outflows.append(exchange_inflows[i] * outflow_ratio * outflow_random)
    
    # Generate transaction-level features
    wallet_ages = np.random.lognormal(4, 1, samples)  # in days
    transaction_sizes = np.random.lognormal(-1, 1, samples)  # in BTC
    transaction_fees = np.random.lognormal(-7, 1, samples)  # in BTC
    
    # Generate fraud labels
    fraud_samples = int(samples * fraud_ratio)
    fraud_indices = np.random.choice(range(samples), size=fraud_samples, replace=False)
    fraud_labels = np.zeros(samples)
    fraud_labels[fraud_indices] = 1
    
    # Modify features for fraudulent transactions
    for idx in fraud_indices:
        # Fraudulent transactions often have newer wallets
        wallet_ages[idx] = wallet_ages[idx] * 0.3
        
        # Unusual transaction sizes
        if np.random.random() < 0.5:
            transaction_sizes[idx] = transaction_sizes[idx] * 5  # Unusually large
        else:
            transaction_sizes[idx] = transaction_sizes[idx] * 0.1  # Unusually small
            
        # Different fee patterns
        if np.random.random() < 0.7:
            transaction_fees[idx] = transaction_fees[idx] * 0.2  # Lower fees
        else:
            transaction_fees[idx] = transaction_fees[idx] * 3  # Higher fees
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'volatility': volatilities,
        'on_chain_volume': volumes,
        'network_hashrate': hashrates,
        'active_addresses': active_addresses,
        'exchange_inflow': exchange_inflows,
        'exchange_outflow': exchange_outflows,
        'wallet_age': wallet_ages,
        'transaction_size': transaction_sizes,
        'transaction_fee': transaction_fees,
        'fraud_label': fraud_labels
    })
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic cryptocurrency data')
    parser.add_argument('--samples', type=int, default=5000, help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='../data/crypto_volatility_fraud_dataset.csv', 
                        help='Output file path')
    parser.add_argument('--fraud_ratio', type=float, default=0.05, 
                        help='Ratio of fraudulent transactions')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Generate data
    print(f"Generating {args.samples} samples with {args.fraud_ratio:.1%} fraud ratio...")
    df = generate_crypto_data(samples=args.samples, fraud_ratio=args.fraud_ratio)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"Data saved to {args.output}")
    print(f"Dataset shape: {df.shape}")
    print(f"Actual fraud ratio: {df['fraud_label'].mean():.2%}")

if __name__ == "__main__":
    main()
