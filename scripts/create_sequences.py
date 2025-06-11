import numpy as np

def create_sequences(data, seq_length=5):
    """
    Create sequences for time series prediction.
    
    Args:
        data (numpy.ndarray): Input data array
        seq_length (int): Length of each sequence
        
    Returns:
        tuple: (X sequences, y targets)
    """
    xs = []
    ys = []
    
    # Check if data is 2D (features only) or includes target
    if len(data.shape) == 2:
        # Assume last column is target
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = data[i+seq_length, -1]  # Last column is target (volatility)
            xs.append(x)
            ys.append(y)
    else:
        # Separate data and target are provided
        X_data, y_data = data
        for i in range(len(X_data) - seq_length):
            x = X_data[i:i+seq_length]
            y = y_data[i+seq_length]
            xs.append(x)
            ys.append(y)
    
    return np.array(xs), np.array(ys)