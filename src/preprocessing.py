import pandas as pd
import numpy as np
import os

def create_sliding_windows(data, window_size=60, horizon=15):
    """
    Converts time-series data into overlapping windows for supervised learning.
    
    Args:
        data (pd.DataFrame): The input metrics dataframe.
        window_size (int): Number of past steps to observe (W).
        horizon (int): Number of future steps to predict (H).
        
    Returns:
        X (np.array): Feature matrix of shape (samples, window_size).
        y (np.array): Target vector of shape (samples,).
    """
    X = []
    y = []
    
    # Extracting raw values for efficiency
    cpu_values = data['cpu_usage'].values
    incident_labels = data['is_incident_incoming'].values
    
    # Iterate through the sequence to create windows
    # We stop at len - W - H to ensure we always have a full horizon to check
    for i in range(len(cpu_values) - window_size - horizon):
        # Observation window (Features)
        window = cpu_values[i : i + window_size]
        
        # Target: 1 if any step in the next 'horizon' contains an incident
        future_horizon = incident_labels[i + window_size : i + window_size + horizon]
        target = 1 if np.any(future_horizon == 1) else 0
        
        X.append(window)
        y.append(target)
        
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Path management
    data_path = os.path.join('..', 'data', 'metrics_synthetic.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        X, y = create_sliding_windows(df)
        
        print("--- Preprocessing Success ---")
        print(f"X shape (Feature windows): {X.shape}")
        print(f"y shape (Target labels):  {y.shape}")
        print(f"Number of incidents detected in windows: {np.sum(y)}")
    else:
        print(f"Error: File not found at {data_path}. Please run data_gen.py first.")