import numpy as np
import pandas as pd
import os

def generate_metrics(n_steps=4000):
    # Base seasonality and noise
    t = np.arange(n_steps)
    cpu_usage = 50 + 15 * np.sin(2 * np.pi * t / (24 * 60)) + np.random.normal(0, 2, n_steps)
    is_incident = np.zeros(n_steps)
    
    # FORCED INJECTION: We place 20 incidents manually
    # 10 Spikes and 10 Memory Leaks
    incident_positions = np.linspace(100, n_steps-200, 20, dtype=int)
    
    for i, pos in enumerate(incident_positions):
        if i % 2 == 0:
            # Inject a Spike (Sudden)
            duration = 15
            cpu_usage[pos : pos + duration] += np.linspace(0, 40, duration)
            is_incident[pos : pos + duration] = 1
        else:
            # Inject a Memory Leak (Gradual)
            duration = 40
            cpu_usage[pos : pos + duration] += np.linspace(0, 30, duration)
            is_incident[pos : pos + duration] = 1
            
    # Clipping to keep it realistic (0-100%)
    cpu_usage = np.clip(cpu_usage, 0, 100)
    
    df = pd.DataFrame({
        'timestamp': t,
        'cpu_usage': cpu_usage,
        'is_incident_incoming': is_incident
    })
    return df

if __name__ == "__main__":
    data = generate_metrics()
    
    output_dir = os.path.join('..', 'data') 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'metrics_synthetic.csv')
    data.to_csv(output_path, index=False)
    
    print(f"--- Forced Data Generation Success ---")
    print(f"Total points: {len(data)}")
    print(f"Total incident-impacted points: {data['is_incident_incoming'].sum()}")