import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import pandas as pd
import numpy as np
import os

# Import custom modules
from preprocessing import create_sliding_windows
from model import AlertingCNN
from baseline import train_baseline

def main():
    # 1. Setup paths
    data_path = os.path.join('data', 'metrics_synthetic.csv')
    if not os.path.exists(data_path):
        print("Error: Dataset not found. Please run data_gen.py and preprocessing.py first.")
        return

    # 2. Data Preparation
    print("--- Loading and Preparing Data ---")
    df = pd.read_csv(data_path)
    X, y = create_sliding_windows(df)
    
    # Split into Train (80%) and Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train Baseline (Random Forest)
    print("\n" + "="*40)
    print("RUNNING BASELINE: Random Forest")
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)
    
    # Calculate metrics for RF
    rf_f1 = f1_score(y_test, rf_preds)
    rf_prec = precision_score(y_test, rf_preds, zero_division=0)
    rf_rec = recall_score(y_test, rf_preds, zero_division=0)
    
    print(classification_report(y_test, rf_preds))

    # 4. Train Candidate (1D-CNN)
    print("\n" + "="*40)
    print("RUNNING CANDIDATE: 1D-CNN (PyTorch)")
    
    num_pos = np.sum(y_train)
    num_neg = len(y_train) - num_pos
    pos_weight_val = torch.tensor([num_neg / num_pos])

    train_ds = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    model = AlertingCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 30
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            
            # Manual weighted loss calculation
            loss = nn.functional.binary_cross_entropy(outputs, batch_y, reduction='none')
            weights = batch_y * pos_weight_val + (1 - batch_y) * 1.0
            weighted_loss = (loss * weights).mean()
            
            weighted_loss.backward()
            optimizer.step()
            epoch_loss += weighted_loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {epoch_loss/len(train_loader):.4f}")

    # 5. Evaluation with Optimized Threshold
    model.eval()
    with torch.no_grad():
        test_probs = model(torch.FloatTensor(X_test_scaled)).squeeze().numpy()
        cnn_preds = (test_probs > 0.3).astype(int)
    
    # Calculate metrics for CNN
    cnn_f1 = f1_score(y_test, cnn_preds)
    cnn_prec = precision_score(y_test, cnn_preds, zero_division=0)
    cnn_rec = recall_score(y_test, cnn_preds, zero_division=0)
    
    print("\n1D-CNN Classification Report:")
    print(classification_report(y_test, cnn_preds))

    # 6. Final Comprehensive Comparison
    print("\n" + "="*55)
    print(f"{'METRIC':<20} | {'RANDOM FOREST':<15} | {'1D-CNN':<10}")
    print("-" * 55)
    print(f"{'Precision':<20} | {rf_prec:<15.4f} | {cnn_prec:<10.4f}")
    print(f"{'Recall':<20} | {rf_rec:<15.4f} | {cnn_rec:<10.4f}")
    print(f"{'F1-Score':<20} | {rf_f1:<15.4f} | {cnn_f1:<10.4f}")
    print("="*55)

if __name__ == "__main__":
    main()