import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import os

# Assuming X and y are already created by preprocessing.py
def train_baseline(X_train, y_train, X_test, y_test):
    print("--- Training Random Forest Baseline ---")
    
    # Initialize the model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train
    rf.fit(X_train, y_train)
    
    # Predict
    preds = rf.predict(X_test)
    
    # Evaluate
    print(classification_report(y_test, preds))
    return f1_score(y_test, preds)

if __name__ == "__main__":
    # This is a placeholder for the logic we will integrate in train.py
    print("Baseline module ready. It will be called during the main training pipeline.")