import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import numpy as np
import pandas as pd
from src.data_loader import load_and_process_data
from src.models import SleepDataset, ANN, CNN, get_sklearn_model
import joblib
import os

# Configuration
DATA_PATH = "sleep_dataset.csv"
N_TRIALS = 15  # Reduced for demonstration/speed
EPOCHS = 10
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_torch_model(model, train_loader, val_loader, epochs, lr, optimizer_name='Adam'):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    best_val_f1 = 0.0
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            
    return best_val_f1

def objective(trial, model_name, X_train, X_test, y_train, y_test, input_dim):
    if model_name == 'KNN':
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
        }
        model = get_sklearn_model('KNN', params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return f1_score(y_test, preds, average='weighted')
        
    elif model_name == 'SVM':
        # Subsample for SVM as it is slow
        idx = np.random.choice(len(X_train), size=min(len(X_train), 5000), replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
        
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
        }
        model = get_sklearn_model('SVM', params)
        model.fit(X_sub, y_sub)
        preds = model.predict(X_test)
        return f1_score(y_test, preds, average='weighted')
        
    elif model_name == 'RF':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
        }
        model = get_sklearn_model('RF', params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return f1_score(y_test, preds, average='weighted')
        
    elif model_name == 'ANN':
        hidden_layers = trial.suggest_int('hidden_layers', 1, 3)
        units = trial.suggest_int('units_per_layer', 16, 128)
        dropout = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        
        model = ANN(input_dim, hidden_layers, units, dropout)
        
        train_dataset = SleepDataset(X_train, y_train)
        val_dataset = SleepDataset(X_test, y_test) # Using test as val for optimization
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        return train_torch_model(model, train_loader, val_loader, EPOCHS, lr)
        
    elif model_name == 'CNN':
        filters = trial.suggest_int('filters', 16, 64)
        kernel_size = trial.suggest_int('kernel_size', 2, 3)
        dropout = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        
        model = CNN(input_dim, filters, kernel_size, dropout)
        
        train_dataset = SleepDataset(X_train, y_train)
        val_dataset = SleepDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        return train_torch_model(model, train_loader, val_loader, EPOCHS, lr)

def evaluate_best_model(model, model_type, X_train, y_train, X_test, y_test, le):
    print(f"\nEvaluating Best {model_type} Model...")
    
    if model_type in ['KNN', 'SVM', 'RF']:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    else: # PyTorch
        train_dataset = SleepDataset(X_train, y_train)
        test_dataset = SleepDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # Train fully for more epochs
        train_torch_model(model, train_loader, test_loader, epochs=20, lr=0.001) # Use fixed LR or optimized?
        
        model.eval()
        preds = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(DEVICE)
                outputs = model(X_batch)
                _, p = torch.max(outputs, 1)
                preds.extend(p.cpu().numpy())
    
    # Calculate Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted')
    rec = recall_score(y_test, preds, average='weighted')
    f1 = f1_score(y_test, preds, average='weighted')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=le.classes_))
    
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # Data Loading
    X_train_full, X_test, y_train_full, y_test, le = load_and_process_data(DATA_PATH)
    input_dim = X_train_full.shape[1]
    print(f"Input Dimension: {input_dim}")
    
    # Split Train into Train/Val for Optimization to avoid Test leakage
    from sklearn.model_selection import train_test_split
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    models_to_run = ['KNN', 'SVM', 'RF', 'ANN', 'CNN']
    results = {}
    
    for model_name in models_to_run:
        print(f"\n--- Optimizing {model_name} ---")
        study = optuna.create_study(direction='maximize')
        # Optimize on X_train_opt -> X_val_opt
        study.optimize(lambda trial: objective(trial, model_name, X_train_opt, X_val_opt, y_train_opt, y_val_opt, input_dim), n_trials=N_TRIALS)
        
        print(f"Best params for {model_name}: {study.best_params}")
        
        # Re-create best model
        if model_name == 'KNN':
            best_model = get_sklearn_model(model_name, study.best_params)
        elif model_name == 'SVM':
            best_model = get_sklearn_model(model_name, study.best_params)
        elif model_name == 'RF':
            best_model = get_sklearn_model(model_name, study.best_params)
        elif model_name == 'ANN':
            p = study.best_params
            best_model = ANN(input_dim, p['hidden_layers'], p['units_per_layer'], p['dropout_rate'])
        elif model_name == 'CNN':
            p = study.best_params
            best_model = CNN(input_dim, p['filters'], p['kernel_size'], p['dropout_rate'])
            
        # Final Evaluation on Held-out Test Set (trained on full X_train)
        metrics = evaluate_best_model(best_model, model_name, X_train_full, y_train_full, X_test, y_test, le)
        results[model_name] = metrics

    # Save Results
    results_df = pd.DataFrame(results).T
    print("\nFinal Results Summary:")
    print(results_df)
    results_df.to_markdown("results_summary.md")

if __name__ == "__main__":
    main()
