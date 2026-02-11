import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from src.data_loader import load_and_process_data, apply_smote
from src.models import SleepDataset, ANN, CNN, get_sklearn_model
import joblib
import os

# Configuration
DATA_PATH = "sleep_dataset.csv"
N_TRIALS = 20
EPOCHS = 20
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = '/content/drive/MyDrive/Sleep_Disorder_Project'


def train_torch_model(model, train_loader, val_loader, epochs, lr):
    """Train a PyTorch model and return best validation F1."""
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
        
        model.eval()
        all_preds, all_labels = [], []
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


def objective(trial, model_name, X_train_raw, X_val, y_train_raw, y_val, input_dim):
    """
    Optuna objective. SMOTE is applied ONLY to the training portion
    inside each trial, NOT to the validation set.
    """
    # Apply SMOTE to training portion only (prevents leakage)
    X_train, y_train = apply_smote(X_train_raw, y_train_raw)
    
    if model_name == 'KNN':
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 15),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan'])
        }
        model = get_sklearn_model('KNN', params)
        model.fit(X_train, y_train)
        return f1_score(y_val, model.predict(X_val), average='weighted')
        
    elif model_name == 'SVM':
        # Use larger subsample for better SVM training
        subsample_size = min(len(X_train), 20000)
        idx = np.random.choice(len(X_train), size=subsample_size, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
        
        params = {
            'C': trial.suggest_float('C', 0.1, 100.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
        }
        model = get_sklearn_model('SVM', params)
        model.fit(X_sub, y_sub)
        return f1_score(y_val, model.predict(X_val), average='weighted')
        
    elif model_name == 'RF':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 10, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
        }
        model = get_sklearn_model('RF', params)
        model.fit(X_train, y_train)
        return f1_score(y_val, model.predict(X_val), average='weighted')
        
    elif model_name in ['ANN', 'CNN']:
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float('dropout_rate', 0.1, 0.5)
        
        train_ds = SleepDataset(X_train, y_train)
        val_ds = SleepDataset(X_val, y_val)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
        
        if model_name == 'ANN':
            model = ANN(input_dim,
                        trial.suggest_int('hidden_layers', 1, 3),
                        trial.suggest_int('units_per_layer', 32, 256),
                        dropout)
        else:
            model = CNN(input_dim,
                        trial.suggest_int('filters', 16, 64),
                        trial.suggest_int('kernel_size', 2, 3),
                        dropout)
        return train_torch_model(model, train_dl, val_dl, 10, lr)


def run_all():
    """Main training pipeline."""
    # 1. Load data (NO SMOTE yet)
    X_train_raw, X_test, y_train_raw, y_test, le = load_and_process_data(DATA_PATH)
    input_dim = X_train_raw.shape[1]
    print(f"Input dimension: {input_dim}")
    print(f"Device: {DEVICE}")
    
    # 2. Split raw training data for optimization (before SMOTE)
    X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
        X_train_raw, y_train_raw, test_size=0.2, random_state=42, stratify=y_train_raw
    )
    print(f"Optimization split: train={X_opt_train.shape[0]}, val={X_opt_val.shape[0]}")
    
    models = ['KNN', 'SVM', 'RF', 'ANN', 'CNN']
    results = {}

    for m in models:
        print(f"\n{'='*50}")
        print(f"Optimizing {m}...")
        print(f"{'='*50}")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda t: objective(t, m, X_opt_train, X_opt_val, y_opt_train, y_opt_val, input_dim),
            n_trials=N_TRIALS
        )
        print(f"Best Params: {study.best_params}")
        print(f"Best Optimization F1: {study.best_value:.4f}")
        
        # 3. Retrain best model on FULL training data (with SMOTE)
        print(f"\nRetraining final {m} on full training data...")
        X_train_full, y_train_full = apply_smote(X_train_raw, y_train_raw)
        
        if m in ['KNN', 'SVM', 'RF']:
            best_model = get_sklearn_model(m, study.best_params)
            best_model.fit(X_train_full, y_train_full)
            preds = best_model.predict(X_test)
            
            # Save
            model_path = os.path.join(SAVE_DIR, f'{m}_best_model.joblib')
            joblib.dump(best_model, model_path)
            print(f"Saved {m} to {model_path}")
        else:
            p = study.best_params
            if m == 'ANN':
                best_model = ANN(input_dim, p['hidden_layers'], p['units_per_layer'], p['dropout_rate'])
            else:
                best_model = CNN(input_dim, p['filters'], p['kernel_size'], p['dropout_rate'])
            
            ds_train = SleepDataset(X_train_full, y_train_full)
            ds_test = SleepDataset(X_test, y_test)
            dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
            dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE)
            
            train_torch_model(best_model, dl_train, dl_test, epochs=EPOCHS, lr=p['lr'])
            
            # Save
            model_path = os.path.join(SAVE_DIR, f'{m}_best_model.pth')
            torch.save(best_model.state_dict(), model_path)
            print(f"Saved {m} to {model_path}")
            
            best_model.eval()
            preds = []
            with torch.no_grad():
                for Xb, _ in dl_test:
                    out = best_model(Xb.to(DEVICE))
                    preds.extend(torch.max(out, 1)[1].cpu().numpy())
        
        # 4. Evaluate on test set
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted')
        rec = recall_score(y_test, preds, average='weighted')
        f1 = f1_score(y_test, preds, average='weighted')
        
        results[m] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
        print(f"\n{m} Test Results:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, preds, target_names=le.classes_))
        
    # 5. Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    results_df = pd.DataFrame(results).T
    print(results_df.to_string())


if __name__ == "__main__":
    run_all()
