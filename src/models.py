import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import Dataset

class SleepDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ANN(nn.Module):
    def __init__(self, input_dim, hidden_layers=1, units_per_layer=24, dropout_rate=0.2, num_classes=3):
        super(ANN, self).__init__()
        layers = []
        
        # Input Layer
        layers.append(nn.Linear(input_dim, units_per_layer))
        layers.append(nn.BatchNorm1d(units_per_layer))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden Layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(units_per_layer, units_per_layer))
            layers.append(nn.BatchNorm1d(units_per_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
        # Output Layer
        layers.append(nn.Linear(units_per_layer, num_classes))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class CNN(nn.Module):
    def __init__(self, input_dim, filters=32, kernel_size=2, dropout_rate=0.3, num_classes=3):
        super(CNN, self).__init__()
        # Input shape: (Batch, 1, Features)
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm1d(filters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate size after conv and pool
        # Conv output size: L_out = L_in - kernel_size + 1
        conv_out_size = input_dim - kernel_size + 1
        # Pool output size: L_out / 2
        pool_out_size = conv_out_size // 2
        
        if pool_out_size <= 0:
             # Fallback if dimensions are too small after pooling
             self.pool = nn.Identity()
             pool_out_size = conv_out_size

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(filters * pool_out_size, num_classes)

    def forward(self, x):
        # Reshape input to (Batch, 1, Features)
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def get_sklearn_model(name, params):
    if name == 'KNN':
        return KNeighborsClassifier(**params, n_jobs=-1)
    elif name == 'SVM':
        return SVC(**params)
    elif name == 'RF':
        return RandomForestClassifier(**params, n_jobs=-1)
    else:
        raise ValueError(f"Unknown sklearn model: {name}")
