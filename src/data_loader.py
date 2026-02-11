import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')


def load_and_process_data(filepath):
    """
    Loads and preprocesses the sleep dataset.
    Returns data WITHOUT SMOTE applied - caller should apply SMOTE where appropriate.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Drop Person ID
    if 'Person ID' in df.columns:
        df = df.drop(columns=['Person ID'])
    
    # 2. Handle Target Variable
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    print("Class distribution:")
    print(df['Sleep Disorder'].value_counts())

    # 3. Feature Engineering: Split Blood Pressure
    if 'Blood Pressure' in df.columns:
        df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
        df = df.drop(columns=['Blood Pressure'])
    
    # 4. Clean Categorical Data
    if 'BMI Category' in df.columns:
        df['BMI Category'] = df['BMI Category'].replace({'Normal Weight': 'Normal'})

    # 5. Define Feature Columns
    target_col = 'Sleep Disorder'
    categorical_cols = ['Gender', 'Occupation', 'BMI Category']
    numeric_cols = [col for col in df.columns if col not in categorical_cols + [target_col]]
    
    # 6. Preprocessing Pipelines
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode Target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 7. Preprocess Features
    print("Preprocessing features...")
    X_processed = preprocessor.fit_transform(X)
    
    # 8. Split Data (NO SMOTE here - handled by caller)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, label_encoder


def apply_smote(X, y):
    """Apply SMOTE to balance classes. Use only on training data."""
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"SMOTE: {X.shape[0]} -> {X_res.shape[0]} samples")
    return X_res, y_res
