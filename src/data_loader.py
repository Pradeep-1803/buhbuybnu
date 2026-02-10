import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

def load_and_process_data(filepath):
    """
    Loads and preprocesses the sleep dataset.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        X_train, X_test, y_train, y_test: Processed split data.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Drop Person ID
    if 'Person ID' in df.columns:
        df = df.drop(columns=['Person ID'])
    
    # 2. Handle Target Variable (Sleep Disorder)
    # Fill NaN with 'None'
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    
    print("Class distribution before balancing:")
    print(df['Sleep Disorder'].value_counts())

    # 3. Feature Engineering: Split Blood Pressure
    if 'Blood Pressure' in df.columns:
        df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
        df = df.drop(columns=['Blood Pressure'])
    
    # 4. Clean Categorical Data
    # 'Normal Weight' and 'Normal' are likely the same in BMI Category
    if 'BMI Category' in df.columns:
        df['BMI Category'] = df['BMI Category'].replace({'Normal Weight': 'Normal'})

    # 5. Define Feature Columns
    target_col = 'Sleep Disorder'
    categorical_cols = ['Gender', 'Occupation', 'BMI Category']
    numeric_cols = [col for col in df.columns if col not in categorical_cols + [target_col]]
    
    # 6. Preprocessing Pipelines
    # Numeric: Standard Scaling
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Categorical: OneHotEncoding
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine
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
    
    # Get feature names after OneHotEncoding for interpretability if needed
    # feature_names = numeric_cols + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols))
    
    # 8. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    # 9. Handle Class Imbalance with SMOTE on Training Data
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("Data processing complete.")
    print(f"Training shape: {X_train_resampled.shape}, Test shape: {X_test.shape}")
    
    return X_train_resampled, X_test, y_train_resampled, y_test, label_encoder

if __name__ == "__main__":
    # Test the loader
    try:
        data_path = "sleep_dataset.csv" # Adjusted relative path for testing
        X_train, X_test, y_train, y_test, le = load_and_process_data(data_path)
        print("Loader test successful.")
    except Exception as e:
        print(f"Loader test failed: {e}")
