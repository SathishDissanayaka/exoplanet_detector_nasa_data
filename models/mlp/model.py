"""MLP Classifier model implementation"""
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def train_mlp_model(X, y, label_encoder, training_params=None):
    """
    Train an MLP model with the given features and target.
    
    Args:
        X: Feature DataFrame (may include 'mission' column for TESS-first splitting)
        y: Target Series (encoded labels)
        label_encoder: LabelEncoder instance for target
        training_params: Dict with training configuration:
            - test_size: float (default 0.2)
            - random_seed: int (default 42)
            - use_tess_first: bool (default True) - use TESS-first splitting strategy
    
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    # Default parameters
    default_params = {
        'test_size': 0.2,
        'random_seed': 42,
        'use_tess_first': True  # Use TESS-first splitting by default
    }
    
    if training_params:
        default_params.update(training_params)
    
    params = default_params
    
    # Check if we should use TESS-First splitting
    use_tess_first = params['use_tess_first'] and 'mission' in X.columns
    
    if use_tess_first:
        print("\n🚀 TESS-First Data Splitting (80% TESS + 20% Kepler)")
        from models.tess_first_training import split_tess_first_data
        
        # Create temporary DataFrame with features + target + mission
        temp_df = X.copy()
        temp_df['_target_'] = y
        
        # Split using TESS-first strategy
        train_data, test_data = split_tess_first_data(
            temp_df,
            tess_train_pct=0.55,
            kepler_train_pct=0.45,
            mission_col='mission',
            target_col='_target_',  # Use the temporary target column name
            random_seed=params['random_seed']
        )
        
        # Separate back into X and y, remove mission column
        X_train = train_data.drop(columns=['_target_', 'mission'])
        y_train = train_data['_target_']
        X_test = test_data.drop(columns=['_target_', 'mission'])
        y_test = test_data['_target_']
    else:
        # Standard random split
        if 'mission' in X.columns:
            X = X.drop(columns=['mission'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params['test_size'], random_state=params['random_seed'], stratify=y
        )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        shuffle=True,
        random_state=params['random_seed'],
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False
    )
    
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True),
        'feature_importance': {},  # MLP doesn't have feature importance
        'n_iterations': model.n_iter_,
        'loss': model.loss_,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'n_features': X_train.shape[1],
        'scaler': scaler  # Store scaler in metrics for later use
    }
    
    return model, metrics


def predict_mlp(model, scaler, data: pd.DataFrame) -> np.ndarray:
    X = data.drop(['label', 'merged_koi_disposition'], axis=1, errors='ignore')
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return predictions
