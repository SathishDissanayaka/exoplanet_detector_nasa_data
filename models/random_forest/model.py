"""Random Forest model training logic with TESS-First strategy"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE


def train_random_forest_model(X, y, label_encoder, training_params=None):
    """
    Train a Random Forest model with the given features and target.
    
    Args:
        X: Feature DataFrame (may include 'mission' column for TESS-first splitting)
        y: Target Series (encoded labels)
        label_encoder: LabelEncoder instance for target
        training_params: Dict with training configuration. Supported keys:
            - test_size: fraction for test set (default 0.2)
            - random_seed: random state (default 42)
            - use_tess_first: bool (default True) - use TESS-first splitting strategy
            - n_estimators: number of trees (default 100)
            - max_depth: maximum tree depth (default 10)
            - min_samples_split: minimum samples to split (default 20)
            - min_samples_leaf: minimum samples per leaf (default 10)
            - max_features: features per split (default 'sqrt')
            - class_weight: class weighting strategy (default 'balanced')
            - use_smote: whether to apply SMOTE oversampling (default False)
    
    Returns:
        Tuple of (trained_model, metrics_dict, stateful_transformers, final_feature_names)
    """
    # Default parameters - OPTIMIZED from modelnewtuning.py
    default_params = {
        'test_size': 0.2,
        'random_seed': 42,
        'use_tess_first': True,  # Use TESS-first splitting by default
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 20,  # More conservative from modelnewtuning
        'min_samples_leaf': 10,  # More conservative from modelnewtuning
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'bootstrap': True,  # Added from modelnewtuning
        'use_smote': False  # Set to True to apply SMOTE for class balancing
    }
    
    if training_params:
        default_params.update(training_params)
    
    params = default_params
    
    # Check if we should use TESS-First splitting
    use_tess_first = params['use_tess_first'] and 'mission' in X.columns
    
    if use_tess_first:
        print("\nTESS-First Data Splitting (80% TESS + 20% Kepler)")
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
        
        # Separate back into X and y (keep mission column for transformers)
        X_train = train_data.drop(columns=['_target_'])
        y_train = train_data['_target_']
        X_test = test_data.drop(columns=['_target_'])
        y_test = test_data['_target_']
    else:
        # Standard random split (keep mission column for transformers)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=params['test_size'],
            stratify=y,
            random_state=params['random_seed']
        )
    
    print(f"\n{'='*60}")
    print(f"Training Random Forest Model")
    print(f"{'='*60}")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Class distribution before SMOTE
    train_dist = pd.Series(y_train).value_counts()
    print(f"\nClass distribution (training set):")
    for label_idx, count in train_dist.items():
        label_name = label_encoder.inverse_transform([label_idx])[0]
        print(f"  {label_name}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # Apply stateful transformers to X_train and X_test
    from common.transformers import RobustStellarScaler, TelescopeDistributionMatcher, RobustUncertaintyCompressor

    # Store fitted stateful transformers for use during inference
    stateful_transformers = {}
    
    robust_stellar_scaler = RobustStellarScaler()
    X_train = robust_stellar_scaler.fit_transform(X_train)
    X_test = robust_stellar_scaler.transform(X_test)
    stateful_transformers['robust_stellar_scaler'] = robust_stellar_scaler

    telescope_matcher = TelescopeDistributionMatcher()
    X_train = telescope_matcher.fit_transform(X_train)
    X_test = telescope_matcher.transform(X_test)
    stateful_transformers['telescope_matcher'] = telescope_matcher

    uncertainty_compressor = RobustUncertaintyCompressor()
    X_train = uncertainty_compressor.fit_transform(X_train)
    X_test = uncertainty_compressor.transform(X_test)
    stateful_transformers['uncertainty_compressor'] = uncertainty_compressor
    
    # Now remove mission column before training (transformers are done with it)
    if 'mission' in X_train.columns:
        X_train = X_train.drop(columns=['mission'])
    if 'mission' in X_test.columns:
        X_test = X_test.drop(columns=['mission'])
    
    # ✅ CRITICAL: Store the final feature names to return to pipeline
    final_feature_names = list(X_train.columns)
    print(f"✅ Final training features: {len(final_feature_names)} columns")
    
    # Apply SMOTE if requested
    if params['use_smote']:
        print(f"\nApplying SMOTE for class balancing...")
        smote = SMOTE(random_state=params['random_seed'])
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        print(f"After SMOTE, training set size: {len(y_train)}")
        train_dist_smote = pd.Series(y_train).value_counts()
        print(f"Class distribution after SMOTE:")
        for label_idx, count in train_dist_smote.items():
            label_name = label_encoder.inverse_transform([label_idx])[0]
            print(f"  {label_name}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # Create Random Forest model with optimized hyperparameters
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        class_weight=params['class_weight'],
        bootstrap=params['bootstrap'],  # Added from modelnewtuning
        random_state=params['random_seed'],
        n_jobs=-1,
        verbose=1
    )
    
    # Train
    print(f"\n🔄 Training Random Forest...")
    model.fit(X_train, y_train)
    
    # Predictions on TEST set (realistic performance)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate COMPREHENSIVE MULTICLASS metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Calculate weighted metrics
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Calculate PER-CLASS metrics
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Generate detailed classification report
    print(f"\n📊 DETAILED MULTICLASS CLASSIFICATION REPORT:")
    print("=" * 50)
    print(classification_report(y_test, y_pred,
                              target_names=label_encoder.classes_,
                              zero_division=0))
    
    # Generate confusion matrix
    print(f"\n🎯 CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
                        index=label_encoder.classes_,
                        columns=label_encoder.classes_)
    print(cm_df)
    
    # Feature importance
    feature_importance = dict(zip(X_train.columns, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n🔍 Top 10 Most Important Features:")
    for feat, importance in top_features:
        print(f"  {feat}: {importance:.4f}")
    
    # Create comprehensive metrics dictionary
    metrics = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_weighted': float(f1_weighted),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(label_encoder.classes_):
        class_name_lower = class_name.lower().replace(' ', '_')
        metrics[f'precision_{class_name_lower}'] = float(precision_per_class[i])
        metrics[f'recall_{class_name_lower}'] = float(recall_per_class[i])
        metrics[f'f1_{class_name_lower}'] = float(f1_per_class[i])
    
    # Add detailed reports
    metrics['classification_report'] = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    
    metrics['confusion_matrix'] = cm.tolist()
    metrics['confusion_matrix_df'] = cm_df.to_dict()
    metrics['feature_importance'] = {k: float(v) for k, v in feature_importance.items()}
    metrics['test_predictions'] = {
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_pred_proba': y_pred_proba.tolist()
    }
    
    # Print summary metrics
    print(f"\n✅ MULTICLASS RESULTS for RANDOM FOREST:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Macro F1: {f1_macro:.4f}")
    print(f"   Macro Precision: {precision_macro:.4f}")
    print(f"   Macro Recall: {recall_macro:.4f}")
    
    print(f"\n🎯 PER-CLASS METRICS:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"   {class_name}:")
        print(f"     Precision: {precision_per_class[i]:.4f}")
        print(f"     Recall: {recall_per_class[i]:.4f}")
        print(f"     F1: {f1_per_class[i]:.4f}")
    
    # Return model, metrics, stateful transformers, AND final feature names
    return model, metrics, stateful_transformers, final_feature_names


