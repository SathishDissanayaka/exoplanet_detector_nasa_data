"""LightGBM model training logic"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from models.tess_first_training import split_tess_first_data
from common.transformers import RobustStellarScaler, TelescopeDistributionMatcher, RobustUncertaintyCompressor


def train_lightgbm_model(X, y, label_encoder, training_params=None):
    """
    Train a LightGBM model with the given features and target.
    
    OPTIMIZATIONS APPLIED:
    1. Enhanced hyperparameters for better generalization:
       - Lower learning rate (0.01) with more rounds for smoother convergence
       - Conservative num_leaves (31) to prevent overfitting
       - Deeper trees (max_depth=12) with strong regularization
       - Conservative min_child_samples (20) for robustness
    
    2. Stronger regularization:
       - L1 (lambda_l1=1.0) for feature selection
       - L2 (lambda_l2=1.5) for weight smoothing
       - Feature fraction (0.7) for randomness
       - Bagging fraction (0.8) for ensemble diversity
       - Path smoothing (0.1) for better generalization
    
    3. Extended training with early stopping:
       - 2000 boosting rounds maximum
       - 150 rounds early stopping patience
       - Automatic best model selection
    
    Args:
        X: Feature DataFrame (may include 'mission' column for TESS-first splitting)
        y: Target Series (encoded labels)
        label_encoder: LabelEncoder instance for target
        training_params: Dict with training configuration:
            - test_size: float (default 0.2)
            - random_seed: int (default 42)
            - use_tess_first: bool (default True) - use TESS-first splitting strategy
            - num_boost_round: int (default 2000)
            - early_stopping_rounds: int (default 150)
            - verbose_eval: int (default 100)
    
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    # Default LightGBM parameters - OPTIMIZED for better precision
    default_lgb_params = {
        'objective': 'multiclass',
        'num_class': len(label_encoder.classes_),
        'metric': 'multi_logloss',
        'learning_rate': 0.01,  # Lower for better convergence
        'num_leaves': 31,  # Smaller to prevent overfitting
        'max_depth': 12,  # Deeper but with regularization
        'min_child_samples': 20,  # More conservative
        'min_child_weight': 0.01,
        'feature_fraction': 0.7,  # More aggressive feature sampling
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 1.0,  # Stronger L1 regularization
        'lambda_l2': 1.5,  # Stronger L2 regularization
        'min_split_gain': 0.02,
        'path_smooth': 0.1,  # Added for better generalization
        'verbosity': -1,
        'random_state': 42,
    }
    
    # Default training parameters
    default_training_params = {
        'test_size': 0.2,
        'random_seed': 42,
        'use_tess_first': True,
        'num_boost_round': 2000,
        'early_stopping_rounds': 150,
        'verbose_eval': 100
    }
    
    # Merge with provided parameters
    if training_params:
        # Separate LightGBM params from training params
        lgb_param_keys = set(default_lgb_params.keys())
        provided_lgb_params = {k: v for k, v in training_params.items() if k in lgb_param_keys}
        provided_training_params = {k: v for k, v in training_params.items() if k not in lgb_param_keys}
        
        lgb_params = {**default_lgb_params, **provided_lgb_params}
        training_params = {**default_training_params, **provided_training_params}
    else:
        lgb_params = default_lgb_params
        training_params = default_training_params
    
    # Check if we should use TESS-First splitting
    use_tess_first = training_params['use_tess_first'] and 'mission' in X.columns
    
    print(f"\n🔍 SPLITTING STRATEGY CHECK:")
    print(f"   use_tess_first parameter: {training_params['use_tess_first']}")
    print(f"   'mission' column present: {'mission' in X.columns}")
    print(f"   Will use TESS-First splitting: {use_tess_first}")
    
    if use_tess_first:
        print(f"\n✅ USING TESS-FIRST SPLITTING STRATEGY")
        print(f"   Importing split_tess_first_data function...")
        
        # Create temporary DataFrame with features + target + mission
        temp_df = X.copy()
        temp_df['_target_'] = y
        
        print(f"   Calling split_tess_first_data with {len(temp_df)} samples...")
        
        # Use the NEW balanced splitting strategy
        train_data, test_data = split_tess_first_data(
            temp_df,
            tess_train_pct=0.55,  # 55% TESS
            kepler_train_pct=0.45,  # 45% Kepler
            mission_col='mission',
            target_col='_target_',  # Use our temporary target column
            random_seed=training_params['random_seed']
        )
        
        # Separate back into X and y (keep mission column for transformers)
        X_train = train_data.drop(columns=['_target_'])
        y_train = train_data['_target_']
        X_test = test_data.drop(columns=['_target_'])
        y_test = test_data['_target_']
        
        print(f"   ✅ TESS-First split completed successfully")
        print(f"   Training set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")

    else:
        print(f"\n⚠️  USING STANDARD RANDOM SPLIT")
        print(f"   Reason: use_tess_first={training_params['use_tess_first']}, mission_in_columns={'mission' in X.columns}")
        
        # Standard random split (keep mission column for transformers)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=training_params['test_size'], 
            stratify=y, 
            random_state=training_params['random_seed']
        )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Store fitted stateful transformers for use during inference
    stateful_transformers = {}
    
    robust_stellar_scaler = RobustStellarScaler()
    X_train = robust_stellar_scaler.fit_transform(X_train)
    print(f"shape of the training set{X_train.shape}")
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
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Training parameters
    num_boost_round = training_params['num_boost_round']
    early_stopping_rounds = training_params['early_stopping_rounds']
    verbose_eval = training_params['verbose_eval']
    
    # Train model
    print("Training LightGBM model with optimized hyperparameters...")
    print(f"  - Learning rate: {lgb_params['learning_rate']}")
    print(f"  - Num leaves: {lgb_params['num_leaves']}")
    print(f"  - Max depth: {lgb_params['max_depth']}")
    print(f"  - Regularization: L1={lgb_params['lambda_l1']}, L2={lgb_params['lambda_l2']}")
    print(f"  - Num boost rounds: {num_boost_round}")
    print(f"  - Early stopping: {early_stopping_rounds}")
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(verbose_eval)
        ]
    )
    
    # Make predictions
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
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
    metrics['best_iteration'] = model.best_iteration
    metrics['feature_importance'] = dict(zip(X_train.columns, model.feature_importance().tolist()))
    metrics['test_predictions'] = {
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_pred_proba': y_pred_proba.tolist()
    }
    
    # Print summary metrics
    print(f"\n✅ MULTICLASS RESULTS for LIGHTGBM:")
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
