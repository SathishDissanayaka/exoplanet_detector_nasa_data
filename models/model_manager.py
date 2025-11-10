"""
Central Model Manager - Orchestrates all model pipelines.
This is the ONLY file that imports from model-specific folders.
"""
from typing import Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

from models.lightgbm.pipeline import LightGBMPipeline
from models.lightgbm.model import train_lightgbm_model
from models.catboost.pipeline import CatBoostPipeline
from models.catboost.model import train_catboost_model
from models.random_forest.pipeline import RandomForestPipeline
from models.random_forest.model import train_random_forest_model
from models.mlp.pipeline import MLPPipeline
from models.mlp.model import train_mlp_model

class ModelManager:
    """
    Central orchestrator for all exoplanet detection models.
    Manages model selection, preprocessing, training, and predictions.
    """
    
    def __init__(self, auto_load_models=False):
        """
        Initialize ModelManager
        
        Args:
            auto_load_models: If True, automatically load pre-trained models from disk
        """
        self.model_dir = Path(__file__).parent  # models/ directory
        
        self.available_models = {
            'lightgbm': {
                'pipeline': LightGBMPipeline(),
                'train_func': train_lightgbm_model,
                'description': 'Fast gradient boosting, handles missing values well',
                'preprocessing_steps': ['Radius ratio features', 'Temperature features', 'No imputation'],
                'trained_model': None,
                'trained': False,
                'model_file': 'lightgbm_model.pkl'
            },
            'catboost': {
                'pipeline': CatBoostPipeline(),
                'train_func': train_catboost_model,
                'description': 'Best for handling mixed data types and missing values',
                'preprocessing_steps': ['Galactic coordinates', 'Radius ratio features', 'Light imputation'],
                'trained_model': None,
                'trained': False,
                'model_file': 'catboost_model.pkl'
            },
            'random_forest': {
                'pipeline': RandomForestPipeline(),
                'train_func': train_random_forest_model,
                'description': 'Robust ensemble method, minimal preprocessing',
                'preprocessing_steps': ['Common cleaning', 'Median imputation'],
                'trained_model': None,
                'trained': False,
                'model_file': 'random_forest_model.pkl'
            },
            'mlp': {
                'pipeline': MLPPipeline(),
                'train_func': train_mlp_model,
                'description': 'Neural network classifier, learns features automatically',
                'preprocessing_steps': ['Common cleaning', 'Median imputation', 'Standard scaling'],
                'trained_model': None,
                'trained': False,
                'model_file': 'mlp_model.pkl'
            }
        }
        
        # Auto-load models if requested
        if auto_load_models:
            self.load_all_models()
    
    def load_model(self, model_name: str) -> bool:
        """
        Load a pre-trained model from disk
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.available_models:
            print(f"⚠️  Unknown model: {model_name}")
            return False
        
        model_config = self.available_models[model_name]
        model_file = self.model_dir / model_config['model_file']
        
        if not model_file.exists():
            print(f"⚠️  Model file not found: {model_file}")
            return False
        
        try:
            # Use joblib for better sklearn compatibility
            saved_data = joblib.load(model_file)
            
            # Load model and pipeline
            model_config['trained_model'] = saved_data['model']
            model_config['pipeline'] = saved_data['pipeline']
            model_config['trained'] = True
            
            # Sync feature names from model to pipeline (critical for CatBoost)
            model_config['pipeline'].set_feature_names_from_model(saved_data['model'])
            
            print(f"✅ Loaded {model_name} model from {model_file.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading {model_name} model: {str(e)}")
            return False
    
    def load_all_models(self) -> int:
        """
        Load all available pre-trained models from disk
        
        Returns:
            Number of models successfully loaded
        """
        loaded_count = 0
        print("\n🔄 Loading pre-trained models...")
        
        for model_name in self.available_models.keys():
            if self.load_model(model_name):
                loaded_count += 1
        
        if loaded_count > 0:
            print(f"\n✅ Successfully loaded {loaded_count}/{len(self.available_models)} models")
        else:
            print("\n⚠️  No pre-trained models found. Models need to be trained first.")
        
        return loaded_count
    
    def save_model(self, model_name: str) -> bool:
        """
        Save a trained model to disk
        
        Args:
            model_name: Name of the model to save
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.available_models:
            print(f"⚠️  Unknown model: {model_name}")
            return False
        
        model_config = self.available_models[model_name]
        
        if not model_config['trained']:
            print(f"⚠️  Model {model_name} is not trained yet")
            return False
        
        model_file = self.model_dir / model_config['model_file']
        
        try:
            # Save both model and pipeline using joblib for better sklearn compatibility
            save_data = {
                'model': model_config['trained_model'],
                'pipeline': model_config['pipeline']
            }
            
            # Use joblib instead of pickle for better sklearn model serialization
            joblib.dump(save_data, model_file, compress=3)
            
            print(f"✅ Saved {model_name} model to {model_file.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving {model_name} model: {str(e)}")
            return False
        
    def get_available_models(self) -> list:
        """Get list of available model names"""
        return list(self.available_models.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_name not in self.available_models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_info = self.available_models[model_name]
        return {
            'name': model_name,
            'description': model_info['description'],
            'preprocessing_steps': model_info['preprocessing_steps'],
            'trained': model_info['trained']
        }
    
    def get_model_preview(self, model_name: str, data: pd.DataFrame) -> tuple:
        """
        Show preprocessing preview before training.
        
        Args:
            model_name: Name of the model
            data: Raw input data
            
        Returns:
            Tuple of (success, processed_data, message)
        """
        if model_name not in self.available_models:
            return False, None, f"Unknown model: {model_name}"
        
        try:
            pipeline = self.available_models[model_name]['pipeline']
            
            # Apply preprocessing for training (this will fit encoders)
            processed_data = pipeline.preprocess_for_training(data.copy())
            
            return True, processed_data, "Preprocessing completed successfully"
        except Exception as e:
            return False, None, f"Preprocessing error: {str(e)}"
    
    def train_model(self, data: pd.DataFrame, model_name: str, training_params: Dict = None) -> tuple:
        """
        Train a model with model-specific preprocessing.
        
        Args:
            data: Raw training data (should include 'mission' column for TESS-First)
            model_name: Name of the model to train
            training_params: Training configuration parameters
            
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        if model_name not in self.available_models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = self.available_models[model_name]
        pipeline = model_config['pipeline']
        train_func = model_config['train_func']
        
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} Model")
        print(f"{'='*60}\n")
        
        # Step 1: Apply model-specific preprocessing for training
        print("Step 1: Preprocessing data...")
        processed_data = pipeline.preprocess_for_training(data.copy())
        
        # Step 2: Separate features and target (keep 'mission' column for TESS-first splitting in models)
        print("\nStep 2: Separating features and target...")
        X, y = pipeline.prepare_features(processed_data)

        # Re-add mission column to X if it exists (models need it for TESS-first splitting)
        if 'mission' in processed_data.columns:
            X['mission'] = processed_data['mission'].values

        print(f"✅ Features shape: {X.shape}")
        print(f"✅ Target shape: {y.shape}")
        print(f"✅ Feature count: {len(X.columns)}")

        # Split data into training and testing sets

        # Step 4: Train the model (TESS-first splitting happens inside each model)
        print(f"\nStep 4: Training {model_name} model...")
        train_result = train_func(
            X, y, 
            pipeline.label_encoder,
            training_params
        )
        
        # Handle different return values (lightgbm, catboost, random_forest return 4 values, others return 2)
        if model_name in ['lightgbm', 'catboost', 'random_forest'] and len(train_result) == 4:
            trained_model, metrics, stateful_transformers, final_feature_names = train_result
            # Store stateful transformers in the pipeline
            pipeline.stateful_transformers = stateful_transformers
            # Store final feature names for inference consistency
            pipeline.feature_names = final_feature_names
            print(f"✅ Stored {len(stateful_transformers)} stateful transformers in pipeline")
            print(f"✅ Stored {len(final_feature_names)} feature names in pipeline for inference")
        elif model_name in ['lightgbm', 'catboost', 'random_forest'] and len(train_result) == 3:
            # Backward compatibility for old training code
            trained_model, metrics, stateful_transformers = train_result
            pipeline.stateful_transformers = stateful_transformers
            print(f"✅ Stored {len(stateful_transformers)} stateful transformers in pipeline")
            print(f"⚠️  Warning: Feature names not returned from training function")
        else:
            trained_model, metrics = train_result
        
        # Store the trained model and pipeline
        self.available_models[model_name]['trained_model'] = trained_model
        self.available_models[model_name]['pipeline'] = pipeline  # Store fitted pipeline
        self.available_models[model_name]['trained'] = True
        
        # Auto-save the model after training
        print(f"\nStep 5: Saving model to disk...")
        if self.save_model(model_name):
            print(f"✅ Model saved successfully!")
        else:
            print(f"⚠️  Warning: Model could not be saved to disk")
        
        print(f"\n{'='*60}")
        print(f"✅ {model_name.upper()} Training Complete!")
        print(f"{'='*60}\n")
        
        return trained_model, metrics
    
    def predict(self, features: Dict[str, float], model_name: str) -> Dict[str, Any]:
        """
        Make a prediction using the specified trained model.
        
        Args:
            features: Dictionary of feature values
            model_name: Name of the model to use
            
        Returns:
            Dictionary with prediction results
        """
        if model_name not in self.available_models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = self.available_models[model_name]
        
        if not model_config['trained']:
            raise ValueError(f"Model {model_name} is not trained yet")
        
        trained_model = model_config['trained_model']
        pipeline = model_config['pipeline']
        
        # Convert features to DataFrame
        df = pd.DataFrame([features])
        
        # Apply preprocessing for inference (does not fit encoders)
        # The inference preprocessing will handle target columns appropriately
        df = pipeline.preprocess_for_inference(df)
        
        # Prepare features (separates X from y if present)
        X, _ = pipeline.prepare_features(df)
        
        # Get prediction
        if model_name == 'lightgbm':
            pred_proba = trained_model.predict(X, num_iteration=trained_model.best_iteration)
            # pred_proba shape: (1, n_classes) for single prediction
            pred_proba_flat = pred_proba[0] if pred_proba.ndim > 1 else pred_proba
            pred_class = np.argmax(pred_proba_flat)
        else:
            # For sklearn-compatible models
            pred_class = trained_model.predict(X)[0]
            pred_proba_flat = trained_model.predict_proba(X)[0]
        
        # Decode prediction
        prediction_label = pipeline.label_encoder.inverse_transform([pred_class])[0]
        
        return {
            'prediction': prediction_label,
            'confidence': float(pred_proba_flat[pred_class]),
            'probabilities': {
                class_name: float(prob)
                for class_name, prob in zip(pipeline.label_encoder.classes_, pred_proba_flat)
            }
        }
    
    def get_feature_names(self) -> list:
        """Returns the list of base features required for prediction"""
        return [
            'merged_koi_prad',
            'merged_koi_srad',
            'merged_koi_teq',
            'merged_koi_steff'
        ]
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Returns descriptions for each base feature"""
        return {
            'merged_koi_prad': 'Planet Radius (Earth radii)',
            'merged_koi_srad': 'Star Radius (Solar radii)',
            'merged_koi_teq': 'Equilibrium Temperature (K)',
            'merged_koi_steff': 'Stellar Effective Temperature (K)'
        }
