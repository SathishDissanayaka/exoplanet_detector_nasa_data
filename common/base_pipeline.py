"""Base pipeline class that all model-specific pipelines inherit from"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder
from .transformers import (
    CrossMissionDuplicateRemover,
    # Stateless transformers for common preprocessing
    TelescopeAgnosticCleaner,
    SafeMultiplicityTransformer,
    SafeOrbitalFeatureTransformer,
    SafeTransitDurationTransformer,
    SafeTransitDepthTransformer,
    SafePlanetRadiusTransformer,
    SafeEnvironmentTransformer,
    ExoplanetPhysicsTransformer,
    SignalToNoiseTransformer,
)

class BasePipeline(ABC):
    """
    Abstract base class for all model pipelines.
    Enforces a common interface while allowing model-specific customization.
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self._is_fitted = False  # Track if encoder has been fitted
        
        # Initialize legacy common transformers (kept for backward compatibility)
        self.duplicate_remover = CrossMissionDuplicateRemover()

        # Store stateful transformers (set during training for models that use them)
        self.stateful_transformers = None
        
        # Initialize stateless transformers for enhanced common preprocessing
        # Removed stateful transformers (distribution_matcher, uncertainty_compressor)
        self.stateless_transformers = [
            ('multiplicity', SafeMultiplicityTransformer()),
            ('orbital', SafeOrbitalFeatureTransformer()),
            ('duration', SafeTransitDurationTransformer()),
            ('depth', SafeTransitDepthTransformer()),
            ('radius', SafePlanetRadiusTransformer()),
            ('environment', SafeEnvironmentTransformer()),
            ('physics', ExoplanetPhysicsTransformer()),
            ('snr', SignalToNoiseTransformer()),
            # Cleaner MUST be last (drops 'mission' column)
            ('cleaner', TelescopeAgnosticCleaner()),
        ]
    
    def set_feature_names_from_model(self, model):
        """
        Sync feature names from a trained model to ensure consistency.
        Critical for models like CatBoost that enforce strict column ordering.
        
        Args:
            model: Trained model with feature_names_ attribute
        """
        if hasattr(model, 'feature_names_'):
            self.feature_names = list(model.feature_names_)
            print(f"✅ Synced {len(self.feature_names)} feature names from model")
        else:
            print("⚠️  Model does not have feature_names_ attribute")
        
    def preprocess(self, data: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Main preprocessing pipeline. Calls common and model-specific steps.
        
        Args:
            data: Raw input DataFrame
            is_training: Whether this is training data (has labels) or inference data
            
        Returns:
            Preprocessed DataFrame ready for training or inference
        """
        df = data.copy()
        
        # Step 1: Common preprocessing (all models)
        df = self._common_preprocessing(df, is_training=is_training)
        
        # Step 2: Model-specific preprocessing (override in child classes)
        df = self._model_specific_preprocessing(df)
        
        # Step 3: Apply stateful transformers if available (for inference with saved transformers)
        if not is_training and self.stateful_transformers:
            print(f"\n🔧 Applying {len(self.stateful_transformers)} stateful transformers for inference...")
            for name, transformer in self.stateful_transformers.items():
                try:
                    print(f"   ├─ Applying: {name}")
                    df = transformer.transform(df)
                except Exception as e:
                    print(f"   ⚠️  Warning: {name} transformer failed: {e}")
                    continue
            print("   └─ ✓ Stateful transformers complete\n")
        
        # Step 4: Final cleaning
        df = self._final_cleaning(df)
        
        return df
    
    def _common_preprocessing(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        UNIVERSAL basic preprocessing that works for all models.
        Only does essential cleaning that helps EVERY algorithm.
        """
        # 1. Keep only predefined merged columns (they always exist)
        # Also preserve 'mission' column if present (needed for TESS-First splitting)
        merged_cols = [col for col in df.columns if col.startswith("merged")]
        if 'mission' in df.columns:
            merged_cols.append('mission')
        df = df[merged_cols].copy()
        
        if 'merged_koi_disposition' in df.columns:
            if is_training:
                # Training: encode labels and fit the encoder
                df = df.dropna(subset=['merged_koi_disposition'])
                df["label"] = self.label_encoder.fit_transform(df["merged_koi_disposition"])
                self._is_fitted = True
            else:
                # Inference: we have labels but shouldn't use them for prediction
                df = df.drop(columns=['merged_koi_disposition'])  
        else:
            # No target column - this is inference data
            pass
        
        # 3. Fix ALL physically impossible values (helps all models)
        # Negative values that break physics across ALL properties
        physical_properties = [
            'merged_koi_prad', 'merged_koi_srad', 'merged_koi_teq', 'merged_koi_steff',
            'merged_koi_period', 'merged_koi_duration', 'merged_koi_insol',
            'merged_koi_slogg', 'merged_koi_time0', 'merged_koi_depth'
        ]
        
        for prop in physical_properties:
            if prop in df.columns:
                df.loc[df[prop] <= 0, prop] = np.nan
        
        
        # 4. Remove exact duplicates (wastes compute for all models)
        df = df.drop_duplicates()
        
        # 5. Apply legacy common transformers (kept for backward compatibility)
        
        # Step 5b: Remove cross-mission duplicates
        df = self.duplicate_remover.transform(df)

        
        # 6. Apply stateless transformers sequentially
        print("\n🔧 Applying stateless transformers...")
        for name, transformer in self.stateless_transformers:
            try:
                    # Stateless transformers - just transform
                    print(f"   ├─ Applying: {name}")
                    df = transformer.transform(df)
            except Exception as e:
                print(f"   ⚠️  Warning: {name} transformer failed: {e}")
                continue
        print("   └─ ✓ Stateless transformers complete\n")
        
        return df
    
    @abstractmethod
    def _model_specific_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Model-specific preprocessing steps.
        Must be implemented by each model's pipeline.
        
        Args:
            df: DataFrame after common preprocessing
            
        Returns:
            DataFrame with model-specific transformations applied
        """
        pass
    
    def _final_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final cleaning steps after all transformations.
        """
        # Drop columns with >90% missing values
        missing_threshold = 0.90  # Drop if more than 90% missing
        
        # Calculate missing percentage for each column
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
        
        if cols_to_drop:
            print(f"\nDropping {len(cols_to_drop)} columns with >{missing_threshold*100:.0f}% missing values:")
            for col in cols_to_drop:
                pct = missing_pct[col] * 100
                print(f"   - {col}: {pct:.1f}% missing")
            df = df.drop(columns=cols_to_drop)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Tuple of (features, target) - target will be None for inference data
        """
        # Columns to drop (not features)
        drop_cols = ["merged_koi_disposition", "mission"]  # Always drop original disposition and mission
        
        # Handle target
        if "label" in df.columns:
            y = df["label"]
            drop_cols.append("label")
        else:
            y = None  # This is inference data
        
        # Get features
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        print(f"\n📊 prepare_features: {len(df.columns)} columns -> {len(X.columns)} features")
        print(f"   Dropped: {[col for col in drop_cols if col in df.columns]}")
        
        # If feature names were already stored (from training), ensure consistency
        # This is critical for models like CatBoost that expect exact column order
        if self.feature_names and y is None:  # Inference mode
            print(f"\n🔧 Aligning features to match training ({len(self.feature_names)} expected features)...")
            
            # Create a DataFrame with all expected features, filled with NaN for missing ones
            X_aligned = pd.DataFrame(index=X.index)
            
            matched_count = 0
            for feature_name in self.feature_names:
                if feature_name in X.columns:
                    X_aligned[feature_name] = X[feature_name]
                    matched_count += 1
                else:
                    # Add missing feature as NaN (CatBoost handles NaN natively)
                    X_aligned[feature_name] = np.nan
            
            print(f"   Matched: {matched_count}/{len(self.feature_names)} features")
            
            # Count and report missing features
            missing_features = [f for f in self.feature_names if f not in X.columns]
            extra_features = [f for f in X.columns if f not in self.feature_names]
            
            if missing_features:
                print(f"   ⚠️  Missing {len(missing_features)} features from training (will be filled with NaN):")
                if len(missing_features) <= 10:
                    for f in missing_features:
                        print(f"      - {f}")
                else:
                    for f in missing_features[:10]:
                        print(f"      - {f}")
                    print(f"      ... and {len(missing_features) - 10} more")
            
            if extra_features:
                print(f"   ⚠️  Extra {len(extra_features)} features not in training (will be ignored):")
                if len(extra_features) <= 10:
                    for f in extra_features:
                        print(f"      - {f}")
                else:
                    for f in extra_features[:10]:
                        print(f"      - {f}")
                    print(f"      ... and {len(extra_features) - 10} more")
            
            X = X_aligned
        else:
            # Training mode - store feature names for later use
            self.feature_names = list(X.columns)
        
        return X, y
    
    def get_preprocessing_info(self) -> dict:
        """
        Return information about the preprocessing steps for this model.
        Useful for logging and debugging.
        """
        info = {
            'model_type': self.__class__.__name__,
            'feature_count': len(self.feature_names),
            'features': self.feature_names,
            'has_label_encoder': self._is_fitted,
            'label_classes': list(self.label_encoder.classes_) if self._is_fitted else None
        }
        return info

    # Convenience methods for clear intent
    def preprocess_for_training(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convenience method for training data"""
        return self.preprocess(data, is_training=True)
    
    def preprocess_for_inference(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convenience method for inference data"""  
        return self.preprocess(data, is_training=False)
