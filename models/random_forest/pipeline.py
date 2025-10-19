"""Random Forest-specific preprocessing pipeline"""
import pandas as pd
import numpy as np
from common.base_pipeline import BasePipeline


class RandomForestPipeline(BasePipeline):
    """
    Random Forest specific preprocessing pipeline.
    Handles error column removal, aggressive missing data filtering, and median imputation.
    """
    
    def __init__(self, missing_threshold=0.5):
        """
        Args:
            missing_threshold: Drop features with more than this fraction of missing values (default 0.5)
        """
        super().__init__()
        self.missing_threshold = missing_threshold
        self.feature_medians = {}  # Store medians for inference
        
    def _model_specific_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Random Forest specific preprocessing:
        1. Drop error columns (_err columns) - not useful for RF
        2. Drop features with >50% missing values (more aggressive than baseline)
        3. Replace infinities with NaN
        4. Median imputation for remaining missing values
        """
        # Step 1: Drop all *_err* columns (measurement error columns)
        # These are metadata and not useful for tree-based models
        err_cols = [c for c in df.columns if "_err" in c.lower()]
        if err_cols:
            print(f"  [RF] Dropping {len(err_cols)} error columns")
            df = df.drop(columns=err_cols)
        
        # Step 2: More aggressive feature filtering for Random Forest
        # RF works better with complete data, so drop features with >50% missing
        # (Baseline only drops >90%, so we need to be more strict)
        numeric_features = df.select_dtypes(include=[np.number]).columns
        
        # Exclude target column from this check
        numeric_features = [col for col in numeric_features if col not in ['label']]
        
        if len(numeric_features) > 0:
            missing_ratio = df[numeric_features].isnull().mean()
            features_to_drop = missing_ratio[missing_ratio > self.missing_threshold].index.tolist()
            
            if features_to_drop:
                print(f"  [RF] Dropping {len(features_to_drop)} features with >{self.missing_threshold*100:.0f}% missing values")
                df = df.drop(columns=features_to_drop)
                
                # Update numeric features list
                numeric_features = [col for col in numeric_features if col not in features_to_drop]
        
        # Step 3: Replace infinities with NaN (tree models can't handle inf)
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Step 4: Median imputation for remaining missing values
        # Random Forests benefit from median imputation rather than mean (more robust to outliers)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['label']]
        
        for col in numeric_cols:
            if df[col].isna().any():
                if col not in self.feature_medians:
                    # Calculate and store median for training
                    self.feature_medians[col] = df[col].median()
                # Fill missing values with median
                df[col].fillna(self.feature_medians[col], inplace=True)
        
        return df
    
    def get_preprocessing_steps(self) -> list:
        """Return list of preprocessing steps for documentation"""
        return [
            'Common baseline preprocessing (merged columns, duplicates, physical constraints)',
            'Advanced transformers (depth-to-radius, cross-mission duplicates, galactic coords)',
            'Drop error columns (_err)',
            f'Drop features with >{self.missing_threshold*100:.0f}% missing',
            'Replace infinities with NaN',
            'Median imputation for missing values'
        ]
    
    def get_feature_medians(self) -> dict:
        """Return the stored feature medians for inference"""
        return self.feature_medians.copy()
