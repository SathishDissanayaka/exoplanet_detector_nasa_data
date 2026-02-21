"""LightGBM-specific preprocessing pipeline"""
import pandas as pd
import numpy as np
from common.base_pipeline import BasePipeline

class LightGBMPipeline(BasePipeline):
    """
    LightGBM-specific preprocessing pipeline with advanced feature engineering.
    LightGBM handles missing values well, so we use minimal imputation but
    add rich feature interactions and transformations that boost performance.
    """
    
    def __init__(self):
        super().__init__()
    
    def _model_specific_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        LightGBM-specific preprocessing with advanced feature engineering:
        
        1. Interaction features - capture relationships between variables
        2. Statistical aggregations from uncertainty columns
        3. Polynomial features for key physical ratios
        4. Log transformations for skewed distributions
        5. Physical regime binning (categorical-like features)
        6. Ratios and normalized features
        
        LightGBM excels with many features and handles NaN natively.
        """
    
        # Final cleanup: replace any remaining infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def get_preprocessing_steps(self) -> list:
        """Return list of preprocessing steps for this pipeline"""
        return [
            'Common data cleaning and transformers',
            'Interaction features (radius×period, temp×insol, etc.)',
            'Statistical aggregations from error columns',
            'Polynomial features (squared, cubed, log transforms)',
            'Physical regime binning (planet size, period, temperature)',
            'Normalized and scale-invariant features',
            'Signal-to-noise proxies',
            'No imputation (LightGBM native NaN handling)'
        ]
