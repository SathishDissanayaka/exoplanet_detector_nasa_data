"""MLP Classifier-specific preprocessing pipeline"""
import pandas as pd
import numpy as np
from common.base_pipeline import BasePipeline

class MLPPipeline(BasePipeline):
    
    def __init__(self):
        super().__init__()
    
    def _model_specific_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def get_preprocessing_steps(self) -> list:
        return [
            'Common data cleaning',
            'Median imputation'
        ]
