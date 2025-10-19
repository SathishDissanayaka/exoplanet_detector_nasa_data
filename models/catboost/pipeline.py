"""CatBoost-specific preprocessing pipeline"""
import pandas as pd
import numpy as np
from common.base_pipeline import BasePipeline


class CatBoostPipeline(BasePipeline):
    """
    CatBoost-specific preprocessing pipeline.
    CatBoost handles missing values and categorical features natively,
    but we can still engineer features to boost performance.
    """
    
    def __init__(self):
        super().__init__()
    
    def _model_specific_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CatBoost-specific preprocessing:
        - Galactic coordinates already added by base pipeline
        - Radius ratio features already added by base pipeline
        - Add advanced feature engineering for CatBoost
        - Light imputation (CatBoost handles NaN well, so minimal imputation)
        """
        
        # Ensure no infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Feature Engineering for CatBoost
        print("🔧 CatBoost: Creating advanced features...")
        
        # 1. Planet-Star Interaction Features
        if 'merged_koi_prad' in df.columns and 'merged_koi_srad' in df.columns:
            if 'radius_ratio_calculated' in df.columns:
                df['radius_ratio_squared'] = df['radius_ratio_calculated'] ** 2
        
        # 2. Orbital Energy Features
        if 'merged_koi_period' in df.columns and 'merged_koi_srad' in df.columns:
            # Semi-major axis proxy (Kepler's 3rd law approximation)
            # a ≈ (P²)^(1/3) for normalized units
            df['orbital_distance_proxy'] = df['merged_koi_period'] ** (2/3)
            
        if 'merged_koi_insol' in df.columns and 'merged_koi_teq' in df.columns:
            # Energy balance ratio
            df['energy_balance'] = df['merged_koi_insol'] / (df['merged_koi_teq'] + 1e-5)
        
        # 3. Stellar Context Features  
        if 'merged_koi_steff' in df.columns and 'merged_koi_srad' in df.columns:
            # Stellar luminosity proxy (Stefan-Boltzmann: L ∝ R² T⁴)
            df['stellar_luminosity_proxy'] = (df['merged_koi_srad'] ** 2) * (df['merged_koi_steff'] ** 4)
            
        if 'merged_koi_slogg' in df.columns and 'merged_koi_srad' in df.columns:
            # Stellar mass proxy from surface gravity (g ∝ M/R²)
            df['stellar_mass_proxy'] = df['merged_koi_slogg'] * (df['merged_koi_srad'] ** 2)
        
        # 4. Detection Quality Features
        if 'merged_koi_duration' in df.columns and 'merged_koi_period' in df.columns:
            # Transit duration to period ratio (quality indicator)
            df['duration_period_ratio'] = df['merged_koi_duration'] / (df['merged_koi_period'] + 1e-5)
            
        if 'merged_koi_depth' in df.columns:
            # Log-scaled depth (easier for model to learn patterns)
            df['log_transit_depth'] = np.log1p(df['merged_koi_depth'].fillna(0))
        
        # 5. Habitability Zone Features
        if 'merged_koi_insol' in df.columns:
            # Habitability zone indicator (Earth receives 1 S_Earth)
            # Venus zone: > 1.78, Habitable: 0.25-1.78, Cold: < 0.25
            df['habitable_zone_indicator'] = pd.cut(
                df['merged_koi_insol'],
                bins=[-np.inf, 0.25, 1.78, np.inf],
                labels=[0, 1, 2]  # 0=cold, 1=habitable, 2=hot
            ).astype(float)
        
        # 6. Planet Size Categories (CatBoost can use as ordinal feature)
        if 'merged_koi_prad' in df.columns:
            # Earth-like (0.5-1.5), Super-Earth (1.5-2), Neptune-like (2-6), Jupiter-like (>6)
            df['planet_size_category'] = pd.cut(
                df['merged_koi_prad'],
                bins=[0, 1.5, 2.0, 6.0, np.inf],
                labels=[0, 1, 2, 3]  # 0=Earth, 1=Super-Earth, 2=Neptune, 3=Jupiter
            ).astype(float)
        
        # 7. Signal-to-Noise Proxy
        if 'merged_koi_depth' in df.columns and 'merged_koi_duration' in df.columns:
            # Deeper, longer transits = stronger signal
            df['signal_strength_proxy'] = df['merged_koi_depth'] * df['merged_koi_duration']
        
        print(f" Added {len([c for c in df.columns if c not in ['merged_koi_disposition', 'label', 'mission']])} features")
        
        return df
    
    def get_preprocessing_steps(self) -> list:
        """Return list of preprocessing steps for this pipeline"""
        return [
            'Advanced feature engineering',
            'Planet-star interaction features',
            'Orbital energy calculations',
            'Stellar context features',
            'Detection quality indicators',
            'Habitability zone features',
            'Planet size categorization',
            'Signal strength proxies',
            'Light imputation (CatBoost handles NaN natively)'
        ]
