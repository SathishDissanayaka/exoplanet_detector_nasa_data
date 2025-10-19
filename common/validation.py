"""
Data validation utilities for merged datasets.
Validates that merged datasets have the required columns after integration.
"""
import pandas as pd
from typing import Tuple, List
from common.data_merger import DatasetMerger


class DatasetValidator:
    """
    Validates merged datasets to ensure they have required columns
    and proper data quality for model training.
    """
    
    def __init__(self):
        self.merger = DatasetMerger()
        # Get the actual column mapping from the merger
        self.column_mapping = self.merger.get_column_mapping()
        
        # Extract the KOI column names that will become merged_ columns
        self.koi_columns = list(set(self.column_mapping.values()))
        
        # Build merged column names
        self.merged_columns = [f"merged_{col}" for col in self.koi_columns]
        
        # Define minimum required columns for any model
        # These are the most critical features that must exist
        self.minimum_required = [
            'merged_koi_disposition',  # Target variable (always needed)
            'merged_ra',               # Right ascension (position)
            'merged_dec',              # Declination (position)
            'merged_koi_period',       # Orbital period
            'merged_koi_time0',        # Transit epoch
            'merged_koi_duration',     # Transit duration
            'merged_koi_depth',        # Transit depth
            'merged_koi_prad',         # Planet radius
            'merged_koi_insol',        # Insolation flux
            'merged_koi_teq',          # Equilibrium temperature
            'merged_koi_steff',        # Stellar effective temperature
            'merged_koi_slogg',        # Stellar surface gravity
            'merged_koi_srad',         # Stellar radius
        ]
        
        # Define recommended columns (warn if missing, but don't fail)
        # These are error columns that are nice to have but not critical
        self.recommended_columns = [
            'merged_koi_time0_err1',   # Transit epoch error 1
            'merged_koi_time0_err2',   # Transit epoch error 2
            'merged_koi_period_err1',  # Period error 1
            'merged_koi_period_err2',  # Period error 2
            'merged_koi_duration_err1', # Duration error 1
            'merged_koi_duration_err2', # Duration error 2
            'merged_koi_depth_err1',   # Depth error 1
            'merged_koi_depth_err2',   # Depth error 2
            'merged_koi_prad_err1',    # Planet radius error 1
            'merged_koi_prad_err2',    # Planet radius error 2
            'merged_koi_insol_err1',   # Insolation flux error 1
            'merged_koi_insol_err2',   # Insolation flux error 2
            'merged_koi_teq_err1',     # Equilibrium temperature error 1
            'merged_koi_teq_err2',     # Equilibrium temperature error 2
            'merged_koi_steff_err1',   # Stellar effective temperature error 1
            'merged_koi_steff_err2',   # Stellar effective temperature error 2
            'merged_koi_slogg_err1',   # Stellar surface gravity error 1
            'merged_koi_slogg_err2',   # Stellar surface gravity error 2
            'merged_koi_srad_err1',    # Stellar radius error 1
            'merged_koi_srad_err2',    # Stellar radius error 2
        ]
        
        # Model-specific preferred columns
        self.model_preferences = {
            'lightgbm': [],
            'catboost': [],
            'random_forest': [],
            'mlp': []
        }
    
    def validate(self, data: pd.DataFrame, model_name: str = None) -> Tuple[bool, str, dict]:
        """
        Validate a merged dataset.
        
        Args:
            data: Merged DataFrame to validate
            model_name: Optional model name for model-specific validation
            
        Returns:
            Tuple of (is_valid, message, validation_info)
            - is_valid: Boolean indicating if dataset passes validation
            - message: Descriptive message about validation result
            - validation_info: Dictionary with detailed validation information
        """
        validation_info = {
            'total_columns': len(data.columns),
            'total_rows': len(data),
            'merged_columns_found': [],
            'missing_minimum': [],
            'missing_recommended': [],
            'missing_model_specific': [],
            'data_quality_issues': []
        }
        
        # Find which merged columns exist
        validation_info['merged_columns_found'] = [
            col for col in data.columns if col.startswith('merged_')
        ]
        
        # Check minimum required columns
        validation_info['missing_minimum'] = [
            col for col in self.minimum_required if col not in data.columns
        ]
        
        if validation_info['missing_minimum']:
            message = f"Missing critical columns: {', '.join(validation_info['missing_minimum'])}"
            return False, message, validation_info
        
        # Check recommended columns (warnings only)
        validation_info['missing_recommended'] = [
            col for col in self.recommended_columns if col not in data.columns
        ]
        
        # Check model-specific requirements
        if model_name and model_name in self.model_preferences:
            validation_info['missing_model_specific'] = [
                col for col in self.model_preferences[model_name] 
                if col not in data.columns
            ]
        
        # Check target variable
        if 'merged_koi_disposition' in data.columns:
            valid_targets = {'CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE'}
            actual_targets = set(data['merged_koi_disposition'].dropna().unique())
            invalid_targets = actual_targets - valid_targets
            
            if invalid_targets:
                validation_info['data_quality_issues'].append(
                    f"Invalid disposition values: {invalid_targets}"
                )
                return False, f"Invalid target values found: {invalid_targets}", validation_info
            
            # Check for null values in target
            null_count = data['merged_koi_disposition'].isnull().sum()
            if null_count > 0:
                validation_info['data_quality_issues'].append(
                    f"Target has {null_count} null values"
                )
        
        # Check for excessive missing data in key columns
        key_columns = [col for col in self.recommended_columns if col in data.columns]
        for col in key_columns:
            null_pct = (data[col].isnull().sum() / len(data)) * 100
            if null_pct > 90:
                validation_info['data_quality_issues'].append(
                    f"{col} has {null_pct:.1f}% missing values"
                )
        
        # Build success message
        message = "Dataset validation passed"
        if validation_info['missing_recommended']:
            message += f" (missing {len(validation_info['missing_recommended'])} recommended columns)"
        
        return True, message, validation_info
    
    def get_available_features(self, data: pd.DataFrame) -> List[str]:
        """Get list of available merged feature columns (excluding target)"""
        merged_cols = [col for col in data.columns if col.startswith('merged_')]
        # Exclude target and raw columns
        feature_cols = [
            col for col in merged_cols 
            if col not in ['merged_koi_disposition'] and 'raw' not in col
        ]
        return feature_cols
    
    def get_validation_report(self, data: pd.DataFrame, model_name: str = None) -> dict:
        """
        Generate a detailed validation report.
        
        Args:
            data: DataFrame to validate
            model_name: Optional model name
            
        Returns:
            Dictionary with comprehensive validation information
        """
        is_valid, message, info = self.validate(data, model_name)
        
        report = {
            'is_valid': is_valid,
            'message': message,
            'details': info,
            'summary': {
                'total_merged_columns': len(info['merged_columns_found']),
                'has_minimum_requirements': len(info['missing_minimum']) == 0,
                'has_all_recommended': len(info['missing_recommended']) == 0,
                'data_quality_ok': len(info['data_quality_issues']) == 0
            }
        }
        
        return report


# Convenience function
def validate_merged_dataset(data: pd.DataFrame, model_name: str = None) -> Tuple[bool, str]:
    """
    Quick validation function for merged datasets.
    
    Args:
        data: Merged DataFrame to validate
        model_name: Optional model name for model-specific checks
        
    Returns:
        Tuple of (is_valid, message)
    """
    validator = DatasetValidator()
    is_valid, message, _ = validator.validate(data, model_name)
    return is_valid, message
