"""Test script to verify stateless transformers integration in base pipeline"""
import pandas as pd
import numpy as np
from common.base_pipeline import BasePipeline
from models.lightgbm.pipeline import LightGBMPipeline


# Create a minimal test pipeline
class TestPipeline(BasePipeline):
    """Test pipeline that uses common preprocessing"""
    
    def _model_specific_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """No model-specific preprocessing for this test"""
        return df


def create_test_data():
    """Create minimal test data matching the exoplanet dataset structure"""
    np.random.seed(42)
    n_samples = 50
    
    data = {
        # Required columns
        'merged_koi_disposition': np.random.choice(['CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE'], n_samples),
        'mission': np.random.choice(['Kepler', 'TESS'], n_samples),
        
        # Physical properties
        'merged_koi_period': np.random.uniform(1, 500, n_samples),
        'merged_koi_period_err1': np.random.uniform(0.01, 0.1, n_samples),
        'merged_koi_period_err2': np.random.uniform(0.01, 0.1, n_samples),
        
        'merged_koi_depth': np.random.uniform(100, 10000, n_samples),
        'merged_koi_depth_err1': np.random.uniform(10, 100, n_samples),
        'merged_koi_depth_err2': np.random.uniform(10, 100, n_samples),
        
        'merged_koi_duration': np.random.uniform(1, 10, n_samples),
        'merged_koi_duration_err1': np.random.uniform(0.1, 0.5, n_samples),
        'merged_koi_duration_err2': np.random.uniform(0.1, 0.5, n_samples),
        
        'merged_koi_prad': np.random.uniform(0.5, 20, n_samples),
        'merged_koi_prad_err1': np.random.uniform(0.1, 1, n_samples),
        'merged_koi_prad_err2': np.random.uniform(0.1, 1, n_samples),
        
        'merged_koi_srad': np.random.uniform(0.5, 2, n_samples),
        'merged_koi_steff': np.random.uniform(3000, 7000, n_samples),
        'merged_koi_teq': np.random.uniform(200, 2000, n_samples),
        'merged_koi_insol': np.random.uniform(0.1, 10, n_samples),
        'merged_koi_slogg': np.random.uniform(3, 5, n_samples),
        
        # Coordinates
        'merged_ra': np.random.uniform(0, 360, n_samples),
        'merged_dec': np.random.uniform(-90, 90, n_samples),
        
        # Time
        'merged_koi_time0': np.random.uniform(2450000, 2460000, n_samples),
        'merged_koi_time0_err1': np.random.uniform(0.001, 0.01, n_samples),
        'merged_koi_time0_err2': np.random.uniform(0.001, 0.01, n_samples),
        
        # Multiplicity
        'merged_multiplicity': np.random.choice([1, 2, 3, 4, 5], n_samples),
    }
    
    return pd.DataFrame(data)


def test_stateless_transformers():
    """Test that stateless transformers are applied correctly"""
    print("="*60)
    print("Testing Stateless Transformers Integration")
    print("="*60)
    
    # Create test data
    print("\n1. Creating test data...")
    df = create_test_data()
    print(f"   Created {len(df)} samples with {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns[:5])}...")
    
    # Initialize pipeline
    print("\n2. Initializing test pipeline...")
    pipeline = TestPipeline()
    print(f"   Pipeline initialized with {len(pipeline.stateless_transformers)} stateless transformers")
    
    # Apply preprocessing
    print("\n3. Applying preprocessing...")
    try:
        df_processed = pipeline.preprocess(df, is_training=True)
        print(f"   ✓ Preprocessing completed successfully!")
        print(f"   Output shape: {df_processed.shape}")
    except Exception as e:
        print(f"   ✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check for expected features
    print("\n4. Checking for expected features...")
    expected_features = [
        'feat_is_single_system',  # From SafeMultiplicityTransformer
        'feat_log_period',  # From SafeOrbitalFeatureTransformer
        'feat_duration_ratio',  # From SafeTransitDurationTransformer
        'feat_log_depth',  # From SafeTransitDepthTransformer
        'feat_radius_ratio',  # From SafePlanetRadiusTransformer
        'feat_log_insol',  # From SafeEnvironmentTransformer
        'feat_temp_ratio',  # From ExoplanetPhysicsTransformer
        'feat_depth_snr',  # From SignalToNoiseTransformer
    ]
    
    found_features = []
    missing_features = []
    
    for feat in expected_features:
        if feat in df_processed.columns:
            found_features.append(feat)
            print(f"   ✓ Found: {feat}")
        else:
            missing_features.append(feat)
            print(f"   ✗ Missing: {feat}")
    
    print(f"\n   Summary: {len(found_features)}/{len(expected_features)} expected features found")
    
    # Check that 'mission' column was removed by cleaner
    print("\n5. Checking telescope-leaking columns...")
    if 'mission' in df_processed.columns:
        print("   ✗ 'mission' column still present (should be removed by TelescopeAgnosticCleaner)")
    else:
        print("   ✓ 'mission' column removed successfully")
    
    if 'merged_ra' in df_processed.columns or 'merged_dec' in df_processed.columns:
        print("   ✗ RA/Dec columns still present (should be removed)")
    else:
        print("   ✓ RA/Dec columns removed successfully")
    
    # Display sample of new features
    print("\n6. Sample of engineered features:")
    feature_cols = [col for col in df_processed.columns if col.startswith('feat_')]
    if feature_cols:
        print(f"   Total engineered features: {len(feature_cols)}")
        print(f"   Sample features: {feature_cols[:10]}")
        print(f"\n   First few rows of engineered features:")
        print(df_processed[feature_cols[:5]].head(3))
    else:
        print("   ✗ No engineered features found!")
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    
    return len(missing_features) == 0


if __name__ == "__main__":
    success = test_stateless_transformers()
    exit(0 if success else 1)
