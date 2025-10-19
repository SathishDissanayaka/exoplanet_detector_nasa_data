"""
Test script to verify LightGBM improvements
Uses the proper data merger workflow (Kepler + TESS → merged dataset)
"""
import pandas as pd
import numpy as np
from models.lightgbm.pipeline import LightGBMPipeline
from models.lightgbm.model import train_lightgbm_model
from common.data_merger import DatasetMerger

def main():
    print("=" * 80)
    print("TESTING LIGHTGBM IMPROVEMENTS")
    print("=" * 80)
    
    # Load Kepler and TESS datasets
    print("\n1. Loading Kepler and TESS datasets...")
    try:
        kepler_df = pd.read_csv('csvs/cumulative_2025.09.30_23.45.15.csv')
        tess_df = pd.read_csv('csvs/TOI_2025.09.30_23.45.34.csv')
        print(f"   ✅ Kepler: {kepler_df.shape[0]} rows, {kepler_df.shape[1]} columns")
        print(f"   ✅ TESS: {tess_df.shape[0]} rows, {tess_df.shape[1]} columns")
    except FileNotFoundError as e:
        print(f"   ❌ Error: Could not find dataset files")
        print(f"   Please ensure the following files exist in csvs/ directory:")
        print(f"     - cumulative_2025.09.30_23.45.15.csv (Kepler)")
        print(f"     - TOI_2025.09.30_23.45.34.csv (TESS)")
        return False
    
    # Merge datasets using DatasetMerger
    print("\n2. Merging datasets with DatasetMerger...")
    merger = DatasetMerger()
    df = merger.merge(kepler_df, tess_df)
    print(f"   ✅ Merged dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Show merge statistics
    merge_info = merger.get_merge_info()
    print(f"\n   Merge Statistics:")
    print(f"     - Kepler rows after merge: {merge_info.get('kepler_rows_after_merge', 0)}")
    print(f"     - TESS rows after merge: {merge_info.get('tess_rows_after_merge', 0)}")
    print(f"     - Unified columns created: {merge_info.get('unified_columns_created', 0)}")
    
    # Check target distribution
    if 'merged_koi_disposition' in df.columns:
        print(f"\n   Target distribution:")
        print(df['merged_koi_disposition'].value_counts().to_string())
    else:
        print(f"   ⚠️  Warning: 'merged_koi_disposition' column not found")
    
    # Initialize pipeline
    print("\n3. Initializing LightGBM pipeline...")
    pipeline = LightGBMPipeline()
    
    # Preprocess data
    print("\n4. Preprocessing data with enhanced feature engineering...")
    df_processed = pipeline.preprocess_for_training(df)
    print(f"   Preprocessed shape: {df_processed.shape}")
    
    # Prepare features
    print("\n5. Preparing features and target...")
    X, y = pipeline.prepare_features(df_processed)
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Number of features: {len(X.columns)}")
    print(f"   Target shape: {y.shape}")
    
    # Show new feature categories
    print("\n6. Analyzing new features...")
    new_features = {
        'Interaction features': [c for c in X.columns if 'interaction' in c],
        'Polynomial features': [c for c in X.columns if any(x in c for x in ['squared', 'cubed', 'log_'])],
        'Category features': [c for c in X.columns if 'category' in c],
        'Uncertainty features': [c for c in X.columns if 'uncertainty' in c or 'asymmetry' in c],
        'SNR features': [c for c in X.columns if 'snr' in c or 'luminosity' in c]
    }
    
    total_new = 0
    for category, features in new_features.items():
        if features:
            print(f"\n   {category}: {len(features)}")
            total_new += len(features)
            if len(features) <= 8:
                for f in features:
                    print(f"     - {f}")
            else:
                print(f"     First 5: {features[:5]}")
                print(f"     ... and {len(features) - 5} more")
    
    print(f"\n   Total new LightGBM-specific features: {total_new}")
    
    # Check for NaN values
    print("\n7. Data quality check...")
    nan_counts = X.isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        print(f"   Columns with NaN values: {len(cols_with_nan)}")
        print(f"   (This is OK - LightGBM handles NaN natively)")
    else:
        print(f"   No NaN values found")
    
    # Train model with new hyperparameters
    print("\n8. Training LightGBM model with optimized hyperparameters...")
    print(f"   Dataset size: {len(df)} samples")
    
    if len(df) < 10:
        print("   ⚠️  WARNING: Very small dataset. Results may not be reliable.")
        print("   Consider using more data for production training.")
    
    try:
        model, metrics = train_lightgbm_model(X, y, pipeline.label_encoder)
        
        print("\n9. Training Results:")
        print("=" * 80)
        print(f"   ✅ Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   ✅ F1 Score:  {metrics['f1']:.4f}")
        print(f"   ✅ Precision: {metrics['precision']:.4f}")
        print(f"   ✅ Recall:    {metrics['recall']:.4f}")
        print(f"   ✅ Best iteration: {metrics['best_iteration']}")
        
        # Feature importance
        print("\n10. Top 10 Most Important Features:")
        feat_imp = sorted(metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (feat, imp) in enumerate(feat_imp, 1):
            print(f"   {i:2d}. {feat:50s} {imp:8.1f}")
        
        print("\n" + "=" * 80)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nIMPROVEMENTS SUMMARY:")
        print(f"  • Used proper data merger workflow (Kepler + TESS)")
        print(f"  • Processed {len(df)} samples from both telescopes")
        print(f"  • Added {total_new} new engineered features")
        print(f"  • Optimized hyperparameters for better generalization")
        print(f"  • Added L1/L2 regularization for robustness")
        print(f"  • Configured for class imbalance handling")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
