#!/usr/bin/env python3
"""
Initial model training script for system setup.
Run this ONCE to train all models and save them to disk.
After running this, the web app will automatically load these pre-trained models.

Usage:
    python train_all_models_initial.py --kepler path/to/kepler.csv --tess path/to/tess.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from models.model_manager import ModelManager
from common.data_merger import DatasetMerger
from common.validation import DatasetValidator


def main():
    parser = argparse.ArgumentParser(
        description='Train all models for initial system setup'
    )
    parser.add_argument(
        '--kepler',
        required=True,
        help='Path to Kepler dataset CSV file'
    )
    parser.add_argument(
        '--tess',
        required=True,
        help='Path to TESS dataset CSV file'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['lightgbm', 'catboost', 'random_forest', 'mlp'],
        help='Models to train (default: all)'
    )
    
    args = parser.parse_args()
    
    # Verify files exist
    kepler_path = Path(args.kepler)
    tess_path = Path(args.tess)
    
    if not kepler_path.exists():
        print(f"❌ Kepler file not found: {kepler_path}")
        sys.exit(1)
    
    if not tess_path.exists():
        print(f"❌ TESS file not found: {tess_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("🚀 INITIAL MODEL TRAINING FOR SYSTEM SETUP")
    print("=" * 80)
    print(f"\n📁 Kepler dataset: {kepler_path}")
    print(f"📁 TESS dataset: {tess_path}")
    print(f"🎯 Models to train: {', '.join(args.models)}")
    print("\n" + "=" * 80)
    
    # Load datasets
    print("\n📊 Loading datasets...")
    try:
        kepler_data = pd.read_csv(kepler_path)
        tess_data = pd.read_csv(tess_path)
        print(f"✅ Kepler: {len(kepler_data)} rows, {len(kepler_data.columns)} columns")
        print(f"✅ TESS: {len(tess_data)} rows, {len(tess_data.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        sys.exit(1)
    
    # Merge datasets
    print("\n🔗 Merging datasets...")
    try:
        merger = DatasetMerger()
        merged_data = merger.merge(kepler_data, tess_data)
        print(f"✅ Merged: {len(merged_data)} rows, {len(merged_data.columns)} columns")
    except Exception as e:
        print(f"❌ Error merging datasets: {e}")
        sys.exit(1)
    
    # Initialize model manager
    print("\n🤖 Initializing model manager...")
    model_manager = ModelManager(auto_load_models=False)
    
    # Validate dataset
    print("\n🔍 Validating dataset...")
    validator = DatasetValidator()
    
    
    # Train each model
    successful_models = []
    failed_models = []
    
    for model_name in args.models:
        if model_name not in model_manager.available_models:
            print(f"\n⚠️  Unknown model: {model_name}")
            failed_models.append(model_name)
            continue
        
        print("\n" + "=" * 80)
        print(f"🎯 Training {model_name.upper()}")
        print("=" * 80)
        
        # Validate for this specific model
        is_valid, message, validation_info = validator.validate(merged_data, model_name)
        
        if not is_valid:
            print(f"❌ Dataset validation failed for {model_name}: {message}")
            failed_models.append(model_name)
            continue
        
        print(f"✅ Dataset validated for {model_name}")
        
        # Train the model
        try:
            print(f"\n🔄 Training {model_name} with TESS-First strategy...")
            trained_model, metrics = model_manager.train_model(
                merged_data, 
                model_name
            )
            
            
            # Save the model to disk
            print(f"\n💾 Saving {model_name} to disk...")
            if model_manager.save_model(model_name):
                print(f"✅ {model_name} saved successfully!")
                successful_models.append(model_name)
            else:
                print(f"❌ Failed to save {model_name}")
                failed_models.append(model_name)
                
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            failed_models.append(model_name)
    
if __name__ == "__main__":
    main()
