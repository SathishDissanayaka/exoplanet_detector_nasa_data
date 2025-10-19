"""
Quick script to check which models are available and their status
"""
from models.model_manager import ModelManager
from pathlib import Path

def check_models():
    print("\n" + "="*60)
    print("EXOPLANET DETECTOR - MODEL STATUS CHECK")
    print("="*60 + "\n")
    
    # Initialize model manager without auto-loading
    model_manager = ModelManager(auto_load_models=False)
    
    model_dir = Path(__file__).parent / "models"
    
    print(f"📁 Model directory: {model_dir}\n")
    
    print("🔍 Checking model files...\n")
    
    for model_name, config in model_manager.available_models.items():
        model_file = model_dir / config['model_file']
        exists = model_file.exists()
        
        status = "✅ EXISTS" if exists else "❌ NOT FOUND"
        size = f"({model_file.stat().st_size / 1024 / 1024:.2f} MB)" if exists else ""
        
        print(f"{status} {model_name:15} → {config['model_file']:30} {size}")
    
    print("\n" + "="*60)
    print("ATTEMPTING TO LOAD MODELS")
    print("="*60 + "\n")
    
    loaded_count = model_manager.load_all_models()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n✅ Successfully loaded: {loaded_count}/{len(model_manager.available_models)} models")
    
    if loaded_count == 0:
        print("\n⚠️  No models are trained yet!")
        print("\n📝 To train models:")
        print("   1. Run: streamlit run app.py")
        print("   2. Login to the application")
        print("   3. Go to 'Train Model' page")
        print("   4. Upload Kepler and TESS datasets")
        print("   5. Train your desired models")
    else:
        print("\n✨ Models are ready for predictions!")
        print("\nTrained models:")
        for model_name, config in model_manager.available_models.items():
            if config['trained']:
                print(f"   ✅ {model_name}")
    
    print("\n")

if __name__ == "__main__":
    check_models()
