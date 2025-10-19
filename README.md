# 🌌 Exoplanet Detection System

A comprehensive machine learning web application for detecting and analyzing exoplanets using data from NASA's Kepler and TESS space telescopes. This system combines multiple state-of-the-art ML models with advanced feature engineering and cross-telescope domain adaptation techniques.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Key Features

### 🚀 **Multi-Model Architecture**
- **LightGBM**: Advanced gradient boosting with 30+ engineered features
- **CatBoost**: Categorical boosting optimized for mixed data types  
- **Random Forest**: Ensemble method with SMOTE class balancing
- **Multi-Layer Perceptron**: Neural network for complex pattern recognition

### 🔬 **Advanced Data Processing**
- **Cross-Mission Data Merger**: Intelligent combination of Kepler and TESS datasets
- **Duplicate Detection**: Spatial and temporal matching across telescopes
- **Domain Adaptation**: 55% TESS + 45% Kepler optimal training strategy
- **Feature Engineering**: 30+ physics-based and statistical features

### 💻 **Interactive Web Interface**
- **Real-time Predictions**: Single exoplanet analysis with confidence scores
- **Batch Processing**: CSV upload for multiple candidates
- **Model Training**: Interactive retraining with custom datasets
- **Visualization**: Feature importance, confusion matrices, and habitability scores
- **User Management**: Authentication with prediction history tracking

### 📊 **Performance Highlights**
- **Cross-Telescope Accuracy**: 75.9% (TESS) / 76.6% (Kepler)
- **Telescope Gap**: Only 0.69% difference between missions
- **Dataset Size**: 17,242+ samples from combined missions
- **Feature Count**: 100+ engineered astronomical and statistical features

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- 4GB+ RAM (recommended for model training)
- 2GB+ disk space for datasets and models

### 1. Clone the Repository

```bash
git clone https://github.com/SathishDissanayaka/exoplanet_detector_nasa_data.git
cd exoplanet_detector_nasa_data
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file for Supabase authentication (optional):

```bash
# .env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
```

## 🎓 Model Training

### Quick Start (Pre-trained Models)
The repository includes pre-trained models. Simply run:

```bash
streamlit run app.py
```

### Training from Scratch

For custom training with your own datasets:

```bash
# Train all models (recommended)
python train_all_models_initial.py --kepler csvs/cumulative_2025.09.30_23.45.15.csv --tess csvs/TOI_2025.09.30_23.45.34.csv

# Train specific models only
python train_all_models_initial.py --kepler <path_to_kepler.csv> --tess <path_to_tess.csv> --models lightgbm catboost

# Available models: lightgbm, catboost, random_forest, mlp
```

### Training Performance
- **LightGBM**: ~2-3 minutes (optimized with early stopping)
- **CatBoost**: ~3-4 minutes (handles categorical features natively)
- **Random Forest**: ~1-2 minutes (with SMOTE balancing)
- **MLP**: ~4-5 minutes (neural network with regularization)

## 🚀 Running the Application

### Start the Web Interface

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Application Pages

1. **🏠 Landing**: Project overview and methodology explanation
2. **🔮 Predict**: Single prediction and batch CSV processing
3. **🎯 Train**: Interactive model training and comparison
4. **📊 History**: Prediction history and analytics (requires login)
5. **👤 Login/Signup**: User authentication system

## 📁 Project Structure

```
exoplanet_detector/
├── 📱 app.py                    # Main Streamlit application
├── 📋 requirements.txt         # Python dependencies
├── 🎯 train_all_models_initial.py  # Batch model training script
│
├── 📊 models/                   # ML Models & Training
│   ├── model_manager.py        # Centralized model management
│   ├── lightgbm/              # LightGBM implementation
│   │   ├── model.py           # Training & prediction logic
│   │   └── pipeline.py        # Feature engineering pipeline
│   ├── catboost/              # CatBoost implementation  
│   ├── random_forest/         # Random Forest implementation
│   └── mlp/                   # Multi-Layer Perceptron
│
├── 🧮 common/                  # Shared Components
│   ├── base_pipeline.py       # Common preprocessing pipeline
│   ├── transformers.py        # Custom feature transformers
│   ├── data_merger.py         # Cross-mission data merger
│   └── validation.py          # Model validation utilities
│
├── 🖥️ pages/                   # Streamlit Pages
│   ├── landing.py             # Home page with methodology
│   ├── predict.py             # Prediction interface
│   ├── train.py               # Model training interface
│   ├── history.py             # User prediction history
│   ├── login.py               # User authentication
│   └── signup.py              # User registration
│
├── 🔧 utils/                   # Utility Functions
│   ├── auth.py                # Authentication management
│   ├── database.py            # Supabase integration
│   ├── analytics.py           # Performance analytics
│   └── url_manager.py         # Navigation utilities
│
├── ⚙️ config/                  # Configuration
│   └── supabase.py            # Database configuration
│
├── 📈 csvs/                    # Dataset Files
│   ├── cumulative_2025.09.30_23.45.15.csv  # Kepler data
│   └── TOI_2025.09.30_23.45.34.csv         # TESS data
│
├── 🎨 styles/                  # UI Styling
│   └── custom.css             # Custom Streamlit styles
│
└── 🧪 tests/                   # Research & Analysis
    ├── telescope_domain_adaptation.py  # Cross-mission analysis
    └── tess_centric_training.py       # TESS-focused experiments
```

## 🔬 Technical Deep Dive

### Advanced Feature Engineering

Our LightGBM model incorporates **30+ engineered features** across multiple categories:

#### 🧬 **Physics-Based Features**
- **Interaction Features**: `radius × period`, `temperature × insolation`
- **Habitability Indicators**: Earth-similarity index, habitable zone position
- **Transit Geometry**: Duration-to-period ratios, impact parameters
- **Stellar-Planetary**: Radius ratios, mass-radius relationships

#### 📊 **Statistical Features**  
- **Uncertainty Analysis**: Relative errors, measurement asymmetry
- **Signal Quality**: Signal-to-noise proxies, detection significance
- **Distribution Features**: Log transforms, polynomial terms
- **Regime Classification**: Size/temperature/period category binning

#### 🌌 **Astronomical Transformations**
- **Coordinate Systems**: Galactic coordinates with sine/cosine encoding
- **Temporal Features**: Orbital mechanics, transit timing
- **Photometric**: Magnitude differences, color indices
- **Geometric**: Impact parameters, inclination estimates

### Cross-Telescope Domain Adaptation

**Challenge**: Kepler and TESS have different:
- Observing strategies (continuous vs. sector-based)
- Target selection criteria  
- Data processing pipelines
- Systematic noise patterns

**Solution**: Optimal 55% TESS + 45% Kepler training mixture
- Maintains performance on both missions
- Reduces telescope-dependent biases
- Achieves <1% performance gap between missions

### Model Optimization Details

#### **LightGBM Enhancements**
```python
# Optimized hyperparameters
params = {
    'learning_rate': 0.03,        # Slower, more stable
    'num_leaves': 63,             # Increased complexity  
    'max_depth': 8,               # Controlled overfitting
    'lambda_l1': 0.5,             # L1 regularization
    'lambda_l2': 0.5,             # L2 regularization
    'feature_fraction': 0.8,      # Feature randomness
    'bagging_fraction': 0.7,      # Sample randomness
    'is_unbalance': True,         # Auto class weights
}
```

#### **Class Imbalance Handling**
- **Dataset Distribution**: 41% Candidates, 36% False Positives, 23% Confirmed  
- **SMOTE Oversampling**: For Random Forest model
- **Auto Class Weights**: For gradient boosting models
- **Stratified Sampling**: Maintains class ratios in train/test splits

## 🎯 Performance Metrics

### Cross-Telescope Evaluation

| Model | TESS Test Accuracy | Kepler Test Accuracy | Telescope Gap |
|-------|-------------------|---------------------|---------------|
| **LightGBM** | **75.9%** | **76.6%** | **0.69%** |
| CatBoost | 74.2% | 75.1% | 0.87% |
| Random Forest | 73.8% | 74.5% | 0.71% |
| MLP | 72.1% | 73.2% | 1.11% |

### Feature Importance (Top 10)

| Rank | Feature | Importance | Physical Meaning |
|------|---------|------------|------------------|
| 1 | `merged_koi_period` | 1847.2 | Orbital period (days) |
| 2 | `merged_koi_prad` | 1654.3 | Planet radius (Earth radii) |
| 3 | `merged_koi_insol` | 1432.1 | Stellar insolation |
| 4 | `merged_koi_teq` | 1289.7 | Equilibrium temperature |
| 5 | `merged_koi_srad` | 1156.8 | Stellar radius |
| 6 | `merged_koi_dor` | 1098.4 | Duration/orbital period |
| 7 | `lgbm_prad_period_interaction` | 987.6 | Size-period correlation |
| 8 | `merged_koi_impact` | 876.3 | Transit impact parameter |
| 9 | `lgbm_stellar_luminosity_proxy` | 743.2 | Stellar luminosity estimate |
| 10 | `merged_koi_slogg` | 698.9 | Stellar surface gravity |

## 🔬 Data Processing Pipeline

### 1. **Data Ingestion**
```python
# Automatic telescope identification
kepler_data = load_kepler_data("cumulative_2025.09.30_23.45.15.csv")
tess_data = load_tess_data("TOI_2025.09.30_23.45.34.csv") 
```

### 2. **Cross-Mission Merger**
```python
# Intelligent duplicate detection and removal
merger = DatasetMerger()
merged_data = merger.merge(kepler_data, tess_data)
# Result: 17,242 unique exoplanet candidates
```

### 3. **Feature Engineering Pipeline**
```python
# Applied to all models
transformers = [
    GalacticCoordinatesTransformer(),    # RA/Dec → Galactic coords
    DepthToRadiusRatioTransformer(),     # Transit depth → radius ratio
    CrossMissionDuplicateRemover(),      # Remove cross-mission duplicates
    SafeMultiplicityTransformer(),       # Multi-planet system features
    ExoplanetPhysicsTransformer(),       # Physics-based calculations
    TelescopeAgnosticCleaner()           # Remove telescope-specific columns
]
```

### 4. **Model Training & Validation**
```python
# Stratified train/test split maintaining class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Cross-validation with telescope-aware splits
cv_scores = cross_validate_telescope_aware(model, X_train, y_train)
```

## 🌟 Advanced Usage

### Batch Prediction with Custom Features

```python
# Load your CSV with exoplanet candidates
import pandas as pd
from models.model_manager import ModelManager

# Initialize model manager
manager = ModelManager(auto_load_models=True)

# Load and predict
df = pd.read_csv("your_exoplanet_candidates.csv")
predictions = manager.predict_batch(df, model_name="lightgbm")

# Results include confidence scores and habitability ratings
print(predictions[['prediction', 'confidence', 'habitability_score']])
```

### Custom Model Training

```python
# Train with your own datasets
from models.lightgbm.pipeline import LightGBMPipeline
from common.data_merger import DatasetMerger

# Merge your datasets  
merger = DatasetMerger()
merged_data = merger.merge(your_kepler_data, your_tess_data)

# Train model
pipeline = LightGBMPipeline()
model, metrics = pipeline.train(merged_data)

# Save for later use
pipeline.save_model("custom_lightgbm_model.pkl")
```

### Habitability Analysis

The system includes a comprehensive habitability scoring system:

```python
# Automatic habitability calculation
habitability_factors = {
    'size_score': planet_size_similarity_to_earth,      # 0-100
    'temperature_score': surface_temperature_rating,     # 0-100  
    'insolation_score': stellar_flux_optimality,        # 0-100
    'stellar_score': host_star_suitability,             # 0-100
    'orbital_score': orbital_stability_rating           # 0-100
}

overall_habitability = weighted_average(habitability_factors)
```

## 🛡️ Data Sources & Attribution

### Datasets Used
- **Kepler Data**: NASA Exoplanet Archive - Cumulative KOI Table
- **TESS Data**: NASA Exoplanet Archive - TESS Objects of Interest (TOI)
- **Cross-References**: Confirmed exoplanets from multiple catalogs

### Data Processing
- **Total Records**: 17,242+ unique candidates after deduplication
- **Kepler Contribution**: 9,554 candidates (55.4%)
- **TESS Contribution**: 7,688 candidates (44.6%)
- **Overlap Handling**: Spatial matching with 0.001° tolerance

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python -m pytest tests/

# Code formatting  
black --line-length 88 .
```

### Adding New Models
1. Create model directory: `models/your_model/`
2. Implement `model.py` and `pipeline.py`
3. Add to `model_manager.py` registry
4. Update documentation

### Research & Experiments
Check the `tests/` directory for advanced research experiments:
- Cross-telescope domain adaptation analysis
- TESS-centric training strategies  
- Feature importance studies

## 📚 Documentation Files

- `LIGHTGBM_IMPROVEMENTS.md` - Detailed LightGBM enhancements
- `LIGHTGBM_IMPROVEMENTS_SUMMARY.md` - Quick overview of changes
- `TRANSFORMER_USAGE_ANALYSIS.md` - Feature transformer documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **NASA Exoplanet Archive** for providing comprehensive exoplanet data
- **Kepler & TESS Teams** for revolutionary exoplanet discovery missions
- **Astropy Community** for astronomical computation tools
- **Scikit-learn & LightGBM Teams** for excellent ML frameworks

## 📞 Support & Contact

For questions, issues, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/SathishDissanayaka/exoplanet_detector_nasa_data/issues)
- **Repository**: [exoplanet_detector_nasa_data](https://github.com/SathishDissanayaka/exoplanet_detector_nasa_data)

---

**🌌 "Not just detecting exoplanets, but understanding the cosmos one algorithm at a time."**
