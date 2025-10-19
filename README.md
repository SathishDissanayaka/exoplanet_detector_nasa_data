# Exoplanet Detector

This project is a web application for detecting exoplanets from Kepler and TESS telescope data using various machine learning models.

## Setup

Follow these steps to set up your local development environment.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd exoplanet-detector-python
```

### 2. Create and Activate a Virtual Environment

It's recommended to use a virtual environment to manage project dependencies.

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate it (on macOS/Linux)
source .venv/bin/activate

# On Windows, use:
# .venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Initial Model Training

Before you can run the application, you need to train the machine learning models. This is a one-time step. The trained models will be saved to the `models/` directory.

Run the following script and provide the paths to your Kepler and TESS datasets. Example CSV files are located in the `csvs/` directory.

```bash
python train_all_models_initial.py --kepler csvs/cumulative_2025.09.30_23.45.15.csv --tess csvs/TOI_2025.09.30_23.45.34.csv
```

This will train all the default models (`lightgbm`, `catboost`, `random_forest`, `mlp`). You can specify which models to train using the `--models` argument:

```bash
python train_all_models_initial.py --kepler <path_to_kepler.csv> --tess <path_to_tess.csv> --models lightgbm random_forest
```

## Running the Application

Once the models are trained and saved, you can start the Streamlit web application.

```bash
streamlit run app.py
```

This will open the application in your web browser.
