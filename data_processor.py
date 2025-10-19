# dataprocessor.py
import streamlit as st
import pandas as pd
from random_forest_model import detect_with_random_forest
# from other_model import run_their_model  # Teammates can add their models

# Map model names to functions
model_map = {
    "random_forest": detect_with_random_forest,
    # "their_model_name": run_their_model,  # Add teammate models here
}

def process_data(df: pd.DataFrame, model_type: str = "random_forest") -> None:
    """
    Process the uploaded CSV data and run the selected model.
    
    Args:
        df (pd.DataFrame): Uploaded Kepler data
        model_type (str): Which model to run ("random_forest", "their_model_name", etc.)
    """
    if df is None or df.empty:
        st.error("No data uploaded or empty file.")
        return

    st.write("Data received successfully. Running model...")

    if model_type in model_map:
        st.write(f"Running {model_type} model...")
        model_map[model_type](df)
    else:
        st.error(f"Model '{model_type}' is not implemented yet.")
