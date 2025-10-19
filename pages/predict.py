import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from config.supabase import supabase
from models.model_manager import ModelManager
from common.data_merger import DatasetMerger
import io

def show_predict_page():
    st.title("🔭 Exoplanet Detection")
    
    # Use the global model manager from session state (loaded at app start)
    if "model_manager" not in st.session_state:
        st.error("⚠️ Model manager not initialized. Please restart the application.")
        return
    
    model_manager = st.session_state.model_manager
    
    # Check if any models are trained
    trained_models = [name for name, config in model_manager.available_models.items() 
                     if config['trained'] and config['trained_model'] is not None]
    
    if not trained_models:
        st.error("⚠️ No pre-trained models found!")
        st.warning("""
        ### � Models need to be trained first
        
        **For System Initialization:**
        - Train models using the scripts in the `models/` directory
        - Or use the Train Model page to create and save models
        - Models will be automatically loaded on app restart
        
        **For Demonstration:**
        - Go to the **Train Model** page to see the training process
        - This is for showing your lecturers how the system works
        
        Pre-trained models should be available at `/models/*.pkl` files.
        """)
        return
    
    st.success(f"✅ {len(trained_models)} pre-trained model(s) loaded and ready!")
    
    # Initialize data merger for column mapping
    merger = DatasetMerger()
    
    # Input method selection
    st.markdown("### Choose Input Method")
    input_method = st.radio(
        "How would you like to provide data?",
        ["Manual Form Input", "CSV Bulk Upload"],
        horizontal=True
    )
    
    # Model selection (only show trained models)
    selected_model = st.selectbox(
        "Select Detection Model",
        trained_models,
        help="Choose the model to use for detection. Different models may have different strengths."
    )
    
    st.markdown("---")
    
    if input_method == "Manual Form Input":
        show_manual_form(model_manager, selected_model, merger)
    else:
        show_csv_upload(model_manager, selected_model, merger)


def show_manual_form(model_manager, selected_model, merger):
    """Display manual form input interface"""
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Enter Celestial Object Data")
        st.write("Input the measurements from your celestial object to check if it might be an exoplanet.")
        
        # Get all possible attributes from the mapping
        all_attributes = get_all_attributes(merger)
        
        # Feature input form with improved layout
        with st.form("prediction_form"):
            feature_values = {}
            
            # Create tabs for different categories
            tab1, tab2, tab3, tab4 = st.tabs(["📍 Position", "🌍 Planet Properties", "⭐ Star Properties", "⏱️ Transit Properties"])
            
            with tab1:
                st.markdown("#### Position Coordinates")
                col_a, col_b = st.columns(2)
                with col_a:
                    feature_values['ra'] = st.number_input(
                        "Right Ascension (degrees)",
                        value=0.0,
                        help="Celestial coordinate - Right Ascension"
                    )
                with col_b:
                    feature_values['dec'] = st.number_input(
                        "Declination (degrees)",
                        value=0.0,
                        help="Celestial coordinate - Declination"
                    )
            
            with tab2:
                st.markdown("#### Planet Properties")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    feature_values['koi_prad'] = st.number_input(
                        "Planet Radius (Earth radii)",
                        min_value=0.0,
                        value=1.0,
                        help="Radius of the planet in Earth radii"
                    )
                    if selected_model not in ['random_forest']:
                        # Initialize session state for error values if not exists
                        if 'koi_prad_err1' not in st.session_state:
                            st.session_state.koi_prad_err1 = 0.0
                        if 'koi_prad_err2' not in st.session_state:
                            st.session_state.koi_prad_err2 = 0.0
                        
                        # Use number_input with session state
                        feature_values['koi_prad_err1'] = st.number_input(
                            "Planet Radius Error (+)",
                            value=st.session_state.koi_prad_err1,
                            format="%.6f",
                            key='koi_prad_err1'
                        )
                        
                        feature_values['koi_prad_err2'] = st.number_input(
                            "Planet Radius Error (-)",
                            value=st.session_state.koi_prad_err2,
                            format="%.6f",
                            key='koi_prad_err2'
                        )
                
                with col_b:
                    feature_values['koi_teq'] = st.number_input(
                        "Equilibrium Temperature (K)",
                        min_value=0.0,
                        value=300.0,
                        help="Estimated equilibrium temperature"
                    )
                    if selected_model not in ['catboost', 'lightgbm', 'random_forest']:
                        # Initialize session state for error values if not exists
                        if 'koi_teq_err1' not in st.session_state:
                            st.session_state.koi_teq_err1 = 0.0
                        if 'koi_teq_err2' not in st.session_state:
                            st.session_state.koi_teq_err2 = 0.0
                        
                        # Use number_input with session state
                        feature_values['koi_teq_err1'] = st.number_input(
                            "Equilibrium Temp Error (+)",
                            value=st.session_state.koi_teq_err1,
                            format="%.6f",
                            key='koi_teq_err1'
                        )
                        
                        feature_values['koi_teq_err2'] = st.number_input(
                            "Equilibrium Temp Error (-)",
                            value=st.session_state.koi_teq_err2,
                            format="%.6f",
                            key='koi_teq_err2'
                        )
                
                st.markdown("#### Insolation Flux")
                col_c, col_d = st.columns(2)
                with col_c:
                    feature_values['koi_insol'] = st.number_input(
                        "Insolation Flux (Earth flux)",
                        min_value=0.0,
                        value=1.0,
                        help="Stellar flux at planet's position"
                    )
                if selected_model not in ['random_forest']:
                    with col_d:
                        # Initialize session state for error values if not exists
                        if 'koi_insol_err1' not in st.session_state:
                            st.session_state.koi_insol_err1 = 0.0
                        if 'koi_insol_err2' not in st.session_state:
                            st.session_state.koi_insol_err2 = 0.0
                        
                        # Use number_input with session state
                        feature_values['koi_insol_err1'] = st.number_input(
                            "Insolation Flux Error (+)",
                            value=st.session_state.koi_insol_err1,
                            format="%.6f",
                            key='koi_insol_err1'
                        )
                        
                        feature_values['koi_insol_err2'] = st.number_input(
                            "Insolation Flux Error (-)",
                            value=st.session_state.koi_insol_err2,
                            format="%.6f",
                            key='koi_insol_err2'
                        )
            
            with tab3:
                st.markdown("#### Star Properties")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    feature_values['koi_srad'] = st.number_input(
                        "Star Radius (Solar radii)",
                        min_value=0.0,
                        value=1.0,
                        help="Radius of the host star"
                    )
                    if selected_model not in ['random_forest']:
                        # Initialize session state for error values if not exists
                        if 'koi_srad_err1' not in st.session_state:
                            st.session_state.koi_srad_err1 = 0.0
                        if 'koi_srad_err2' not in st.session_state:
                            st.session_state.koi_srad_err2 = 0.0
                        
                        # Use number_input with session state
                        feature_values['koi_srad_err1'] = st.number_input(
                            "Star Radius Error (+)",
                            value=st.session_state.koi_srad_err1,
                            format="%.6f",
                            key='koi_srad_err1'
                        )
                        
                        feature_values['koi_srad_err2'] = st.number_input(
                            "Star Radius Error (-)",
                            value=st.session_state.koi_srad_err2,
                            format="%.6f",
                            key='koi_srad_err2'
                        )
                    
                    feature_values['koi_steff'] = st.number_input(
                        "Stellar Effective Temperature (K)",
                        min_value=0.0,
                        value=5778.0,
                        help="Effective temperature of the host star"
                    )
                    if selected_model not in ['random_forest']:
                        # Initialize session state for error values if not exists
                        if 'koi_steff_err1' not in st.session_state:
                            st.session_state.koi_steff_err1 = 0.0
                        if 'koi_steff_err2' not in st.session_state:
                            st.session_state.koi_steff_err2 = 0.0
                        
                        # Use number_input with session state
                        feature_values['koi_steff_err1'] = st.number_input(
                            "Stellar Temp Error (+)",
                            value=st.session_state.koi_steff_err1,
                            format="%.6f",
                            key='koi_steff_err1'
                        )
                        
                        feature_values['koi_steff_err2'] = st.number_input(
                            "Stellar Temp Error (-)",
                            value=st.session_state.koi_steff_err2,
                            format="%.6f",
                            key='koi_steff_err2'
                        )
                
                with col_b:
                    feature_values['koi_slogg'] = st.number_input(
                        "Stellar Surface Gravity (log10(cm/s²))",
                        value=4.4,
                        help="Surface gravity of the host star"
                    )
                    if selected_model not in ['random_forest']:
                        # Initialize session state for error values if not exists
                        if 'koi_slogg_err1' not in st.session_state:
                            st.session_state.koi_slogg_err1 = 0.0
                        if 'koi_slogg_err2' not in st.session_state:
                            st.session_state.koi_slogg_err2 = 0.0
                        
                        # Use number_input with session state
                        feature_values['koi_slogg_err1'] = st.number_input(
                            "Surface Gravity Error (+)",
                            value=st.session_state.koi_slogg_err1,
                            format="%.6f",
                            key='koi_slogg_err1'
                        )
                        
                        feature_values['koi_slogg_err2'] = st.number_input(
                            "Surface Gravity Error (-)",
                            value=st.session_state.koi_slogg_err2,
                            format="%.6f",
                            key='koi_slogg_err2'
                        )
            
            with tab4:
                st.markdown("#### Transit Properties")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    feature_values['koi_time0'] = st.number_input(
                        "Transit Epoch (BJD)",
                        value=0.0,
                        help="Time of first transit"
                    )
                    if selected_model not in ['random_forest']:
                        # Initialize session state for error values if not exists
                        if 'koi_time0_err1' not in st.session_state:
                            st.session_state.koi_time0_err1 = 0.0
                        if 'koi_time0_err2' not in st.session_state:
                            st.session_state.koi_time0_err2 = 0.0
                        
                        # Use number_input with session state
                        feature_values['koi_time0_err1'] = st.number_input(
                            "Transit Epoch Error (+)",
                            value=st.session_state.koi_time0_err1,
                            format="%.6f",
                            key='koi_time0_err1'
                        )
                        
                        feature_values['koi_time0_err2'] = st.number_input(
                            "Transit Epoch Error (-)",
                            value=st.session_state.koi_time0_err2,
                            format="%.6f",
                            key='koi_time0_err2'
                        )
                    
                    feature_values['koi_period'] = st.number_input(
                        "Orbital Period (days)",
                        min_value=0.0,
                        value=1.0,
                        help="Time between transits"
                    )
                    if selected_model not in ['random_forest']:
                        # Initialize session state for error values if not exists
                        if 'koi_period_err1' not in st.session_state:
                            st.session_state.koi_period_err1 = 0.0
                        if 'koi_period_err2' not in st.session_state:
                            st.session_state.koi_period_err2 = 0.0
                        
                        # Use number_input with session state
                        feature_values['koi_period_err1'] = st.number_input(
                            "Orbital Period Error (+)",
                            value=st.session_state.koi_period_err1,
                            format="%.6f",
                            key='koi_period_err1'
                        )
                        
                        feature_values['koi_period_err2'] = st.number_input(
                            "Orbital Period Error (-)",
                            value=st.session_state.koi_period_err2,
                            format="%.6f",
                            key='koi_period_err2'
                        )
                
                with col_b:
                    feature_values['koi_duration'] = st.number_input(
                        "Transit Duration (hours)",
                        min_value=0.0,
                        value=3.0,
                        help="Duration of the transit"
                    )
                    if selected_model not in ['random_forest']:
                        # Initialize session state for error values if not exists
                        if 'koi_duration_err1' not in st.session_state:
                            st.session_state.koi_duration_err1 = 0.0
                        if 'koi_duration_err2' not in st.session_state:
                            st.session_state.koi_duration_err2 = 0.0
                        
                        # Use number_input with session state
                        feature_values['koi_duration_err1'] = st.number_input(
                            "Transit Duration Error (+)",
                            value=st.session_state.koi_duration_err1,
                            format="%.6f",
                            key='koi_duration_err1'
                        )
                        
                        feature_values['koi_duration_err2'] = st.number_input(
                            "Transit Duration Error (-)",
                            value=st.session_state.koi_duration_err2,
                            format="%.6f",
                            key='koi_duration_err2'
                        )
                    
                    feature_values['koi_depth'] = st.number_input(
                        "Transit Depth (ppm)",
                        min_value=0.0,
                        value=1000.0,
                        help="Depth of transit in parts per million"
                    )
                    if selected_model not in ['random_forest']:
                        # Initialize session state for error values if not exists
                        if 'koi_depth_err1' not in st.session_state:
                            st.session_state.koi_depth_err1 = 0.0
                        if 'koi_depth_err2' not in st.session_state:
                            st.session_state.koi_depth_err2 = 0.0
                        
                        # Use number_input with session state
                        feature_values['koi_depth_err1'] = st.number_input(
                            "Transit Depth Error (+)",
                            value=st.session_state.koi_depth_err1,
                            format="%.6f",
                            key='koi_depth_err1'
                        )
                        
                        feature_values['koi_depth_err2'] = st.number_input(
                            "Transit Depth Error (-)",
                            value=st.session_state.koi_depth_err2,
                            format="%.6f",
                            key='koi_depth_err2'
                        )
            
            submitted = st.form_submit_button("🚀 Detect Exoplanet", use_container_width=True)
    
    with col2:
        st.markdown("### Detection Guidelines")
        st.info("""
        **Tips for accurate detection:**
        
        1. Ensure all measurements are in correct units
        2. Double-check the values before submission
        3. Consider using multiple models for verification
        """)
        
        # Show feature categories
        st.markdown("### Input Categories")
        st.write("📍 **Position**: RA, Dec")
        st.write("🌍 **Planet**: Radius, Temperature, Insolation")
        st.write("⭐ **Star**: Radius, Temperature, Surface Gravity")
        st.write("⏱️ **Transit**: Epoch, Period, Duration, Depth")
    
    # Handle form submission
    if submitted:
        process_single_prediction(model_manager, selected_model, feature_values)


def show_csv_upload(model_manager, selected_model, merger):
    """Display CSV upload and column mapping interface"""
    
    # SHOW SAVE BUTTON FIRST if results exist (at top level to avoid nesting issues)
    if 'batch_results' in st.session_state and st.session_state.batch_results is not None:
        st.success(f"✅ Batch prediction completed! {len(st.session_state.batch_results)} results ready.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"📊 **{len(st.session_state.batch_results)}** predictions from **{st.session_state.get('batch_model', 'unknown')}** model")
        with col2:
            if st.button("🗑️ Clear", key="clear_results"):
                st.session_state.batch_results = None
                st.rerun()
        
        # Save button at TOP LEVEL
        if st.button("💾 SAVE ALL RESULTS TO DATABASE", use_container_width=True, key="save_batch_db", type="primary"):
            user = st.session_state.get('user')
            
            if not user:
                st.error("🔒 Please log in to save predictions to your account.")
            else:
                with st.spinner("Saving predictions..."):
                    try:
                        results_df = st.session_state.batch_results
                        saved_model = st.session_state.get('batch_model', 'unknown')
                        success_count = 0
                        
                        for idx, row in results_df.iterrows():
                            data = {
                                'user_id': str(user.id),
                                'model_name': saved_model,
                                'prediction': row['Prediction'],
                                'confidence': float(row['Confidence']),
                                'features': row.get('Features', {})  # Include features if available
                            }
                            response = supabase.table('predictions').insert(data).execute()
                            success_count += 1
                        
                        st.success(f"✅ Saved {success_count} predictions!")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        with st.expander("Details"):
                            st.code(str(e))
        
        st.markdown("---")
    
    st.markdown("### Upload CSV File")
    st.write("Upload a CSV file with multiple celestial objects for batch prediction.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with your data"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
            
            # Show preview
            with st.expander("📊 Preview Uploaded Data (first 5 rows)", expanded=True):
                st.dataframe(df.head())
            
            # Initialize session state for mappings if not exists (MUST BE FIRST!)
            if 'column_mappings' not in st.session_state:
                st.session_state.column_mappings = {}
            
            # Column mapping interface
            st.markdown("### 🔄 Map Your Columns")
            st.write("Map your CSV columns to the attributes expected by our models. Type the column name or use the pick button.")
            
            # Show mapping progress
            total_attrs = 32  # Total possible attributes
            mapped_count = len([v for v in st.session_state.column_mappings.values() if v])
            
            progress_col1, progress_col2, progress_col3 = st.columns([2, 1, 1])
            with progress_col1:
                st.progress(mapped_count / total_attrs, text=f"Mapped: {mapped_count}/{total_attrs} attributes")
            with progress_col2:
                st.metric("✅ Mapped", mapped_count)
            with progress_col3:
                st.metric("⏳ Remaining", total_attrs - mapped_count)
            
            st.markdown("---")
            
            # Get all possible attributes
            all_attributes = get_all_attributes(merger)
            
            # Create mapping interface with tabs
            map_tab1, map_tab2, map_tab3, map_tab4 = st.tabs([
                "📍 Position", "🌍 Planet Properties", "⭐ Star Properties", "⏱️ Transit Properties"
            ])
            
            column_mappings = {}
            
            with map_tab1:
                st.markdown("#### Position Coordinates")
                column_mappings['ra'] = create_column_selector(df, 'ra', "Right Ascension", all_attributes)
                column_mappings['dec'] = create_column_selector(df, 'dec', "Declination", all_attributes)
            
            with map_tab2:
                st.markdown("#### Planet Properties")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Planet Radius**")
                    column_mappings['koi_prad'] = create_column_selector(df, 'koi_prad', "Planet Radius", all_attributes)
                    if selected_model not in ['random_forest']:
                        column_mappings['koi_prad_err1'] = create_column_selector(df, 'koi_prad_err1', "Planet Radius Error (+)", all_attributes)
                        column_mappings['koi_prad_err2'] = create_column_selector(df, 'koi_prad_err2', "Planet Radius Error (-)", all_attributes)
                
                with col_b:
                    st.markdown("**Equilibrium Temperature**")
                    column_mappings['koi_teq'] = create_column_selector(df, 'koi_teq', "Equilibrium Temperature", all_attributes)
                    if selected_model not in ['catboost', 'lightgbm', 'random_forest']:
                        column_mappings['koi_teq_err1'] = create_column_selector(df, 'koi_teq_err1', "Equilibrium Temp Error (+)", all_attributes)
                        column_mappings['koi_teq_err2'] = create_column_selector(df, 'koi_teq_err2', "Equilibrium Temp Error (-)", all_attributes)
                
                st.markdown("**Insolation Flux**")
                col_c, col_d = st.columns(2)
                with col_c:
                    column_mappings['koi_insol'] = create_column_selector(df, 'koi_insol', "Insolation Flux", all_attributes)
                if selected_model not in ['random_forest']:
                    column_mappings['koi_insol_err1'] = create_column_selector(df, 'koi_insol_err1', "Insolation Flux Error (+)", all_attributes)
                    with col_d:
                        column_mappings['koi_insol_err2'] = create_column_selector(df, 'koi_insol_err2', "Insolation Flux Error (-)", all_attributes)
            
            with map_tab3:
                st.markdown("#### Star Properties")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Star Radius**")
                    column_mappings['koi_srad'] = create_column_selector(df, 'koi_srad', "Star Radius", all_attributes)
                    if selected_model not in ['random_forest']:
                        column_mappings['koi_srad_err1'] = create_column_selector(df, 'koi_srad_err1', "Star Radius Error (+)", all_attributes)
                        column_mappings['koi_srad_err2'] = create_column_selector(df, 'koi_srad_err2', "Star Radius Error (-)", all_attributes)
                    
                    st.markdown("**Stellar Effective Temperature**")
                    column_mappings['koi_steff'] = create_column_selector(df, 'koi_steff', "Stellar Effective Temperature", all_attributes)
                    if selected_model not in ['random_forest']:
                        column_mappings['koi_steff_err1'] = create_column_selector(df, 'koi_steff_err1', "Stellar Temp Error (+)", all_attributes)
                        column_mappings['koi_steff_err2'] = create_column_selector(df, 'koi_steff_err2', "Stellar Temp Error (-)", all_attributes)
                
                with col_b:
                    st.markdown("**Stellar Surface Gravity**")
                    column_mappings['koi_slogg'] = create_column_selector(df, 'koi_slogg', "Stellar Surface Gravity", all_attributes)
                    if selected_model not in ['random_forest']:
                        column_mappings['koi_slogg_err1'] = create_column_selector(df, 'koi_slogg_err1', "Surface Gravity Error (+)", all_attributes)
                        column_mappings['koi_slogg_err2'] = create_column_selector(df, 'koi_slogg_err2', "Surface Gravity Error (-)", all_attributes)
            
            with map_tab4:
                st.markdown("#### Transit Properties")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Transit Epoch**")
                    column_mappings['koi_time0'] = create_column_selector(df, 'koi_time0', "Transit Epoch", all_attributes)
                    if selected_model not in ['random_forest']:
                        column_mappings['koi_time0_err1'] = create_column_selector(df, 'koi_time0_err1', "Transit Epoch Error (+)", all_attributes)
                        column_mappings['koi_time0_err2'] = create_column_selector(df, 'koi_time0_err2', "Transit Epoch Error (-)", all_attributes)
                    
                    st.markdown("**Orbital Period**")
                    column_mappings['koi_period'] = create_column_selector(df, 'koi_period', "Orbital Period", all_attributes)
                    if selected_model not in ['random_forest']:
                        column_mappings['koi_period_err1'] = create_column_selector(df, 'koi_period_err1', "Orbital Period Error (+)", all_attributes)
                        column_mappings['koi_period_err2'] = create_column_selector(df, 'koi_period_err2', "Orbital Period Error (-)", all_attributes)
                
                with col_b:
                    st.markdown("**Transit Duration**")
                    column_mappings['koi_duration'] = create_column_selector(df, 'koi_duration', "Transit Duration", all_attributes)
                    if selected_model not in ['random_forest']:
                        column_mappings['koi_duration_err1'] = create_column_selector(df, 'koi_duration_err1', "Transit Duration Error (+)", all_attributes)
                        column_mappings['koi_duration_err2'] = create_column_selector(df, 'koi_duration_err2', "Transit Duration Error (-)", all_attributes)
                    
                    st.markdown("**Transit Depth**")
                    column_mappings['koi_depth'] = create_column_selector(df, 'koi_depth', "Transit Depth", all_attributes)
                    if selected_model not in ['random_forest']:
                        column_mappings['koi_depth_err1'] = create_column_selector(df, 'koi_depth_err1', "Transit Depth Error (+)", all_attributes)
                        column_mappings['koi_depth_err2'] = create_column_selector(df, 'koi_depth_err2', "Transit Depth Error (-)", all_attributes)
            
            # Action buttons
            st.markdown("---")
            
            # Utility buttons in 3 columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Auto-Map", use_container_width=True, help="Automatically map columns based on name similarity"):
                    auto_mapped = auto_map_columns(df, all_attributes)
                    st.session_state.column_mappings = auto_mapped
                    st.success(f"✅ Auto-mapped {len([v for v in auto_mapped.values() if v])} columns!")
                    st.rerun()
            
            with col2:
                if st.button("👁️ Show Mapped", use_container_width=True, help="Show only mapped columns"):
                    mapped_count = len([k for k, v in column_mappings.items() if v])
                    st.info(f"**Mapped Columns:** {mapped_count}\n\n" + 
                           "\n".join([f"✓ {k} → {v}" for k, v in column_mappings.items() if v]))
            
            with col3:
                if st.button("🧹 Clear All", use_container_width=True, help="Clear all column mappings"):
                    st.session_state.column_mappings = {}
                    st.success("✅ Mappings cleared!")
                    st.rerun()
            
            # Predict button on its own row
            st.markdown("---")
            if st.button("🚀 Predict", use_container_width=True, type="primary", help="Run batch prediction with current mappings", key="predict_button"):
                process_batch_prediction(df, column_mappings, model_manager, selected_model, merger)
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
    else:
        # Show instructions when no file is uploaded
        st.info("""
        ### 📋 CSV Upload Instructions
        
        1. **Prepare your CSV file** with columns containing celestial object measurements
        2. **Upload the file** using the uploader above
        3. **Map your columns** to the expected attributes
        4. **Run batch prediction** to get results for all objects
        
        #### Expected Data Format
        Your CSV can have any column names, but should include measurements for:
        - Position (RA, Dec)
        - Planet properties (radius, temperature, insolation)
        - Star properties (radius, temperature, surface gravity)
        - Transit properties (epoch, period, duration, depth)
        
        #### Sample CSV Structure
        ```
        ra, dec, planet_radius, star_radius, star_temp, planet_temp, ...
        123.45, 67.89, 1.2, 0.98, 5500, 300, ...
        ```
        """)


def get_all_attributes(merger):
    """Get all possible attribute names from the merger mapping"""
    # Get base attributes from KOI naming
    base_attrs = list(merger.TESS_TO_KOI_MAPPING.values())
    # Add merged_ prefix for unified names
    merged_attrs = [f"merged_{attr}" for attr in base_attrs]
    return base_attrs + merged_attrs


def create_column_selector(df, target_attr, label, all_attributes):
    """Create a column selector for mapping with dropdown selection"""
    # Check if there's a saved mapping
    saved_mapping = st.session_state.column_mappings.get(target_attr, "")
    
    # Try to find a matching column automatically
    suggested_value = saved_mapping if saved_mapping else ""
    
    if not suggested_value:
        if target_attr in df.columns:
            suggested_value = target_attr
        else:
            # Try to find similar column names
            for col in df.columns:
                if target_attr.lower() in col.lower() or col.lower() in target_attr.lower():
                    suggested_value = col
                    break
    
    # Create options list with empty option first
    options = [""] + list(df.columns)
    
    # Find the index of the suggested value
    try:
        default_index = options.index(suggested_value) if suggested_value else 0
    except ValueError:
        default_index = 0
    
    # Use selectbox for simple dropdown selection
    selected = st.selectbox(
        label,
        options=options,
        index=default_index,
        key=f"map_{target_attr}",
        help=f"Select the column from your CSV that corresponds to {label}"
    )
    
    # Return None if empty
    if not selected or selected.strip() == "":
        return None
    
    return selected


def auto_map_columns(df, all_attributes):
    """Automatically map CSV columns to expected attributes"""
    mappings = {}
    df_columns_lower = {col.lower(): col for col in df.columns}
    
    # Try to match based on attribute names
    for attr in all_attributes:
        attr_lower = attr.lower()
        
        # Direct match
        if attr_lower in df_columns_lower:
            mappings[attr] = df_columns_lower[attr_lower]
            continue
        
        # Partial match
        for df_col_lower, df_col in df_columns_lower.items():
            if attr_lower in df_col_lower or df_col_lower in attr_lower:
                mappings[attr] = df_col
                break
    
    return mappings


def apply_column_mapping(df, column_mappings):
    """Apply column mappings to create a standardized DataFrame"""
    mapped_df = pd.DataFrame()
    
    for target_attr, source_col in column_mappings.items():
        if source_col is not None:
            mapped_df[target_attr] = df[source_col]
    
    return mapped_df


def process_single_prediction(model_manager, selected_model, feature_values):
    """Process a single prediction from form input"""
    model_config = model_manager.available_models[selected_model]
    
    if not model_config['trained'] or model_config['trained_model'] is None:
        st.error(f"❌ Model **{selected_model}** is not trained yet.")
        st.info("💡 Please go to the **Train Model** page to train this model first, or select a different model.")
        
        # Show which models are available
        trained_models = [name for name, config in model_manager.available_models.items() 
                         if config['trained'] and config['trained_model'] is not None]
        
        if trained_models:
            st.success(f"✅ Available trained models: {', '.join(trained_models)}")
        else:
            st.warning("⚠️ No models are currently trained. Please train at least one model.")
        return
    
    try:
        with st.spinner("Analyzing celestial object..."):
            # Convert to DataFrame for preprocessing
            df = pd.DataFrame([feature_values])
            
            # Add mission tag (required for preprocessing)
            df['mission'] = 'UNKNOWN'
            
            # Get the pipeline for preprocessing
            pipeline = model_manager.available_models[selected_model]['pipeline']
            
            # Apply preprocessing (same as training)
            # The pipeline expects columns to start with 'merged_'
            df.columns = [f'merged_{col}' if col != 'mission' else col for col in df.columns]
            processed_df = pipeline.preprocess_for_inference(df)
            
            # Prepare features
            X, _ = pipeline.prepare_features(processed_df)
            
            # Get trained model
            trained_model = model_manager.available_models[selected_model]['trained_model']
            
            # Make prediction
            if selected_model == 'lightgbm':
                pred_proba = trained_model.predict(X, num_iteration=trained_model.best_iteration)
                pred_proba_flat = pred_proba[0] if pred_proba.ndim > 1 else pred_proba
                pred_class = np.argmax(pred_proba_flat)
            else:
                pred_class = trained_model.predict(X)[0]
                pred_proba_flat = trained_model.predict_proba(X)[0]
            
            # Decode prediction
            prediction_label = pipeline.label_encoder.inverse_transform([pred_class])[0]
            
            result = {
                'prediction': prediction_label,
                'probability': float(pred_proba_flat[pred_class]),
                'probabilities': {
                    class_name: float(prob)
                    for class_name, prob in zip(pipeline.label_encoder.classes_, pred_proba_flat)
                }
            }
            
            display_prediction_results(result, selected_model, feature_values)
    
    except Exception as e:
        st.error(f"❌ Error during detection: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def process_batch_prediction(df, column_mappings, model_manager, selected_model, merger):
    """Process batch predictions from CSV upload"""
    model_config = model_manager.available_models[selected_model]
    
    if not model_config['trained'] or model_config['trained_model'] is None:
        st.error(f"Model **{selected_model}** is not trained yet.")
        st.info("💡 Please go to the **Train Model** page to train this model first, or select a different model.")
        
        # Show which models are available
        trained_models = [name for name, config in model_manager.available_models.items() 
                         if config['trained'] and config['trained_model'] is not None]
        
        if trained_models:
            st.success(f"Available trained models: {', '.join(trained_models)}")
        else:
            st.warning("No models are currently trained. Please train at least one model.")
        return
    
    try:
        with st.spinner(f"Processing {len(df)} objects..."):
            # Apply column mapping
            mapped_df = apply_column_mapping(df, column_mappings)
            
            if mapped_df.empty:
                st.error("No columns were mapped. Please map at least some columns before running prediction.")
                return
            
            # Add mission tag
            mapped_df['mission'] = 'UNKNOWN'
            
            # Get the pipeline for preprocessing
            pipeline = model_manager.available_models[selected_model]['pipeline']

            # The pipeline expects columns to start with 'merged_'
            # We rename the columns from the mapping to include the prefix
            rename_dict = {
                target_attr: f'merged_{target_attr}'
                for target_attr in column_mappings.keys()
                if target_attr != 'mission'
            }
            renamed_df = mapped_df.rename(columns=rename_dict)
            
            # Apply preprocessing (same as training)
            processed_df = pipeline.preprocess_for_inference(renamed_df)
            
            # Prepare features
            X, _ = pipeline.prepare_features(processed_df)
            
            # Get trained model
            trained_model = model_manager.available_models[selected_model]['trained_model']
            
            # Make predictions
            if selected_model == 'lightgbm':
                pred_proba = trained_model.predict(X, num_iteration=trained_model.best_iteration)
                pred_classes = np.argmax(pred_proba, axis=1)
            else:
                pred_classes = trained_model.predict(X)
                pred_proba = trained_model.predict_proba(X)
            
            # Ensure pred_classes is 1D array
            pred_classes = np.array(pred_classes).flatten()
            
            # Decode predictions
            predictions = pipeline.label_encoder.inverse_transform(pred_classes)
            
            # Extract confidences safely as scalars
            confidences = []
            for i in range(len(pred_classes)):
                class_idx = int(pred_classes[i])
                confidence = float(pred_proba[i][class_idx])
                confidences.append(confidence)
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'Prediction': predictions,
                'Confidence': confidences
            })
            
            # Store the mapped features for database insertion
            # Convert each row of mapped_df to a dict for the features column
            features_list = []
            for idx in range(len(mapped_df)):
                feature_dict = mapped_df.iloc[idx].to_dict()
                # Convert numpy types to native Python types for JSON serialization
                # Handle NaN, None, and infinity values
                serializable_features = {}
                for k, v in feature_dict.items():
                    if k == 'mission':
                        continue
                    if pd.isna(v) or v is None:
                        serializable_features[k] = None
                    elif isinstance(v, (int, float, np.number)):
                        # Check for infinity
                        if np.isinf(v):
                            serializable_features[k] = None
                        else:
                            serializable_features[k] = float(v)
                    else:
                        serializable_features[k] = str(v)
                features_list.append(serializable_features)
            
            results_df['Features'] = features_list
            
            # Calculate habitability scores for CONFIRMED and CANDIDATE predictions
            habitability_scores = []
            habitability_details = []
            for idx in range(len(results_df)):
                if predictions[idx] in ['CONFIRMED', 'CANDIDATE']:
                    # Calculate habitability score
                    feature_dict = features_list[idx]
                    hab_data = calculate_habitability_score(feature_dict)
                    habitability_scores.append(hab_data['total_score'])
                    habitability_details.append(hab_data)
                else:
                    habitability_scores.append(None)
                    habitability_details.append(None)
            
            results_df['Habitability_Score'] = habitability_scores
            results_df['Habitability_Details'] = habitability_details
            
            # Add original data (first few columns) for display
            display_cols = list(df.columns)[:5]  # Show first 5 columns
            for col in display_cols:
                results_df[f'Original_{col}'] = df[col].values[:len(results_df)]
            
            # Display results
            st.success(f"✅ Processed {len(results_df)} objects successfully!")
            
            st.markdown("### 📊 Batch Prediction Results")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                confirmed = (results_df['Prediction'] == 'CONFIRMED').sum()
                st.metric("Confirmed", confirmed)
            
            with col2:
                candidates = (results_df['Prediction'] == 'CANDIDATE').sum()
                st.metric("Candidates", candidates)
            
            with col3:
                false_pos = (results_df['Prediction'] == 'FALSE POSITIVE').sum()
                st.metric("False Positives", false_pos)
            
            with col4:
                # Calculate average habitability for confirmed/candidate planets
                habitable_scores = results_df[results_df['Habitability_Score'].notna()]['Habitability_Score']
                if len(habitable_scores) > 0:
                    avg_hab = habitable_scores.mean()
                    st.metric("🌱 Avg Habitability", f"{avg_hab:.1f}/100")
                else:
                    st.metric("🌱 Avg Habitability", "N/A")
            
            # Add visualizations
            st.markdown("### 📈 Batch Analysis Visualizations")
            
            batch_viz_col1, batch_viz_col2 = st.columns(2)
            
            with batch_viz_col1:
                # Prediction distribution pie chart
                pred_counts = results_df['Prediction'].value_counts().reset_index()
                pred_counts.columns = ['Prediction', 'Count']
                pred_counts['Percentage'] = (pred_counts['Count'] / pred_counts['Count'].sum() * 100).round(1)
                
                pie = alt.Chart(pred_counts).mark_arc(innerRadius=60, outerRadius=110).encode(
                    theta=alt.Theta('Count:Q'),
                    color=alt.Color('Prediction:N',
                                  scale=alt.Scale(domain=['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'],
                                                range=['#2ecc71', '#f39c12', '#e74c3c']),
                                  legend=alt.Legend(title="Prediction")),
                    tooltip=['Prediction', 'Count', alt.Tooltip('Percentage:Q', format='.1f', title='Percentage (%)')]
                ).properties(
                    title='Prediction Distribution',
                    height=300
                )
                
                st.altair_chart(pie, use_container_width=True)
            
            with batch_viz_col2:
                # Confidence distribution histogram
                conf_hist = alt.Chart(results_df).mark_bar().encode(
                    alt.X('Confidence:Q', bin=alt.Bin(maxbins=20), title='Confidence Level'),
                    y=alt.Y('count()', title='Number of Objects'),
                    color=alt.value('#3498db'),
                    tooltip=[alt.Tooltip('Confidence:Q', bin=alt.Bin(maxbins=20), title='Confidence Range'),
                            alt.Tooltip('count()', title='Count')]
                ).properties(
                    title='Confidence Distribution',
                    height=300
                )
                
                st.altair_chart(conf_hist, use_container_width=True)
            
            # Habitability analysis for confirmed/candidates
            if len(habitable_scores) > 0:
                st.markdown("### 🌱 Habitability Analysis")
                
                hab_viz_col1, hab_viz_col2 = st.columns(2)
                
                with hab_viz_col1:
                    # Habitability score distribution
                    hab_df = results_df[results_df['Habitability_Score'].notna()].copy()
                    
                    hab_hist = alt.Chart(hab_df).mark_bar().encode(
                        alt.X('Habitability_Score:Q', bin=alt.Bin(maxbins=15), title='Habitability Score'),
                        y=alt.Y('count()', title='Number of Objects'),
                        color=alt.value('#27ae60'),
                        tooltip=[alt.Tooltip('Habitability_Score:Q', bin=alt.Bin(maxbins=15), title='Score Range'),
                                alt.Tooltip('count()', title='Count')]
                    ).properties(
                        title='Habitability Score Distribution',
                        height=300
                    )
                    
                    st.altair_chart(hab_hist, use_container_width=True)
                
                with hab_viz_col2:
                    # Confidence vs Habitability scatter
                    scatter = alt.Chart(hab_df).mark_circle(size=60).encode(
                        x=alt.X('Confidence:Q', title='Prediction Confidence', scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y('Habitability_Score:Q', title='Habitability Score', scale=alt.Scale(domain=[0, 100])),
                        color=alt.Color('Prediction:N',
                                      scale=alt.Scale(domain=['CONFIRMED', 'CANDIDATE'],
                                                    range=['#2ecc71', '#f39c12']),
                                      legend=alt.Legend(title="Type")),
                        tooltip=['Prediction', 
                                alt.Tooltip('Confidence:Q', format='.2%'),
                                alt.Tooltip('Habitability_Score:Q', format='.1f')]
                    ).properties(
                        title='Confidence vs Habitability',
                        height=300
                    )
                    
                    st.altair_chart(scatter, use_container_width=True)
                
                # Statistics breakdown
                hab_stats_col1, hab_stats_col2, hab_stats_col3, hab_stats_col4 = st.columns(4)
                
                with hab_stats_col1:
                    high_hab = (hab_df['Habitability_Score'] >= 70).sum()
                    st.metric("High Habitability (≥70)", high_hab)
                
                with hab_stats_col2:
                    medium_hab = ((hab_df['Habitability_Score'] >= 40) & (hab_df['Habitability_Score'] < 70)).sum()
                    st.metric("Medium (40-69)", medium_hab)
                
                with hab_stats_col3:
                    low_hab = (hab_df['Habitability_Score'] < 40).sum()
                    st.metric("Low (<40)", low_hab)
                
                with hab_stats_col4:
                    max_hab = hab_df['Habitability_Score'].max()
                    st.metric("Best Score", f"{max_hab:.1f}/100")
            
            # Show results table with key columns
            st.markdown("### 📋 Detailed Results Table")
            display_df = results_df[['Prediction', 'Confidence', 'Habitability_Score'] + [col for col in results_df.columns if col.startswith('Original_')]].copy()
            
            # Format the display - ensure confidence is always a scalar
            display_df['Confidence'] = display_df['Confidence'].apply(
                lambda x: f"{float(x):.2%}" if x is not None else "N/A"
            )
            display_df['Habitability_Score'] = display_df['Habitability_Score'].apply(
                lambda x: f"{x:.1f}/100" if pd.notna(x) else "N/A"
            )
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download button - prepare CSV with relevant columns
            download_df = results_df.copy()
            # Remove the habitability details object (too complex for CSV)
            download_df = download_df.drop(columns=['Habitability_Details', 'Features'])
            csv = download_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"exoplanet_predictions_{selected_model}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Visualization
            st.markdown("### 📈 Results Visualization")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Create prediction distribution chart
                pred_counts = results_df['Prediction'].value_counts().reset_index()
                pred_counts.columns = ['Category', 'Count']
                
                chart = alt.Chart(pred_counts).mark_bar().encode(
                    x='Category',
                    y='Count',
                    color=alt.Color('Category', scale=alt.Scale(scheme='category10')),
                    tooltip=['Category', 'Count']
                ).properties(
                    height=300,
                    title='Prediction Distribution'
                )
                
                st.altair_chart(chart, use_container_width=True)
            
            with viz_col2:
                # Create habitability score distribution for confirmed/candidates
                habitable_df = results_df[results_df['Habitability_Score'].notna()].copy()
                
                if len(habitable_df) > 0:
                    # Add color category for habitability levels
                    def get_hab_category(score):
                        if score >= 70:
                            return 'High (70-100)'
                        elif score >= 40:
                            return 'Moderate (40-69)'
                        else:
                            return 'Low (0-39)'
                    
                    habitable_df['Hab_Category'] = habitable_df['Habitability_Score'].apply(get_hab_category)
                    
                    hab_chart = alt.Chart(habitable_df).mark_bar().encode(
                        x=alt.X('Habitability_Score:Q', bin=alt.Bin(step=10), title='Habitability Score'),
                        y=alt.Y('count()', title='Count'),
                        color=alt.Color('Hab_Category:N', 
                                       scale=alt.Scale(
                                           domain=['High (70-100)', 'Moderate (40-69)', 'Low (0-39)'],
                                           range=['green', 'orange', 'red']
                                       ),
                                       legend=alt.Legend(title='Habitability Level')),
                        tooltip=['count()', 'Hab_Category']
                    ).properties(
                        height=300,
                        title='Habitability Score Distribution'
                    )
                    
                    st.altair_chart(hab_chart, use_container_width=True)
                else:
                    st.info("No confirmed or candidate planets found to analyze habitability.")
            
            # Detailed habitability analysis for top candidates
            if len(habitable_df) > 0:
                st.markdown("---")
                st.markdown("### 🌱 Top Habitable Candidates")
                
                # Sort by habitability score
                top_habitable = habitable_df.nlargest(min(5, len(habitable_df)), 'Habitability_Score')
                
                st.write(f"Showing top {len(top_habitable)} most habitable candidates:")
                
                for idx, row in top_habitable.iterrows():
                    with st.expander(f"🌍 Object #{idx + 1} - Habitability Score: {row['Habitability_Score']:.1f}/100"):
                        detail_col1, detail_col2 = st.columns([1, 1])
                        
                        with detail_col1:
                            st.write("**Prediction Details:**")
                            st.write(f"- Classification: {row['Prediction']}")
                            st.write(f"- Confidence: {row['Confidence']:.2%}")
                            st.write(f"- Habitability: {row['Habitability_Score']:.1f}/100")
                        
                        with detail_col2:
                            st.write("**Original Data:**")
                            for col in results_df.columns:
                                if col.startswith('Original_'):
                                    display_name = col.replace('Original_', '')
                                    st.write(f"- {display_name}: {row[col]}")
                        
                        # Show detailed habitability breakdown if available
                        if row['Habitability_Details'] is not None:
                            st.markdown("**Habitability Factor Breakdown:**")
                            hab_detail = row['Habitability_Details']
                            
                            for component_name, data in hab_detail['components'].items():
                                score_color = "🟢" if data['score'] >= 70 else "🟡" if data['score'] >= 40 else "🔴"
                                st.write(f"{score_color} **{component_name}**: {data['score']:.1f}/100 (Current: {data['value']:.2f} {data['unit']}, Optimal: {data['optimal']})")
            
            st.markdown("---")
            
            # Store results in session state
            st.session_state.batch_results = results_df
            st.session_state.batch_model = selected_model
            
            # Save button at BOTTOM (after results display)
            st.markdown("---")
            st.markdown("### 💾 Save Results to Database")
            
            if st.button("💾 SAVE ALL RESULTS TO DATABASE", use_container_width=True, key='save_batch_bottom', type='primary'):
                user = st.session_state.get('user')
                
                if not user:
                    st.error("🔒 Please log in to save predictions to your account.")
                    st.info("💡 Go to the Login page to create an account or sign in.")
                else:
                    with st.spinner("Saving predictions..."):
                        try:
                            success_count = 0
                            
                            for idx, row in results_df.iterrows():
                                data = {
                                    'user_id': str(user.id),
                                    'model_name': selected_model,
                                    'prediction': row['Prediction'],
                                    'confidence': float(row['Confidence']),
                                    'features': row['Features']  # Include the features from mapped data
                                }
                                
                                # Add habitability score if available
                                if pd.notna(row['Habitability_Score']):
                                    data['habitability_score'] = float(row['Habitability_Score'])
                                
                                response = supabase.table('predictions').insert(data).execute()
                                success_count += 1
                            
                            st.success(f"✅ Saved {success_count} predictions to your account!")
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            with st.expander("Show Error Details"):
                                st.code(str(e))
    except Exception as e:
        st.error(f"❌ Error during batch prediction: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def calculate_habitability_score(feature_values):
    """
    Calculate habitability score based on planetary and stellar parameters.
    Returns a score from 0-100 and a breakdown of contributing factors.
    """
    scores = {}
    weights = {
        'temperature': 0.30,
        'radius': 0.25,
        'insolation': 0.25,
        'stellar_temp': 0.15,
        'orbital_stability': 0.05
    }
    
    # 1. Temperature Score (0-100)
    # Ideal: 273-373K (liquid water), acceptable: 200-400K
    temp = feature_values.get('koi_teq', 0) or 0
    if temp and 273 <= temp <= 373:
        scores['temperature'] = 100
    elif temp and 200 <= temp <= 450:
        # Gaussian falloff
        if temp < 273:
            scores['temperature'] = 100 * np.exp(-((273 - temp) / 50) ** 2)
        else:
            scores['temperature'] = 100 * np.exp(-((temp - 373) / 50) ** 2)
    elif temp:
        scores['temperature'] = max(0, 100 * np.exp(-((temp - 323) / 100) ** 2))
    else:
        scores['temperature'] = 0
    
    # 2. Radius Score (0-100)
    # Ideal: 0.5-2.0 Earth radii (rocky planets)
    radius = feature_values.get('koi_prad', 0) or 0
    if radius and 0.5 <= radius <= 2.0:
        scores['radius'] = 100
    elif radius and 0.3 <= radius <= 3.0:
        if radius < 0.5:
            scores['radius'] = 100 * (radius / 0.5)
        else:
            scores['radius'] = 100 * np.exp(-((radius - 2.0) / 1.0) ** 2)
    elif radius:
        scores['radius'] = max(0, 100 * np.exp(-((radius - 1.25) / 2.0) ** 2))
    else:
        scores['radius'] = 0
    
    # 3. Insolation Flux Score (0-100)
    # Ideal: 0.3-1.8 Earth flux (habitable zone)
    insol = feature_values.get('koi_insol', 0) or 0
    if insol and 0.3 <= insol <= 1.8:
        scores['insolation'] = 100
    elif insol and 0.1 <= insol <= 3.0:
        if insol < 0.3:
            scores['insolation'] = 100 * (insol / 0.3)
        else:
            scores['insolation'] = 100 * np.exp(-((insol - 1.8) / 1.5) ** 2)
    elif insol:
        scores['insolation'] = max(0, 100 * np.exp(-((insol - 1.0) / 2.0) ** 2))
    else:
        scores['insolation'] = 0
    
    # 4. Stellar Temperature Score (0-100)
    # Ideal: 4000-7000K (K, G, F type stars - stable and long-lived)
    stellar_temp = feature_values.get('koi_steff', 0) or 0
    if stellar_temp and 4000 <= stellar_temp <= 7000:
        scores['stellar_temp'] = 100
    elif stellar_temp and 3000 <= stellar_temp <= 8000:
        if stellar_temp < 4000:
            scores['stellar_temp'] = 100 * ((stellar_temp - 3000) / 1000)
        else:
            scores['stellar_temp'] = 100 * np.exp(-((stellar_temp - 7000) / 1000) ** 2)
    elif stellar_temp:
        scores['stellar_temp'] = max(0, 100 * np.exp(-((stellar_temp - 5500) / 2000) ** 2))
    else:
        scores['stellar_temp'] = 0
    
    # 5. Orbital Stability Score (0-100)
    # Based on orbital period - too short or too long reduces score
    period = feature_values.get('koi_period', 0) or 0
    if period and 10 <= period <= 500:
        scores['orbital_stability'] = 100
    elif period and 1 <= period <= 1000:
        if period < 10:
            scores['orbital_stability'] = 100 * (period / 10)
        else:
            scores['orbital_stability'] = 100 * np.exp(-((period - 500) / 300) ** 2)
    else:
        scores['orbital_stability'] = max(0, 50) if period else 0
    
    # Calculate weighted total score
    total_score = sum(scores[key] * weights[key] for key in weights.keys())
    
    # Create detailed breakdown
    breakdown = {
        'total_score': round(total_score, 1),
        'components': {
            'Temperature': {
                'score': round(scores['temperature'], 1),
                'value': temp,
                'unit': 'K',
                'optimal': '273-373 K'
            },
            'Planet Radius': {
                'score': round(scores['radius'], 1),
                'value': radius,
                'unit': 'Earth radii',
                'optimal': '0.5-2.0 R⊕'
            },
            'Insolation Flux': {
                'score': round(scores['insolation'], 1),
                'value': insol,
                'unit': 'Earth flux',
                'optimal': '0.3-1.8 S⊕'
            },
            'Stellar Temperature': {
                'score': round(scores['stellar_temp'], 1),
                'value': stellar_temp,
                'unit': 'K',
                'optimal': '4000-7000 K'
            },
            'Orbital Stability': {
                'score': round(scores['orbital_stability'], 1),
                'value': period,
                'unit': 'days',
                'optimal': '10-500 days'
            }
        }
    }
    
    return breakdown


def display_habitability_analysis(habitability_data):
    """Display habitability score and analysis"""
    st.markdown("---")
    st.markdown("### 🌱 Habitability Analysis")
    
    total_score = habitability_data['total_score']
    
    # Overall score with color coding
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Color based on score
        if total_score >= 70:
            score_color = "🟢"
            score_label = "High Habitability Potential"
            st.success(f"{score_color} **{score_label}**")
        elif total_score >= 40:
            score_color = "🟡"
            score_label = "Moderate Habitability Potential"
            st.warning(f"{score_color} **{score_label}**")
        else:
            score_color = "🔴"
            score_label = "Low Habitability Potential"
            st.error(f"{score_color} **{score_label}**")
        
        # Large score display
        st.markdown(f"<h1 style='text-align: center;'>{total_score}/100</h1>", unsafe_allow_html=True)
        
        # Progress bar
        st.progress(total_score / 100)
    
    st.markdown("---")
    
    # Detailed breakdown
    st.markdown("#### 📊 Score Breakdown")
    
    components = habitability_data['components']
    
    for component_name, data in components.items():
        with st.expander(f"{component_name}: {data['score']}/100", expanded=False):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Current Value", f"{data['value']:.2f} {data['unit']}")
                st.progress(data['score'] / 100)
            
            with col_b:
                st.metric("Optimal Range", data['optimal'])
                
                # Status indicator
                if data['score'] >= 70:
                    st.success("✅ Favorable")
                elif data['score'] >= 40:
                    st.warning("⚠️ Marginal")
                else:
                    st.error("❌ Unfavorable")
    
    # Additional context
    st.markdown("---")
    st.markdown("#### 💡 Habitability Factors Explained")
    
    with st.expander("What makes a planet habitable?"):
        st.markdown("""
        **Key Factors for Habitability:**
        
        1. **🌡️ Temperature (30% weight)**
           - Must allow liquid water (273-373 K)
           - Too hot: water evaporates
           - Too cold: water freezes
        
        2. **🌍 Planet Radius (25% weight)**
           - Rocky planets: 0.5-2.0 Earth radii
           - Smaller: weak gravity, can't hold atmosphere
           - Larger: likely gas giant, no solid surface
        
        3. **☀️ Insolation Flux (25% weight)**
           - Amount of stellar energy received
           - "Goldilocks zone": 0.3-1.8 Earth flux
           - Too much: runaway greenhouse
           - Too little: frozen surface
        
        4. **⭐ Stellar Temperature (15% weight)**
           - Stable, long-lived stars: 4000-7000 K
           - Too hot: short-lived, high radiation
           - Too cold: very close orbit needed
        
        5. **🔄 Orbital Stability (5% weight)**
           - Moderate period: 10-500 days
           - Too short: tidally locked, extreme temps
           - Too long: distant, cold
        
        **Note:** This is a simplified model. Real habitability involves many more factors including:
        - Atmospheric composition
        - Magnetic field
        - Tidal forces
        - System age
        - Water availability
        """)


def display_prediction_results(result, selected_model, feature_values):
    """Display prediction results in a formatted way"""
    # Create a results container
    results_container = st.container()
    
    with results_container:
        st.markdown("### 🌟 Detection Results")
        
        # Create columns for main results
        res_col1, res_col2, res_col3 = st.columns([2, 1, 1])
        
        with res_col1:
            prediction = result['prediction']
            confidence = result['probability']
            
            # Different styling based on prediction
            if prediction == "CONFIRMED":
                st.success(f"🌍 Potential Exoplanet Detected! ({confidence:.1%} confidence)")
            elif prediction == "FALSE POSITIVE":
                st.warning(f"⚠️ Likely Not an Exoplanet ({confidence:.1%} confidence)")
            else:
                st.info(f"❓ Candidate Object ({confidence:.1%} confidence)")
        
        with res_col2:
            # Confidence gauge
            st.metric("Confidence Score", f"{confidence*100:.1f}%", 
                     delta=f"{(confidence-0.5)*100:+.1f}%" if confidence > 0.5 else None)
        
        with res_col3:
            # Create a button to save the result
            save_single = st.button("📋 Save Result", use_container_width=True, key="save_single_result")
            
            if save_single:
                print(f"🔘 Single save button clicked!")  # Debug print
                user = st.session_state.get('user')
                print(f"User object: {user}")  # Debug print
                
                if not user:
                    st.error("🔒 Please log in to save predictions.")
                    st.info("💡 Login to access your prediction history.")
                else:
                    try:
                        # Convert feature_values to a JSON-serializable format
                        # Handle NaN, None, and infinity values
                        serializable_features = {}
                        for k, v in feature_values.items():
                            if pd.isna(v) or v is None:
                                serializable_features[k] = None
                            elif isinstance(v, (int, float, np.number)):
                                # Check for infinity
                                if np.isinf(v):
                                    serializable_features[k] = None
                                else:
                                    serializable_features[k] = float(v)
                            else:
                                serializable_features[k] = str(v)
                        
                        data = {
                            'user_id': str(user.id),  # Always include user_id when logged in
                            'model_name': selected_model,
                            'features': serializable_features,
                            'prediction': result['prediction'],
                            'confidence': float(result['probability'])
                        }
                        
                        # Add habitability score if it's a confirmed or candidate planet
                        if prediction in ["CONFIRMED", "CANDIDATE"]:
                            habitability_data = calculate_habitability_score(feature_values)
                            data['habitability_score'] = float(habitability_data['total_score'])
                        
                        print(f"💾 Inserting data: {data}")  # Debug print
                        response = supabase.table('predictions').insert(data).execute()
                        print(f"✅ Response: {response.data}")  # Debug print
                        st.success("✅ Result saved to your account!")
                    except Exception as e:
                        st.error(f"❌ Error saving result: {str(e)}")
                        print(f"❌ Exception: {str(e)}")  # Debug print
                        import traceback
                        traceback.print_exc()  # Print to console
                        with st.expander("Show Error Details"):
                            st.code(traceback.format_exc())
                            st.info("💡 Make sure the predictions table allows user_id and has proper constraints:\n\n```sql\nALTER TABLE predictions ALTER COLUMN user_id DROP NOT NULL;\n```")
        
        # Show detailed probabilities with multiple visualizations
        st.markdown("#### Probability Analysis")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Create probability bar chart
            prob_data = pd.DataFrame({
                'Category': list(result['probabilities'].keys()),
                'Probability': [v * 100 for v in result['probabilities'].values()]
            })
            
            bar_chart = alt.Chart(prob_data).mark_bar().encode(
                x=alt.X('Probability:Q', title='Probability (%)', scale=alt.Scale(domain=[0, 100])),
                y=alt.Y('Category:N', sort='-x', title='Classification'),
                color=alt.Color('Category:N', 
                              scale=alt.Scale(domain=['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'],
                                            range=['#2ecc71', '#f39c12', '#e74c3c']),
                              legend=None),
                tooltip=['Category', alt.Tooltip('Probability:Q', format='.2f', title='Probability (%)')]
            ).properties(
                height=200,
                title='Probability Distribution'
            )
            
            st.altair_chart(bar_chart, use_container_width=True)
        
        with viz_col2:
            # Donut chart for probabilities
            donut_chart = alt.Chart(prob_data).mark_arc(innerRadius=50).encode(
                theta=alt.Theta('Probability:Q'),
                color=alt.Color('Category:N',
                              scale=alt.Scale(domain=['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'],
                                            range=['#2ecc71', '#f39c12', '#e74c3c']),
                              legend=alt.Legend(title="Classification")),
                tooltip=['Category', alt.Tooltip('Probability:Q', format='.2f', title='Probability (%)')]
            ).properties(
                height=200,
                title='Probability Split'
            )
            
            st.altair_chart(donut_chart, use_container_width=True)
        
        # Show probability breakdown
        prob_breakdown_col1, prob_breakdown_col2, prob_breakdown_col3 = st.columns(3)
        
        with prob_breakdown_col1:
            conf_prob = result['probabilities'].get('CONFIRMED', 0) * 100
            st.metric("🌍 Confirmed", f"{conf_prob:.1f}%")
        
        with prob_breakdown_col2:
            cand_prob = result['probabilities'].get('CANDIDATE', 0) * 100
            st.metric("❓ Candidate", f"{cand_prob:.1f}%")
        
        with prob_breakdown_col3:
            fp_prob = result['probabilities'].get('FALSE POSITIVE', 0) * 100
            st.metric("⚠️ False Positive", f"{fp_prob:.1f}%")
        
        # Show habitability score for CONFIRMED or CANDIDATE predictions
        if prediction in ["CONFIRMED", "CANDIDATE"]:
            habitability_data = calculate_habitability_score(feature_values)
            display_habitability_analysis(habitability_data)
        
        # Technical details in expander
        with st.expander("View Technical Details"):
            st.write("**Input Features:**")
            features_df = pd.DataFrame([
                {'Feature': feature, 'Value': value}
                for feature, value in feature_values.items()
            ])
            st.dataframe(features_df, use_container_width=True)
            
            st.write("\n**Model Information:**")
            st.write(f"- Model Used: **{selected_model}**")
            st.write(f"- Confidence Score: **{confidence:.4f}**")
            st.write(f"- Prediction: **{prediction}**")
