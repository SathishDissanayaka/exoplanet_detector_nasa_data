# Updated train_page.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from config.supabase import supabase
from models.model_manager import ModelManager
from common.data_merger import DatasetMerger
from common.validation import DatasetValidator

# Updated train_page.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from config.supabase import supabase
from models.model_manager import ModelManager
from common.data_merger import DatasetMerger
from common.validation import DatasetValidator

def show_preprocessing_comparison(original_data, processed_data, model_name):
    """Show before/after preprocessing comparison"""
    
    st.markdown("### 🔧 Preprocessing Changes")
    
    st.metric("Original Features", len(original_data.columns))
    st.metric("Original Rows", len(original_data))
    st.metric("Processed Features", len(processed_data.columns))
    st.metric("Processed Rows", len(processed_data))
    
    st.write("**Original Columns:**")
    st.write(list(original_data.columns))
    
    st.write("**New Columns:**")
    new_cols = list(set(processed_data.columns) - set(original_data.columns))
    st.write(new_cols if new_cols else ["No new columns added"])
    
    # Show sample of processed data
    with st.expander("📊 Processed Data Preview"):
        st.dataframe(processed_data.head(), use_container_width=True)
    
    # Show feature statistics
    with st.expander("📈 Feature Statistics"):
        st.write("**Original Data Stats**")
        st.dataframe(original_data.describe(), use_container_width=True)
        st.write("---")
        st.write("**Processed Data Stats**")
        st.dataframe(processed_data.describe(), use_container_width=True)

def show_train_page():
    st.title("🚀 Model Training Demo")
    
    # Use the global model manager from session state (loaded at app start)
    if "model_manager" not in st.session_state:
        st.error("⚠️ Model manager not initialized. Please restart the application.")
        return
    
    # Use the SAME model manager for demonstration purposes
    model_manager = st.session_state.model_manager

    # Workflow info at the top
    with st.expander("📚 Training Demonstration Guide"):
        st.info("""
        **⚠️ IMPORTANT: This is a DEMONSTRATION feature for educational purposes.**
        
        The system already has pre-trained models loaded and ready to use for predictions.
        This page allows you to:
        - See how the training process works
        - Experiment with different datasets
        - Understand model preprocessing steps
        - View training metrics and visualizations
        
        **Note:** Training a model here will temporarily update that model in memory,
        but won't affect the pre-trained models on disk unless you explicitly save them.
        
        ---
        
        **Demonstration Steps:**
        1. Upload both datasets (Kepler + TESS)
        2. Merge datasets to see data integration
        3. Review preprocessing transformations
        4. Train model and view performance metrics
        """)
    
    with st.expander("ℹ️ About Pre-trained Models"):
        st.success("""
        **✅ Your app already has trained models ready to use!**
        
        - Models are automatically loaded when the app starts
        - No user training required for normal usage
        - Go to the Predict page to use them immediately
        - This Training page is purely for demonstration/education
        """)
    
    with st.expander("📋 Required Columns After Merging"):
        st.info("""
        **After merging, dataset must have:**
        
        1. **merged_koi_disposition** - Target variable (CONFIRMED, FALSE POSITIVE, CANDIDATE)
        2. **merged_koi_prad** - Planet Radius (Earth radii)
        3. **merged_koi_srad** - Star Radius (Solar radii)
        4. **merged_koi_teq** - Equilibrium Temperature (K)
        5. **merged_koi_steff** - Stellar Temperature (K)
        """)
    
    st.markdown("---")
    
    # Model selection
    available_models = list(model_manager.available_models.keys())
    selected_model = st.selectbox(
        "Select Model Architecture",
        available_models,
        help="Choose the type of model to train"
    )
    
    # Show model-specific info
    model_info = model_manager.available_models[selected_model]
    st.info(f"""
    **{selected_model.upper()} Model**
    - {model_info['description']}
    - **Preprocessing Steps:** {', '.join(model_info['preprocessing_steps'])}
    """)
    
    # Show model-specific tips
    model_tips = {
        'catboost': "✅ Handles missing values automatically\n✅ No need for one-hot encoding\n✅ Great with categorical features",
        'random_forest': "✅ Robust to outliers\n✅ Feature importance available\n✅ Works well with small datasets",
        'xgboost': "✅ High performance\n✅ Good for large datasets\n✅ Regularization prevents overfitting"
    }
    st.success(f"**💡 {selected_model.upper()} Tips:**\n\n{model_tips.get(selected_model, '')}")
    
    # Dataset upload section
    st.markdown("---")
    st.markdown("### 📁 Upload Datasets")
    
    st.markdown("#### Kepler Dataset")
    kepler_file = st.file_uploader(
        "Upload Kepler CSV",
        type="csv",
        key="kepler_upload",
        help="Upload the Kepler telescope dataset"
    )
    
    st.markdown("#### TESS Dataset")
    tess_file = st.file_uploader(
        "Upload TESS CSV",
        type="csv",
        key="tess_upload",
        help="Upload the TESS telescope dataset"
    )
    
    # Only proceed if both datasets are uploaded
    if kepler_file is not None and tess_file is not None:
        try:
            # Load both datasets
            kepler_data = pd.read_csv(kepler_file)
            tess_data = pd.read_csv(tess_file)
        
            st.success(f"📊 Kepler: {len(kepler_data)} rows, {len(kepler_data.columns)} columns")
            st.success(f"📊 TESS: {len(tess_data)} rows, {len(tess_data.columns)} columns")

            # Merge datasets button
            st.markdown("---")
            st.markdown("### 🔗 Merge Datasets")
        
            if st.button("🔗 Merge Kepler + TESS", use_container_width=True):
                with st.spinner("Merging datasets..."):
                    try:
                        merger = DatasetMerger()
                        data = merger.merge(kepler_data, tess_data)
                        
                        # Store merged data
                        st.session_state.merged_data = data
                        st.session_state.datasets_merged = True
                        
                        st.success(f"✅ Datasets merged successfully!")
                        st.info(f"📊 Merged dataset: {len(data)} rows, {len(data.columns)} columns")
                        
                        # Show merge debug info
                        merge_info = merger.get_merge_info()
                        with st.expander("🔍 Merge Details"):
                            st.write("**Merge Statistics:**")
                            st.metric("Kepler Rows", merge_info.get('kepler_rows', 0))
                            st.metric("TESS Rows", merge_info.get('tess_rows', 0))
                            st.metric("Merged Rows", merge_info.get('merged_rows', 0))
                            st.metric("Unified Columns", merge_info.get('unified_columns_created', 0))
                            
                            st.write("**Columns Created:**")
                            merged_cols = [col for col in data.columns if col.startswith('merged_')]
                            st.write(f"Found {len(merged_cols)} merged columns")
                            st.code('\n'.join(sorted(merged_cols)))
                            
                    except Exception as merge_error:
                        st.error(f"❌ Error merging datasets: {str(merge_error)}")
                        st.exception(merge_error)
                        st.session_state.datasets_merged = False
            
            # Only show training UI if datasets are merged
            if st.session_state.get('datasets_merged', False):
                data = st.session_state.merged_data
                
                # Validate dataset with model-specific checks using new validator
                validator = DatasetValidator()
                is_valid, message, validation_info = validator.validate(data, selected_model)
            
                # Show validation details
                with st.expander("🔍 Validation Details"):
                    st.write(f"**Total merged columns found:** {len(validation_info['merged_columns_found'])}")
                    
                    if validation_info['missing_recommended']:
                        st.warning(f"⚠️ Missing recommended columns: {', '.join(validation_info['missing_recommended'][:5])}")
                    
                    if validation_info['missing_model_specific']:
                        st.info(f"ℹ️ Model-specific columns missing: {', '.join(validation_info['missing_model_specific'])}")
                    
                    if validation_info['data_quality_issues']:
                        st.warning("⚠️ Data quality issues:")
                        for issue in validation_info['data_quality_issues']:
                            st.write(f"  - {issue}")
            
                if is_valid:
                    st.success(f"✅ {message}")
                    
                    # Show data preview in tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Preview", "📈 Statistics", "📉 Distribution", "🔧 Preprocessing"])
                
                    with tab1:
                        st.dataframe(data.head(), use_container_width=True)
                    
                    with tab2:
                        st.write("Basic Statistics:")
                        st.dataframe(data.describe(), use_container_width=True)
                    
                    with tab3:
                        # Create distribution plot
                        target_dist = pd.DataFrame(
                            data['merged_koi_disposition'].value_counts()
                        ).reset_index()
                        target_dist.columns = ['Category', 'Count']
                        
                        chart = alt.Chart(target_dist).mark_bar().encode(
                            x='Category',
                            y='Count',
                            color='Category',
                            tooltip=['Category', 'Count']
                        ).properties(
                            title='Distribution of Target Classes'
                        )
                        st.altair_chart(chart, use_container_width=True)
                    
                    with tab4:
                        # Preprocessing preview
                        st.markdown("#### Preprocessing Preview")
                        st.write(f"See how {selected_model} will transform your data:")
                        
                        if st.button("🔍 Preview Preprocessing", key="preview_preprocessing"):
                            with st.spinner("Applying preprocessing..."):
                                success, processed_data, preview_message = model_manager.get_model_preview(selected_model, data)
                                
                                if success:
                                    st.success("Preprocessing preview completed!")
                                    show_preprocessing_comparison(data, processed_data, selected_model)
                                    
                                    # Store processed data for training
                                    st.session_state.processed_data = processed_data
                                    st.session_state.preprocessing_success = True
                                else:
                                    st.error(preview_message)
                                    st.session_state.preprocessing_success = False
                
                    # Training options - using model defaults with TESS-First strategy
                    st.markdown("---")
                    st.markdown("### Training Configuration")
                    st.info("🔧 Using optimized default parameters with TESS-First strategy (80% TESS + 20% Kepler)")
                    
                    # Enable TESS-First strategy by default for all models
                    training_params = {
                        'use_tess_first': True,
                        'random_seed': 42
                    }
                    
                    train_button = st.button(
                        "🚀 Train Model",
                        use_container_width=True,
                        disabled=not getattr(st.session_state, 'preprocessing_success', False),
                        key="train_button"
                    )
                    
                    if not getattr(st.session_state, 'preprocessing_success', False):
                        st.warning("⚠️ Please run preprocessing preview before training")
                    
                    if train_button and st.session_state.get('preprocessing_success', False):
                        try:
                            with st.spinner(f"Training {selected_model}..."):
                                # Get the original merged data (not preprocessed)
                                # The model_manager.train_model() will handle preprocessing
                                original_data = st.session_state.merged_data
                                
                                # Train the model (which will preprocess internally)
                                trained_model, metrics = model_manager.train_model(
                                    original_data, selected_model, training_params
                                )
                                
                                st.success(f"✨ {selected_model} trained successfully!")
                                
                                # Show training results in columns
                                st.markdown("### 📊 Model Performance")
                                
                                # Primary metrics (Macro - treats all classes equally)
                                st.markdown("#### Macro-averaged Metrics (Equal Weight Per Class)")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                                with col2:
                                    st.metric("F1 Score", f"{metrics.get('f1_macro', 0):.3f}")
                                with col3:
                                    st.metric("Precision", f"{metrics.get('precision_macro', 0):.3f}")
                                with col4:
                                    st.metric("Recall", f"{metrics.get('recall_macro', 0):.3f}")
                                
                                # Weighted metrics (accounts for class imbalance)
                                st.markdown("#### Weighted Metrics (Adjusted for Class Imbalance)")
                                col5, col6, col7 = st.columns(3)
                                
                                with col5:
                                    st.metric("F1 (Weighted)", f"{metrics.get('f1_weighted', 0):.3f}")
                                with col6:
                                    st.metric("Precision (Weighted)", f"{metrics.get('precision_weighted', 0):.3f}")
                                with col7:
                                    st.metric("Recall (Weighted)", f"{metrics.get('recall_weighted', 0):.3f}")
                                
                                # Create tabs for different visualizations
                                viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                                    "🎯 Confusion Matrix",
                                    "📈 Per-Class Metrics",
                                    "🔥 Feature Importance",
                                    "📋 Detailed Report"
                                ])
                                
                                with viz_tab1:
                                    st.markdown("#### Confusion Matrix")
                                    st.write("Shows how well the model predicted each class")
                                    
                                    # Create confusion matrix heatmap
                                    conf_matrix = np.array(metrics.get('confusion_matrix', []))
                                    if conf_matrix.size > 0:
                                        # Get class names
                                        class_names = model_manager.available_models[selected_model]['pipeline'].label_encoder.classes_
                                        
                                        # Create two columns - one for matrix, one for statistics
                                        cm_col1, cm_col2 = st.columns([2, 1])
                                        
                                        with cm_col1:
                                            # Create dataframe for heatmap
                                            conf_df = pd.DataFrame(
                                                conf_matrix,
                                                index=[f"True: {cls}" for cls in class_names],
                                                columns=[f"Pred: {cls}" for cls in class_names]
                                            )
                                            
                                            # Display as a styled dataframe with custom container
                                            st.markdown('<div class="confusion-matrix-container">', unsafe_allow_html=True)
                                            st.dataframe(
                                                conf_df.style.background_gradient(cmap='Blues', axis=None)
                                                .set_properties(**{
                                                    'font-weight': 'bold',
                                                    'font-size': '14px',
                                                    'text-align': 'center',
                                                    'border': '2px solid black'
                                                }),
                                                use_container_width=False,
                                                height=400
                                            )
                                            st.markdown('</div>', unsafe_allow_html=True)
                                        
                                        with cm_col2:
                                            st.markdown("##### Matrix Statistics")
                                            
                                            # Calculate per-class accuracy
                                            total_per_class = conf_matrix.sum(axis=1)
                                            correct_per_class = np.diag(conf_matrix)
                                            
                                            for i, cls in enumerate(class_names):
                                                accuracy = (correct_per_class[i] / total_per_class[i] * 100) if total_per_class[i] > 0 else 0
                                                st.metric(
                                                    f"{cls}",
                                                    f"{accuracy:.1f}%",
                                                    f"{int(correct_per_class[i])}/{int(total_per_class[i])}"
                                                )
                                            
                                            st.markdown("---")
                                            total_correct = np.diag(conf_matrix).sum()
                                            total_samples = conf_matrix.sum()
                                            overall_acc = (total_correct / total_samples * 100) if total_samples > 0 else 0
                                            st.metric("Overall", f"{overall_acc:.1f}%", f"{int(total_correct)}/{int(total_samples)}")
                                        
                                        # Add interpretation
                                        st.info("💡 Diagonal values (top-left to bottom-right) represent correct predictions. Higher values are better!")
                                        
                                        # Add normalized confusion matrix
                                        with st.expander("📊 View Normalized Confusion Matrix (Percentages)"):
                                            conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
                                            conf_df_norm = pd.DataFrame(
                                                conf_matrix_norm,
                                                index=[f"True: {cls}" for cls in class_names],
                                                columns=[f"Pred: {cls}" for cls in class_names]
                                            )
                                            
                                            st.dataframe(
                                                conf_df_norm.style.background_gradient(cmap='RdYlGn', axis=None)
                                                .format("{:.1f}%")
                                                .set_properties(**{'font-weight': 'bold', 'text-align': 'center'}),
                                                use_container_width=True
                                            )
                                
                                with viz_tab2:
                                    st.markdown("#### Per-Class Performance")
                                    st.write("Detailed metrics for each exoplanet category")
                                    
                                    # Extract per-class metrics from classification report
                                    class_report = metrics.get('classification_report', {})
                                    if class_report:
                                        # Create dataframe for per-class metrics
                                        class_metrics = []
                                        for class_name, class_data in class_report.items():
                                            if isinstance(class_data, dict) and 'precision' in class_data:
                                                class_metrics.append({
                                                    'Class': class_name,
                                                    'Precision': class_data['precision'],
                                                    'Recall': class_data['recall'],
                                                    'F1-Score': class_data['f1-score'],
                                                    'Support': int(class_data['support'])
                                                })
                                        
                                        if class_metrics:
                                            class_df = pd.DataFrame(class_metrics)
                                            
                                            # Create bar chart using altair
                                            metrics_melted = class_df.melt(
                                                id_vars=['Class', 'Support'],
                                                value_vars=['Precision', 'Recall', 'F1-Score'],
                                                var_name='Metric',
                                                value_name='Score'
                                            )
                                            
                                            chart = alt.Chart(metrics_melted).mark_bar().encode(
                                                x=alt.X('Metric:N', title='Metric'),
                                                y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
                                                color=alt.Color('Metric:N', legend=None),
                                                column=alt.Column('Class:N', title='Exoplanet Class'),
                                                tooltip=['Class', 'Metric', alt.Tooltip('Score:Q', format='.3f'), 'Support']
                                            ).properties(
                                                width=150,
                                                height=300
                                            )
                                            
                                            st.altair_chart(chart, use_container_width=True)
                                            
                                            # Show table
                                            st.dataframe(
                                                class_df.style.format({
                                                    'Precision': '{:.3f}',
                                                    'Recall': '{:.3f}',
                                                    'F1-Score': '{:.3f}'
                                                }).background_gradient(subset=['Precision', 'Recall', 'F1-Score'], cmap='RdYlGn'),
                                                use_container_width=True
                                            )
                                
                                with viz_tab3:
                                    st.markdown("#### Feature Importance")
                                    st.write("Which features contributed most to predictions?")
                                    
                                    feature_importance = metrics.get('feature_importance', {})
                                    if feature_importance:
                                        # Create dataframe and sort by importance
                                        feat_df = pd.DataFrame([
                                            {'Feature': k, 'Importance': v}
                                            for k, v in feature_importance.items()
                                        ]).sort_values('Importance', ascending=False).head(20)
                                        
                                        # Create horizontal bar chart
                                        chart = alt.Chart(feat_df).mark_bar().encode(
                                            y=alt.Y('Feature:N', sort='-x', title='Feature'),
                                            x=alt.X('Importance:Q', title='Importance Score'),
                                            color=alt.Color('Importance:Q', scale=alt.Scale(scheme='viridis'), legend=None),
                                            tooltip=['Feature', alt.Tooltip('Importance:Q', format='.4f')]
                                        ).properties(
                                            height=500
                                        )
                                        
                                        st.altair_chart(chart, use_container_width=True)
                                        
                                        st.info(f"🔍 Showing top 20 most important features out of {len(feature_importance)} total features")
                                    else:
                                        st.warning("Feature importance not available for this model")
                                
                                with viz_tab4:
                                    st.markdown("#### Detailed Classification Report")
                                    
                                    class_report = metrics.get('classification_report', {})
                                    if class_report:
                                        # Display as formatted JSON
                                        st.json(class_report)
                                    
                                    st.markdown("#### Training Information")
                                    st.write(f"**Model Type:** {selected_model}")
                                    st.write(f"**Dataset Size:** {len(data)} samples")
                                    st.write(f"**Training Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                    st.write(f"**Preprocessing Steps:** {', '.join(model_info['preprocessing_steps'])}")
                                
                                # Show option to save this as the new pre-trained model
                                st.markdown("---")
                                st.markdown("### � Save Trained Model")
                                st.warning("""
                                **⚠️ Optional: Save this model to disk**
                                
                                This will replace the existing pre-trained model file.
                                Only do this if you want this newly trained model to become the default.
                                """)
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if st.button("💾 Save to Disk (Permanent)", use_container_width=True, type="secondary"):
                                        if model_manager.save_model(selected_model):
                                            st.success(f"✅ {selected_model} model saved to disk! It will be loaded automatically on next app restart.")
                                        else:
                                            st.error(f"❌ Failed to save {selected_model} model")
                                
                                with col2:
                                    if st.button("🔄 Use for Current Session Only", use_container_width=True, type="primary"):
                                        st.success(f"✅ {selected_model} model is now active for this session!")
                                        st.info("This model will be available in the Predict page until you restart the app.")
                                        st.balloons()
                                
                                # Optional: Save model metrics to Supabase (disabled - table doesn't exist)
                                # Uncomment this section after creating the model_metrics table in Supabase
                                # supabase_metrics = {
                                #     'model_name': selected_model,
                                #     'dataset_size': len(data),
                                #     'training_date': pd.Timestamp.now().isoformat(),
                                #     'accuracy': metrics.get('accuracy', 0),
                                #     'f1_score': metrics.get('f1', 0),
                                #     'metadata': {
                                #         'test_size': 0.2,  # Model default
                                #         'random_seed': 42,  # Model default
                                #         'cv_folds': 5,  # Model default
                                #         'preprocessing_steps': model_info['preprocessing_steps'],
                                #         'data_distribution': data['merged_koi_disposition'].value_counts().to_dict()
                                #     }
                                # }
                                # supabase.table('model_metrics').insert(supabase_metrics).execute()
                                
                        except Exception as e:
                            st.error(f"❌ Error during training: {str(e)}")
            
                else:
                    st.error(f"❌ Dataset validation failed: {message}")
                    st.write("**Validation details:**")
                    st.write(f"- Missing critical columns: {validation_info['missing_minimum']}")
                    st.write(f"- Available merged columns: {len(validation_info['merged_columns_found'])}")
                    st.write(f"- Data quality issues: {validation_info['data_quality_issues']}")
                    
                    with st.expander("📋 All available columns"):
                        st.write(validation_info['merged_columns_found'])
                
        except Exception as e:
            st.error(f"❌ Error loading datasets: {str(e)}")

# Add this to your main app
if __name__ == "__main__":
    show_train_page()