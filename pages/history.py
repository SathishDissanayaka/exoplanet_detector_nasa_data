"""
Prediction History Page - View saved predictions for the logged-in user
"""
import streamlit as st
import pandas as pd
import altair as alt
from config.supabase import supabase
from datetime import datetime

def show_history_page():
    st.title("📜 Prediction History")
    
    # Check if user is logged in
    user = st.session_state.get('user')
    
    if not user:
        st.warning("Please log in to view your prediction history.")
        st.info("💡 Go to the Login page to access your saved predictions.")
        return
    
    st.markdown(f"### Your Saved Predictions")
    st.write(f"Welcome back! Here are all predictions saved to your account.")
    
    try:
        # Fetch predictions for the logged-in user
        response = supabase.table('predictions')\
            .select("*")\
            .eq('user_id', str(user.id))\
            .order('created_at', desc=True)\
            .execute()
        
        if not response.data or len(response.data) == 0:
            st.info("No saved predictions yet. Start making predictions and save them to see your history!")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(response.data)
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Predictions", len(df))
        
        with col2:
            confirmed = len(df[df['prediction'] == 'CONFIRMED'])
            st.metric("Confirmed", confirmed)
        
        with col3:
            candidates = len(df[df['prediction'] == 'CANDIDATE'])
            st.metric("Candidates", candidates)
        
        with col4:
            false_pos = len(df[df['prediction'] == 'FALSE POSITIVE'])
            st.metric("False Positives", false_pos)
        
        st.markdown("---")
        
        # Filters
        st.markdown("### 🔍 Filters")
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            # Model filter
            models = ['All'] + sorted(df['model_name'].unique().tolist())
            selected_model = st.selectbox("Filter by Model", models)
        
        with filter_col2:
            # Prediction filter
            predictions = ['All'] + sorted(df['prediction'].unique().tolist())
            selected_prediction = st.selectbox("Filter by Prediction", predictions)
        
        # Apply filters
        filtered_df = df.copy()
        if selected_model != 'All':
            filtered_df = filtered_df[filtered_df['model_name'] == selected_model]
        if selected_prediction != 'All':
            filtered_df = filtered_df[filtered_df['prediction'] == selected_prediction]
        
        st.markdown(f"**Showing {len(filtered_df)} of {len(df)} predictions**")
        
        # Visualization
        st.markdown("### 📈 Prediction Analytics")
        
        vis_col1, vis_col2, vis_col3 = st.columns(3)
        
        with vis_col1:
            # Prediction type distribution - Donut chart
            pred_counts = filtered_df['prediction'].value_counts().reset_index()
            pred_counts.columns = ['Prediction', 'Count']
            pred_counts['Percentage'] = (pred_counts['Count'] / pred_counts['Count'].sum() * 100).round(1)
            
            chart1 = alt.Chart(pred_counts).mark_arc(innerRadius=60, outerRadius=100).encode(
                theta=alt.Theta('Count:Q'),
                color=alt.Color('Prediction:N', 
                              scale=alt.Scale(domain=['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'],
                                            range=['#2ecc71', '#f39c12', '#e74c3c']),
                              legend=alt.Legend(title="Prediction")),
                tooltip=['Prediction', 'Count', alt.Tooltip('Percentage:Q', format='.1f', title='Percentage (%)')]
            ).properties(
                width=250,
                height=250,
                title='By Prediction Type'
            )
            
            st.altair_chart(chart1, use_container_width=True)
        
        with vis_col2:
            # Model distribution - Bar chart
            model_counts = filtered_df['model_name'].value_counts().reset_index()
            model_counts.columns = ['Model', 'Count']
            
            chart2 = alt.Chart(model_counts).mark_bar().encode(
                x=alt.X('Count:Q', title='Number of Predictions'),
                y=alt.Y('Model:N', sort='-x', title='Model'),
                color=alt.Color('Model:N', scale=alt.Scale(scheme='tableau10'), legend=None),
                tooltip=['Model', 'Count']
            ).properties(
                height=250,
                title='Predictions by Model'
            )
            
            st.altair_chart(chart2, use_container_width=True)
        
        with vis_col3:
            # Confidence distribution - Histogram
            if 'confidence' in filtered_df.columns:
                conf_chart = alt.Chart(filtered_df).mark_bar().encode(
                    alt.X('confidence:Q', bin=alt.Bin(maxbins=20), title='Confidence'),
                    y=alt.Y('count()', title='Count'),
                    color=alt.value('#3498db'),
                    tooltip=[alt.Tooltip('confidence:Q', bin=alt.Bin(maxbins=20), title='Confidence Range'),
                            alt.Tooltip('count()', title='Count')]
                ).properties(
                    height=250,
                    title='Confidence Distribution'
                )
                
                st.altair_chart(conf_chart, use_container_width=True)
        
        # Time series analysis
        if 'created_at' in filtered_df.columns and len(filtered_df) > 1:
            st.markdown("### 📅 Predictions Over Time")
            
            time_df = filtered_df.copy()
            time_df['date'] = pd.to_datetime(time_df['created_at']).dt.date
            time_counts = time_df.groupby(['date', 'prediction']).size().reset_index(name='count')
            
            time_chart = alt.Chart(time_counts).mark_line(point=True).encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('count:Q', title='Number of Predictions'),
                color=alt.Color('prediction:N',
                              scale=alt.Scale(domain=['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'],
                                            range=['#2ecc71', '#f39c12', '#e74c3c']),
                              legend=alt.Legend(title="Prediction Type")),
                tooltip=['date:T', 'prediction:N', 'count:Q']
            ).properties(
                height=300
            )
            
            st.altair_chart(time_chart, use_container_width=True)
        
        # Confidence statistics
        if 'confidence' in filtered_df.columns:
            st.markdown("### 📊 Confidence Statistics")
            conf_stats_col1, conf_stats_col2, conf_stats_col3, conf_stats_col4 = st.columns(4)
            
            with conf_stats_col1:
                avg_conf = filtered_df['confidence'].mean()
                st.metric("Average Confidence", f"{avg_conf:.1f}%")
            
            with conf_stats_col2:
                median_conf = filtered_df['confidence'].median()
                st.metric("Median Confidence", f"{median_conf:.1f}%")
            
            with conf_stats_col3:
                high_conf = len(filtered_df[filtered_df['confidence'] >= 80])
                st.metric("High Confidence (≥80%)", high_conf)
            
            with conf_stats_col4:
                low_conf = len(filtered_df[filtered_df['confidence'] < 50])
                st.metric("Low Confidence (<50%)", low_conf)
        
        # Display table - Grouped by Date with Expandable Sections
        st.markdown("### 📋 Detailed History by Date")
        st.markdown("*Click on any date to expand and view predictions made that day*")
        
        # Prepare display dataframe
        display_df = filtered_df.copy()
        
        # Extract date and time separately
        if 'created_at' in display_df.columns:
            display_df['date'] = pd.to_datetime(display_df['created_at']).dt.date
            display_df['time'] = pd.to_datetime(display_df['created_at']).dt.strftime('%H:%M:%S')
        
        # Group by date
        dates = sorted(display_df['date'].unique(), reverse=True)
        
        for date in dates:
            date_df = display_df[display_df['date'] == date].copy()
            
            # Calculate statistics for this date
            total_count = len(date_df)
            confirmed_count = len(date_df[date_df['prediction'] == 'CONFIRMED'])
            candidate_count = len(date_df[date_df['prediction'] == 'CANDIDATE'])
            false_pos_count = len(date_df[date_df['prediction'] == 'FALSE POSITIVE'])
            avg_confidence = date_df['confidence'].mean() if 'confidence' in date_df.columns else 0
            
            # Create summary string
            summary_parts = []
            if confirmed_count > 0:
                summary_parts.append(f"{confirmed_count} Confirmed")
            if candidate_count > 0:
                summary_parts.append(f"{candidate_count} Candidate")
            if false_pos_count > 0:
                summary_parts.append(f"{false_pos_count} False Positive")
            
            summary = " | ".join(summary_parts)
            if avg_confidence > 0:
                summary += f" | Avg Confidence: {avg_confidence:.1f}%"
            
            # Create expander with summary
            with st.expander(
                f"**{date.strftime('%A, %B %d, %Y')}** — {total_count} prediction{'s' if total_count != 1 else ''} | {summary}",
                expanded=False
            ):
                # Show mini statistics for this date
                mini_col1, mini_col2, mini_col3, mini_col4 = st.columns(4)
                
                with mini_col1:
                    st.metric("Total", total_count)
                with mini_col2:
                    st.metric("Confirmed", confirmed_count)
                with mini_col3:
                    st.metric("Candidates", candidate_count)
                with mini_col4:
                    st.metric("False Positives", false_pos_count)
                
                # Show predictions for this date
                st.markdown("##### Predictions")
                
                # Prepare columns for display
                display_columns = ['time', 'model_name', 'prediction', 'confidence']
                
                # Add features preview if available
                if 'features' in date_df.columns:
                    date_df['features_preview'] = date_df['features'].apply(
                        lambda x: str(x)[:50] + '...' if x and len(str(x)) > 50 else str(x) if x else 'N/A'
                    )
                    display_columns.append('features_preview')
                
                # Show dataframe for this date
                st.dataframe(
                    date_df[display_columns],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'time': st.column_config.TextColumn('Time', width='small'),
                        'model_name': st.column_config.TextColumn('Model', width='medium'),
                        'prediction': st.column_config.TextColumn('Prediction', width='medium'),
                        'confidence': st.column_config.NumberColumn('Confidence', format="%.2f%%", width='small'),
                        'features_preview': st.column_config.TextColumn('Features', width='large')
                    }
                )
                
                # Show prediction distribution for this date if there are multiple predictions
                if len(date_df) > 1:
                    st.markdown("##### Distribution")
                    
                    dist_col1, dist_col2 = st.columns(2)
                    
                    with dist_col1:
                        # Prediction type pie chart
                        pred_counts = date_df['prediction'].value_counts().reset_index()
                        pred_counts.columns = ['Prediction', 'Count']
                        
                        pie_chart = alt.Chart(pred_counts).mark_arc().encode(
                            theta=alt.Theta('Count:Q'),
                            color=alt.Color('Prediction:N', 
                                          scale=alt.Scale(domain=['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'],
                                                        range=['#2ecc71', '#f39c12', '#e74c3c'])),
                            tooltip=['Prediction', 'Count']
                        ).properties(
                            width=200,
                            height=200,
                            title='Predictions'
                        )
                        
                        st.altair_chart(pie_chart, use_container_width=True)
                    
                    with dist_col2:
                        # Model usage bar chart
                        model_counts = date_df['model_name'].value_counts().reset_index()
                        model_counts.columns = ['Model', 'Count']
                        
                        bar_chart = alt.Chart(model_counts).mark_bar().encode(
                            x=alt.X('Count:Q'),
                            y=alt.Y('Model:N', sort='-x'),
                            color=alt.Color('Model:N', legend=None),
                            tooltip=['Model', 'Count']
                        ).properties(
                            height=200,
                            title='Models Used'
                        )
                        
                        st.altair_chart(bar_chart, use_container_width=True)
                
                st.markdown("---")
        
        # Download option
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Delete options
        st.markdown("---")
        st.markdown("### 🗑️ Manage History")
        
        del_col1, del_col2 = st.columns(2)
        
        with del_col1:
            if st.button("🗑️ Clear All History", use_container_width=True, type="secondary"):
                if st.session_state.get('confirm_delete'):
                    try:
                        supabase.table('predictions')\
                            .delete()\
                            .eq('user_id', str(user.id))\
                            .execute()
                        st.success("✅ All history cleared!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing history: {str(e)}")
                else:
                    st.session_state.confirm_delete = True
                    st.warning("⚠️ Click again to confirm deletion")
        
        with del_col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_delete = False
                st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error loading prediction history: {str(e)}")
        import traceback
        with st.expander("Show Error Details"):
            st.code(traceback.format_exc())
