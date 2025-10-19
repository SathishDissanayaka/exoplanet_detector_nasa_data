import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Path to the pretrained model bundle
MODEL_PATH = "models/random_forest_model.pkl"

def detect_with_random_forest(df: pd.DataFrame):
    """
    Apply pretrained Random Forest model to detect exoplanets from uploaded CSV.
    Shows all plots, top predictions, and realistic test accuracy in Streamlit.
    """

    # --- Load pretrained model bundle ---
    try:
        bundle = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Pretrained model not found at {MODEL_PATH}. Please train the model first.")
        return

    rf_model = bundle["model"]
    features = bundle["features"]
    medians = bundle["medians"]

    st.write("Note: This Random Forest model was trained using SMOTE for balanced classes.")

    # --- Prepare uploaded data ---
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.warning(f"Missing features in uploaded CSV: {missing_features}")
    X = df[features].copy().fillna(pd.Series(medians))

    # Optional: use existing labels if available
    y = df.get("merged_koi_disposition")

    # --- Predictions ---
    y_pred = rf_model.predict(X)
    probs = rf_model.predict_proba(X)

    st.write(" Predictions complete!")

    # --- Show realistic test accuracy if saved ---
    X_test = bundle.get("X_test")
    y_test = bundle.get("y_test")
    if X_test is not None and y_test is not None:
        y_pred_test = rf_model.predict(X_test)
        acc_test = (y_pred_test == y_test).mean()
        st.write(f"**Test Accuracy (realistic): {acc_test*100:.2f}%**")
    elif y is not None:
        # fallback: compute accuracy on uploaded labels (dummy dataset)
        acc = (y_pred == y).mean()
        st.write(f"**Accuracy on uploaded CSV: {acc*100:.2f}%**")

    # --- Classification report & confusion matrix if labels exist ---
    if y is not None:
        st.write("Classification Report:")
        st.text(classification_report(y, y_pred))

        cm = confusion_matrix(y, y_pred, labels=['CANDIDATE','CONFIRMED','FALSE POSITIVE'])
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        plt.figure(figsize=(8,6))
        sns.heatmap(cm_percent, annot=cm, fmt='d', cmap='Blues',
                    xticklabels=['CANDIDATE','CONFIRMED','FALSE POSITIVE'],
                    yticklabels=['CANDIDATE','CONFIRMED','FALSE POSITIVE'])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(plt.gcf())
        plt.close()

    # --- Feature importance plot ---
    feat_importances = pd.Series(rf_model.feature_importances_, index=features)
    top_features = feat_importances.sort_values(ascending=False)[:20]
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
    plt.title("Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    st.pyplot(plt.gcf())
    plt.close()

    # --- Top 20 CONFIRMED planets ---
    probs_df = pd.DataFrame(probs, columns=rf_model.classes_)
    probs_df['planet'] = df.get('kepler_name', df.get('toi', range(len(df))))
    top_confirmed = probs_df[['planet','CONFIRMED']].sort_values(by='CONFIRMED', ascending=False).head(20)

    st.write("Top 20 planets most likely to be CONFIRMED:")
    st.dataframe(top_confirmed)

    plt.figure(figsize=(12,6))
    sns.barplot(x='CONFIRMED', y='planet', data=top_confirmed, palette="viridis")
    plt.title("Top 20 Planets Most Likely to Be Confirmed Exoplanets")
    plt.xlabel("Predicted Probability (CONFIRMED)")
    plt.ylabel("Planet")
    st.pyplot(plt.gcf())
    plt.close()
