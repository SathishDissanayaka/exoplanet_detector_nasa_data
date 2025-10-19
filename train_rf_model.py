import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from imblearn.over_sampling import SMOTE

# --- CONFIG ---
TRAIN_CSV = r"C:\Users\CHAMA COMPUTERS\Downloads\merged_exoplanets.csv"
OUT_PKL = os.path.join("models", "random_forest_model.pkl")
# ----------------

def main():
    print("Loading training CSV:", TRAIN_CSV)
    df = pd.read_csv(TRAIN_CSV, low_memory=False)

       # --- DROP ALL *_err* COLUMNS ---
    err_cols = [c for c in df.columns if "_err" in c]
    if err_cols:
        print(f"Dropping {len(err_cols)} _err columns: {err_cols[:5]}{'...' if len(err_cols)>5 else ''}")
        df = df.drop(columns=err_cols)

    # Select merged numeric features
    numeric_features = [col for col in df.select_dtypes(include="number") if col.startswith("merge")]

    # Drop features with >50% missing
    missing_ratio = df[numeric_features].isnull().mean()
    numeric_features = missing_ratio[missing_ratio <= 0.5].index.tolist()
    print(f"Using {len(numeric_features)} numeric features.")

    if not numeric_features:
        raise ValueError("No usable merged numeric features found in training data.")

    X = df[numeric_features].copy()
    feature_medians = X.median().to_dict()
    X = X.fillna(feature_medians)

    y = df["merged_koi_disposition"]

    # --- Split into training and test sets ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # --- Apply SMOTE to training data ---
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE, training set size: {len(y_train_res)}")
    print(f"Class distribution after SMOTE:\n{pd.Series(y_train_res).value_counts()}")

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,        # limit depth to reduce overfitting
        min_samples_leaf=5,  # minimum samples per leaf
        random_state=42
    )
    print("Training RandomForest on resampled data...")
    rf.fit(X_train_res, y_train_res)

    # --- Evaluate on test set only ---
    y_pred_test = rf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f" Test Set Accuracy: {test_acc*100:.2f}%")
    print("\n--- Test Set Evaluation ---")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_test))
    cm = confusion_matrix(y_test, y_pred_test, labels=['CANDIDATE','CONFIRMED','FALSE POSITIVE'])
    print("Confusion Matrix:")
    print(cm)

    # Save model bundle including test split
    bundle = {
        "model": rf,
        "features": numeric_features,
        "medians": feature_medians,
        "X_test": X_test,
        "y_test": y_test
    }
    os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)
    joblib.dump(bundle, OUT_PKL)
    print("Saved model bundle to:", OUT_PKL)

if __name__ == "__main__":
    main()
