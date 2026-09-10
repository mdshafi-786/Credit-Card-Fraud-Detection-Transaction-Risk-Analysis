"""
Credit Card Fraud Detection — Model Training Pipeline
=====================================================
Trains a Random Forest and XGBoost model on transaction data,
handles class imbalance with SMOTE, and saves the best model.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, accuracy_score,
    precision_score, recall_score
)
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Using Random Forest only.")


# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "transactions.csv")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.json")

# Features to use
CATEGORICAL_FEATURES = [
    'merchant_category', 'customer_gender', 'transaction_city',
    'transaction_state', 'card_type', 'device_type', 'channel', 'day_of_week'
]

NUMERICAL_FEATURES = [
    'transaction_amount_inr', 'transaction_hour', 'customer_age',
    'customer_income_monthly_inr', 'customer_tenure_months',
    'avg_transaction_amount_inr', 'transaction_count_30d',
    'distance_from_home_km', 'international_transaction',
    'failed_attempts_24h', 'previous_fraud_count',
    'account_age_months', 'amount_to_customer_avg_ratio'
]

TARGET = 'is_fraud'


def load_data():
    """Load and validate the dataset."""
    print("=" * 70)
    print("STEP 1: Loading Data")
    print("=" * 70)
    df = pd.read_csv(DATA_PATH)
    print(f"  ✓ Loaded {len(df)} transactions with {len(df.columns)} features")
    print(f"  ✓ Fraud: {df[TARGET].sum()} ({df[TARGET].mean()*100:.2f}%)")
    print(f"  ✓ Legitimate: {(df[TARGET]==0).sum()} ({(1-df[TARGET].mean())*100:.2f}%)")
    return df


def feature_engineering(df):
    """Create new features from existing data."""
    print("\n" + "=" * 70)
    print("STEP 2: Feature Engineering")
    print("=" * 70)

    # Night transaction flag (10 PM - 5 AM)
    df['is_night_transaction'] = df['transaction_hour'].apply(
        lambda h: 1 if h >= 22 or h <= 5 else 0
    )
    print("  ✓ Created is_night_transaction")

    # High amount flag
    df['is_high_amount'] = (df['transaction_amount_inr'] > 10000).astype(int)
    print("  ✓ Created is_high_amount (> ₹10,000)")

    # Amount deviation from customer average
    df['amount_deviation'] = df['transaction_amount_inr'] / (df['avg_transaction_amount_inr'] + 1)
    print("  ✓ Created amount_deviation")

    # Velocity score: high transaction count + high amount
    df['velocity_score'] = df['transaction_count_30d'] * df['amount_to_customer_avg_ratio']
    print("  ✓ Created velocity_score")

    # Failed attempt risk
    df['failed_attempt_risk'] = df['failed_attempts_24h'] * df['is_night_transaction']
    print("  ✓ Created failed_attempt_risk")

    # Account risk (new accounts = higher risk)
    df['account_risk_score'] = 1 / (df['account_age_months'] + 1)
    print("  ✓ Created account_risk_score")

    # Add engineered features to the numerical list
    engineered_features = [
        'is_night_transaction', 'is_high_amount', 'amount_deviation',
        'velocity_score', 'failed_attempt_risk', 'account_risk_score'
    ]

    print(f"\n  ✓ Total engineered features: {len(engineered_features)}")
    return df, engineered_features


def encode_and_scale(df, engineered_features):
    """Encode categorical variables and scale numerical features."""
    print("\n" + "=" * 70)
    print("STEP 3: Encoding & Scaling")
    print("=" * 70)

    # Label encode categorical features
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"  ✓ Encoded {col} ({len(le.classes_)} classes)")

    # Build feature list
    encoded_cat_features = [col + '_encoded' for col in CATEGORICAL_FEATURES]
    all_features = NUMERICAL_FEATURES + engineered_features + encoded_cat_features

    print(f"\n  Total features for model: {len(all_features)}")

    # Prepare X and y
    X = df[all_features].copy()
    y = df[TARGET].copy()

    # Scale numerical features
    scaler = StandardScaler()
    scale_cols = NUMERICAL_FEATURES + engineered_features
    X[scale_cols] = scaler.fit_transform(X[scale_cols])
    print("  ✓ StandardScaler applied to numerical features")

    return X, y, label_encoders, scaler, all_features


def handle_imbalance(X_train, y_train):
    """Apply SMOTE to handle class imbalance."""
    print("\n" + "=" * 70)
    print("STEP 4: Handling Class Imbalance (SMOTE)")
    print("=" * 70)
    print(f"  Before SMOTE: {dict(pd.Series(y_train).value_counts())}")

    smote = SMOTE(random_state=42, sampling_strategy=0.5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"  After SMOTE:  {dict(pd.Series(y_resampled).value_counts())}")
    print(f"  ✓ Resampled from {len(X_train)} to {len(X_resampled)} samples")

    return X_resampled, y_resampled


def train_models(X_train, y_train, X_test, y_test):
    """Train Random Forest and XGBoost, return the best model."""
    print("\n" + "=" * 70)
    print("STEP 5: Training Models")
    print("=" * 70)

    models = {}
    results = {}

    # ─── Random Forest ───────────────────────────────────────────────────
    print("\n  ▸ Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    rf_metrics = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'f1': f1_score(y_test, rf_pred),
        'auc_roc': roc_auc_score(y_test, rf_prob)
    }
    models['RandomForest'] = rf
    results['RandomForest'] = rf_metrics

    print(f"    Accuracy:  {rf_metrics['accuracy']:.4f}")
    print(f"    Precision: {rf_metrics['precision']:.4f}")
    print(f"    Recall:    {rf_metrics['recall']:.4f}")
    print(f"    F1-Score:  {rf_metrics['f1']:.4f}")
    print(f"    AUC-ROC:   {rf_metrics['auc_roc']:.4f}")

    # ─── XGBoost ─────────────────────────────────────────────────────────
    if HAS_XGBOOST:
        print("\n  ▸ Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=30,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_prob = xgb.predict_proba(X_test)[:, 1]

        xgb_metrics = {
            'accuracy': accuracy_score(y_test, xgb_pred),
            'precision': precision_score(y_test, xgb_pred),
            'recall': recall_score(y_test, xgb_pred),
            'f1': f1_score(y_test, xgb_pred),
            'auc_roc': roc_auc_score(y_test, xgb_prob)
        }
        models['XGBoost'] = xgb
        results['XGBoost'] = xgb_metrics

        print(f"    Accuracy:  {xgb_metrics['accuracy']:.4f}")
        print(f"    Precision: {xgb_metrics['precision']:.4f}")
        print(f"    Recall:    {xgb_metrics['recall']:.4f}")
        print(f"    F1-Score:  {xgb_metrics['f1']:.4f}")
        print(f"    AUC-ROC:   {xgb_metrics['auc_roc']:.4f}")

    # ─── Select Best Model ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6: Model Comparison & Selection")
    print("=" * 70)

    best_name = max(results, key=lambda k: results[k]['f1'])
    best_model = models[best_name]
    best_metrics = results[best_name]

    print(f"\n  ★ Best Model: {best_name} (F1={best_metrics['f1']:.4f})")

    # Detailed classification report
    if best_name == 'RandomForest':
        best_pred = rf_pred
        best_prob = rf_prob
    else:
        best_pred = xgb_pred
        best_prob = xgb_prob

    print("\n  Classification Report:")
    print(classification_report(y_test, best_pred, target_names=['Legitimate', 'Fraud']))

    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, best_pred)
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

    # Store confusion matrix in metrics
    best_metrics['confusion_matrix'] = cm.tolist()
    best_metrics['model_name'] = best_name

    # Also store all model results
    all_results = {name: {k: float(v) if isinstance(v, (np.floating, float)) else v
                         for k, v in metrics.items()}
                  for name, metrics in results.items()}

    return best_model, best_metrics, all_results, best_prob


def extract_feature_importance(model, feature_names, model_name):
    """Extract and save feature importance."""
    print("\n" + "=" * 70)
    print("STEP 7: Feature Importance")
    print("=" * 70)

    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    importance_data = []
    for name, imp in feat_imp[:15]:
        print(f"  {name:40s} {imp:.4f}")
        importance_data.append({'feature': name, 'importance': float(imp)})

    return importance_data


def save_artifacts(model, scaler, label_encoders, feature_names, metrics, all_results, importance_data):
    """Save model, scaler, encoders, and metrics."""
    print("\n" + "=" * 70)
    print("STEP 8: Saving Artifacts")
    print("=" * 70)

    joblib.dump(model, MODEL_PATH)
    print(f"  ✓ Model saved: {MODEL_PATH}")

    joblib.dump(scaler, SCALER_PATH)
    print(f"  ✓ Scaler saved: {SCALER_PATH}")

    joblib.dump(label_encoders, ENCODERS_PATH)
    print(f"  ✓ Encoders saved: {ENCODERS_PATH}")

    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    print(f"  ✓ Feature names saved: {FEATURE_NAMES_PATH}")

    # Save metrics
    metrics_to_save = {
        'best_model': metrics,
        'all_models': all_results
    }
    # Convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    metrics_json = json.loads(json.dumps(metrics_to_save, default=convert_numpy))
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  ✓ Metrics saved: {METRICS_PATH}")

    with open(FEATURE_IMPORTANCE_PATH, 'w') as f:
        json.dump(importance_data, f, indent=2)
    print(f"  ✓ Feature importance saved: {FEATURE_IMPORTANCE_PATH}")


def main():
    """Run the complete training pipeline."""
    print("\n" + "█" * 70)
    print("  CREDIT CARD FRAUD DETECTION — MODEL TRAINING PIPELINE")
    print("█" * 70)

    # Step 1: Load data
    df = load_data()

    # Step 2: Feature engineering
    df, engineered_features = feature_engineering(df)

    # Step 3: Encode and scale
    X, y, label_encoders, scaler, feature_names = encode_and_scale(df, engineered_features)

    # Step 4: Train-test split
    print("\n" + "=" * 70)
    print("STEP 3.5: Train-Test Split")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} samples (Fraud: {y_train.sum()})")
    print(f"  Test:  {len(X_test)} samples (Fraud: {y_test.sum()})")

    # Step 5: Handle imbalance
    X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train)

    # Step 6: Train models
    best_model, best_metrics, all_results, best_prob = train_models(
        X_train_balanced, y_train_balanced, X_test, y_test
    )

    # Step 7: Feature importance
    importance_data = extract_feature_importance(best_model, feature_names, best_metrics['model_name'])

    # Step 8: Save everything
    save_artifacts(best_model, scaler, label_encoders, feature_names, best_metrics, all_results, importance_data)

    print("\n" + "█" * 70)
    print("  ✅ TRAINING COMPLETE!")
    print(f"  Best Model: {best_metrics['model_name']}")
    print(f"  F1-Score: {best_metrics['f1']:.4f}")
    print(f"  AUC-ROC: {best_metrics['auc_roc']:.4f}")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    main()
