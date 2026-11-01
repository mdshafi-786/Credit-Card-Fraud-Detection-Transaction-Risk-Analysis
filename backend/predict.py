"""
Credit Card Fraud Detection — Prediction Service
==================================================
Loads the trained model and provides prediction functions.
"""

import os
import joblib
import numpy as np
import pandas as pd

# Paths
ML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "ml")
MODEL_PATH = os.path.join(ML_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(ML_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(ML_DIR, "label_encoders.pkl")
FEATURE_NAMES_PATH = os.path.join(ML_DIR, "feature_names.pkl")

# Categorical and numerical feature lists (must match training)
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

ENGINEERED_FEATURES = [
    'is_night_transaction', 'is_high_amount', 'amount_deviation',
    'velocity_score', 'failed_attempt_risk', 'account_risk_score'
]


_CACHED_ARTIFACTS = None


def load_model_artifacts(force_reload=False):
    """Load saved model, scaler, and encoders with auto-training fallback and caching."""
    global _CACHED_ARTIFACTS
    if _CACHED_ARTIFACTS is not None and not force_reload:
        return _CACHED_ARTIFACTS

    files_exist = (
        os.path.exists(MODEL_PATH) and
        os.path.exists(SCALER_PATH) and
        os.path.exists(ENCODERS_PATH) and
        os.path.exists(FEATURE_NAMES_PATH)
    )

    if not files_exist:
        from backend.model_training import main as train_pipeline
        train_pipeline()

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
    except Exception:
        # Cross-environment pickle version fallback
        from backend.model_training import main as train_pipeline
        train_pipeline()
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)

    _CACHED_ARTIFACTS = (model, scaler, encoders, feature_names)
    return _CACHED_ARTIFACTS


def engineer_features(data):
    """Apply the same feature engineering as training."""
    data = data.copy()

    # Night transaction (10 PM - 5 AM)
    data['is_night_transaction'] = int(
        data.get('transaction_hour', 12) >= 22 or data.get('transaction_hour', 12) <= 5
    )

    # High amount flag
    data['is_high_amount'] = int(data.get('transaction_amount_inr', 0) > 10000)

    # Amount deviation
    avg_amt = data.get('avg_transaction_amount_inr', 1)
    data['amount_deviation'] = data.get('transaction_amount_inr', 0) / (avg_amt + 1)

    # Velocity score
    data['velocity_score'] = (
        data.get('transaction_count_30d', 0) *
        data.get('amount_to_customer_avg_ratio', 1)
    )

    # Failed attempt risk
    data['failed_attempt_risk'] = (
        data.get('failed_attempts_24h', 0) * data['is_night_transaction']
    )

    # Account risk score
    data['account_risk_score'] = 1 / (data.get('account_age_months', 1) + 1)

    return data


def predict_fraud(transaction_data):
    """
    Predict if a transaction is fraudulent.

    Args:
        transaction_data: dict with transaction fields

    Returns:
        dict with prediction, risk_score, risk_level, and contributing_factors
    """
    model, scaler, encoders, feature_names = load_model_artifacts()

    # Apply feature engineering
    data = engineer_features(transaction_data)

    # Encode categorical features
    for col in CATEGORICAL_FEATURES:
        if col in data and col in encoders:
            le = encoders[col]
            val = str(data[col])
            if val in le.classes_:
                data[col + '_encoded'] = le.transform([val])[0]
            else:
                # Unknown category — use most frequent class
                data[col + '_encoded'] = 0
        else:
            data[col + '_encoded'] = 0

    # Build feature vector
    feature_vector = []
    for feat in feature_names:
        if feat in data:
            feature_vector.append(float(data[feat]))
        else:
            feature_vector.append(0.0)

    feature_vector = np.array(feature_vector).reshape(1, -1)

    scale_cols = [f for f in NUMERICAL_FEATURES + ENGINEERED_FEATURES if f in feature_names]
    temp_df = pd.DataFrame(feature_vector, columns=feature_names)
    temp_df[scale_cols] = scaler.transform(temp_df[scale_cols])

    # Predict with feature names preserved
    prediction = int(model.predict(temp_df)[0])
    probabilities = model.predict_proba(temp_df)[0]
    fraud_probability = float(probabilities[1])

    # Calculate risk score (0-100)
    risk_score = round(fraud_probability * 100, 2)

    # Determine risk level
    if risk_score < 25:
        risk_level = "Low"
    elif risk_score < 50:
        risk_level = "Medium"
    elif risk_score < 75:
        risk_level = "High"
    else:
        risk_level = "Critical"

    # Get contributing factors
    contributing_factors = get_contributing_factors(data, model, feature_names)

    return {
        'prediction': prediction,
        'prediction_label': 'Fraud' if prediction == 1 else 'Genuine',
        'risk_score': risk_score,
        'risk_level': risk_level,
        'fraud_probability': round(fraud_probability, 4),
        'contributing_factors': contributing_factors
    }


def get_contributing_factors(data, model, feature_names):
    """Identify top contributing factors for the prediction."""
    importances = model.feature_importances_
    feature_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    factors = []
    readable_names = {
        'distance_from_home_km': 'Distance from Home',
        'amount_to_customer_avg_ratio': 'Amount vs Average Ratio',
        'failed_attempts_24h': 'Failed Login Attempts (24h)',
        'transaction_amount_inr': 'Transaction Amount',
        'account_age_months': 'Account Age',
        'is_night_transaction': 'Night Transaction',
        'is_high_amount': 'High Amount Flag',
        'amount_deviation': 'Amount Deviation',
        'international_transaction': 'International Transaction',
        'previous_fraud_count': 'Previous Fraud History',
        'velocity_score': 'Transaction Velocity',
        'failed_attempt_risk': 'Failed Attempt Risk',
        'account_risk_score': 'Account Risk Score',
        'transaction_hour_encoded': 'Transaction Hour',
        'merchant_category_encoded': 'Merchant Category',
        'channel_encoded': 'Transaction Channel',
        'customer_age': 'Customer Age',
        'customer_income_monthly_inr': 'Customer Income',
    }

    for feat, imp in feature_imp[:6]:
        name = readable_names.get(feat, feat.replace('_', ' ').title())
        val = data.get(feat, 'N/A')
        if isinstance(val, float):
            val = round(val, 2)
        factors.append({
            'feature': name,
            'importance': round(float(imp) * 100, 2),
            'value': val
        })

    return factors


if __name__ == "__main__":
    # Test prediction
    test_txn = {
        'transaction_amount_inr': 25000,
        'merchant_category': 'Jewelry',
        'transaction_hour': 2,
        'day_of_week': 'Tuesday',
        'customer_age': 35,
        'customer_gender': 'Male',
        'customer_income_monthly_inr': 50000,
        'customer_tenure_months': 12,
        'avg_transaction_amount_inr': 3000,
        'transaction_count_30d': 15,
        'transaction_city': 'Delhi',
        'transaction_state': 'Delhi',
        'distance_from_home_km': 150,
        'card_type': 'Visa',
        'device_type': 'Web',
        'channel': 'Online',
        'international_transaction': 1,
        'failed_attempts_24h': 3,
        'previous_fraud_count': 0,
        'account_age_months': 6,
        'amount_to_customer_avg_ratio': 8.33
    }

    result = predict_fraud(test_txn)
    print("\nTest Prediction Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
