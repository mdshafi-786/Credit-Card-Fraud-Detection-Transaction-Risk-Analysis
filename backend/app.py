"""
Credit Card Fraud Detection — Flask Backend API
=================================================
REST API for fraud prediction, transaction management,
dashboard analytics, and alert management.
"""

import os
import sys
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import (
    init_db, seed_data, get_all_transactions, get_transaction_by_id,
    save_prediction, save_alert, get_alerts, resolve_alert,
    get_dashboard_stats, get_fraud_by_category, get_fraud_by_hour,
    get_fraud_by_city, get_fraud_by_channel, get_fraud_by_card_type,
    get_fraud_trend, get_recent_predictions
)
from backend.predict import predict_fraud

app = Flask(__name__)
CORS(app)


# ─── Health Check ────────────────────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


# ─── Prediction Endpoint ────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict fraud for a new transaction."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Run prediction
        result = predict_fraud(data)

        # Generate transaction ID
        txn_id = data.get('transaction_id', f"TXN_PRED_{datetime.now().strftime('%Y%m%d%H%M%S')}")

        # Save prediction to database
        save_prediction(
            txn_id, result['prediction'],
            result['risk_score'], result['risk_level']
        )

        # Create alert if suspicious (risk_score > 70)
        if result['risk_score'] > 70:
            severity = 'Critical' if result['risk_score'] > 90 else 'High'
            message = (
                f"Suspicious transaction detected: ₹{data.get('transaction_amount_inr', 0):,.2f} "
                f"at {data.get('merchant_category', 'Unknown')} — "
                f"Risk Score: {result['risk_score']}%"
            )
            save_alert(txn_id, 'Fraud Detection', severity, message, result['risk_score'])

        result['transaction_id'] = txn_id
        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


# ─── Transaction Endpoints ──────────────────────────────────────────────────
@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    """Fetch transactions with pagination and filters."""
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)

    filters = {}
    if request.args.get('city'):
        filters['city'] = request.args.get('city')
    if request.args.get('category'):
        filters['category'] = request.args.get('category')
    if request.args.get('is_fraud') is not None:
        filters['is_fraud'] = request.args.get('is_fraud', type=int)

    df = get_all_transactions(limit, offset, filters if filters else None)
    return jsonify(df.to_dict(orient='records'))


@app.route('/api/transactions/<txn_id>', methods=['GET'])
def get_transaction(txn_id):
    """Fetch a single transaction."""
    df = get_transaction_by_id(txn_id)
    if df.empty:
        return jsonify({'error': 'Transaction not found'}), 404
    return jsonify(df.to_dict(orient='records')[0])


# ─── Dashboard Endpoints ────────────────────────────────────────────────────
@app.route('/api/dashboard/stats', methods=['GET'])
def dashboard_stats():
    """Get overview KPI stats."""
    stats = get_dashboard_stats()
    return jsonify(stats)


@app.route('/api/dashboard/fraud-by-category', methods=['GET'])
def fraud_by_category():
    """Get fraud stats by merchant category."""
    df = get_fraud_by_category()
    return jsonify(df.to_dict(orient='records'))


@app.route('/api/dashboard/fraud-by-hour', methods=['GET'])
def fraud_by_hour():
    """Get fraud stats by hour."""
    df = get_fraud_by_hour()
    return jsonify(df.to_dict(orient='records'))


@app.route('/api/dashboard/fraud-by-city', methods=['GET'])
def fraud_by_city():
    """Get fraud stats by city."""
    df = get_fraud_by_city()
    return jsonify(df.to_dict(orient='records'))


@app.route('/api/dashboard/fraud-by-channel', methods=['GET'])
def fraud_by_channel():
    """Get fraud stats by channel."""
    df = get_fraud_by_channel()
    return jsonify(df.to_dict(orient='records'))


@app.route('/api/dashboard/fraud-by-card', methods=['GET'])
def fraud_by_card():
    """Get fraud stats by card type."""
    df = get_fraud_by_card_type()
    return jsonify(df.to_dict(orient='records'))


@app.route('/api/dashboard/fraud-trend', methods=['GET'])
def fraud_trend():
    """Get daily fraud trend."""
    df = get_fraud_trend()
    return jsonify(df.to_dict(orient='records'))


# ─── Alert Endpoints ────────────────────────────────────────────────────────
@app.route('/api/alerts', methods=['GET'])
def alerts():
    """Get active alerts."""
    resolved = request.args.get('resolved', 'false').lower() == 'true'
    df = get_alerts(resolved=resolved)
    return jsonify(df.to_dict(orient='records'))


@app.route('/api/alerts/<int:alert_id>/resolve', methods=['PUT'])
def resolve_alert_endpoint(alert_id):
    """Resolve an alert."""
    resolve_alert(alert_id)
    return jsonify({'status': 'resolved', 'alert_id': alert_id})


@app.route('/api/predictions/recent', methods=['GET'])
def recent_predictions():
    """Get recent predictions."""
    limit = request.args.get('limit', 20, type=int)
    df = get_recent_predictions(limit)
    return jsonify(df.to_dict(orient='records'))


# ─── Initialize and Run ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Initializing database...")
    init_db()
    seed_data()
    print("\n🚀 Starting Flask API on http://localhost:5000")
    app.run(debug=True, port=5000)
