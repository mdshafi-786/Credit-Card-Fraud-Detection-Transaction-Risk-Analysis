"""
Credit Card Fraud Detection — Database Module
===============================================
SQLite database setup, schema, seeding, and query functions.
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "fraud_detection.db")
CSV_PATH = os.path.join(BASE_DIR, "transactions.csv")


def get_connection():
    """Get a SQLite connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database schema."""
    conn = get_connection()
    cursor = conn.cursor()

    # Transactions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            customer_id TEXT,
            transaction_amount_inr REAL,
            merchant_category TEXT,
            merchant_id TEXT,
            transaction_datetime TEXT,
            transaction_hour INTEGER,
            day_of_week TEXT,
            customer_age INTEGER,
            customer_gender TEXT,
            customer_income_monthly_inr INTEGER,
            customer_tenure_months INTEGER,
            avg_transaction_amount_inr REAL,
            transaction_count_30d INTEGER,
            transaction_city TEXT,
            transaction_state TEXT,
            distance_from_home_km REAL,
            card_type TEXT,
            device_type TEXT,
            channel TEXT,
            international_transaction INTEGER,
            failed_attempts_24h INTEGER,
            previous_fraud_count INTEGER,
            account_age_months INTEGER,
            amount_to_customer_avg_ratio REAL,
            is_fraud INTEGER
        )
    """)

    # Predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            prediction INTEGER,
            risk_score REAL,
            risk_level TEXT,
            predicted_at TEXT,
            model_name TEXT,
            FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
        )
    """)

    # Alerts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            alert_type TEXT,
            severity TEXT,
            message TEXT,
            risk_score REAL,
            is_resolved INTEGER DEFAULT 0,
            created_at TEXT,
            resolved_at TEXT,
            FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
        )
    """)

    conn.commit()
    conn.close()
    print("  ✓ Database schema initialized")


def seed_data():
    """Seed the database with CSV data."""
    conn = get_connection()
    cursor = conn.cursor()

    # Check if already seeded
    cursor.execute("SELECT COUNT(*) FROM transactions")
    count = cursor.fetchone()[0]

    if count > 0:
        print(f"  ✓ Database already has {count} transactions")
        conn.close()
        return

    # Load CSV and insert
    df = pd.read_csv(CSV_PATH)
    df.to_sql('transactions', conn, if_exists='append', index=False)

    print(f"  ✓ Seeded {len(df)} transactions into database")
    conn.commit()
    conn.close()


def get_all_transactions(limit=100, offset=0, filters=None):
    """Fetch transactions with optional filters."""
    conn = get_connection()
    query = "SELECT * FROM transactions WHERE 1=1"
    params = []

    if filters:
        if filters.get('city'):
            query += " AND transaction_city = ?"
            params.append(filters['city'])
        if filters.get('category'):
            query += " AND merchant_category = ?"
            params.append(filters['category'])
        if filters.get('is_fraud') is not None:
            query += " AND is_fraud = ?"
            params.append(filters['is_fraud'])
        if filters.get('min_amount'):
            query += " AND transaction_amount_inr >= ?"
            params.append(filters['min_amount'])
        if filters.get('max_amount'):
            query += " AND transaction_amount_inr <= ?"
            params.append(filters['max_amount'])

    query += " ORDER BY transaction_datetime DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_transaction_by_id(txn_id):
    """Fetch a single transaction."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM transactions WHERE transaction_id = ?",
        conn, params=[txn_id]
    )
    conn.close()
    return df


def save_prediction(txn_id, prediction, risk_score, risk_level, model_name="RandomForest"):
    """Save a prediction result."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (transaction_id, prediction, risk_score, risk_level, predicted_at, model_name)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (txn_id, prediction, risk_score, risk_level, datetime.now().isoformat(), model_name))
    conn.commit()
    conn.close()


def save_alert(txn_id, alert_type, severity, message, risk_score):
    """Save an alert for a suspicious transaction."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO alerts (transaction_id, alert_type, severity, message, risk_score, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (txn_id, alert_type, severity, message, risk_score, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_alerts(resolved=False, limit=50):
    """Fetch alerts."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM alerts WHERE is_resolved = ? ORDER BY created_at DESC LIMIT ?",
        conn, params=[int(resolved), limit]
    )
    conn.close()
    return df


def resolve_alert(alert_id):
    """Mark an alert as resolved."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE alerts SET is_resolved = 1, resolved_at = ? WHERE id = ?",
        (datetime.now().isoformat(), alert_id)
    )
    conn.commit()
    conn.close()


def get_dashboard_stats():
    """Get overview statistics for the dashboard."""
    conn = get_connection()
    cursor = conn.cursor()

    stats = {}
    cursor.execute("SELECT COUNT(*) FROM transactions")
    stats['total_transactions'] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = 1")
    stats['total_fraud'] = cursor.fetchone()[0]

    cursor.execute("SELECT SUM(transaction_amount_inr) FROM transactions")
    stats['total_amount'] = cursor.fetchone()[0] or 0

    cursor.execute("SELECT AVG(transaction_amount_inr) FROM transactions")
    stats['avg_amount'] = cursor.fetchone()[0] or 0

    cursor.execute("SELECT COUNT(*) FROM alerts WHERE is_resolved = 0")
    stats['active_alerts'] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions")
    stats['total_predictions'] = cursor.fetchone()[0]

    if stats['total_transactions'] > 0:
        stats['fraud_rate'] = (stats['total_fraud'] / stats['total_transactions']) * 100
    else:
        stats['fraud_rate'] = 0

    conn.close()
    return stats


def get_fraud_by_category():
    """Get fraud stats grouped by merchant category."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT merchant_category,
               COUNT(*) as total,
               SUM(is_fraud) as fraud_count,
               ROUND(AVG(is_fraud) * 100, 2) as fraud_rate
        FROM transactions
        GROUP BY merchant_category
        ORDER BY fraud_count DESC
    """, conn)
    conn.close()
    return df


def get_fraud_by_hour():
    """Get fraud stats grouped by transaction hour."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT transaction_hour,
               COUNT(*) as total,
               SUM(is_fraud) as fraud_count,
               ROUND(AVG(is_fraud) * 100, 2) as fraud_rate
        FROM transactions
        GROUP BY transaction_hour
        ORDER BY transaction_hour
    """, conn)
    conn.close()
    return df


def get_fraud_by_city():
    """Get fraud stats grouped by city."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT transaction_city,
               COUNT(*) as total,
               SUM(is_fraud) as fraud_count,
               ROUND(AVG(is_fraud) * 100, 2) as fraud_rate
        FROM transactions
        GROUP BY transaction_city
        ORDER BY fraud_count DESC
    """, conn)
    conn.close()
    return df


def get_fraud_by_channel():
    """Get fraud stats grouped by channel."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT channel,
               COUNT(*) as total,
               SUM(is_fraud) as fraud_count,
               ROUND(AVG(is_fraud) * 100, 2) as fraud_rate
        FROM transactions
        GROUP BY channel
        ORDER BY fraud_count DESC
    """, conn)
    conn.close()
    return df


def get_fraud_by_card_type():
    """Get fraud stats grouped by card type."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT card_type,
               COUNT(*) as total,
               SUM(is_fraud) as fraud_count,
               ROUND(AVG(is_fraud) * 100, 2) as fraud_rate
        FROM transactions
        GROUP BY card_type
        ORDER BY fraud_count DESC
    """, conn)
    conn.close()
    return df


def get_fraud_trend():
    """Get daily fraud trend."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT DATE(transaction_datetime) as date,
               COUNT(*) as total,
               SUM(is_fraud) as fraud_count,
               ROUND(AVG(is_fraud) * 100, 2) as fraud_rate
        FROM transactions
        WHERE transaction_datetime IS NOT NULL
        GROUP BY DATE(transaction_datetime)
        HAVING date IS NOT NULL
        ORDER BY date
    """, conn)
    conn.close()

    # Fallback to pandas date parsing if SQLite DATE() returned empty or invalid
    if df.empty or df['date'].isna().all() or len(df) <= 1:
        conn = get_connection()
        raw_df = pd.read_sql_query("SELECT transaction_datetime, is_fraud FROM transactions", conn)
        conn.close()
        if not raw_df.empty:
            raw_df['date'] = pd.to_datetime(raw_df['transaction_datetime'], errors='coerce').dt.strftime('%Y-%m-%d')
            raw_df = raw_df.dropna(subset=['date'])
            df = raw_df.groupby('date').agg(
                total=('is_fraud', 'count'),
                fraud_count=('is_fraud', 'sum'),
                fraud_rate=('is_fraud', lambda x: round(x.mean() * 100, 2))
            ).reset_index().sort_values('date')
    return df


def get_amount_distribution():
    """Get transaction amount distribution for fraud vs legitimate."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT transaction_amount_inr, is_fraud
        FROM transactions
    """, conn)
    conn.close()
    return df


def get_recent_predictions(limit=20):
    """Get recent predictions."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT p.*, t.transaction_amount_inr, t.merchant_category,
               t.transaction_city, t.customer_id
        FROM predictions p
        LEFT JOIN transactions t ON p.transaction_id = t.transaction_id
        ORDER BY p.predicted_at DESC
        LIMIT ?
    """, conn, params=[limit])
    conn.close()
    return df


# Initialize on import
if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    seed_data()
    stats = get_dashboard_stats()
    print(f"\nDashboard Stats: {stats}")
