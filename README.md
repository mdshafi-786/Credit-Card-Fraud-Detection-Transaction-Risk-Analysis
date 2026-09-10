# 🛡️ Credit Card Fraud Detection & Transaction Risk Analysis

A full-stack **machine learning-powered** fraud detection system with a real-time Streamlit dashboard, Flask REST API, and an automated ML training pipeline. The system analyzes credit card transactions using **Random Forest** and **XGBoost** classifiers to detect fraudulent activity, assigns risk scores, and raises alerts — all presented through an interactive dark-themed analytics dashboard.

---

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Dashboard Pages](#-dashboard-pages)
- [Screenshots](#-screenshots)
- [License](#-license)

---

## ✨ Features

- **Real-Time Fraud Prediction** — Submit transaction details and get instant fraud/genuine classification with a risk score (0–100%).
- **Interactive Analytics Dashboard** — KPI cards, trend charts, category breakdowns, hourly heatmaps, city-wise analysis, and more via Plotly.
- **Dual-Model Training** — Trains both Random Forest and XGBoost; automatically selects the best model based on F1-score.
- **SMOTE Oversampling** — Handles severe class imbalance (≈3.17% fraud rate) using Synthetic Minority Oversampling.
- **Feature Engineering** — 6 custom-engineered features: night transaction flag, high amount flag, amount deviation, velocity score, failed attempt risk, and account risk score.
- **Alert Management System** — Automatic alert generation for high-risk transactions (>70% risk score) with Critical/High severity tiers and one-click resolution.
- **Transaction History Browser** — Filter, search, and download transactions by city, merchant category, fraud status, and more.
- **Model Performance Dashboard** — Confusion matrix, feature importance, model comparison table, and recent prediction log.
- **REST API** — Flask-based API for programmatic access to predictions, transactions, dashboard stats, and alerts.
- **SQLite Database** — Persistent storage for transactions, predictions, and alerts with auto-seeding from CSV.

---

## 🛠️ Tech Stack

| Layer          | Technology                                                        |
| -------------- | ----------------------------------------------------------------- |
| **Frontend**   | Streamlit 1.41, Plotly 5.24, Custom CSS (dark glassmorphism)      |
| **Backend**    | Flask 3.1, Flask-CORS 5.0                                        |
| **ML/AI**      | Scikit-learn 1.6, XGBoost 2.1, Imbalanced-learn 0.12 (SMOTE)    |
| **Data**       | Pandas 2.2, NumPy 1.26                                           |
| **Database**   | SQLite 3 (via Python `sqlite3`)                                   |
| **Viz**        | Plotly, Matplotlib 3.9, Seaborn 0.13                              |
| **Serialization** | Joblib 1.4                                                    |

---

## 📁 Project Structure

```
Credit card fraud detection & Transaction Risk Analysis/
│
├── streamlit_app.py          # Main Streamlit dashboard (frontend)
├── analyze_data.py           # Exploratory data analysis script
├── transactions.csv          # Dataset — 12,000 transactions (26 features)
├── fraud_detection.db        # SQLite database (auto-generated)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── .streamlit/
│   └── config.toml           # Streamlit theme config (dark mode)
│
└── backend/
    ├── app.py                # Flask REST API server
    ├── database.py           # SQLite schema, seeding & query functions
    ├── model_training.py     # ML training pipeline (RF + XGBoost + SMOTE)
    ├── predict.py            # Prediction service (loads model, runs inference)
    │
    └── ml/                   # Trained model artifacts (auto-generated)
        ├── fraud_model.pkl         # Best trained model (serialized)
        ├── scaler.pkl              # StandardScaler
        ├── label_encoders.pkl      # LabelEncoders for categorical features
        ├── feature_names.pkl       # Ordered feature name list
        ├── model_metrics.json      # Evaluation metrics (accuracy, F1, AUC, etc.)
        └── feature_importance.json # Top-15 feature importances
```

---

## 📊 Dataset

The project uses a synthetic Indian credit card transaction dataset.

| Property           | Value                                         |
| ------------------ | --------------------------------------------- |
| **Records**        | 12,000 transactions                           |
| **Features**       | 26 columns                                    |
| **Target**         | `is_fraud` (binary: 0 = Legitimate, 1 = Fraud)|
| **Fraud Rate**     | 3.17% (381 fraud / 11,619 legitimate)         |
| **Currency**       | INR (₹)                                       |

### Key Features

| Feature                          | Type        | Description                                 |
| -------------------------------- | ----------- | ------------------------------------------- |
| `transaction_amount_inr`         | Numerical   | Transaction amount in Indian Rupees         |
| `merchant_category`              | Categorical | e.g., Online Shopping, Grocery, Jewelry     |
| `transaction_hour`               | Numerical   | Hour of day (0–23)                          |
| `day_of_week`                    | Categorical | Day name (Monday–Sunday)                    |
| `customer_age`                   | Numerical   | Customer's age                              |
| `customer_income_monthly_inr`    | Numerical   | Monthly income in INR                       |
| `distance_from_home_km`          | Numerical   | Distance of transaction from customer home  |
| `card_type`                      | Categorical | Visa, Mastercard, RuPay, Amex              |
| `channel`                        | Categorical | POS, Online, Mobile App, ATM               |
| `international_transaction`      | Binary      | 1 = international, 0 = domestic             |
| `failed_attempts_24h`            | Numerical   | Failed transaction attempts in last 24h     |
| `previous_fraud_count`           | Numerical   | Historical fraud count for the customer     |
| `amount_to_customer_avg_ratio`   | Numerical   | Ratio of transaction amount to customer avg |

---

## 🤖 Machine Learning Pipeline

### Pipeline Overview

```
CSV Data → Feature Engineering → Label Encoding → StandardScaler
    → Train/Test Split (80/20, stratified)
    → SMOTE Oversampling (on training set only)
    → Model Training (Random Forest + XGBoost)
    → Model Comparison (F1-Score)
    → Best Model Selection → Save Artifacts
```

### Feature Engineering (6 custom features)

| Feature                | Logic                                              |
| ---------------------- | -------------------------------------------------- |
| `is_night_transaction` | 1 if hour ≥ 22 or hour ≤ 5                        |
| `is_high_amount`       | 1 if amount > ₹10,000                             |
| `amount_deviation`     | `amount / (avg_amount + 1)`                        |
| `velocity_score`       | `transaction_count_30d × amount_to_avg_ratio`      |
| `failed_attempt_risk`  | `failed_attempts_24h × is_night_transaction`       |
| `account_risk_score`   | `1 / (account_age_months + 1)`                     |

### Models

| Model           | Key Hyperparameters                                                |
| --------------- | ------------------------------------------------------------------ |
| **Random Forest** | 200 trees, max_depth=15, class_weight=balanced                   |
| **XGBoost**       | 200 estimators, max_depth=8, lr=0.1, scale_pos_weight=30        |

### Class Imbalance Handling

- **SMOTE** (Synthetic Minority Oversampling Technique) with `sampling_strategy=0.5` is applied to the training set only to prevent data leakage.

### Evaluation Metrics

The best model is selected by **F1-Score**. Reported metrics include:

- Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Confusion Matrix (TN, FP, FN, TP)
- Top-15 Feature Importance

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **pip** (Python package manager)

### Installation

1. **Clone or download** the project:

   ```bash
   git clone <repository-url>
   cd "Credit card fraud detection & Transaction Risk Analysis"
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the ML model** (required before first run):

   ```bash
   python backend/model_training.py
   ```

   This will:
   - Load `transactions.csv`
   - Engineer features, encode, and scale
   - Apply SMOTE and train Random Forest + XGBoost
   - Save the best model and artifacts to `backend/ml/`

4. **Launch the Streamlit dashboard**:

   ```bash
   streamlit run streamlit_app.py
   ```

   The dashboard auto-initializes the SQLite database and seeds it from the CSV on first launch.

### (Optional) Run the Flask API

```bash
python backend/app.py
```

The API server starts on `http://localhost:5000`.

---

## 💡 Usage

### Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Navigate between pages using the **sidebar** or the **top segmented navigation bar**.

### Exploratory Data Analysis

```bash
python analyze_data.py
```

Prints a comprehensive terminal report covering: dataset shape, missing values, statistics, fraud distribution, category breakdowns, hourly analysis, city-wise fraud rates, and feature correlations.

### Flask API (Standalone)

```bash
python backend/app.py
```

Use any HTTP client (curl, Postman, fetch) to interact with the REST endpoints.

---

## 🔌 API Endpoints

### Health

| Method | Endpoint         | Description          |
| ------ | ---------------- | -------------------- |
| GET    | `/api/health`    | Health check         |

### Prediction

| Method | Endpoint         | Description                                 |
| ------ | ---------------- | ------------------------------------------- |
| POST   | `/api/predict`   | Predict fraud for a transaction (JSON body) |

**Example request body:**

```json
{
  "transaction_amount_inr": 25000,
  "merchant_category": "Jewelry",
  "transaction_hour": 2,
  "day_of_week": "Tuesday",
  "customer_age": 35,
  "customer_gender": "Male",
  "customer_income_monthly_inr": 50000,
  "customer_tenure_months": 12,
  "avg_transaction_amount_inr": 3000,
  "transaction_count_30d": 15,
  "transaction_city": "Delhi",
  "transaction_state": "Delhi",
  "distance_from_home_km": 150,
  "card_type": "Visa",
  "device_type": "Web",
  "channel": "Online",
  "international_transaction": 1,
  "failed_attempts_24h": 3,
  "previous_fraud_count": 0,
  "account_age_months": 6,
  "amount_to_customer_avg_ratio": 8.33
}
```

### Transactions

| Method | Endpoint                      | Description                       |
| ------ | ----------------------------- | --------------------------------- |
| GET    | `/api/transactions`           | List transactions (with filters)  |
| GET    | `/api/transactions/<txn_id>`  | Get single transaction            |

**Query params:** `limit`, `offset`, `city`, `category`, `is_fraud`

### Dashboard Analytics

| Method | Endpoint                            | Description                 |
| ------ | ----------------------------------- | --------------------------- |
| GET    | `/api/dashboard/stats`              | KPI overview stats          |
| GET    | `/api/dashboard/fraud-by-category`  | Fraud by merchant category  |
| GET    | `/api/dashboard/fraud-by-hour`      | Fraud by hour of day        |
| GET    | `/api/dashboard/fraud-by-city`      | Fraud by city               |
| GET    | `/api/dashboard/fraud-by-channel`   | Fraud by channel            |
| GET    | `/api/dashboard/fraud-by-card`      | Fraud by card type          |
| GET    | `/api/dashboard/fraud-trend`        | Daily fraud trend           |

### Alerts

| Method | Endpoint                              | Description          |
| ------ | ------------------------------------- | -------------------- |
| GET    | `/api/alerts`                         | Get alerts           |
| PUT    | `/api/alerts/<alert_id>/resolve`      | Resolve an alert     |
| GET    | `/api/predictions/recent`             | Recent predictions   |

---

## 📄 Dashboard Pages

### 📊 Dashboard
The main analytics hub displaying:
- **5 KPI cards** — Total Transactions, Fraud Detected, Fraud Rate, Total Volume, Active Alerts
- **Fraud Trend Over Time** — Dual-axis line chart (total vs fraud)
- **Fraud vs Legitimate** — Donut chart with fraud percentage
- **Fraud by Hour** — Color-graded bar chart showing risky hours
- **Fraud by Merchant Category** — Horizontal bar chart
- **Fraud by City** — Bar chart of top fraud cities
- **Fraud by Channel** — Donut chart (POS, Online, Mobile, ATM)
- **Transaction Amount Distribution** — Overlaid histogram (fraud vs legit)
- **Fraud by Card Type** — Grouped bar chart
- **Recent Alerts** — Live alert feed with severity badges

### 🔍 Predict Transaction
Interactive form to test fraud predictions:
- Input transaction details (amount, merchant, hour, location, etc.)
- Input customer profile (age, gender, income, tenure)
- Set risk indicators (failed attempts, previous fraud, distance)
- Receive: **risk gauge**, **risk level**, **contributing factors** chart

### 📋 Transaction History
Filterable data table:
- Filter by city, merchant category, fraud status
- Color-coded fraud rows (red highlight)
- Summary metrics row
- CSV download option

### 🚨 Alerts
Alert management center:
- **Active Alerts** tab with resolve buttons
- **Resolved Alerts** tab with history
- Severity-coded display (🔴 Critical / 🟠 High)

### 🤖 Model Performance
ML evaluation dashboard:
- Best model badge and KPI metrics
- Model comparison table (RF vs XGBoost)
- Interactive confusion matrix heatmap
- Top-15 feature importance bar chart
- Training configuration details
- Recent predictions log

---

## ⚙️ Configuration

### Streamlit Theme (`.streamlit/config.toml`)

The dashboard uses a custom dark theme:

```toml
[theme]
base = "dark"
primaryColor = "#667eea"
backgroundColor = "#0a0e27"
secondaryBackgroundColor = "#1a1f4e"
textColor = "#e2e8f0"
font = "sans serif"
```

---

## 📦 Requirements

```
flask==3.1.1
flask-cors==5.0.1
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.6.1
xgboost==2.1.4
imbalanced-learn==0.12.4
joblib==1.4.2
streamlit==1.41.1
plotly==5.24.1
matplotlib==3.9.4
seaborn==0.13.2
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║          ⚡  D E S I G N E D   &   D E V E L O P E D  ⚡        ║
║                         — BY —                                   ║
║                                                                  ║
║       ███╗   ███╗ ██████╗ ██╗  ██╗ █████╗ ███╗   ███╗           ║
║       ████╗ ████║██╔═══██╗██║  ██║██╔══██╗████╗ ████║           ║
║       ██╔████╔██║██║   ██║███████║███████║██╔████╔██║           ║
║       ██║╚██╔╝██║██║   ██║██╔══██║██╔══██║██║╚██╔╝██║           ║
║       ██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║██║ ╚═╝ ██║           ║
║       ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝           ║
║                                                                  ║
║           ███████╗██╗  ██╗ █████╗ ███████╗██╗                    ║
║           ██╔════╝██║  ██║██╔══██╗██╔════╝██║                    ║
║           ███████╗███████║███████║█████╗  ██║                    ║
║           ╚════██║██╔══██║██╔══██║██╔══╝  ██║                    ║
║           ███████║██║  ██║██║  ██║██║     ██║                    ║
║           ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝                    ║
║                                                                  ║
║            ░█░█░█░  MOHAMMED SHAFIULLA  ░█░█░█░                  ║
║                                                                  ║
║   ─────────────────────────────────────────────────────────────   ║
║          🛡️ Fraud Detection  •  🤖 Machine Learning              ║
║          📊 Data Analytics   •  🚀 Full-Stack Python              ║
║   ─────────────────────────────────────────────────────────────   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

**Built with ❤️ using Python, Streamlit & Scikit-learn**

