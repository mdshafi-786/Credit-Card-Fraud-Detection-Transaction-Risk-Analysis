"""
Credit Card Fraud Detection & Transaction Risk Analysis
========================================================
Streamlit Dashboard — Real-time Monitoring, Prediction & Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from backend.database import (
    init_db, seed_data, get_dashboard_stats, get_fraud_by_category,
    get_fraud_by_hour, get_fraud_by_city, get_fraud_by_channel,
    get_fraud_by_card_type, get_fraud_trend, get_all_transactions,
    get_alerts, resolve_alert, save_prediction, save_alert,
    get_amount_distribution, get_recent_predictions
)
from backend.predict import predict_fraud

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Dark Theme Enforced */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    section.main,
    .block-container {
        background-color: #0a0e27 !important;
        color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* All general typography */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
    }

    p, span, div {
        color: #e2e8f0;
    }

    /* Hide default streamlit branding but keep sidebar toggle button visible */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar collapse & expand toggle button */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        color: #ffffff !important;
        background: #1a1f4e !important;
        border: 1px solid rgba(102, 126, 234, 0.4) !important;
        border-radius: 10px !important;
        margin: 10px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4) !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg {
        fill: #667eea !important;
    }

    /* Top Navigation styling */
    [data-testid="stSegmentedControl"] {
        background: rgba(16, 20, 56, 0.85) !important;
        padding: 6px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        margin-bottom: 1.2rem !important;
        width: 100% !important;
    }

    [data-testid="stSegmentedControl"] button {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border-radius: 10px !important;
    }

    [data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.35) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070a1e 0%, #101438 100%) !important;
        border-right: 1px solid rgba(102, 126, 234, 0.2) !important;
    }

    /* High contrast text inside sidebar */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label {
        color: #f1f5f9 !important;
    }

    /* Sidebar Radio item styling for clear readability */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        gap: 8px;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        background: rgba(102, 126, 234, 0.08) !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 10px !important;
        padding: 9px 14px !important;
        margin-bottom: 4px !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background: rgba(102, 126, 234, 0.25) !important;
        border-color: rgba(102, 126, 234, 0.5) !important;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label p,
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label span {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Active selection in sidebar */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.35) 0%, rgba(118, 75, 162, 0.35) 100%) !important;
        border: 1px solid #667eea !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25) !important;
    }

    /* Inputs, Selectboxes, Sliders high contrast */
    label[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] span {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="input"] input {
        background-color: #12173d !important;
        color: #ffffff !important;
        border-color: rgba(102, 126, 234, 0.3) !important;
    }

    div[data-baseweb="select"] span {
        color: #ffffff !important;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f4e 50%, #2d1b69 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(102, 126, 234, 0.25);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    .main-header h1 {
        color: #ffffff !important;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        color: #cbd5e1 !important;
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a1f4e 0%, #0d1137 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.25);
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }

    .kpi-label {
        color: #cbd5e1 !important;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }

    .kpi-icon {
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
    }

    .kpi-blue .kpi-value { color: #818cf8 !important; }
    .kpi-red .kpi-value { color: #ff5252 !important; }
    .kpi-green .kpi-value { color: #2ed573 !important; }
    .kpi-orange .kpi-value { color: #ffa502 !important; }
    .kpi-purple .kpi-value { color: #c084fc !important; }

    /* Section headers */
    .section-header {
        color: #f1f5f9 !important;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.35);
    }

    /* Alert cards */
    .alert-card {
        background: rgba(255, 71, 87, 0.1);
        border: 1px solid rgba(255, 71, 87, 0.35);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.6rem;
    }

    .alert-card.critical {
        border-color: rgba(255, 71, 87, 0.6);
        background: rgba(255, 71, 87, 0.16);
    }

    .alert-card.high {
        border-color: rgba(255, 165, 2, 0.6);
        background: rgba(255, 165, 2, 0.12);
    }

    /* Prediction result cards */
    .result-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .result-fraud {
        background: linear-gradient(135deg, rgba(255, 71, 87, 0.2) 0%, rgba(255, 71, 87, 0.08) 100%);
        border: 2px solid rgba(255, 71, 87, 0.5);
    }

    .result-genuine {
        background: linear-gradient(135deg, rgba(46, 213, 115, 0.2) 0%, rgba(46, 213, 115, 0.08) 100%);
        border: 2px solid rgba(46, 213, 115, 0.5);
    }

    /* Metric boxes */
    .metric-panel {
        background: linear-gradient(135deg, #1a1f4e 0%, #0d1137 100%);
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    /* Table styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .badge-critical { background: #ff4757; color: white; }
    .badge-high { background: #ffa502; color: white; }
    .badge-medium { background: #667eea; color: white; }
    .badge-low { background: #2ed573; color: white; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        color: #cbd5e1 !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* Custom Metric Overrides */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f4e 0%, #0d1137 100%) !important;
        border: 1px solid rgba(102, 126, 234, 0.25) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25) !important;
    }

    [data-testid="stMetricValue"] {
        font-weight: 800 !important;
        color: #ffffff !important;
    }

    [data-testid="stMetricLabel"] p {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize System ───────────────────────────────────────────────────────
@st.cache_resource
def initialize_system():
    init_db()
    seed_data()
    model_path = os.path.join(ROOT_DIR, "backend", "ml", "fraud_model.pkl")
    metrics_path = os.path.join(ROOT_DIR, "backend", "ml", "model_metrics.json")
    if not (os.path.exists(model_path) and os.path.exists(metrics_path)):
        try:
            from backend.model_training import main as train_pipeline
            train_pipeline()
        except Exception as e:
            print(f"Auto-training on startup: {e}")
    return True

initialize_system()


PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#f1f5f9', size=12),
    margin=dict(l=45, r=30, t=40, b=40),
)

COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'danger': '#ff4757',
    'success': '#2ed573',
    'warning': '#ffa502',
    'info': '#3498db',
    'gradient': ['#667eea', '#764ba2', '#a855f7', '#3498db', '#2ed573', '#ffa502',
                 '#ff4757', '#e84393', '#00b894', '#fdcb6e', '#6c5ce7', '#00cec9']
}


PAGES = [
    "📊 Dashboard",
    "🔍 Predict Transaction",
    "📋 Transaction History",
    "🚨 Alerts",
    "🤖 Model Performance"
]

if "current_page" not in st.session_state:
    st.session_state["current_page"] = PAGES[0]

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    font-size: 1.4rem; font-weight: 800; margin: 0;">
            🛡️ FraudGuard AI
        </h2>
        <p style="color: #a0aec0; font-size: 0.75rem; margin-top: 0.3rem;">
            Real-time Fraud Detection System
        </p>
    </div>
    <hr style="border-color: rgba(102, 126, 234, 0.2); margin: 0.5rem 0 1rem 0;">
    """, unsafe_allow_html=True)

    sidebar_idx = PAGES.index(st.session_state["current_page"]) if st.session_state["current_page"] in PAGES else 0
    sidebar_selected = st.radio(
        "Navigation",
        PAGES,
        index=sidebar_idx,
        key=f"sidebar_nav_{st.session_state['current_page']}",
        label_visibility="collapsed"
    )
    if sidebar_selected != st.session_state["current_page"]:
        st.session_state["current_page"] = sidebar_selected
        st.rerun()

    st.markdown("<hr style='border-color: rgba(102, 126, 234, 0.2);'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 0.8rem; background: rgba(102, 126, 234, 0.08);
                border-radius: 12px; border: 1px solid rgba(102, 126, 234, 0.15);">
        <p style="color: #667eea; font-weight: 600; font-size: 0.8rem; margin: 0;">
            ⚡ System Status
        </p>
        <p style="color: #2ed573; font-size: 0.75rem; margin: 0.3rem 0 0 0;">
            ● Model Active &nbsp; ● DB Connected
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── Top Navigation Menu ─────────────────────────────────────────────────────
# Prominently displayed right at top of page so options are never hidden
top_selected = st.segmented_control(
    "Navigation Menu",
    PAGES,
    default=st.session_state["current_page"],
    key=f"top_nav_{st.session_state['current_page']}",
    label_visibility="collapsed"
)
if top_selected and top_selected != st.session_state["current_page"]:
    st.session_state["current_page"] = top_selected
    st.rerun()

page = st.session_state["current_page"]


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Dashboard":

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Fraud Analytics Dashboard</h1>
        <p>Real-time monitoring of credit card transactions and fraud detection metrics</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── KPI Cards ───────────────────────────────────────────────────────
    stats = get_dashboard_stats()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="kpi-card kpi-blue">
            <div class="kpi-icon">💳</div>
            <div class="kpi-value">{stats['total_transactions']:,}</div>
            <div class="kpi-label">Total Transactions</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card kpi-red">
            <div class="kpi-icon">🚫</div>
            <div class="kpi-value">{stats['total_fraud']:,}</div>
            <div class="kpi-label">Fraud Detected</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card kpi-orange">
            <div class="kpi-icon">📈</div>
            <div class="kpi-value">{stats['fraud_rate']:.2f}%</div>
            <div class="kpi-label">Fraud Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card kpi-green">
            <div class="kpi-icon">💰</div>
            <div class="kpi-value">₹{stats['total_amount']/10000000:.1f}Cr</div>
            <div class="kpi-label">Total Volume</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="kpi-card kpi-purple">
            <div class="kpi-icon">🔔</div>
            <div class="kpi-value">{stats['active_alerts']}</div>
            <div class="kpi-label">Active Alerts</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Row 1: Fraud Trend + Risk Donut ─────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">📈 Fraud Trend Over Time</div>', unsafe_allow_html=True)
        trend_df = get_fraud_trend()
        if not trend_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_df['date'], y=trend_df['total'],
                name='Total Transactions', mode='lines',
                line=dict(color='#818cf8', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(129, 140, 248, 0.08)'
            ))
            fig.add_trace(go.Scatter(
                x=trend_df['date'], y=trend_df['fraud_count'],
                name='Fraud Detected', mode='lines',
                line=dict(color='#ff5252', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(255, 82, 82, 0.12)',
                yaxis='y2'
            ))
            fig.update_layout(
                **PLOT_LAYOUT,
                height=360,
                xaxis=dict(
                    title=dict(text='Date', font=dict(color='#cbd5e1', size=12)),
                    tickfont=dict(color='#94a3b8', size=11),
                    gridcolor='rgba(102, 126, 234, 0.12)',
                    showline=True,
                    linecolor='rgba(102, 126, 234, 0.25)'
                ),
                yaxis=dict(
                    title=dict(text='Total Transactions', font=dict(color='#818cf8', size=12)),
                    tickfont=dict(color='#818cf8', size=11),
                    gridcolor='rgba(102, 126, 234, 0.12)',
                    showline=True,
                    linecolor='rgba(102, 126, 234, 0.25)'
                ),
                yaxis2=dict(
                    title=dict(text='Fraud Detected', font=dict(color='#ff5252', size=12)),
                    tickfont=dict(color='#ff5252', size=11),
                    overlaying='y',
                    side='right',
                    gridcolor='rgba(255, 71, 87, 0.05)',
                    showline=True,
                    linecolor='rgba(255, 71, 87, 0.25)'
                ),
                legend=dict(
                    orientation='h',
                    y=1.12,
                    x=0.0,
                    xanchor='left',
                    font=dict(color='#f1f5f9', size=11),
                    bgcolor='rgba(10, 14, 39, 0.7)'
                ),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col2:
        st.markdown('<div class="section-header">🎯 Fraud vs Legitimate</div>', unsafe_allow_html=True)
        fig = go.Figure(data=[go.Pie(
            labels=['Legitimate', 'Fraud'],
            values=[stats['total_transactions'] - stats['total_fraud'], stats['total_fraud']],
            hole=0.65,
            marker=dict(colors=[COLORS['success'], COLORS['danger']]),
            textinfo='percent+label',
            textfont=dict(size=12, color='white'),
            hoverinfo='label+value+percent'
        )])
        fig.update_layout(
            **PLOT_LAYOUT,
            height=350,
            showlegend=False,
            annotations=[dict(
                text=f"<b>{stats['fraud_rate']:.1f}%</b><br>Fraud",
                x=0.5, y=0.5, font_size=16, font_color='#ff4757',
                showarrow=False
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

    # ─── Row 2: Hourly Fraud + Category Fraud ────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">🕐 Fraud by Transaction Hour</div>', unsafe_allow_html=True)
        hour_df = get_fraud_by_hour()
        if not hour_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hour_df['transaction_hour'],
                y=hour_df['fraud_rate'],
                marker=dict(
                    color=hour_df['fraud_rate'],
                    colorscale=[[0, COLORS['success']], [0.5, COLORS['warning']], [1, COLORS['danger']]],
                    line=dict(width=0)
                ),
                hovertemplate='Hour: %{x}<br>Fraud Rate: %{y:.1f}%<extra></extra>'
            ))
            fig.update_layout(
                **PLOT_LAYOUT,
                height=350,
                xaxis=dict(title='Hour of Day', dtick=1, gridcolor='rgba(102, 126, 234, 0.1)'),
                yaxis=dict(title='Fraud Rate (%)', gridcolor='rgba(102, 126, 234, 0.1)'),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🏷️ Fraud by Merchant Category</div>', unsafe_allow_html=True)
        cat_df = get_fraud_by_category()
        if not cat_df.empty:
            cat_df = cat_df.sort_values('fraud_rate', ascending=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=cat_df['merchant_category'],
                x=cat_df['fraud_rate'],
                orientation='h',
                marker=dict(
                    color=cat_df['fraud_rate'],
                    colorscale=[[0, COLORS['primary']], [1, COLORS['danger']]],
                    line=dict(width=0)
                ),
                text=cat_df['fraud_rate'].apply(lambda x: f'{x:.1f}%'),
                textposition='outside',
                textfont=dict(color='#e2e8f0', size=11),
                hovertemplate='%{y}<br>Fraud Rate: %{x:.2f}%<extra></extra>'
            ))
            fig.update_layout(
                **PLOT_LAYOUT,
                height=350,
                xaxis=dict(title='Fraud Rate (%)', gridcolor='rgba(102, 126, 234, 0.1)'),
                yaxis=dict(gridcolor='rgba(102, 126, 234, 0.1)'),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    # ─── Row 3: City Fraud + Channel/Card ────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">🌍 Fraud by City</div>', unsafe_allow_html=True)
        city_df = get_fraud_by_city()
        if not city_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=city_df['transaction_city'],
                y=city_df['fraud_count'],
                name='Fraud Count',
                marker=dict(color=COLORS['danger'], opacity=0.8),
                text=city_df['fraud_count'],
                textposition='outside',
                textfont=dict(color='#e2e8f0')
            ))
            fig.update_layout(
                **PLOT_LAYOUT,
                height=350,
                xaxis=dict(title='City', tickangle=-45, gridcolor='rgba(102, 126, 234, 0.1)'),
                yaxis=dict(title='Fraud Count', gridcolor='rgba(102, 126, 234, 0.1)'),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">📡 Fraud by Channel</div>', unsafe_allow_html=True)
        channel_df = get_fraud_by_channel()
        if not channel_df.empty:
            fig = go.Figure(data=[go.Pie(
                labels=channel_df['channel'],
                values=channel_df['fraud_count'],
                marker=dict(colors=COLORS['gradient'][:len(channel_df)]),
                textinfo='label+percent',
                textfont=dict(size=12, color='white'),
                hole=0.4
            )])
            fig.update_layout(
                **PLOT_LAYOUT,
                height=350,
                showlegend=True,
                legend=dict(orientation='h', y=-0.1, x=0.5, xanchor='center')
            )
            st.plotly_chart(fig, use_container_width=True)

    # ─── Row 4: Amount Distribution + Card Type ─────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">💰 Transaction Amount Distribution</div>', unsafe_allow_html=True)
        amt_df = get_amount_distribution()
        if not amt_df.empty:
            fig = go.Figure()
            legit = amt_df[amt_df['is_fraud'] == 0]['transaction_amount_inr']
            fraud = amt_df[amt_df['is_fraud'] == 1]['transaction_amount_inr']
            fig.add_trace(go.Histogram(x=legit, name='Legitimate', marker_color=COLORS['success'],
                                        opacity=0.6, nbinsx=50))
            fig.add_trace(go.Histogram(x=fraud, name='Fraud', marker_color=COLORS['danger'],
                                        opacity=0.8, nbinsx=50))
            fig.update_layout(
                **PLOT_LAYOUT,
                height=350,
                barmode='overlay',
                xaxis=dict(title='Amount (₹)', gridcolor='rgba(102, 126, 234, 0.1)', range=[0, 50000]),
                yaxis=dict(title='Count', gridcolor='rgba(102, 126, 234, 0.1)'),
                legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center')
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">💳 Fraud by Card Type</div>', unsafe_allow_html=True)
        card_df = get_fraud_by_card_type()
        if not card_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=card_df['card_type'],
                y=card_df['total'],
                name='Total',
                marker_color=COLORS['primary'],
                opacity=0.6
            ))
            fig.add_trace(go.Bar(
                x=card_df['card_type'],
                y=card_df['fraud_count'],
                name='Fraud',
                marker_color=COLORS['danger']
            ))
            fig.update_layout(
                **PLOT_LAYOUT,
                height=350,
                barmode='group',
                xaxis=dict(title='Card Type', gridcolor='rgba(102, 126, 234, 0.1)'),
                yaxis=dict(title='Count', gridcolor='rgba(102, 234, 234, 0.1)'),
                legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center')
            )
            st.plotly_chart(fig, use_container_width=True)

    # ─── Recent Alerts ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🚨 Recent Alerts</div>', unsafe_allow_html=True)
    alerts_df = get_alerts(resolved=False, limit=5)
    if alerts_df.empty:
        st.info("✅ No active alerts — all transactions appear normal.")
    else:
        for _, alert in alerts_df.iterrows():
            severity_color = '#ff4757' if alert['severity'] == 'Critical' else '#ffa502'
            st.markdown(f"""
            <div class="alert-card {'critical' if alert['severity'] == 'Critical' else 'high'}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: {severity_color}; font-weight: 700;">
                            {'🔴' if alert['severity'] == 'Critical' else '🟠'} {alert['severity']}
                        </span>
                        <span style="color: #a0aec0; margin-left: 1rem; font-size: 0.8rem;">
                            {alert['transaction_id']}
                        </span>
                    </div>
                    <span style="color: #a0aec0; font-size: 0.75rem;">{alert['created_at'][:19]}</span>
                </div>
                <p style="color: #e2e8f0; margin: 0.5rem 0 0 0; font-size: 0.85rem;">{alert['message']}</p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICT TRANSACTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Predict Transaction":

    st.markdown("""
    <div class="main-header">
        <h1>🔍 Transaction Risk Predictor</h1>
        <p>Enter transaction details to check for fraud risk</p>
    </div>
    """, unsafe_allow_html=True)

    # Ensure model exists
    model_path = os.path.join(ROOT_DIR, "backend", "ml", "fraud_model.pkl")
    if not os.path.exists(model_path):
        with st.spinner("🔄 Initializing ML model (one-time setup)..."):
            try:
                from backend.model_training import main as train_pipeline
                train_pipeline()
            except Exception as e:
                st.error(f"⚠️ Unable to load model: {e}")
                st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💳 Transaction Details")
        transaction_amount = st.number_input("Transaction Amount (₹)", min_value=0.0,
                                              max_value=500000.0, value=5000.0, step=100.0)
        merchant_category = st.selectbox("Merchant Category",
            ["Online Shopping", "Grocery", "Restaurant", "Electronics", "Fuel",
             "Clothing", "Utilities", "Entertainment", "Travel", "Healthcare",
             "Education", "Jewelry"])
        transaction_hour = st.slider("Transaction Hour", 0, 23, 14)
        day_of_week = st.selectbox("Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        channel = st.selectbox("Channel", ["POS", "Online", "Mobile App", "ATM"])
        card_type = st.selectbox("Card Type", ["Visa", "Mastercard", "RuPay", "Amex"])
        device_type = st.selectbox("Device Type", ["Android", "iOS", "Web", "ATM"])
        international = st.selectbox("International Transaction", ["No", "Yes"])

    with col2:
        st.markdown("### 👤 Customer Profile")
        customer_age = st.number_input("Customer Age", min_value=18, max_value=80, value=35)
        customer_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        customer_income = st.number_input("Monthly Income (₹)", min_value=10000,
                                           max_value=500000, value=50000, step=5000)
        customer_tenure = st.number_input("Tenure (months)", min_value=1,
                                           max_value=200, value=60)
        account_age = st.number_input("Account Age (months)", min_value=1,
                                       max_value=200, value=48)
        avg_txn_amount = st.number_input("Avg Transaction Amount (₹)", min_value=100.0,
                                          max_value=50000.0, value=2500.0, step=100.0)

        st.markdown("### 📍 Transaction Location")
        transaction_city = st.selectbox("City",
            ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad",
             "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Kochi", "Indore"])
        city_state_map = {
            "Mumbai": "Maharashtra", "Delhi": "Delhi", "Bengaluru": "Karnataka",
            "Chennai": "Tamil Nadu", "Kolkata": "West Bengal", "Hyderabad": "Telangana",
            "Pune": "Maharashtra", "Ahmedabad": "Gujarat", "Jaipur": "Rajasthan",
            "Lucknow": "Uttar Pradesh", "Kochi": "Kerala", "Indore": "Madhya Pradesh"
        }
        transaction_state = city_state_map.get(transaction_city, "Maharashtra")
        distance_from_home = st.number_input("Distance from Home (km)", min_value=0.0,
                                              max_value=500.0, value=10.0, step=1.0)

    st.markdown("### ⚠️ Risk Indicators")
    r1, r2, r3 = st.columns(3)
    with r1:
        failed_attempts = st.number_input("Failed Attempts (24h)", min_value=0, max_value=10, value=0)
    with r2:
        previous_fraud = st.number_input("Previous Fraud Count", min_value=0, max_value=5, value=0)
    with r3:
        txn_count_30d = st.number_input("Transaction Count (30d)", min_value=1, max_value=50, value=13)

    amount_ratio = round(transaction_amount / (avg_txn_amount + 1), 3)

    st.markdown("---")

    if st.button("🔍 Analyze Transaction", use_container_width=True, type="primary"):
        with st.spinner("Running fraud detection model..."):
            txn_data = {
                'transaction_amount_inr': transaction_amount,
                'merchant_category': merchant_category,
                'transaction_hour': transaction_hour,
                'day_of_week': day_of_week,
                'customer_age': customer_age,
                'customer_gender': customer_gender,
                'customer_income_monthly_inr': customer_income,
                'customer_tenure_months': customer_tenure,
                'avg_transaction_amount_inr': avg_txn_amount,
                'transaction_count_30d': txn_count_30d,
                'transaction_city': transaction_city,
                'transaction_state': transaction_state,
                'distance_from_home_km': distance_from_home,
                'card_type': card_type,
                'device_type': device_type,
                'channel': channel,
                'international_transaction': 1 if international == "Yes" else 0,
                'failed_attempts_24h': failed_attempts,
                'previous_fraud_count': previous_fraud,
                'account_age_months': account_age,
                'amount_to_customer_avg_ratio': amount_ratio
            }

            try:
                result = predict_fraud(txn_data)

                # Save prediction
                txn_id = f"TXN_PRED_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                save_prediction(txn_id, result['prediction'], result['risk_score'], result['risk_level'])

                # Alert if suspicious
                if result['risk_score'] > 70:
                    severity = 'Critical' if result['risk_score'] > 90 else 'High'
                    message = (
                        f"Suspicious transaction: ₹{transaction_amount:,.2f} at {merchant_category} "
                        f"in {transaction_city} — Risk: {result['risk_score']}%"
                    )
                    save_alert(txn_id, 'Fraud Detection', severity, message, result['risk_score'])

                # Display result
                st.markdown("---")

                is_fraud = result['prediction'] == 1
                card_class = "result-fraud" if is_fraud else "result-genuine"
                verdict_emoji = "🚫" if is_fraud else "✅"
                verdict_text = "FRAUD DETECTED" if is_fraud else "GENUINE TRANSACTION"
                verdict_color = "#ff4757" if is_fraud else "#2ed573"

                st.markdown(f"""
                <div class="result-card {card_class}">
                    <h1 style="font-size: 3rem; margin: 0;">{verdict_emoji}</h1>
                    <h2 style="color: {verdict_color}; font-weight: 800; margin: 0.5rem 0;">
                        {verdict_text}
                    </h2>
                </div>
                """, unsafe_allow_html=True)

                # Risk Score Gauge and Metrics
                r1, r2, r3 = st.columns(3)
                with r1:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result['risk_score'],
                        title={'text': "Risk Score", 'font': {'color': '#ffffff', 'size': 18, 'family': 'Inter'}},
                        number={'font': {'color': '#ffffff', 'size': 38, 'family': 'Inter'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': '#818cf8', 'tickfont': {'color': '#cbd5e1'}},
                            'bar': {'color': verdict_color},
                            'steps': [
                                {'range': [0, 25], 'color': 'rgba(46, 213, 115, 0.3)'},
                                {'range': [25, 50], 'color': 'rgba(102, 126, 234, 0.3)'},
                                {'range': [50, 75], 'color': 'rgba(255, 165, 2, 0.3)'},
                                {'range': [75, 100], 'color': 'rgba(255, 71, 87, 0.3)'}
                            ],
                            'threshold': {
                                'line': {'color': "#ffffff", 'width': 3},
                                'thickness': 0.8,
                                'value': result['risk_score']
                            }
                        }
                    ))
                    fig.update_layout(
                        paper_bgcolor='rgba(26, 31, 78, 0.7)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Inter', color='#ffffff'),
                        margin=dict(l=30, r=30, t=40, b=20),
                        height=280
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with r2:
                    risk_colors = {'Low': '#2ed573', 'Medium': '#818cf8',
                                   'High': '#ffa502', 'Critical': '#ff4757'}
                    rc = risk_colors.get(result['risk_level'], '#818cf8')
                    st.markdown(f"""
                    <div class="metric-panel">
                        <p style="color: #94a3b8; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 0.4rem 0;">Risk Level</p>
                        <h2 style="color: {rc} !important; font-size: 3rem; font-weight: 900; margin: 0.2rem 0; text-shadow: 0 0 25px {rc}44;">
                            {result['risk_level']}
                        </h2>
                        <p style="color: #cbd5e1; font-size: 0.95rem; font-weight: 500; margin-top: 0.8rem;">
                            Fraud Probability: <strong style="color: #ffffff;">{result['fraud_probability']*100:.2f}%</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                with r3:
                    st.markdown(f"""
                    <div class="metric-panel">
                        <p style="color: #94a3b8; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 0.3rem 0;">Transaction ID</p>
                        <p style="color: #818cf8; font-size: 0.92rem; font-weight: 700; word-break: break-all; margin: 0 0 1rem 0;">{txn_id}</p>
                        <p style="color: #94a3b8; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 0.3rem 0;">Amount</p>
                        <h3 style="color: #ffffff !important; font-size: 2.2rem; font-weight: 800; margin: 0;">
                            ₹{transaction_amount:,.2f}
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)

                # Contributing Factors
                st.markdown('<div class="section-header">🔬 Contributing Factors</div>', unsafe_allow_html=True)
                factors = result.get('contributing_factors', [])
                if factors:
                    factor_df = pd.DataFrame(factors)
                    fig = go.Figure(go.Bar(
                        x=factor_df['importance'],
                        y=factor_df['feature'],
                        orientation='h',
                        marker=dict(
                            color=factor_df['importance'],
                            colorscale=[[0, COLORS['primary']], [1, COLORS['danger']]]
                        ),
                        text=factor_df['importance'].apply(lambda x: f'{x:.1f}%'),
                        textposition='outside',
                        textfont=dict(color='#e2e8f0')
                    ))
                    fig.update_layout(
                        **PLOT_LAYOUT, height=300,
                        xaxis=dict(title='Importance (%)', gridcolor='rgba(102, 126, 234, 0.1)'),
                        yaxis=dict(autorange='reversed'),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TRANSACTION HISTORY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Transaction History":

    st.markdown("""
    <div class="main-header">
        <h1>📋 Transaction History</h1>
        <p>Browse, search, and filter all transactions</p>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        filter_city = st.selectbox("Filter by City", ["All"] +
            ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad",
             "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Kochi", "Indore"])
    with fc2:
        filter_category = st.selectbox("Filter by Category", ["All"] +
            ["Online Shopping", "Grocery", "Restaurant", "Electronics", "Fuel",
             "Clothing", "Utilities", "Entertainment", "Travel", "Healthcare",
             "Education", "Jewelry"])
    with fc3:
        filter_fraud = st.selectbox("Fraud Status", ["All", "Fraud Only", "Legitimate Only"])
    with fc4:
        num_records = st.selectbox("Records per page", [50, 100, 200, 500], index=1)

    filters = {}
    if filter_city != "All":
        filters['city'] = filter_city
    if filter_category != "All":
        filters['category'] = filter_category
    if filter_fraud == "Fraud Only":
        filters['is_fraud'] = 1
    elif filter_fraud == "Legitimate Only":
        filters['is_fraud'] = 0

    df = get_all_transactions(limit=num_records, filters=filters if filters else None)

    if not df.empty:
        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Showing", f"{len(df)} records")
        m2.metric("Fraud in View", f"{df['is_fraud'].sum()}")
        m3.metric("Avg Amount", f"₹{df['transaction_amount_inr'].mean():,.0f}")
        m4.metric("Total Amount", f"₹{df['transaction_amount_inr'].sum():,.0f}")

        # Display columns
        display_cols = [
            'transaction_id', 'customer_id', 'transaction_amount_inr',
            'merchant_category', 'transaction_datetime', 'transaction_city',
            'card_type', 'channel', 'distance_from_home_km',
            'international_transaction', 'is_fraud'
        ]
        display_df = df[[c for c in display_cols if c in df.columns]].copy()

        # Color-code fraud
        def highlight_fraud(row):
            if row['is_fraud'] == 1:
                return ['background-color: rgba(255, 71, 87, 0.15)'] * len(row)
            return [''] * len(row)

        styled_df = display_df.style.apply(highlight_fraud, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=500)

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download as CSV", csv, "transactions.csv",
            "text/csv", use_container_width=True
        )
    else:
        st.info("No transactions found with the selected filters.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ALERTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🚨 Alerts":

    st.markdown("""
    <div class="main-header">
        <h1>🚨 Alert Center</h1>
        <p>Monitor and manage suspicious transaction alerts</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔴 Active Alerts", "✅ Resolved Alerts"])

    with tab1:
        active_alerts = get_alerts(resolved=False, limit=50)
        if active_alerts.empty:
            st.markdown("""
            <div style="text-align: center; padding: 3rem;">
                <h2 style="color: #2ed573;">✅ All Clear!</h2>
                <p style="color: #a0aec0;">No active alerts at the moment. The system is running smoothly.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"⚠️ {len(active_alerts)} active alert(s) requiring attention")

            for _, alert in active_alerts.iterrows():
                with st.container():
                    c1, c2, c3 = st.columns([5, 1, 1])
                    with c1:
                        severity_emoji = "🔴" if alert['severity'] == 'Critical' else "🟠"
                        st.markdown(f"""
                        **{severity_emoji} {alert['severity']}** — `{alert['transaction_id']}`

                        {alert['message']}

                        *Risk Score: {alert['risk_score']}% | Created: {alert['created_at'][:19]}*
                        """)
                    with c2:
                        st.metric("Risk", f"{alert['risk_score']}%")
                    with c3:
                        if st.button("✅ Resolve", key=f"resolve_{alert['id']}"):
                            resolve_alert(alert['id'])
                            st.rerun()
                    st.divider()

    with tab2:
        resolved_alerts = get_alerts(resolved=True, limit=50)
        if resolved_alerts.empty:
            st.info("No resolved alerts yet.")
        else:
            st.dataframe(resolved_alerts, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Model Performance":

    st.markdown("""
    <div class="main-header">
        <h1>🤖 Model Performance</h1>
        <p>Machine learning model metrics, evaluation, and feature importance</p>
    </div>
    """, unsafe_allow_html=True)

    metrics_path = os.path.join(ROOT_DIR, "backend", "ml", "model_metrics.json")
    importance_path = os.path.join(ROOT_DIR, "backend", "ml", "feature_importance.json")

    if not os.path.exists(metrics_path) or not os.path.exists(importance_path):
        with st.spinner("🔄 Computing model evaluation metrics (one-time setup)..."):
            try:
                from backend.model_training import main as train_pipeline
                train_pipeline()
            except Exception as e:
                st.error(f"⚠️ Model metrics not available: {e}")
                st.stop()

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    with open(importance_path, 'r') as f:
        importance = json.load(f)

    best = metrics.get('best_model', {})
    all_models = metrics.get('all_models', {})

    # Model info
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                border: 1px solid rgba(102, 126, 234, 0.2); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
        <h3 style="color: #667eea; margin: 0;">🏆 Best Model: {best.get('model_name', 'N/A')}</h3>
        <p style="color: #a0aec0; margin: 0.3rem 0 0 0;">Selected based on F1-Score performance</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{best.get('accuracy', 0)*100:.2f}%")
    m2.metric("Precision", f"{best.get('precision', 0)*100:.2f}%")
    m3.metric("Recall", f"{best.get('recall', 0)*100:.2f}%")
    m4.metric("F1-Score", f"{best.get('f1', 0)*100:.2f}%")
    m5.metric("AUC-ROC", f"{best.get('auc_roc', 0)*100:.2f}%")

    # Model comparison
    if len(all_models) > 1:
        st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)
        comparison_data = []
        for model_name, model_metrics in all_models.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{model_metrics.get('accuracy', 0)*100:.2f}%",
                'Precision': f"{model_metrics.get('precision', 0)*100:.2f}%",
                'Recall': f"{model_metrics.get('recall', 0)*100:.2f}%",
                'F1-Score': f"{model_metrics.get('f1', 0)*100:.2f}%",
                'AUC-ROC': f"{model_metrics.get('auc_roc', 0)*100:.2f}%"
            })
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        st.markdown('<div class="section-header">📋 Confusion Matrix</div>', unsafe_allow_html=True)
        cm = best.get('confusion_matrix', [[0, 0], [0, 0]])
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Legit', 'Predicted Fraud'],
            y=['Actual Legit', 'Actual Fraud'],
            text=[[str(v) for v in row] for row in cm],
            texttemplate='%{text}',
            textfont=dict(size=20, color='white'),
            colorscale=[[0, '#0a0e27'], [0.5, '#667eea'], [1, '#764ba2']],
            showscale=False
        ))
        fig.update_layout(**PLOT_LAYOUT, height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    with col2:
        st.markdown('<div class="section-header">🎯 Feature Importance (Top 15)</div>', unsafe_allow_html=True)
        if importance:
            imp_df = pd.DataFrame(importance)
            imp_df = imp_df.sort_values('importance', ascending=True)
            fig = go.Figure(go.Bar(
                x=imp_df['importance'],
                y=imp_df['feature'],
                orientation='h',
                marker=dict(
                    color=imp_df['importance'],
                    colorscale=[[0, COLORS['primary']], [0.5, COLORS['secondary']], [1, COLORS['danger']]]
                ),
                text=imp_df['importance'].apply(lambda x: f'{x:.4f}'),
                textposition='outside',
                textfont=dict(color='#e2e8f0', size=10)
            ))
            fig.update_layout(
                **PLOT_LAYOUT, height=450,
                xaxis=dict(title='Importance', gridcolor='rgba(102, 126, 234, 0.1)'),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    # Model Details
    st.markdown('<div class="section-header">📑 Model Details</div>', unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("""
        **Training Configuration:**
        - Algorithm: Random Forest / XGBoost
        - Class Imbalance: SMOTE Oversampling
        - Train/Test Split: 80/20 (stratified)
        - Feature Scaling: StandardScaler
        - Categorical Encoding: LabelEncoder
        """)
    with dc2:
        st.markdown("""
        **Feature Engineering:**
        - `is_night_transaction` — Transactions between 10 PM–5 AM
        - `is_high_amount` — Transactions > ₹10,000
        - `amount_deviation` — Amount / Customer average
        - `velocity_score` — Transaction frequency × amount ratio
        - `failed_attempt_risk` — Failed attempts × night flag
        - `account_risk_score` — 1 / (account age + 1)
        """)

    # Recent Predictions
    st.markdown('<div class="section-header">🔄 Recent Predictions</div>', unsafe_allow_html=True)
    recent = get_recent_predictions(limit=10)
    if not recent.empty:
        st.dataframe(recent, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions made yet. Use the Predict Transaction page to make predictions.")
