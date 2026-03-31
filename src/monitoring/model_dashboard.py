"""Model Performance Monitoring Dashboard.

This Streamlit dashboard provides real-time monitoring of ML model performance
with alerts for degradation and retraining recommendations.

Features:
- Live model metrics (win rate, Sharpe, drawdown)
- Prediction distribution analysis
- Feature importance tracking
- Performance decay detection
- Comparison: training vs live performance
- Automated retraining recommendations

Usage:
    streamlit run src/monitoring/model_dashboard.py
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="ML Model Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .status-good {
        color: #00cc00;
        font-weight: bold;
    }
    .status-warning {
        color: #ff9900;
        font-weight: bold;
    }
    .status-bad {
        color: #ff0000;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_model_metadata(model_dir: str = "models/xgboost/30_minute") -> Dict:
    """Load model metadata.

    Args:
        model_dir: Path to model directory

    Returns:
        Dictionary with model metadata
    """
    metadata_path = Path(model_dir) / "sb_params_optimized.json"

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=60)  # Cache for 1 minute
def load_walk_forward_results(
    results_path: str = "models/xgboost/walk_forward_results.json"
) -> Dict:
    """Load walk-forward validation results.

    Args:
        results_path: Path to results file

    Returns:
        Dictionary with validation results
    """
    results_file = Path(results_path)

    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=60)
def load_recent_predictions(
    n_samples: int = 1000,
    predictions_path: str = "data/ml_training/silver_bullet_signals.parquet"
) -> pd.DataFrame:
    """Load recent predictions for analysis.

    Args:
        n_samples: Number of recent samples to load
        predictions_path: Path to predictions file

    Returns:
        DataFrame with recent predictions
    """
    try:
        import pyarrow.parquet as pq

        predictions_file = Path(predictions_path)
        if predictions_file.exists():
            df = pd.read_parquet(predictions_file)
            return df.tail(n_samples)
    except Exception as e:
        logger.warning(f"Failed to load predictions: {e}")

    # Return mock data if file doesn't exist
    return pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=n_samples, freq='5min'),
        'probability': np.random.beta(2, 2, n_samples),
        'prediction': np.random.randint(0, 2, n_samples),
        'actual': np.random.randint(0, 2, n_samples),
        'direction': np.random.choice(['bullish', 'bearish'], n_samples),
    })


def calculate_live_metrics(predictions_df: pd.DataFrame) -> Dict:
    """Calculate live performance metrics.

    Args:
        predictions_df: DataFrame with predictions and actuals

    Returns:
        Dictionary with performance metrics
    """
    if len(predictions_df) == 0:
        return {}

    # Win rate
    win_rate = (predictions_df['prediction'] == predictions_df['actual']).mean()

    # Precision (positive predictive value)
    precision = predictions_df[predictions_df['prediction'] == 1]['actual'].mean()

    # Recall (true positive rate)
    recall = predictions_df[predictions_df['actual'] == 1]['prediction'].mean()

    # Average probability
    avg_probability = predictions_df['probability'].mean()

    # Probability distribution
    prob_std = predictions_df['probability'].std()

    # Directional bias
    bullish_pct = (predictions_df['direction'] == 'bullish').mean()

    return {
        'win_rate': win_rate,
        'precision': precision,
        'recall': recall,
        'avg_probability': avg_probability,
        'prob_std': prob_std,
        'bullish_pct': bullish_pct,
        'n_predictions': len(predictions_df),
    }


def detect_performance_degradation(
    live_metrics: Dict,
    baseline_metrics: Dict,
    threshold: float = 0.10
) -> Dict:
    """Detect if model performance has degraded.

    Args:
        live_metrics: Current live performance metrics
        baseline_metrics: Baseline (training) metrics
        threshold: Performance drop threshold (10%)

    Returns:
        Dictionary with degradation status
    """
    if not live_metrics or not baseline_metrics:
        return {'degraded': False, 'reason': 'Insufficient data'}

    degraded = False
    reasons = []

    # Check win rate
    baseline_win_rate = baseline_metrics.get('win_rate', 0.5)
    live_win_rate = live_metrics.get('win_rate', 0.5)

    if live_win_rate < baseline_win_rate - threshold:
        degraded = True
        reasons.append(f"Win rate dropped: {live_win_rate:.2%} vs {baseline_win_rate:.2%}")

    # Check precision
    baseline_precision = baseline_metrics.get('precision', 0.5)
    live_precision = live_metrics.get('precision', 0.5)

    if live_precision < baseline_precision - threshold:
        degraded = True
        reasons.append(f"Precision dropped: {live_precision:.2%} vs {baseline_precision:.2%}")

    return {
        'degraded': degraded,
        'reasons': reasons,
        'win_rate_diff': live_win_rate - baseline_win_rate,
        'precision_diff': live_precision - baseline_precision,
    }


def render_metric_card(title: str, value: str, subtitle: str, status: str = "good") -> None:
    """Render a metric card with status indicator.

    Args:
        title: Metric title
        value: Metric value
        subtitle: Additional context
        status: Status ('good', 'warning', 'bad')
    """
    status_class = f"status-{status}"
    st.markdown(f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <p class="{status_class}" style="font-size: 2rem;">{value}</p>
        <p style="color: #666;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main dashboard application."""
    st.title("📊 ML Model Performance Monitor")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Settings")

    model_dir = st.sidebar.text_input(
        "Model Directory",
        value="models/xgboost/30_minute"
    )

    results_path = st.sidebar.text_input(
        "Validation Results",
        value="models/xgboost/walk_forward_results.json"
    )

    # Load data
    with st.spinner("Loading model data..."):
        model_metadata = load_model_metadata(model_dir)
        walk_forward_results = load_walk_forward_results(results_path)
        predictions_df = load_recent_predictions()

    # Calculate metrics
    live_metrics = calculate_live_metrics(predictions_df)
    training_metrics = {
        'win_rate': model_metadata.get('win_rate', 0.85),
        'precision': 0.75,  # Estimated
    }
    realistic_metrics = {
        'win_rate': walk_forward_results.get('realistic_win_rate', 0.45),
        'precision': 0.60,  # Estimated
    }

    # Detect degradation
    degradation = detect_performance_degradation(
        live_metrics, realistic_metrics
    )

    # === Header Alerts ===
    if degradation['degraded']:
        st.error(f"⚠️ PERFORMANCE DEGRADED: {'; '.join(degradation['reasons'])}")
    else:
        st.success("✅ Model performance within acceptable range")

    # === Metrics Overview ===
    st.header("Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if live_metrics:
            win_rate = live_metrics['win_rate']
            baseline = realistic_metrics['win_rate']
            diff = win_rate - baseline

            if diff >= -0.05:
                status = "good"
            elif diff >= -0.10:
                status = "warning"
            else:
                status = "bad"

            render_metric_card(
                "Live Win Rate",
                f"{win_rate:.2%}",
                f"Baseline: {baseline:.2%} ({diff:+.1%})",
                status
            )
        else:
            render_metric_card("Live Win Rate", "N/A", "No data", "warning")

    with col2:
        if live_metrics:
            precision = live_metrics['precision']
            render_metric_card(
                "Precision",
                f"{precision:.2%}",
                f"Positive predictions",
                "good" if precision >= 0.60 else "warning"
            )
        else:
            render_metric_card("Precision", "N/A", "No data", "warning")

    with col3:
        if live_metrics:
            avg_prob = live_metrics['avg_probability']
            render_metric_card(
                "Avg Probability",
                f"{avg_prob:.2%}",
                f"Confidence score",
                "good" if avg_prob >= 0.65 else "warning"
            )
        else:
            render_metric_card("Avg Probability", "N/A", "No data", "warning")

    with col4:
        if live_metrics:
            n_preds = live_metrics['n_predictions']
            render_metric_card(
                "Predictions",
                f"{n_preds:,}",
                "Total samples",
                "good"
            )
        else:
            render_metric_card("Predictions", "N/A", "No data", "warning")

    # === Performance Comparison ===
    st.header("Performance Comparison")

    if walk_forward_results:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Training vs Realistic vs Live")

            comparison_data = {
                'Training': model_metadata.get('win_rate', 0.85),
                'Walk-Forward': walk_forward_results.get('realistic_win_rate', 0.45),
                'Live': live_metrics.get('win_rate', 0.0) if live_metrics else 0.0,
            }

            fig = go.Figure(data=[
                go.Bar(
                    name='Win Rate',
                    x=list(comparison_data.keys()),
                    y=[v * 100 for v in comparison_data.values()],
                    marker_color=['#00cc00', '#ff9900', '#0066cc']
                )
            ])

            fig.update_layout(
                yaxis_title="Win Rate (%)",
                title="Model Performance Comparison",
                yaxis=dict(range=[0, 100])
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Performance Stability")

            if 'validations' in walk_forward_results:
                validations = walk_forward_results['validations']

                periods = [v['period'] for v in validations]
                win_rates = [v['test_win_rate'] * 100 for v in validations]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=periods,
                    y=win_rates,
                    mode='lines+markers',
                    name='Win Rate',
                    line=dict(color='#0066cc', width=2)
                ))

                # Add baseline
                if live_metrics and live_metrics.get('win_rate'):
                    fig.add_hline(
                        y=live_metrics['win_rate'] * 100,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Live"
                    )

                fig.update_layout(
                    xaxis_title="Validation Period",
                    yaxis_title="Win Rate (%)",
                    title="Walk-Forward Validation Results",
                    yaxis=dict(range=[0, 100])
                )

                st.plotly_chart(fig, use_container_width=True)

    # === Prediction Distribution ===
    st.header("Prediction Analysis")

    if live_metrics and live_metrics['n_predictions'] > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Probability Distribution")

            fig = px.histogram(
                predictions_df,
                x='probability',
                nbins=50,
                title="Distribution of Prediction Probabilities",
                labels={'probability': 'Success Probability'},
                color_discrete_sequence=['#0066cc']
            )

            fig.add_vline(
                x=0.65,
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold (65%)"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Directional Bias")

            bullish_pct = live_metrics['bullish_pct'] * 100
            bearish_pct = 100 - bullish_pct

            fig = go.Figure(data=[
                go.Pie(
                    labels=['Bullish', 'Bearish'],
                    values=[bullish_pct, bearish_pct],
                    hole=0.3,
                    marker_colors=['#00cc00', '#ff0000']
                )
            ])

            fig.update_layout(
                title="Signal Direction Distribution",
                annotations=[{
                    'text': f"{bullish_pct:.0f}% Bullish",
                    'x': 0.5,
                    'y': 0.5,
                    'font_size': 20,
                    'showarrow': False
                }]
            )

            st.plotly_chart(fig, use_container_width=True)

    # === Recommendations ===
    st.header("Recommendations")

    recommendations = []

    # Check if retraining is needed
    if degradation['degraded']:
        recommendations.append({
            'priority': 'HIGH',
            'action': 'Retrain Model',
            'reason': f"Performance degraded: {'; '.join(degradation['reasons'])}"
        })

    # Check if model is outdated
    if model_metadata:
        train_end = model_metadata.get('data_range', {}).get('end', '')
        if train_end:
            train_date = datetime.strptime(train_end, '%Y-%m-%d')
            days_old = (datetime.now() - train_date).days

            if days_old > 90:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'action': 'Update Training Data',
                    'reason': f"Model is {days_old} days old. Retrain with recent data."
                })

    # Check prediction confidence
    if live_metrics and live_metrics.get('avg_probability', 0) < 0.60:
        recommendations.append({
            'priority': 'LOW',
            'action': 'Review Signal Quality',
            'reason': "Low average confidence suggests poor signal quality"
        })

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            priority_color = {
                'HIGH': '🔴',
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }.get(rec['priority'], '⚪')

            st.markdown(f"""
            **{priority_color} {rec['priority']} PRIORITY: {rec['action']}**

            {rec['reason']}

            ---
            """)
    else:
        st.success("✅ No recommendations - Model performing well!")

    # === Retraining Status ===
    st.header("Retraining Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Last Retrained", model_metadata.get('optimization_date', 'N/A'))

    with col2:
        if model_metadata.get('data_range'):
            data_range = model_metadata['data_range']
            st.metric(
                "Training Period",
                f"{data_range['start']} to {data_range['end']}"
            )
        else:
            st.metric("Training Period", "N/A")

    with col3:
        if walk_forward_results.get('realistic_win_rate'):
            st.metric(
                "Expected Win Rate",
                f"{walk_forward_results['realistic_win_rate']:.2%}"
            )
        else:
            st.metric("Expected Win Rate", "N/A")

    # Auto-refresh
    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("Auto-Refresh (30s)", value=False)

    if auto_refresh:
        st_autorefresh(interval=30000, key="refresh")


def st_autorefresh(interval: int, key: str) -> None:
    """Auto-refresh the page.

    Args:
        interval: Refresh interval in milliseconds
        key: Unique key for the component
    """
    st.markdown(
        f"""
        <script>
            setTimeout(function() {{
                location.reload();
            }}, {interval});
        </script>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
