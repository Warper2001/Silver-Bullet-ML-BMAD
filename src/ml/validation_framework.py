"""Validation framework for 1-minute trading system.

Provides tools for:
- Temporal split validation (prevent data leakage)
- Data leakage detection
- Performance validation with realistic metrics
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class TemporalSplitValidator:
    """Validate temporal splits to prevent data leakage.

    Ensures strict temporal separation between training and validation data.
    """

    def __init__(
        self,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None
    ):
        """Initialize temporal split validator.

        Args:
            train_start: Training start date (YYYY-MM-DD)
            train_end: Training end date (YYYY-MM-DD)
            val_start: Validation start date (YYYY-MM-DD)
            val_end: Validation end date (YYYY-MM-DD)
            test_start: Optional test start date (YYYY-MM-DD)
            test_end: Optional test end date (YYYY-MM-DD)
        """
        self.train_start = pd.to_datetime(train_start)
        self.train_end = pd.to_datetime(train_end)
        self.val_start = pd.to_datetime(val_start)
        self.val_end = pd.to_datetime(val_end)

        if test_start and test_end:
            self.test_start = pd.to_datetime(test_start)
            self.test_end = pd.to_datetime(test_end)
        else:
            self.test_start = None
            self.test_end = None

        # Validate temporal ordering
        self._validate_temporal_ordering()

    def _validate_temporal_ordering(self):
        """Ensure proper temporal ordering of splits."""
        errors = []

        if self.train_start >= self.train_end:
            errors.append("Train start >= train end")

        if self.train_end >= self.val_start:
            errors.append(f"Train end ({self.train_end}) >= val start ({self.val_start}) - DATA LEAKAGE RISK")

        if self.val_start >= self.val_end:
            errors.append("Val start >= val end")

        if self.test_start and self.test_end:
            if self.val_end >= self.test_start:
                errors.append(f"Val end ({self.val_end}) >= test start ({self.test_start}) - DATA LEAKAGE RISK")
            if self.test_start >= self.test_end:
                errors.append("Test start >= test end")

        if errors:
            raise ValueError(f"Temporal split validation failed:\n" + "\n".join(f"  ❌ {e}" for e in errors))

        logger.info("✅ Temporal ordering validated")

    def validate_no_leakage(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: Optional[pd.DataFrame] = None,
        timestamp_col: str = 'timestamp'
    ) -> Dict:
        """Check for temporal overlap between datasets.

        Args:
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Optional test dataset
            timestamp_col: Name of timestamp column

        Returns:
            Dictionary with validation results
        """
        results = {
            'timestamp_col': timestamp_col,
            'overlaps': [],
            'gaps': [],
            'passed': True
        }

        # Get actual date ranges from data
        if timestamp_col in train_data.columns:
            train_actual_start = pd.to_datetime(train_data[timestamp_col]).min()
            train_actual_end = pd.to_datetime(train_data[timestamp_col]).max()
        else:
            train_actual_start = train_data.index.min()
            train_actual_end = train_data.index.max()

        if timestamp_col in val_data.columns:
            val_actual_start = pd.to_datetime(val_data[timestamp_col]).min()
            val_actual_end = pd.to_datetime(val_data[timestamp_col]).max()
        else:
            val_actual_start = val_data.index.min()
            val_actual_end = val_data.index.max()

        # Check for overlap between train and val
        if train_actual_end >= val_actual_start:
            results['overlaps'].append({
                'type': 'train_val_overlap',
                'train_end': str(train_actual_end),
                'val_start': str(val_actual_start),
                'overlap_days': (train_actual_end - val_actual_start).days
            })
            results['passed'] = False
            logger.error(f"❌ OVERLAP: Train data extends {train_actual_end} into validation {val_actual_start}")

        # Check for gap (acceptable but worth noting)
        gap_days = (val_actual_start - train_actual_end).days
        if gap_days > 1:
            results['gaps'].append({
                'type': 'train_val_gap',
                'gap_days': gap_days
            })
            logger.info(f"ℹ️  GAP: {gap_days} days between train and val")

        # Check test data if provided
        if test_data is not None:
            if timestamp_col in test_data.columns:
                test_actual_start = pd.to_datetime(test_data[timestamp_col]).min()
                test_actual_end = pd.to_datetime(test_data[timestamp_col]).max()
            else:
                test_actual_start = test_data.index.min()
                test_actual_end = test_data.index.max()

            if val_actual_end >= test_actual_start:
                results['overlaps'].append({
                    'type': 'val_test_overlap',
                    'val_end': str(val_actual_end),
                    'test_start': str(test_actual_start),
                    'overlap_days': (val_actual_end - test_actual_start).days
                })
                results['passed'] = False
                logger.error(f"❌ OVERLAP: Val data extends {val_actual_end} into test {test_actual_start}")

        if results['passed']:
            logger.info("✅ No temporal overlaps detected")

        return results

    def summary(self) -> str:
        """Generate summary of temporal splits."""
        lines = [
            "Temporal Split Configuration:",
            f"  Train: {self.train_start.date()} to {self.train_end.date()}",
            f"  Validation: {self.val_start.date()} to {self.val_end.date()}"
        ]

        if self.test_start:
            lines.append(f"  Test: {self.test_start.date()} to {self.test_end.date()}")

        train_days = (self.train_end - self.train_start).days
        val_days = (self.val_end - self.val_start).days

        lines.extend([
            f"\nDuration:",
            f"  Train: {train_days} days",
            f"  Validation: {val_days} days",
            f"  Ratio: {val_days/train_days:.1%}"
        ])

        return "\n".join(lines)


class DataLeakageDetector:
    """Detect various types of data leakage in ML pipelines.

    Checks for:
    - Temporal leakage (future data in training)
    - Feature leakage (future information in features)
    - Target leakage (labels contaminated by future)
    """

    def __init__(self, timestamp_col: str = 'timestamp'):
        """Initialize data leakage detector.

        Args:
            timestamp_col: Name of timestamp column
        """
        self.timestamp_col = timestamp_col

    def detect_temporal_leakage(
        self,
        df: pd.DataFrame,
        feature_windows: Dict[str, int]
    ) -> Dict:
        """Check if features use future data (look-ahead bias).

        Args:
            df: Dataset with features and timestamps
            feature_windows: Dictionary mapping feature names to their lookback windows

        Returns:
            Dictionary with leakage detection results
        """
        results = {
            'leakage_detected': False,
            'leaky_features': [],
            'safe_features': []
        }

        if self.timestamp_col not in df.columns:
            logger.warning(f"Timestamp column '{self.timestamp_col}' not found")
            return results

        df = df.sort_values(self.timestamp_col).reset_index(drop=True)

        for feature, window in feature_windows.items():
            if feature not in df.columns:
                continue

            # Check if feature has NaN values at start (expected for rolling windows)
            initial_nans = df[feature].iloc[:window].isna().sum()

            if initial_nans == 0 and window > 0:
                # Feature has no NaNs at start but should - potential leakage
                results['leakage_detected'] = True
                results['leaky_features'].append({
                    'feature': feature,
                    'reason': f'Window={window}, but no initial NaNs (uses future data?)',
                    'initial_nans': initial_nans
                })
                logger.warning(f"⚠️  LEAKAGE: {feature} may use future data")
            else:
                results['safe_features'].append(feature)

        if not results['leakage_detected']:
            logger.info("✅ No temporal leakage detected")

        return results

    def detect_target_leakage(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str]
    ) -> Dict:
        """Check if target is leaked into features.

        Args:
            df: Dataset
            target_col: Target column name
            feature_cols: List of feature column names

        Returns:
            Dictionary with leakage detection results
        """
        results = {
            'leakage_detected': False,
            'correlations': []
        }

        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found")
            return results

        for feature in feature_cols:
            if feature not in df.columns:
                continue

            # Calculate correlation
            corr = df[feature].corr(df[target_col])

            if abs(corr) > 0.95:
                # Extremely high correlation - potential leakage
                results['leakage_detected'] = True
                results['correlations'].append({
                    'feature': feature,
                    'correlation': corr,
                    'risk': 'HIGH'
                })
                logger.error(f"❌ LEAKAGE: {feature} has {corr:.3f} correlation with target")
            elif abs(corr) > 0.8:
                results['correlations'].append({
                    'feature': feature,
                    'correlation': corr,
                    'risk': 'MEDIUM'
                })
                logger.warning(f"⚠️  SUSPICIOUS: {feature} has {corr:.3f} correlation with target")

        if not results['leakage_detected']:
            logger.info("✅ No target leakage detected")

        return results

    def generate_audit_report(
        self,
        df: pd.DataFrame,
        feature_windows: Dict[str, int],
        target_col: str,
        output_path: Optional[Path] = None
    ) -> str:
        """Generate comprehensive audit report.

        Args:
            df: Dataset to audit
            feature_windows: Feature lookback windows
            target_col: Target column name
            output_path: Optional path to save report

        Returns:
            Report text
        """
        lines = [
            "# Data Leakage Audit Report",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Dataset Shape:** {df.shape}",
            f"**Date Range:** {df.index.min()} to {df.index.max()}",
            "\n---\n"
        ]

        # Temporal leakage check
        lines.append("## Temporal Leakage Check\n")
        feature_cols = list(feature_windows.keys())
        temporal_results = self.detect_temporal_leakage(df, feature_windows)

        if temporal_results['leakage_detected']:
            lines.append("❌ **TEMPORAL LEAKAGE DETECTED**\n")
            for leak in temporal_results['leaky_features']:
                lines.append(f"- **{leak['feature']}**: {leak['reason']}")
        else:
            lines.append("✅ No temporal leakage detected\n")

        # Target leakage check
        lines.append("\n## Target Leakage Check\n")
        target_results = self.detect_target_leakage(df, target_col, feature_cols)

        if target_results['leakage_detected']:
            lines.append("❌ **TARGET LEAKAGE DETECTED**\n")
            for corr in target_results['correlations']:
                if corr['risk'] == 'HIGH':
                    lines.append(f"- **{corr['feature']}**: r={corr['correlation']:.3f} ({corr['risk']} risk)")
        else:
            lines.append("✅ No target leakage detected\n")

        report = "\n".join(lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"✅ Audit report saved to: {output_path}")

        return report


class PerformanceValidator:
    """Validate model performance with realistic metrics.

    Includes transaction costs, risk-adjusted returns, and robustness checks.
    """

    def __init__(
        self,
        commission_per_contract: float = 2.50,
        slippage_ticks: float = 0.50,
        contracts_per_trade: int = 5
    ):
        """Initialize performance validator.

        Args:
            commission_per_contract: Commission cost per contract
            slippage_ticks: Slippage in ticks
            contracts_per_trade: Number of contracts per trade
        """
        self.commission_per_contract = commission_per_contract
        self.slippage_ticks = slippage_ticks
        self.contracts_per_trade = contracts_per_trade

        # Realistic performance targets
        self.targets = {
            'win_rate': (45.0, 55.0),  # 45-55%
            'trades_per_day': (5, 25),  # 5-25 trades/day
            'expectation_per_trade': (20.0, None),  # ≥$20
            'sharpe_ratio': (0.6, None),  # ≥0.6
            'profit_factor': (1.3, None),  # ≥1.3
            'max_drawdown': (None, 1000.0)  # ≤$1000
        }

    def calculate_metrics(
        self,
        trades_df: pd.DataFrame,
        include_costs: bool = True
    ) -> Dict:
        """Calculate performance metrics from trade results.

        Args:
            trades_df: DataFrame with trade results (must have 'pnl' column)
            include_costs: Whether to include transaction costs

        Returns:
            Dictionary with performance metrics
        """
        if len(trades_df) == 0:
            return {
                'trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'trades_per_day': 0.0
            }

        # Add transaction costs if not already included
        if include_costs:
            cost_per_trade = (
                self.commission_per_contract * self.contracts_per_trade +
                self.slippage_ticks * 0.25 * self.contracts_per_trade
            )
            # If costs not already in P&L, subtract them
            # (assuming input P&L already includes costs)

        # Calculate metrics
        trades_df = trades_df.copy()
        trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date

        win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()

        if 'date' in trades_df.columns:
            trades_per_day = trades_df.groupby('date').size().mean()
        else:
            trades_per_day = 0.0

        winners = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        losers = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = winners / losers if losers > 0 else 0.0

        returns_std = trades_df['pnl'].std()
        sharpe_ratio = (avg_pnl / returns_std) if returns_std > 0 else 0.0

        cumulative_returns = trades_df['pnl'].cumsum()
        max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()

        return {
            'trades': len(trades_df),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'expectation_per_trade': avg_pnl,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'trades_per_day': trades_per_day
        }

    def validate_against_targets(
        self,
        metrics: Dict
    ) -> Dict:
        """Check if metrics meet realistic performance targets.

        Args:
            metrics: Dictionary of performance metrics

        Returns:
            Dictionary with validation results
        """
        results = {
            'passed': [],
            'failed': [],
            'overall_passed': True
        }

        for metric_name, (min_val, max_val) in self.targets.items():
            if metric_name not in metrics:
                continue

            actual = metrics[metric_name]

            # Check against targets
            passed = True
            if min_val is not None and actual < min_val:
                passed = False
            if max_val is not None and actual > max_val:
                passed = False

            status = {
                'metric': metric_name,
                'actual': actual,
                'target_min': min_val,
                'target_max': max_val,
                'passed': passed
            }

            if passed:
                results['passed'].append(status)
            else:
                results['failed'].append(status)
                results['overall_passed'] = False

        return results

    def check_red_flags(
        self,
        metrics: Dict
    ) -> List[str]:
        """Check for unrealistic performance (indicates overfitting).

        Args:
            metrics: Dictionary of performance metrics

        Returns:
            List of red flag warnings
        """
        red_flags = []

        # Unrealistic win rate
        if metrics['win_rate'] > 70:
            red_flags.append(f"❌ Win rate {metrics['win_rate']:.1f}% > 70% - UNREALISTIC for 1-min data")

        # Unrealistic Sharpe ratio
        if metrics['sharpe_ratio'] > 3.0:
            red_flags.append(f"❌ Sharpe ratio {metrics['sharpe_ratio']:.1f} > 3.0 - INDICATES OVERFITTING")

        # Too many trades per day
        if metrics['trades_per_day'] > 30:
            red_flags.append(f"❌ Trades/day {metrics['trades_per_day']:.1f} > 30 - TOO FREQUENT, likely noise")

        # No trades
        if metrics['trades'] == 0:
            red_flags.append("❌ Zero trades generated - model too conservative or broken")

        # Very few trades
        if metrics['trades'] < 10:
            red_flags.append(f"⚠️  Only {metrics['trades']} trades - INSUFFICIENT DATA")

        return red_flags

    def generate_validation_report(
        self,
        trades_df: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> str:
        """Generate comprehensive validation report.

        Args:
            trades_df: DataFrame with trade results
            output_path: Optional path to save report

        Returns:
            Report text
        """
        metrics = self.calculate_metrics(trades_df)
        target_validation = self.validate_against_targets(metrics)
        red_flags = self.check_red_flags(metrics)

        lines = [
            "# Performance Validation Report",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Trades:** {metrics['trades']}",
            "\n---\n"
        ]

        # Performance Metrics
        lines.append("## Performance Metrics\n")
        lines.append(f"| Metric | Value | Target | Status |")
        lines.append(f"|--------|-------|--------|--------|")

        for metric, (min_val, max_val) in self.targets.items():
            if metric not in metrics:
                continue

            actual = metrics[metric]
            target_str = f"{min_val}-{max_val}" if max_val else f"≥{min_val}"

            # Format value
            if metric in ['win_rate']:
                value_str = f"{actual:.1f}%"
            elif metric in ['total_pnl', 'avg_pnl', 'expectation_per_trade', 'max_drawdown']:
                value_str = f"${actual:,.2f}"
            elif metric in ['trades_per_day']:
                value_str = f"{actual:.1f}"
            else:
                value_str = f"{actual:.2f}"

            # Check status
            passed = any(p['metric'] == metric for p in target_validation['passed'])
            status = "✅" if passed else "❌"

            lines.append(f"| {metric} | {value_str} | {target_str} | {status} |")

        # Red Flags
        if red_flags:
            lines.append("\n## ⚠️ RED FLAGS\n")
            for flag in red_flags:
                lines.append(f"{flag}")
        else:
            lines.append("\n## ✅ No Red Flags Detected\n")

        # Overall Assessment
        lines.append("\n## Overall Assessment\n")
        if target_validation['overall_passed'] and not red_flags:
            lines.append("✅ **PASS:** All targets met, no red flags")
        elif target_validation['overall_passed']:
            lines.append("⚠️  **CAUTION:** Targets met but red flags detected")
        else:
            lines.append("❌ **FAIL:** Targets not met")

        report = "\n".join(lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"✅ Validation report saved to: {output_path}")

        return report


def lock_validation_data(
    df: pd.DataFrame,
    output_path: Path,
    checksum_algorithm: str = 'sha256'
) -> str:
    """Lock validation data with checksum to prevent modifications.

    Args:
        df: Validation dataset
        output_path: Path to save locked data
        checksum_algorithm: Hash algorithm to use (default: sha256)

    Returns:
        Checksum hash
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save data
    df.to_csv(output_path, index=False)

    # Calculate checksum
    hash_func = getattr(hashlib, checksum_algorithm)()
    with open(output_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    checksum = hash_func.hexdigest()

    # Save checksum
    checksum_path = output_path.with_suffix(f'.{checksum_algorithm}')
    with open(checksum_path, 'w') as f:
        f.write(checksum)

    logger.info(f"✅ Validation data locked: {output_path}")
    logger.info(f"✅ Checksum ({checksum_algorithm}): {checksum}")

    return checksum
