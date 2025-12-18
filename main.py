#!/usr/bin/env python3
"""
Insider Trading Anomaly Detection System - Main Entry Point

This is the main entry point for the Insider Trading Anomaly Detection System.
It provides both GUI and CLI interfaces for running the analysis.

Usage:
    # GUI mode (default)
    python main.py

    # CLI mode with sample data
    python main.py --cli --generate-sample

    # CLI mode with custom data
    python main.py --cli --input data/raw/transactions.csv --output reports/

Examples:
    python main.py
    python main.py --cli --generate-sample --window 14
    python main.py --cli -i data/raw/my_data.csv -o reports/
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def run_gui():
    """Run the GUI application."""
    from src.gui_main import run_gui as _run_gui
    _run_gui()


def run_cli(args):
    """Run the CLI analysis."""
    from src.data_preprocessor import DataPreprocessor
    from src.feature_engineer import FeatureEngineer
    from src.model_factory import ModelFactory
    from src.anomaly_scorer import AnomalyScorer
    from src.excel_reporter import ExcelReporter
    from src import constants

    print("=" * 60)
    print("Insider Trading Anomaly Detection System")
    print("=" * 60)

    # Generate sample data if requested
    if args.generate_sample:
        from src.sample_data import generate_sample_data
        input_path = generate_sample_data()
        print(f"\nGenerated sample data: {input_path}")
    else:
        input_path = args.input

    if not input_path or not Path(input_path).exists():
        print("Error: No input file specified or file not found.")
        print("Use --generate-sample to create sample data or --input to specify a file.")
        return 1

    # Step 1: Load and preprocess data
    print("\n[1/5] Loading data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_raw_data(input_path)
    stats = preprocessor.get_summary_stats()
    print(f"  Loaded {stats['total_transactions']:,} transactions")
    print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(f"  Unique executives: {stats['executives']['unique_count']}")

    # Step 2: Feature engineering
    print("\n[2/5] Engineering features...")
    window_days = args.window
    feature_engineer = FeatureEngineer(df, window_days)
    feature_engineer.aggregate_by_window()
    feature_matrix = feature_engineer.create_feature_matrix()
    print(f"  Created {feature_matrix.shape[0]} windows × {feature_matrix.shape[1]} features")

    # Step 3: Train models
    print("\n[3/5] Training models...")
    model_factory = ModelFactory()
    models_to_train = ['IsolationForest', 'DBSCAN', 'LocalOutlierFactor']

    if args.all_models:
        models_to_train = constants.SUPPORTED_MODELS

    training_metrics = {}
    X = feature_matrix.values
    feature_names = list(feature_matrix.columns)

    for model_type in models_to_train:
        try:
            model_name = f"{model_type}_cli"
            params = constants.DEFAULT_HYPERPARAMETERS.get(model_type, {})
            model_factory.create_model(model_type, params, model_name)
            _, metrics = model_factory.train_model(model_name, X, feature_names)
            training_metrics[model_name] = metrics
            print(f"  {model_type}: {metrics['anomalies_detected']} anomalies "
                  f"({metrics['training_time']:.2f}s)")
        except Exception as e:
            print(f"  {model_type}: Error - {e}")

    # Step 4: Detect anomalies
    print("\n[4/5] Detecting anomalies...")
    scorer = AnomalyScorer()

    # Use first model for primary detection
    primary_model = list(model_factory.models.keys())[0]
    anomalies_df = model_factory.detect_historical_anomalies(
        primary_model, X, feature_matrix.index, percentile=args.percentile
    )

    # Add risk levels
    anomalies_df['risk_level'] = anomalies_df['anomaly_score'].apply(
        lambda s: scorer.get_risk_level(s)
    )

    n_anomalies = (anomalies_df['is_anomaly'] == 1).sum()
    print(f"  Detected {n_anomalies} anomalous windows")

    # Show top anomalies
    top_anomalies = anomalies_df[anomalies_df['is_anomaly'] == 1].head(5)
    if len(top_anomalies) > 0:
        print("\n  Top anomalies:")
        for _, row in top_anomalies.iterrows():
            print(f"    {row['window_date'].strftime('%Y-%m-%d')}: "
                  f"score={row['anomaly_score']:.3f}, risk={row['risk_level']}")

    # Step 5: Generate report
    print("\n[5/5] Generating report...")
    output_dir = Path(args.output) if args.output else constants.REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.utils import get_timestamp
    report_path = output_dir / f"{get_timestamp()}_report.xlsx"

    reporter = ExcelReporter(report_path)

    summary_data = {
        'total_transactions': stats['total_transactions'],
        'n_windows': len(feature_matrix),
        'n_features': feature_matrix.shape[1],
        'n_models': len(training_metrics),
        'n_anomalies': n_anomalies,
        'anomaly_rate': n_anomalies / len(feature_matrix),
        'window_days': window_days,
        'date_range': stats['date_range'],
        'key_findings': [
            f"Detected {n_anomalies} anomalous trading windows",
            f"Used {len(training_metrics)} ML models for detection",
            f"Analyzed {stats['total_transactions']:,} transactions"
        ],
        'recommendations': [
            "Review top-scoring anomalies with compliance team",
            "Investigate executives involved in flagged windows",
            "Compare anomaly dates with corporate announcements"
        ]
    }

    feature_stats = feature_engineer.get_feature_statistics()

    output_path = reporter.generate_full_report(
        summary_data=summary_data,
        anomalies_df=anomalies_df,
        feature_stats=feature_stats,
        training_metrics=training_metrics,
        specs_data={
            'n_windows': len(feature_matrix),
            'window_days': window_days,
            'n_features': len(feature_names)
        }
    )

    print(f"  Report saved: {output_path}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Insider Trading Anomaly Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--cli', action='store_true',
        help='Run in command-line mode (default: GUI mode)'
    )

    parser.add_argument(
        '-i', '--input', type=str,
        help='Input CSV file path'
    )

    parser.add_argument(
        '-o', '--output', type=str,
        help='Output directory for reports'
    )

    parser.add_argument(
        '--generate-sample', action='store_true',
        help='Generate sample data for testing'
    )

    parser.add_argument(
        '--window', type=int, default=14,
        help='Aggregation window size in days (default: 14)'
    )

    parser.add_argument(
        '--percentile', type=int, default=95,
        help='Percentile threshold for anomaly detection (default: 95)'
    )

    parser.add_argument(
        '--all-models', action='store_true',
        help='Train all available models (default: top 3 only)'
    )

    args = parser.parse_args()

    if args.cli:
        return run_cli(args)
    else:
        run_gui()
        return 0


if __name__ == '__main__':
    sys.exit(main())
