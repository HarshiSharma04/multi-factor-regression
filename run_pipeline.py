#!/usr/bin/env python3
"""
Run the end-to-end pipeline:
- optionally generate sample data
- load & preprocess
- train models (static + rolling)
- compute attributions
- save JSON + plots + HTML report
"""
import argparse
from pathlib import Path
import json

from src.data_processing import (
    generate_sample_dataset,
    load_and_prepare,
)
from src.modeling import (
    fit_static_models,
    rolling_regression,
    compute_static_contributions,
    compute_rolling_contributions,
    evaluate_model,
)
from src.visualization import (
    plot_stacked_contributions,
    plot_coefficients_over_time,
    plot_actual_vs_predicted,
)
from src.report import generate_html_report
from src.utils import ensure_output_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/sample_data.csv",
                        help="Path to CSV (if exists) or where generator saved sample.")
    parser.add_argument("--generate-sample", action="store_true",
                        help="Generate a synthetic sample dataset at data/sample_data.csv")
    parser.add_argument("--model", type=str, default="ridge",
                        choices=["ols", "ridge", "lasso"], help="Model type for static fit")
    parser.add_argument("--rolling-model", type=str, default="ridge",
                        choices=["ols", "ridge", "lasso"], help="Model type for rolling fit")
    parser.add_argument("--rolling-window", type=int, default=252,
                        help="Rolling window size in days")
    args = parser.parse_args()

    project_root = Path(__file__).parent
    data_path = project_root / args.input
    outputs = project_root / "outputs"
    ensure_output_dirs(outputs)

    # ---- DATA ----
    if args.generate_sample or not data_path.exists():
        print("Generating sample dataset...")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        generate_sample_dataset(str(data_path))
        print(f"Sample data generated at {data_path}")

    print("Loading and preparing data...")
    df_index, X, y = load_and_prepare(str(data_path), index_col="Index")

    # ---- STATIC MODEL ----
    print(f"Fitting static model ({args.model}) ...")
    model, model_info = fit_static_models(X, y, model_type=args.model)
    metrics = evaluate_model(model, X, y)
    print("Static model metrics:", metrics)

    contributions_static = compute_static_contributions(model, X, y, index=X.index)
    static_json = contributions_static.to_dict(orient="index")

    # ---- ROLLING REGRESSION ----
    print(f"Running rolling regression ({args.rolling_model}) with window {args.rolling_window} ...")
    coeffs_df, intercepts = rolling_regression(X, y, window=args.rolling_window, model_type=args.rolling_model)
    contributions_rolling = compute_rolling_contributions(coeffs_df, X)
    rolling_json = contributions_rolling.to_dict(orient="index")

    # 🔧 FIX: convert Timestamp keys → string for JSON compatibility
    rolling_json_str = {str(k): v for k, v in rolling_json.items()}
    static_json_str = {str(k): v for k, v in static_json.items()}

    # ---- JSON OUTPUT ----
    payload = {
        "metadata": {
            "model": args.model,
            "rolling_model": args.rolling_model,
            "rolling_window": args.rolling_window,
        },
        "static_attributions": static_json_str,
        "rolling_attributions": rolling_json_str,
        "metrics": metrics
    }
    out_json = outputs / "factor_contributions.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print("Saved JSON:", out_json)

    # ---- VISUALIZATIONS ----
    print("Plotting contributions (static & rolling)...")
    plots = []

    p1 = outputs / "plots" / "stacked_contributions_static.png"
    plot_stacked_contributions(
        contributions_static.drop(columns=["predicted_return", "actual_return"], errors="ignore"),
        p1
    )
    plots.append(p1)

    p2 = outputs / "plots" / "coeffs_rolling.png"
    plot_coefficients_over_time(coeffs_df, p2)
    plots.append(p2)

    p3 = outputs / "plots" / "actual_vs_pred.png"
    plot_actual_vs_predicted(X.index, y, model.predict(X), p3)
    plots.append(p3)

    # ---- SUMMARY ----
    cumulative = contributions_static.drop(columns=["predicted_return", "actual_return"], errors="ignore").sum().sort_values(ascending=False)
    top_pos = cumulative.head(3).to_dict()
    top_neg = cumulative.tail(3).to_dict()

    summary_text = (
        f"🌸 Static model ({args.model}) results: RMSE={metrics['rmse']:.5f}, R2={metrics['r2']:.3f}.\n"
        f"✨ Top positive cumulative contributors: {top_pos}.\n"
        f"🌙 Top negative cumulative contributors: {top_neg}.\n"
        "🎀 See the attached pastel-styled plots for time-series attribution and coefficient dynamics."
    )

    # ---- HTML REPORT ----
        # ---- HTML REPORT ----
    report_out = outputs / "report.html"
    generate_html_report(
        report_out,
        title="🌸 Multi-Factor Regression — Pastel Factor Attribution Report 🌸",
        summary=summary_text,
        plot_paths=[str(p) for p in plots],
        json_path=out_json,   # ✅ FIXED
    )
    print("Report generated:", report_out)



if __name__ == "__main__":
    main()
