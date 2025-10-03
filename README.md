# 🧾 Multi-Factor Regression Model — AI Engineer Assignment

This repository contains a full end-to-end implementation of a **Multi-Factor Regression Model** for explaining stock market index returns using multiple explanatory factors (macroeconomic, market-based, and technical). The project evaluates factor contributions over time and generates machine-readable outputs, human-readable summaries, and visualizations.

---

## 🔹 Project Overview

- **Objective:** Attribute stock index returns (e.g., NIFTY 50, S&P 500) to multiple explanatory factors and quantify contributions using regression models.
- **Approach:** 
  - Data preprocessing and alignment of multiple factor time series.
  - Regression modeling using **OLS**, **Ridge**, or **Lasso**.
  - Static and rolling-window factor attribution.
  - Machine-readable JSON outputs and pastel-themed human-readable reports.
  - Visualizations of factor contributions over time.

---

## 🛠️ Features / Deliverables

1. **Codebase**  
   All scripts are modular and organized under `src/`:
   - `data_processing.py` → Data loading, cleaning, alignment, and sample dataset generation.
   - `modeling.py` → Static and rolling regression models, factor attribution.
   - `visualization.py` → Pastel-themed plots for contributions and coefficients.
   - `report.py` → Human-readable HTML report generation.
   - `run_pipeline.py` → Orchestrates the end-to-end workflow.

2. **Sample Dataset**
   - Located in `data/sample_data.csv`.
   - Includes index prices and multiple factors (OilPrice, FX, VIX, CPI, InterestRate).
   - Can be generated using:  
     ```bash
     python run_pipeline.py --generate-sample
     ```

3. **Sample Outputs**
   - `outputs/factor_contributions.json` → Machine-readable factor contributions.
   - `outputs/report.html` → Human-readable summary and plots.
   - `outputs/plots/` → Visualizations of contributions over time.
   - `outputs/SUMMARY.md` → Markdown summary of top contributors and metrics.

4. **Visualizations**
   - Stacked contributions of factors over time.
   - Rolling coefficients over time.
   - Actual vs predicted index returns.
   - Pastel/cutesy theme for aesthetic appeal.

5. **Evaluation Metrics**
   - RMSE, MAE, R² score for static regression models.

---


