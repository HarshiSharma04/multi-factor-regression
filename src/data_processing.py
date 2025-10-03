"""
Data loading, preprocessing, sample data generator, frequency alignment, transforms.
"""
import pandas as pd
import numpy as np

def generate_sample_dataset(path="data/sample_data.csv",
                            start="2018-01-01",
                            end="2021-12-31",
                            seed=42):
    """
    Create a synthetic but realistic daily dataset with:
    - Index (price level)
    - OilPrice, FX, VIX (daily numeric)
    - CPI, InterestRate (monthly; upsampled to daily via forward-fill)
    The Index returns are generated as a weighted combination of factor moves + noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)  # business days
    n = len(dates)

    # create base factor series
    # daily random walks with drift
    oil = 50 + np.cumsum(rng.normal(0.01, 0.3, size=n))
    fx = 70 + np.cumsum(rng.normal(0.005, 0.1, size=n))
    vix = 12 + np.abs(np.cumsum(rng.normal(0, 0.05, size=n)))  # positive
    # create monthly CPI / InterestRate then forward fill to daily
    monthly_dates = pd.date_range(start=start, end=end, freq="MS")
    cpi_monthly = 100 + np.cumsum(rng.normal(0.05, 0.1, size=len(monthly_dates)))
    ir_monthly = 3 + np.cumsum(rng.normal(0.0, 0.02, size=len(monthly_dates)))

    df = pd.DataFrame(index=dates)
    df["OilPrice"] = oil
    df["FX"] = fx
    df["VIX"] = vix

    cpi_series = pd.Series(cpi_monthly, index=monthly_dates).reindex(df.index, method="ffill")
    ir_series = pd.Series(ir_monthly, index=monthly_dates).reindex(df.index, method="ffill")
    df["CPI"] = cpi_series.values
    df["InterestRate"] = ir_series.values

    # simulate index log returns influenced by percent-changes of factors:
    # define true coefficients (hidden)
    true_coefs = {
        "OilPrice": 0.0008,
        "FX": -0.0006,
        "VIX": -0.0012,
        "CPI": -0.0002,
        "InterestRate": -0.0005,
    }

    # compute factor returns/diffs
    oil_ret = pd.Series(df["OilPrice"]).pct_change().fillna(0).values
    fx_ret = pd.Series(df["FX"]).pct_change().fillna(0).values
    vix_ret = pd.Series(df["VIX"]).pct_change().fillna(0).values
    cpi_diff = pd.Series(df["CPI"]).diff().fillna(0).values
    ir_diff = pd.Series(df["InterestRate"]).diff().fillna(0).values

    # build index daily returns
    noise = rng.normal(0, 0.005, size=n)
    idx_ret = (true_coefs["OilPrice"] * oil_ret +
               true_coefs["FX"] * fx_ret +
               true_coefs["VIX"] * vix_ret +
               true_coefs["CPI"] * (cpi_diff / 100.0) +
               true_coefs["InterestRate"] * (ir_diff / 100.0) +
               0.0005 + noise)

    # convert to price level
    index_level = 1000 * np.exp(np.cumsum(idx_ret))  # log accumulation
    df["Index"] = index_level

    # save
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Date"}, inplace=True)
    df.to_csv(path, index=False)
    return df

def load_and_prepare(path, index_col="Index", date_col="Date", factor_cols=None):
    """
    Loads CSV, parses dates, computes returns, aligns factors, and returns:
      - original index series (price)
      - X: DataFrame with factor features aligned to index returns (same index)
      - y: series of index returns (pct_change)
    By default auto-detect factor columns as all columns except date/index.
    """
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)

    if factor_cols is None:
        factor_cols = [c for c in df.columns if c != index_col]

    # Basic alignment: forward-fill missing factor values (sensible for monthly->daily)
    df[factor_cols] = df[factor_cols].ffill().bfill()

    # compute target returns (daily)
    y = df[index_col].pct_change().fillna(0)
    # feature transforms: percent-change for price-like factors, diff for rates
    X = pd.DataFrame(index=df.index)
    for col in factor_cols:
        if col.lower().startswith(("cpi", "infl", "ir", "interest")) or "rate" in col.lower():
            # rates/CPI: use difference (level changes matter)
            X[col] = df[col].diff().fillna(0)
        else:
            # price-like: use percent change
            X[col] = df[col].pct_change().fillna(0)

    # Optionally standardize features (z-score)
    X = (X - X.mean()) / (X.std().replace(0, 1))

    # align X and y (drop initial days where y might be NaN)
    aligned_idx = X.index.intersection(y.index)
    X = X.loc[aligned_idx].copy()
    y = y.loc[aligned_idx].copy()

    return df[index_col].loc[aligned_idx], X, y
