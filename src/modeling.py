"""
Modeling: static OLS (statsmodels), Ridge/Lasso (sklearn with CV), rolling regression.
Also functions to compute factor contributions (static and rolling).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def fit_static_models(X, y, model_type="ridge", alphas=None):
    """
    Fit a static model on full X/y.
    Returns (model_obj, info_dict)
    - for 'ols' uses statsmodels OLS (with intercept)
    - for 'ridge' uses RidgeCV
    - for 'lasso' uses LassoCV
    """
    if alphas is None:
        alphas = np.logspace(-4, 2, 50)
    if model_type == "ols":
        Xc = sm.add_constant(X)
        ols_res = sm.OLS(y, Xc).fit()
        return ols_res, {"type": "ols", "params": ols_res.params}
    elif model_type == "ridge":
        model = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error').fit(X, y)
        return model, {"type": "ridge", "alpha": float(model.alpha_)}
    elif model_type == "lasso":
        model = LassoCV(alphas=alphas, cv=5, n_jobs=-1, max_iter=5000).fit(X, y)
        return model, {"type": "lasso", "alpha": float(model.alpha_)}

def evaluate_model(model, X, y):
    """
    Returns RMSE, MAE, R2. Works for statsmodels (has predict) and sklearn.
    """
    if hasattr(model, "predict"):
        try:
            preds = model.predict(X)
        except Exception:
            # statsmodels: add const
            preds = model.predict(sm.add_constant(X))
    else:
        raise ValueError("Model has no predict method")
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))
    return {"rmse": rmse, "mae": mae, "r2": r2}

def extract_coefs_from_model(model, X):
    """
    Return series of coefficients indexed by X.columns and an intercept value.
    Supports statsmodels OLS and sklearn estimators with coef_ / intercept_.
    """
    if hasattr(model, "params"):  # statsmodels
        params = model.params
        if "const" in params.index:
            intercept = float(params["const"])
            coef = params.drop("const")
        elif "const" in params.index.str.lower():
            # fallback
            intercept = float(params.iloc[0])
            coef = params.iloc[1:]
        else:
            intercept = 0.0
            coef = params
        coef = coef.reindex(X.columns).fillna(0.0)
        return pd.Series(coef, index=X.columns), intercept
    else:
        # sklearn-style
        intercept = float(model.intercept_) if hasattr(model, "intercept_") else 0.0
        coefs = np.array(model.coef_)
        return pd.Series(coefs, index=X.columns), intercept

def compute_static_contributions(model, X, y=None, index=None):
    """
    For a static model, compute per-date contributions: X * coef for each factor.
    Returns DataFrame: index aligned with X.index; columns = factors, and predicted_return, actual_return (if y provided)
    """
    coefs, intercept = extract_coefs_from_model(model, X)
    contrib = X.multiply(coefs, axis=1)
    pred = contrib.sum(axis=1) + intercept
    contrib["predicted_return"] = pred
    if y is not None:
        contrib["actual_return"] = y
    if index is not None:
        contrib.index = index
    return contrib

def rolling_regression(X, y, window=252, model_type="ridge", alphas=None):
    """
    Perform a rolling-window regression (train on each window) and return:
      - coeffs_df: DataFrame of coefficients (rows indexed by end-of-window date)
      - intercepts: Series of intercepts
    Supported model_type: 'ols', 'ridge', 'lasso'
    Implementation uses sklearn LinearRegression for 'ols' when rolling (speed), statsmodels used in static fit.
    """
    if alphas is None:
        alphas = np.logspace(-4, 2, 20)

    dates = X.index
    cols = X.columns.tolist()
    rows = []
    intercepts = []
    idx_dates = []

    for end in range(window - 1, len(X)):
        start = end - window + 1
        X_train = X.iloc[start:end+1]
        y_train = y.iloc[start:end+1]

        if model_type == "ols":
            model = LinearRegression().fit(X_train, y_train)
        elif model_type == "ridge":
            model = Ridge(alpha=1.0).fit(X_train, y_train)  # you can replace with cv for each window (slow)
        elif model_type == "lasso":
            model = Lasso(alpha=0.001, max_iter=5000).fit(X_train, y_train)
        else:
            raise ValueError("Unknown model type")

        coef = pd.Series(model.coef_, index=cols)
        rows.append(coef.values)
        intercepts.append(float(model.intercept_) if hasattr(model, "intercept_") else 0.0)
        idx_dates.append(dates[end])

    coeffs_df = pd.DataFrame(rows, index=idx_dates, columns=cols)
    intercepts = pd.Series(intercepts, index=idx_dates)
    return coeffs_df, intercepts

def compute_rolling_contributions(coeffs_df, X):
    """
    Given coefficient time-series (coeffs_df indexed by date) and X (daily features),
    compute attributions for each date that has a coefficient row.
    Returns DataFrame indexed by coeffs_df.index with columns = factors + predicted_return.
    """
    # ensure alignment: coeffs_df.index subset of X.index
    common_idx = coeffs_df.index.intersection(X.index)
    contributions = []
    for dt in common_idx:
        coef = coeffs_df.loc[dt]
        xrow = X.loc[dt]
        contrib = xrow * coef
        row = contrib.to_dict()
        row["predicted_return"] = float(contrib.sum())
        contributions.append(row)
    return pd.DataFrame(contributions, index=common_idx)
