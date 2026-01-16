"""
Bayesian Baseline and Marketing Mix Model (MMM) Utilities for Rossmann Sales Forecasting

This module provides a comprehensive suite of functions for building, fitting, and analyzing 
Bayesian models for Rossmann store sales. It supports the entire pipeline from data preparation 
to future forecasting and counterfactual analysis.

Key stages covered:
1. Baseline Preparation: prepare_rossmann_baseline_df(), build_baseline_arrays()
2. Baseline Modeling: fit_bayesian_baseline(), baseline_posterior_predict()
3. Lift Calculation: compute_empirical_lift(), compute_hybrid_discount_intensity()
4. MMM Preparation: prepare_mmm_df(), build_mmm_arrays()
5. MMM Modeling: fit_mmm_pymc()
6. Forecasting Utilities: build_future_calendar(), assign_promos(), sample_discount_intensity(), map_last_year_holidays(), forecast_baseline_spline()
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from typing import Tuple
from scipy.interpolate import UnivariateSpline
import arviz as az
import matplotlib.pyplot as plt

def prepare_rossmann_baseline_df(
    df: pd.DataFrame
    ) -> Tuple:
    """
    Prepare Rossmann sales data for Bayesian baseline modeling.
    
    This function performs comprehensive data preprocessing including:
    - Filtering for open days with valid sales data
    - Creating time and store indices for hierarchical modeling
    - Engineering features (day of week, month, holidays, competition)
    - Log-transforming the target variable
    
    The preprocessing ensures data is in the correct format for PyMC model fitting,
    with proper indexing for hierarchical effects and time series components.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing merged train and store data with columns:
        - Store: Store identifier
        - Date: Date of observation
        - Sales: Daily sales amount
        - Open: Binary indicator (1=open, 0=closed)
        - DayOfWeek: Day of week (1=Monday, 7=Sunday)
        - StateHoliday: State holiday indicator ('0', 'a', 'b', 'c')
        - SchoolHoliday: School holiday indicator (0/1)
        - CompetitionDistance: Distance to nearest competitor
        - Promo, Promo2: Promotional indicators (kept for lift calculation)
        
    Returns
    -------
    tuple of (pd.DataFrame, int, int)
        df : pd.DataFrame
            Preprocessed DataFrame with additional columns:
            - y: log(Sales + 1) - log-transformed target
            - t_idx: Time index (0 to T-1)
            - store_idx: Store index (0 to S-1)
            - dow_idx: Day of week index (0-6)
            - month_idx: Month index (0-11)
            - state_holiday: Binary state holiday indicator
            - school_holiday: Binary school holiday indicator
            - log_comp_dist: Log-transformed competition distance
        S : int
            Number of unique stores in the dataset
        T : int
            Number of unique time points (days) in the dataset
            
    Notes
    -----
    - Only keeps observations where Open=1 and Sales >= 0
    - Sorts data by Date and Store for time series alignment
    - Uses int64 for indices to avoid PyTensor overflow issues
    - Competition distance NaNs are filled with median before log-transform
    - StateHoliday is converted from categorical ('0','a','b','c') to binary
    
    Examples
    --------
    >>> df_full = train_df.merge(store_df, on='Store')
    >>> df, S, T = prepare_rossmann_baseline_df(df_full)
    >>> print(f"Stores: {S}, Time points: {T}")
    Stores: 1115, Time points: 942
    """
    df = df.copy()

    # Ensure datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Keep only open days with positive sales
    print(f"Before filtering: {len(df):,} observations")
    df = df[df["Open"] == 1].copy()
    df = df[df["Sales"].notna()].copy()
    df = df[df["Sales"] >= 0].copy()
    print(f"After filtering (Open=1, Sales>=0): {len(df):,} observations")

    # Target: log(Sales + 1)
    df["y"] = np.log(df["Sales"].astype(float) + 1.0)

    # Sort by date and store for time series alignment
    df = df.sort_values(["Date", "Store"])
    
    # Create global time index (0 to T-1)
    unique_dates = pd.Index(df["Date"].unique()).sort_values()
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    df["t_idx"] = df["Date"].map(date_to_idx).astype(int)
    T = len(unique_dates)
    print(f"Time points (T): {T}")

    # Create store index (0 to S-1)
    store_ids = pd.Index(df["Store"].unique()).sort_values()
    store_to_idx = {s: i for i, s in enumerate(store_ids)}
    df["store_idx"] = df["Store"].map(store_to_idx).astype(int)
    S = len(store_ids)
    print(f"Number of stores (S): {S}")

    # Day of week: 1-7 -> 0-6
    df["dow_idx"] = (df["DayOfWeek"].astype(int) - 1).clip(0, 6)

    # Month: 1-12 -> 0-11
    df["month_idx"] = (df["Date"].dt.month.astype(int) - 1).clip(0, 11)

    # State Holiday: handle '0', 'a', 'b', 'c'
    sh = df["StateHoliday"].astype(str)
    df["state_holiday"] = (sh != "0").astype(int)
    
    # School Holiday
    df["school_holiday"] = df["SchoolHoliday"].astype(int)

    # Competition distance (log-transformed, fill NaN with median)
    comp = df["CompetitionDistance"].astype(float)
    comp_filled = comp.fillna(comp.median())
    df["log_comp_dist"] = np.log1p(comp_filled)

    # Keep Promo columns for later lift calculation (NOT used in baseline)
    for c in ["Promo", "Promo2"]:
        if c in df.columns:
            df[c] = df[c].astype(int)
        else:
            df[c] = 0

    return df, S, T


def build_baseline_arrays(
    df: pd.DataFrame
    ) -> Tuple:
    """
    Extract numpy arrays from preprocessed DataFrame for PyMC baseline model.
    
    This function converts the preprocessed DataFrame into numpy arrays with
    appropriate dtypes for efficient PyMC model fitting. All indices use int64
    to avoid overflow issues with large store counts.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame from prepare_rossmann_baseline_df() containing:
        - y: Log-transformed sales
        - store_idx, t_idx, dow_idx, month_idx: Integer indices
        - state_holiday, school_holiday: Binary indicators
        - log_comp_dist: Log-transformed competition distance
        
    Returns
    -------
    tuple of numpy.ndarray
        y : np.ndarray (float32)
            Log-transformed sales target, shape (N,)
        store_idx : np.ndarray (int64)
            Store indices for hierarchical effects, shape (N,)
        t_idx : np.ndarray (int64)
            Time indices for Gaussian Random Walk, shape (N,)
        dow_idx : np.ndarray (int64)
            Day of week indices (0-6), shape (N,)
        month_idx : np.ndarray (int64)
            Month indices (0-11), shape (N,)
        state_h : np.ndarray (float32)
            State holiday binary indicator, shape (N,)
        school_h : np.ndarray (float32)
            School holiday binary indicator, shape (N,)
        log_comp : np.ndarray (float32)
            Log-transformed competition distance, shape (N,)
            
    Notes
    -----
    - Uses int64 for all indices (not int8/int16) to prevent PyTensor overflow
    - With 1115 stores, int8 (max=127) would cause errors
    - Float32 is used for continuous variables to reduce memory usage
    - All arrays are 1-dimensional with length N (number of observations)
    
    Important
    ---------
    The use of int64 for indices is critical. PyTensor graph optimization can
    fail with smaller integer types when store counts exceed the type's maximum
    value (e.g., 1115 stores > 127 max for int8).
    
    Examples
    --------
    >>> arrays = build_baseline_arrays(df)
    >>> y, store_idx, t_idx, dow_idx, month_idx, state_h, school_h, log_comp = arrays
    >>> print(f"Target shape: {y.shape}, dtype: {y.dtype}")
    Target shape: (844392,), dtype: float32
    >>> print(f"Store index range: [{store_idx.min()}, {store_idx.max()}]")
    Store index range: [0, 1114]
    """
    y = df["y"].to_numpy(dtype=np.float32)
    
    # Explicitly cast to int64 to avoid overflow in PyTensor graph optimization
    store_idx = df["store_idx"].to_numpy(dtype=np.int64)
    t_idx = df["t_idx"].to_numpy(dtype=np.int64)
    dow_idx = df["dow_idx"].to_numpy(dtype=np.int64)
    month_idx = df["month_idx"].to_numpy(dtype=np.int64)

    state_h = df["state_holiday"].to_numpy(dtype=np.float32)
    school_h = df["school_holiday"].to_numpy(dtype=np.float32)
    log_comp = df["log_comp_dist"].to_numpy(dtype=np.float32)

    return y, store_idx, t_idx, dow_idx, month_idx, state_h, school_h, log_comp


def fit_bayesian_baseline(
    y: np.ndarray,
    store_idx: np.ndarray,
    t_idx: np.ndarray,
    dow_idx: np.ndarray,
    month_idx: np.ndarray,
    state_h: np.ndarray,
    school_h: np.ndarray,
    log_comp: np.ndarray,
    S: int,
    T: int,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9
    ) -> Tuple:
    """
    Fit hierarchical Bayesian baseline model for Rossmann sales.
    
    This function implements a Bayesian hierarchical model that captures baseline
    sales patterns WITHOUT promotional effects. The model uses:
    - Hierarchical store intercepts with partial pooling
    - Gaussian Random Walk for smooth time trends
    - Day-of-week and monthly seasonality with sum-to-zero constraints
    - Holiday and competition effects
    
    Model Structure
    ---------------
    log(Sales) = α + α_s[store] + level[time] + β_dow[dow] + β_month[month]
                 + β_state_h * StateHoliday + β_school_h * SchoolHoliday
                 + β_comp * log(CompetitionDistance) + ε
    
    Where:
    - α: Global intercept
    - α_s: Store-specific random intercepts (hierarchical)
    - level: Gaussian Random Walk capturing time trends
    - β_dow, β_month: Day-of-week and monthly effects (sum-to-zero)
    - β_state_h, β_school_h, β_comp: Holiday and competition coefficients
    - ε ~ Normal(0, σ): Observation noise
    
    Parameters
    ----------
    y : np.ndarray
        Log-transformed sales (target variable)
    store_idx : np.ndarray
        Store indices (0 to S-1)
    t_idx : np.ndarray
        Time indices (0 to T-1)
    dow_idx : np.ndarray
        Day of week indices (0-6)
    month_idx : np.ndarray
        Month indices (0-11)
    state_h : np.ndarray
        State holiday binary indicator
    school_h : np.ndarray
        School holiday binary indicator
    log_comp : np.ndarray
        Log-transformed competition distance
    S : int
        Number of stores
    T : int
        Number of time points
    draws : int, default=1000
        Number of posterior samples per chain
    tune : int, default=1000
        Number of tuning steps for sampler adaptation
    chains : int, default=4
        Number of MCMC chains
    target_accept : float, default=0.9
        Target acceptance rate for NUTS sampler
        
    Returns
    -------
    tuple of (pm.Model, az.InferenceData)
        model : pm.Model
            Compiled PyMC model object
        idata : az.InferenceData
            Posterior samples and diagnostics
            
    Priors
    ------
    - α ~ Normal(0, 2): Global intercept
    - τ_store ~ HalfNormal(0.5): Store effect standard deviation
    - α_s ~ Normal(0, τ_store): Store random intercepts (non-centered)
    - σ_level ~ HalfNormal(0.1): Random walk innovation std
    - level[t] ~ GaussianRandomWalk(σ_level): Time trend
    - β_dow_raw ~ Normal(0, 0.3): Day-of-week effects (before centering)
    - β_month_raw ~ Normal(0, 0.3): Monthly effects (before centering)
    - β_state_h, β_school_h, β_comp ~ Normal(0, 0.5): Covariate effects
    - σ ~ HalfNormal(0.5): Observation noise
    
    Notes
    -----
    - Uses non-centered parameterization for store effects (better sampling)
    - Sum-to-zero constraints ensure identifiability of seasonal effects
    - Gaussian Random Walk allows flexible time trends without overfitting
    - NO promotional variables included (this is the baseline model)
    - High target_accept (0.9) helps with complex hierarchical geometry
    
    Convergence
    -----------
    Check convergence using:
    - R-hat < 1.01 for all parameters
    - Effective sample size > 400 (for 4 chains × 1000 draws)
    - Visual inspection of trace plots
    
    Examples
    --------
    >>> model, idata = fit_bayesian_baseline(
    ...     y, store_idx, t_idx, dow_idx, month_idx, 
    ...     state_h, school_h, log_comp, S, T
    ... )
    >>> summary = az.summary(idata, var_names=['alpha', 'tau_store', 'sigma'])
    >>> print(summary)
    """
    with pm.Model() as model:
        # --- Using ConstantData to maintain explicit int64 dtypes in the graph ---
        s_idx = pm.Data("s_idx", store_idx.astype("int64"))
        time_idx = pm.Data("time_idx", t_idx.astype("int64"))
        d_idx = pm.Data("d_idx", dow_idx.astype("int64"))
        m_idx = pm.Data("m_idx", month_idx.astype("int64"))
        
        st_h = pm.Data("st_h", state_h)
        sc_h = pm.Data("sc_h", school_h)
        l_comp = pm.Data("l_comp", log_comp)
        
        # ============================================
        # HYPERPRIORS FOR STORE INTERCEPTS
        # Non-centered parameterization for better sampling
        # ============================================
        
        tau_store = pm.HalfNormal("tau_store", sigma=0.5)
        store_offset = pm.Normal("store_offset", mu=0.0, sigma=1.0, shape=S)
        alpha_store = pm.Deterministic("alpha_store", store_offset * tau_store)

        # ============================================
        # GLOBAL INTERCEPT
        # ============================================
        alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)

        # ============================================
        # DAY-OF-WEEK & MONTH SEASONALITY
        # Sum-to-zero constraint for identifiability
        # ============================================
        dow_raw = pm.Normal("dow_raw", mu=0.0, sigma=0.3, shape=7)
        dow = pm.Deterministic("dow", dow_raw - pt.mean(dow_raw))

        mon_raw = pm.Normal("mon_raw", mu=0.0, sigma=0.3, shape=12)
        mon = pm.Deterministic("mon", mon_raw - pt.mean(mon_raw))

        # ============================================
        # HOLIDAY EFFECTS
        # ============================================
        beta_state_h = pm.Normal("beta_state_h", mu=0.0, sigma=0.5)
        beta_school_h = pm.Normal("beta_school_h", mu=0.0, sigma=0.5)

        # ============================================
        # COMPETITION DISTANCE EFFECT
        # ============================================
        beta_comp = pm.Normal("beta_comp", mu=0.0, sigma=0.5)

        # ============================================
        # GLOBAL TIME TREND: GAUSSIAN RANDOM WALK
        # Captures dynamic macro drift in baseline
        # ============================================
        sigma_level = pm.HalfNormal("sigma_level", sigma=0.1)
        level = pm.GaussianRandomWalk("level", sigma=sigma_level, shape=T, init_dist=pm.Normal.dist(0.0, 1.0))

        # ============================================
        # LINEAR PREDICTOR (BASELINE - NO PROMO)
        # ============================================
        mu = (
            alpha
            + alpha_store[s_idx]
            + dow[d_idx]
            + mon[m_idx]
            + beta_state_h * st_h
            + beta_school_h * sc_h
            + beta_comp * l_comp
            + level[time_idx]
        )

        # ============================================
        # LIKELIHOOD
        # ============================================
        sigma = pm.HalfNormal("sigma", sigma=0.5)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        # ============================================
        # SAMPLING
        # ============================================
        print("Starting MCMC sampling...")
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=10,
            target_accept=target_accept,
            random_seed=42,
            return_inferencedata=True
        )

    return model, idata


def baseline_posterior_predict(
    model: pm.Model,
    idata,
    df: pd.DataFrame,
    return_intervals: bool = True
    ) -> pd.DataFrame:
    """
    Generate counterfactual baseline predictions from the Bayesian model.
    
    This function generates posterior predictive samples for baseline sales
    (without promotional effects) and computes summary statistics including
    mean predictions and credible intervals.
    
    Parameters
    ----------
    model : pm.Model
        Fitted PyMC model from fit_bayesian_baseline()
    idata : az.InferenceData
        Posterior samples from MCMC sampling
    df : pd.DataFrame
        Original preprocessed DataFrame with Store, Date, Sales columns
    return_intervals : bool, default=True
        If True, compute and return 5th, 50th, and 95th percentiles
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - Store: Store identifier
        - Date: Date of observation
        - Sales: Actual sales
        - baseline_mean: Mean baseline prediction (no promo)
        - baseline_p05: 5th percentile (if return_intervals=True)
        - baseline_p50: 50th percentile/median (if return_intervals=True)
        - baseline_p95: 95th percentile (if return_intervals=True)
        
    Notes
    -----
    - Predictions are in original sales units (not log-space)
    - Uses np.expm1() to back-transform from log(Sales+1) to Sales
    - Credible intervals capture posterior uncertainty
    - Mean is computed across all chains and draws
    
    Examples
    --------
    >>> baseline_df = baseline_posterior_predict(model, idata, df)
    >>> print(baseline_df[['Sales', 'baseline_mean', 'baseline_p05', 'baseline_p95']].head())
    """
    print("Generating posterior predictive samples...")
    
    with model:
        ppc = pm.sample_posterior_predictive(idata, var_names=["y_obs"], random_seed=42)

    y_obs = ppc.posterior_predictive["y_obs"]
    sales_draws = np.expm1(y_obs)
    
    # Calculate statistics across chains and draws
    base_mean = sales_draws.mean(dim=["chain", "draw"]).values
    
    out = df[["Store", "Date", "Sales"]].copy()
    out["baseline_mean"] = base_mean
    if return_intervals:
        out["baseline_p05"] = sales_draws.quantile(0.05, dim=["chain", "draw"]).values
        out["baseline_p50"] = sales_draws.quantile(0.50, dim=["chain", "draw"]).values
        out["baseline_p95"] = sales_draws.quantile(0.95, dim=["chain", "draw"]).values

    return out


def compute_empirical_lift(
    df: pd.DataFrame, 
    baseline_pred: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Calculate empirical lift by comparing actual sales to baseline predictions.
    
    Empirical lift measures the multiplicative increase in sales due to promotions:
    lift = (Actual Sales - Baseline Sales) / Baseline Sales
    
    Parameters
    ----------
    df : pd.DataFrame
        Original data with Store, Date, Sales, Promo columns
    baseline_pred : pd.DataFrame
        Baseline predictions from baseline_posterior_predict()
        Must contain Store, Date, baseline_mean columns
        
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with additional columns:
        - baseline_mean: Baseline prediction
        - lift_emp: Empirical lift (clipped at 0)
        - lift_abs: Absolute lift (Sales - Baseline)
        
    Notes
    -----
    - Lift is clipped at 0 (no negative lift allowed)
    - Uses small epsilon (1e-6) to avoid division by zero
    - Merges on Store and Date, ensuring proper alignment
    
    Formula
    -------
    lift_emp = max(0, (Sales - Baseline) / (Baseline + ε))
    lift_abs = Sales - Baseline
    
    Examples
    --------
    >>> df_with_lift = compute_empirical_lift(df, baseline_df)
    >>> print(df_with_lift.groupby('Promo')['lift_emp'].describe())
    """
    df['Date'] = pd.to_datetime(df['Date'])
    baseline_pred['Date'] = pd.to_datetime(baseline_pred['Date'])
    merged = df.merge(baseline_pred, on=["Store", "Date"], how="left", suffixes=('', '_pred'))
    
    # Calculate lift
    eps = 1e-6
    merged["lift_emp"] = np.maximum(
        0.0,
        (merged["Sales"] - merged["baseline_mean"]) / (merged["baseline_mean"] + eps)
    )
    
    # Absolute lift
    merged["lift_abs"] = merged["Sales"] - merged["baseline_mean"]
    
    return merged


def compute_hybrid_discount_intensity(
    df_with_lift: pd.DataFrame, 
    w_struct: float = 0.6, 
    w_emp: float = 0.4
    ) -> pd.DataFrame:
    """
    Compute hybrid discount intensity combining structural and empirical components.
    
    The hybrid approach combines:
    1. Structural component: Based on promotional flags (Promo, Promo2)
    2. Empirical component: Based on observed lift, normalized by 95th percentile
    
    This creates a more robust promotional intensity metric that captures both
    planned promotional activity and actual market response.
    
    Parameters
    ----------
    df_with_lift : pd.DataFrame
        DataFrame from compute_empirical_lift() with lift_emp column
    w_struct : float, default=0.6
        Weight for structural component (0 to 1)
    w_emp : float, default=0.4
        Weight for empirical component (0 to 1)
        Note: w_struct + w_emp should equal 1.0
        
    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns:
        - di_struct: Structural discount intensity
        - di_emp: Empirical discount intensity (normalized)
        - discount_intensity: Weighted combination
        
    Notes
    -----
    - Structural: di_struct = 1.0 * Promo + 0.5 * Promo2
    - Empirical: di_emp = lift_emp / P95(lift_emp)
    - Hybrid: discount_intensity = w_struct * di_struct + w_emp * di_emp
    - P95 normalization makes empirical component robust to outliers
    
    Examples
    --------
    >>> df_final = compute_hybrid_discount_intensity(df_with_lift)
    >>> print(df_final[['di_struct', 'di_emp', 'discount_intensity']].describe())
    """
    df = df_with_lift.copy()

    # Structural promo index
    df["di_struct"] = 1.0 * df["Promo"].astype(float) + 0.5 * df["Promo2"].astype(float)

    # Empirical component normalized by P95 (robust to outliers)
    p95 = np.quantile(df["lift_emp"].to_numpy(), 0.95) if df["lift_emp"].notna().any() else 1.0
    p95 = max(p95, 1e-6)
    df["di_emp"] = df["lift_emp"] / p95

    # Hybrid intensity (weighted combination)
    df["discount_intensity"] = w_struct * df["di_struct"] + w_emp * df["di_emp"]
    
    return df


# ---------------------------------------
# MMM Functions
# ---------------------------------------
def prepare_mmm_df(
    df_method: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Prepare the final Marketing Mix Model (MMM) DataFrame by applying necessary transformations to the input data.

    This function processes the input DataFrame (`df_method`) and prepares it for further modeling steps. 
    The transformations include handling missing values, computing log-transformed sales and baseline, 
    creating time and store indices, adding holiday and competition features, and ensuring that the necessary 
    columns (e.g., 'discount_intensity') are present.

    Parameters:
    -----------
    df_method : pd.DataFrame
        The input DataFrame containing the raw data. It must include columns like:
        - 'baseline_mean': Baseline prediction of sales.
        - 'Sales': Actual sales data.
        - 'Date': The date of the observation.
        - 'Store': The store identifier.
        - 'DayOfWeek': The day of the week (1-7).
        - 'StateHoliday': Binary indicator of state holidays.
        - 'SchoolHoliday': Binary indicator of school holidays.
        - 'CompetitionDistance': Distance to the nearest competitor.
        - 'discount_intensity': Discount intensity (must be present).

    Returns:
    --------
    pd.DataFrame
        A processed DataFrame with the following transformations:
        - Rows with missing 'baseline_mean' or 'Sales' are dropped.
        - Log-transformed sales and baseline are added as 'y' and 'baseline_log' respectively.
        - Date is converted to datetime format and sorted.
        - Store index ('store_idx') is added, mapping store IDs to integer indices.
        - 'dow_idx' and 'month_idx' are added as indices for the day of the week and month.
        - Binary holiday features ('state_holiday' and 'school_holiday') are created.
        - Log-transformed competition distance ('log_comp_dist') is added.
        - A check is performed to ensure that 'discount_intensity' column is present.

    Notes:
    ------
    - Missing 'baseline_mean' or 'Sales' values are dropped from the DataFrame.
    - If the 'Open' column is present, only rows with 'Open' == 1 are kept.
    - A small epsilon (1e-6) is used when applying the log transformation to avoid taking the log of zero.
    - The column 'discount_intensity' must be present in the input DataFrame. If not, a ValueError is raised.
    - The output DataFrame includes several additional features like log-transformed sales and baseline, 
      indices for day of week, month, store, and competition distance, and binary holiday indicators.

    Example:
    --------
    # Assuming df_model is a DataFrame containing the necessary columns:
    df_mmm, stores_mmm = prepare_mmm_df(df_model)
    
    # df_mmm is the transformed DataFrame ready for modeling
    # stores_mmm is the number of unique stores (length of store_ids)
    """

    df = df_method.copy()

    # Drop rows with missing baseline or sales
    df = df[df["baseline_mean"].notna()].copy()
    df = df[df["Sales"].notna()].copy()

    # Optional: keep only open days
    if "Open" in df.columns:
        df = df[df["Open"] == 1].copy()

    # Target: log-sales
    df["y"] = np.log(df["Sales"].astype(float) + 1.0)

    # Baseline offset in log space
    # We model incremental factor on top of baseline_mean
    eps = 1e-6
    df["baseline_log"] = np.log(df["baseline_mean"].astype(float) + 1.0 + eps)

    # Time & store indices
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Store"])

    store_ids = pd.Index(df["Store"].unique()).sort_values()
    store_to_idx = {s: i for i, s in enumerate(store_ids)}
    df["store_idx"] = df["Store"].map(store_to_idx).astype(int)
    S = len(store_ids)

    # DOW, month (optional controls)
    df["dow_idx"] = (df["DayOfWeek"].astype(int) - 1).clip(0, 6)
    df["month_idx"] = (df["Date"].dt.month.astype(int) - 1).clip(0, 11)

    # Holidays (binary)
    sh = df["StateHoliday"].astype(str)
    df["state_holiday"] = (sh != "0").astype(int)
    df["school_holiday"] = df["SchoolHoliday"].astype(int)

    # Competition
    comp = df["CompetitionDistance"].astype(float)
    comp_filled = comp.fillna(comp.median())
    df["log_comp_dist"] = np.log1p(comp_filled)

    # For now, we use discount_intensity as a single feature.
    # If you want adstock + Hill outside PyMC, replace this with that transformed column.
    if "discount_intensity" not in df.columns:
        raise ValueError("Expected 'discount_intensity' column from Method 4.")

    return df, S


def build_mmm_arrays(
    df_mmm: pd.DataFrame
    )-> Tuple:
    """
    Build the design arrays required for the Marketing Mix Model (MMM) from the given DataFrame.

    This function extracts specific columns from the input DataFrame (`df_mmm`) and converts them into NumPy arrays
    of appropriate data types. These arrays represent the features and target variable needed for MMM modeling.

    Parameters:
    -----------
    df_mmm : pd.DataFrame
        The input DataFrame containing the processed data. It must include the following columns:
        - 'y': Log-transformed sales target variable.
        - 'baseline_log': Log-transformed baseline sales.
        - 'discount_intensity': The intensity of discounts.
        - 'store_idx': Store indices (numeric).
        - 'dow_idx': Day of the week indices (numeric, 0-6).
        - 'month_idx': Month indices (numeric, 0-11).
        - 'state_holiday': Binary indicator for state holidays (1 or 0).
        - 'school_holiday': Binary indicator for school holidays (1 or 0).
        - 'log_comp_dist': Log-transformed competition distance.

    Returns:
    --------
    tuple
        A tuple of NumPy arrays corresponding to the following features and target:
        - y (np.ndarray): Log-transformed sales target variable.
        - baseline_log (np.ndarray): Log-transformed baseline sales.
        - di (np.ndarray): Discount intensity.
        - store_idx (np.ndarray): Store indices (integer values).
        - dow_idx (np.ndarray): Day of the week indices (integer values, 0-6).
        - month_idx (np.ndarray): Month indices (integer values, 0-11).
        - state_h (np.ndarray): Binary indicator for state holidays (float type).
        - school_h (np.ndarray): Binary indicator for school holidays (float type).
        - log_comp (np.ndarray): Log-transformed competition distance (float type).

    Notes:
    ------
    - All output arrays are NumPy arrays with appropriate data types.
    - The function assumes the presence of the necessary columns in the `df_mmm` DataFrame. 
    - The data types for the arrays are optimized for efficient computation during modeling:
      - `float32` for continuous variables and binary features.
      - `int32` for indices (store, day of the week, month).
      
    Example:
    --------
    # Assuming df_mmm is a processed DataFrame ready for model fitting:
    y, baseline_log, di, store_idx, dow_idx, month_idx, state_h, school_h, log_comp = build_mmm_arrays(df_mmm)
    
    # y: Log-transformed sales
    # baseline_log: Log-transformed baseline sales
    # di: Discount intensity
    # store_idx: Store indices
    # dow_idx: Day of the week indices
    # month_idx: Month indices
    # state_h: Binary indicator for state holidays
    # school_h: Binary indicator for school holidays
    # log_comp: Log-transformed competition distance
    """
    y = df_mmm["y"].to_numpy(dtype=np.float32)
    baseline_log = df_mmm["baseline_log"].to_numpy(dtype=np.float32)
    di = df_mmm["discount_intensity"].to_numpy(dtype=np.float32)

    store_idx = df_mmm["store_idx"].to_numpy(dtype=np.int32)
    dow_idx = df_mmm["dow_idx"].to_numpy(dtype=np.int32)
    month_idx = df_mmm["month_idx"].to_numpy(dtype=np.int32)

    state_h = df_mmm["state_holiday"].to_numpy(dtype=np.float32)
    school_h = df_mmm["school_holiday"].to_numpy(dtype=np.float32)
    log_comp = df_mmm["log_comp_dist"].to_numpy(dtype=np.float32)

    return (
        y,
        baseline_log,
        di,
        store_idx,
        dow_idx,
        month_idx,
        state_h,
        school_h,
        log_comp,
    )


def fit_mmm_pymc(
    y: np.ndarray,
    baseline_log: np.ndarray,
    di: np.ndarray,
    store_idx: np.ndarray,
    dow_idx: np.ndarray,
    month_idx: np.ndarray,
    state_h: np.ndarray,
    school_h: np.ndarray,
    log_comp: np.ndarray,
    S: int,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    cores: int = 10,
    random_seed: int = 123,
    ) -> Tuple:
    """
    Fit a Marketing Mix Model (MMM) using PyMC with promotional discount intensity as a media channel.

    This function fits a Bayesian model that predicts log-transformed sales (`log(Sales)`) as a function of
    various factors, including baseline sales, store-specific random intercepts, promotional effects, day-of-week
    (DOW), month, holiday indicators, and competition distance.

    The model is hierarchical and includes random effects for each store, as well as a global effect for promo intensity.
    It uses Markov Chain Monte Carlo (MCMC) to sample from the posterior distribution of the model parameters.

    Parameters:
    -----------
    y : np.ndarray
        The log-transformed sales target variable (dependent variable).
    baseline_log : np.ndarray
        The baseline log-transformed sales, used as a fixed offset in the model.
    di : np.ndarray
        The discount intensity, representing the media channel (independent variable).
    store_idx : np.ndarray
        The indices for the stores (for random intercepts).
    dow_idx : np.ndarray
        The indices for the days of the week (0-6).
    month_idx : np.ndarray
        The indices for the months (0-11).
    state_h : np.ndarray
        Binary indicator for state holidays (1 for holiday, 0 otherwise).
    school_h : np.ndarray
        Binary indicator for school holidays (1 for holiday, 0 otherwise).
    log_comp : np.ndarray
        The log-transformed competition distance (independent variable).
    S : int
        The total number of unique stores (used for defining the size of random effects).
    draws : int, optional
        The number of MCMC samples to draw (default is 1000).
    tune : int, optional
        The number of tuning (burn-in) steps to perform before sampling (default is 1000).
    chains : int, optional
        The number of chains to run in parallel (default is 4).
    target_accept : float, optional
        The target acceptance rate for the sampler (default is 0.9).
    cores : int, optional
        The number of cores to use for parallelization (default is 10).
    random_seed : int, optional
        The random seed for reproducibility (default is 123).

    Returns:
    --------
    tuple
        A tuple containing:
        - `mmm_model` : The PyMC model object used for sampling.
        - `idata_mmm` : The InferenceData object containing the sampled posterior distributions of the model parameters.

    Notes:
    ------
    - The model includes several components:
        - A global intercept (`alpha`).
        - Store-specific random intercepts (`alpha_store`).
        - Random effects for promotional discount intensity at the store level (`beta_promo_store`).
        - Day-of-week and month effects (`dow` and `mon`), with sum-to-zero constraints for identifiability.
        - Binary indicators for state holidays (`state_h`), school holidays (`school_h`), and competition distance (`log_comp`).
    - The model uses `pm.sample()` to perform MCMC sampling. The resulting `idata` contains the posterior distributions for the model parameters.
    - Sampling is done with 4 chains, each running for `draws` iterations with `tune` tuning steps.
    - This function requires the PyMC library to be installed.

    Example:
    --------
    # Assuming mmm_arrays contains the necessary design arrays (y, baseline_log, di, etc.) and stores_mmm is the number of stores:
    model_mmm, idata_mmm = fit_mmm_pymc(*mmm_arrays, stores_mmm)
    
    # `model_mmm` is the PyMC model object, and `idata_mmm` contains the posterior samples of the model parameters.
    """

    with pm.Model() as mmm_model:

        # --- Data containers ---
        Y = pm.Data("Y", y)
        BASE = pm.Data("BASE", baseline_log)
        DI = pm.Data("DI", di)

        STORE_IDX = pm.Data("STORE_IDX", store_idx)
        DOW_IDX = pm.Data("DOW_IDX", dow_idx)
        MONTH_IDX = pm.Data("MONTH_IDX", month_idx)

        STATE_H = pm.Data("STATE_H", state_h)
        SCHOOL_H = pm.Data("SCHOOL_H", school_h)
        LOG_COMP = pm.Data("LOG_COMP", log_comp)

        # --- Hyperpriors for store intercepts ---
        tau_store = pm.HalfNormal("tau_store", sigma=0.5)
        store_offset = pm.Normal("store_offset", mu=0.0, sigma=1.0, shape=S)
        alpha_store = pm.Deterministic("alpha_store", store_offset * tau_store)

        # --- Global intercept on top of baseline ---
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)

        # --- Store-level random promo effect (hierarchical β_promo_s) ---
        beta_promo_global = pm.Normal("beta_promo_global", mu=0.0, sigma=1.0)
        tau_promo = pm.HalfNormal("tau_promo", sigma=0.5)

        promo_offset = pm.Normal("promo_offset", mu=0.0, sigma=1.0, shape=S)
        beta_promo_store = pm.Deterministic(
            "beta_promo_store", beta_promo_global + promo_offset * tau_promo
        )

        # --- Day-of-week & month effects (sum-to-zero for identifiability) ---
        dow_raw = pm.Normal("dow_raw", mu=0.0, sigma=0.3, shape=7)
        dow = pm.Deterministic("dow", dow_raw - pt.mean(dow_raw))

        mon_raw = pm.Normal("mon_raw", mu=0.0, sigma=0.3, shape=12)
        mon = pm.Deterministic("mon", mon_raw - pt.mean(mon_raw))

        # --- Holiday & competition effects ---
        beta_state_h = pm.Normal("beta_state_h", mu=0.0, sigma=0.5)
        beta_school_h = pm.Normal("beta_school_h", mu=0.0, sigma=0.5)
        beta_comp = pm.Normal("beta_comp", mu=0.0, sigma=0.5)

        
        # Here we just rename it for clarity:
        DI_final = DI

        # --- Linear predictor ---
        mu = (
            BASE  # baseline offset from previous Bayesian model
            + alpha
            + alpha_store[STORE_IDX]
            + beta_promo_store[STORE_IDX] * DI_final
            + dow[DOW_IDX]
            + mon[MONTH_IDX]
            + beta_state_h * STATE_H
            + beta_school_h * SCHOOL_H
            + beta_comp * LOG_COMP
        )

        # --- Likelihood ---
        sigma = pm.HalfNormal("sigma", sigma=0.5)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=Y)

        # --- Sampling ---
        idata_mmm = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores = cores,
            target_accept=target_accept,
            random_seed=random_seed,
        )

    return mmm_model, idata_mmm


def build_future_calendar(
    start_date: str,
    end_date: str,
    store_id: int,
    store_idx: int,
    ) -> pd.DataFrame:
    """
    Build a future calendar DataFrame for a given store over a specified date range.

    This function generates a DataFrame that includes future dates for a given store and assigns relevant 
    time-related indices, such as day of the week (`dow_idx`) and month index (`month_idx`).

    Parameters:
    -----------
    start_date : str or datetime-like
        The last date in the historical data (used as the starting point for the future dates).
    end_date : str or datetime-like
        The end date for the forecast period (inclusive).
    store_id : int
        The identifier for the store (e.g., store number or name).
    store_idx : int
        The index corresponding to the store in the model data.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the future dates, store identifiers, and relevant time indices:
        - 'Date': The future date.
        - 'Store': The store identifier (store_id).
        - 'store_idx': The store index (store_idx).
        - 'dow_idx': The day of the week index (0=Monday, 6=Sunday).
        - 'month_idx': The month index (0=January, 11=December).

    Notes:
    ------
    - The day of the week index (`dow_idx`) is derived from the future dates using `.dt.dayofweek`.
    - The month index (`month_idx`) is derived from the month of the future dates using `.dt.month`.
    - The generated DataFrame can be used to assign holidays, promos, and other time-related effects.

    Example:
    --------
    # Assuming the last_date is '2023-12-31' and the end_date is '2024-12-31':
    df_future = build_future_calendar(
        start_date="2023-12-31", 
        end_date="2024-12-31", 
        store_id=82, 
        store_idx=0
    )
    """
    dates = pd.date_range(start_date + pd.Timedelta(days=1), end_date, freq="D")

    df_future = pd.DataFrame({
        "Date": dates,
        "Store": store_id,
        "store_idx": store_idx,
    })

    df_future["dow_idx"] = df_future["Date"].dt.dayofweek
    df_future["month_idx"] = df_future["Date"].dt.month - 1

    return df_future


def assign_promos(
    df: pd.DataFrame, 
    n_promo: int, 
    n_promo2: int, 
    length_promo: int, 
    length_promo2: int, 
    seed: int = 42
    ) -> pd.DataFrame:
    """
    Randomly assigns promotional blocks to the future period for two types of promotions.
    This function distributes a specified number of promotional days across a given timeframe 
    by assigning them in consecutive blocks whenever possible. It handles two independent 
    promotions (promo and promo2) and maintains the original DataFrame structure.
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame (usually a future calendar) where promotions will be assigned.
    n_promo : int
        Total number of promotional days for the first promotion (promo).
    n_promo2 : int
        Total number of promotional days for the second promotion (promo2).
    length_promo : int
        The length of consecutive days for the first promotion's blocks.
    length_promo2 : int
        The length of consecutive days for the second promotion's blocks.
    seed : int, optional
        The random seed for shuffling available days (default is 42).
    Returns:
    --------
    pd.DataFrame
        The modified DataFrame with 'promo' and 'promo2' columns containing 1s for 
        promotional days and 0s otherwise.
    Notes:
    ------
    - Promotions are assigned by randomly picking starting days for blocks of the specified length.
    - If the number of days isn't perfectly divisible by the block length, the remainder 
      is assigned randomly as single days.
    - Overlap between 'promo' and 'promo2' is allowed.
    Example:
    --------
    # Assign 10 days of Promo 1 (5-day blocks) and 20 days of Promo 2 (10-day blocks):
    df_future = assign_promos(df_future, n_promo=10, n_promo2=20, length_promo=5, length_promo2=10)
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    df = df.copy()
    df["promo"] = 0
    df["promo2"] = 0
    
    num_blocks_p1 = n_promo // length_promo
    remainder_p1 = n_promo % length_promo
    num_blocks_p2 = n_promo2 // length_promo2
    remainder_p2 = n_promo2 % length_promo2
    
    available_days = list(range(n))
    rng.shuffle(available_days)
    
    p1_indices = []
    for _ in range(num_blocks_p1):
        for day in available_days:
            if day + length_promo <= n:
                block = list(range(day, day + length_promo))
                if not any(d in p1_indices for d in block):
                    p1_indices.extend(block)
                    break
    
    if remainder_p1 > 0:
        remaining = [d for d in range(n) if d not in p1_indices]
        if len(remaining) >= remainder_p1:
            p1_indices.extend(rng.choice(remaining, size=remainder_p1, replace=False))
    
    rng.shuffle(available_days)
    p2_indices = []
    for _ in range(num_blocks_p2):
        for day in available_days:
            if day + length_promo2 <= n:
                block = list(range(day, day + length_promo2))
                if not any(d in p2_indices for d in block):
                    p2_indices.extend(block)
                    break
    
    if remainder_p2 > 0:
        remaining = [d for d in range(n) if d not in p2_indices]
        if len(remaining) >= remainder_p2:
            p2_indices.extend(rng.choice(remaining, size=remainder_p2, replace=False))
    
    df.loc[df.index[p1_indices[:n_promo]], "promo"] = 1
    df.loc[df.index[p2_indices[:n_promo2]], "promo2"] = 1
    
    return df


def sample_discount_intensity(
    df_future: pd.DataFrame, 
    df_history: pd.DataFrame, 
    Store: int, 
    seed: int = 42
    ) -> pd.DataFrame:
    """
    Sample discount intensity values for the future period based on historical data.

    This function assigns discount intensity values to the future DataFrame (`df_future`) for days when promotions 
    are active. It uses historical discount intensity values from the specified store (`Store`) and assigns them 
    to future dates with promo activity.

    Parameters:
    -----------
    df_future : pd.DataFrame
        The DataFrame containing the future dates and promo indicators (`promo` and `promo2`).
    df_history : pd.DataFrame
        The DataFrame containing historical sales data with the `discount_intensity` and `Store` columns.
    Store : int
        The store identifier for which to sample the discount intensity.
    seed : int, optional
        The random seed for reproducibility (default is 42).

    Returns:
    --------
    pd.DataFrame
        The `df_future` DataFrame with an additional column 'discount_intensity' assigned for promo days.

    Notes:
    ------
    - The function assigns discount intensity values only for the days when either `promo` or `promo2` is active.
    - Discount intensity values are sampled from the historical values for the specified store.

    Example:
    --------
    # Sample discount intensity for future dates:
    df_future = sample_discount_intensity(df_future, df_history, Store=82)
    """
    rng = np.random.default_rng(seed)
    
    hist_di = df_history.loc[
        df_history["discount_intensity"] > 0 & (df_history["Store"] == Store),
        "discount_intensity"
    ].values

    df_future["discount_intensity"] = 0.0

    promo_mask = (df_future["promo"] == 1) | (df_future["promo2"] == 1)
    df_future.loc[promo_mask, "discount_intensity"] = rng.choice(
        hist_di,
        size=promo_mask.sum(),
        replace=True
    )

    return df_future


def map_last_year_holidays(
    df_future: pd.DataFrame, 
    df_history: pd.DataFrame, 
    Store: int
    ) -> pd.DataFrame:
    """
    Map last year's holidays to the future DataFrame.

    This function maps the last year's holidays (state_holiday and school_holiday) from the historical data 
    to the future DataFrame for a specific store. It uses the `Date` column from the future DataFrame and 
    adds a new column `Date_ly` to align with the historical data.

    Parameters:
    -----------
    df_future : pd.DataFrame
        The DataFrame containing the future dates and promo indicators (`promo` and `promo2`).
    df_history : pd.DataFrame
        The DataFrame containing historical sales data with the `state_holiday`, `school_holiday`, and `Store` columns.
    Store : int
        The store identifier for which to map the holidays.

    Returns:
    --------
    pd.DataFrame
        The `df_future` DataFrame with additional columns 'state_holiday' and 'school_holiday' mapped from the 
        historical data for the specified store.

    Notes:
    ------
    - The function maps holidays from the last year (i.e., `Date_ly` is `Date` + 1 year).
    - The function uses the `Date` column from the future DataFrame to align with the historical data.
    - The function fills missing values with 0 and converts the columns to integer type.

    Example:
    --------
    # Map last year's holidays for future dates:
    df_future = map_last_year_holidays(df_future, df_history, Store=82)
    """
    hist = df_history.copy()
    # filter history to one store
    hist = df_history[df_history["Store"] == Store].copy()
    hist["Date_ly"] = hist["Date"] + pd.DateOffset(years=1)

    holiday_map = hist.set_index("Date_ly")[["state_holiday", "school_holiday"]]

    df_future = df_future.join(holiday_map, on="Date")
    df_future[["state_holiday", "school_holiday"]] = (
        df_future[["state_holiday", "school_holiday"]]
        .fillna(0)
        .astype(int)
    )

    return df_future


def forecast_baseline_spline(
    df: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    lookback_days: int = 365,
    ) -> pd.DataFrame:
    """
    Forecast baseline sales using a smoothing spline based on historical baseline data.

    This function fits a smoothing spline model to the historical baseline sales data and uses it to forecast 
    the baseline sales for future dates.

    Parameters:
    -----------
    df : pd.DataFrame
        The historical DataFrame containing at least two columns: `Date` and `baseline_mean`.
    forecast_dates : pd.DatetimeIndex
        The future dates for which the baseline sales should be predicted.
    lookback_days : int, optional
        The number of past days to use for fitting the spline model (default is 365).

    Returns:
    --------
    np.ndarray
        The forecasted baseline sales for the future dates.

    Notes:
    ------
    - The spline is fit using the past `lookback_days` of data and then used to predict the baseline for the future.
    - The baseline forecast is returned in the original scale (not log-transformed).

    Example:
    --------
    # Forecast baseline sales for future dates:
    df_future["baseline_mean"] = forecast_baseline_spline(
        df=df_mmm[df_mmm["Store"] == 82],
        forecast_dates=df_future["Date"]
    )
    df_future["baseline_log"] = np.log(df_future["baseline_mean"] + 1)
    """

    df_hist = (
        df.sort_values("Date")
          .tail(lookback_days)
          .copy()
    )

    # Time index
    t_hist = np.arange(len(df_hist))
    y_hist = np.log(df_hist["baseline_mean"].values + 1)

    # Fit spline (slight smoothing)
    spline = UnivariateSpline(
        t_hist,
        y_hist,
        k=3,
        s=len(t_hist) * 0.5
    )

    # Future time index
    n_future = len(forecast_dates)
    t_future = np.arange(len(t_hist), len(t_hist) + n_future)

    y_future_log = spline(t_future)
    baseline_future = np.exp(y_future_log) - 1

    return baseline_future


def compute_sales_forecast(df_sim: pd.DataFrame, 
                           model_mmm: pm.Model, 
                           idata_mmm: az.InferenceData) -> pd.DataFrame:
    """
    Computes the daily sales forecast for a given dataframe, model & posterior data.
    This function uses the Bayesian MMM model to sample posterior predictive distributions
    for two scenarios: factual (with original discount intensity) and 
    counterfactual (without promotions). It updates the input DataFrame with 
    mean forecasted sales, baseline sales, and incremental lift.
    Parameters:
    -----------
    df_sim : pd.DataFrame
        The simulation input DataFrame containing promotional indicators ('promo', 'promo2'), 
        'discount_intensity', and categorical feature indices.
    model_mmm : pm.Model
        The Bayesian MMM model used for forecasting.
    idata_mmm : az.InferenceData
        The inference data containing the posterior samples from the MMM model.
    Returns:
    --------
    pd.DataFrame
        The input DataFrame updated with 'sales', 'sales_no_promo', and 'incremental' columns.
    Notes:
    ------
    - Performs sampling from the posterior predictive distribution using PyMC.
    - Sales are converted back from log-scale (log(y+1)) to original scale.
    Example:
    --------
    df_forecast = compute_sales_forecast(df_sim, model_mmm, idata_mmm)
    """
    with model_mmm:
        pm.set_data({
            "Y": np.zeros(len(df_sim)),
            "BASE": df_sim["baseline_log"].values,
            "DI": df_sim["discount_intensity"].values,
            "STORE_IDX": df_sim["store_idx"].values,
            "DOW_IDX": df_sim["dow_idx"].values,
            "MONTH_IDX": df_sim["month_idx"].values,
            "STATE_H": df_sim["state_holiday"].values,
            "SCHOOL_H": df_sim["school_holiday"].values,
            "LOG_COMP": df_sim["log_comp_dist"].values,
        })
        ppc = pm.sample_posterior_predictive(idata_mmm, var_names=["y_obs"], progressbar=False)
        pm.set_data({"DI": np.zeros_like(df_sim["discount_intensity"].values)})
        ppc_no_promo = pm.sample_posterior_predictive(idata_mmm, var_names=["y_obs"], progressbar=False)
    
    # Compute DAILY sales (not total)
    daily_sales = np.exp(ppc.posterior_predictive["y_obs"]) - 1
    daily_sales_no_promo = np.exp(ppc_no_promo.posterior_predictive["y_obs"]) - 1
    df_sim['sales'] = daily_sales.mean(dim=("chain", "draw")).values
    df_sim['sales_no_promo'] = daily_sales_no_promo.mean(dim=("chain", "draw")).values
    df_sim['incremental'] = df_sim['sales'] - df_sim['sales_no_promo']
    return df_sim


def visualize_calendar(df_res: pd.DataFrame):
    """
    Visualizes the optimal promotional calendar alongside the sales forecast.
    Generates a two-panel plot showing timelines for active promotions and 
    the resulting sales forecast vs. baseline.
    Parameters:
    -----------
    df_res : pd.DataFrame
        Results DataFrame containing 'Date', 'promo', 'promo2', 'sales', and 'sales_no_promo'.
    Returns:
    --------
    None
        Displays a matplotlib plot and prints a summary to the console.
    Notes:
    ------
    - Shaded regions indicate Promo 1 (coral), Promo 2 (teal), and Overlap (purple).
    Example:
    --------
    visualize_calendar(best_df)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    dates = df_res['Date']
    
    # Top plot: Promo schedule
    ax1.fill_between(dates, 0, 1, where=(df_res['promo'] == 1), 
                     color='coral', alpha=0.3, label='Promo 1', 
                     step='post', transform=ax1.get_xaxis_transform())
    ax1.fill_between(dates, 0, 1, where=(df_res['promo2'] == 1), 
                     color='teal', alpha=0.2, label='Promo 2', 
                     step='post', transform=ax1.get_xaxis_transform())
    overlap = (df_res['promo'] == 1) & (df_res['promo2'] == 1)
    ax1.fill_between(dates, 0, 1, where=overlap, 
                     color='purple', alpha=0.4, label='Both Promos', 
                     step='post', transform=ax1.get_xaxis_transform())
    ax1.set_title('Promotional Calendar Schedule', fontsize=13)
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Promo Status')
    
    # Bottom plot: Sales forecast
    ax2.plot(dates, df_res['sales_no_promo'], 'k--', lw=1.5, alpha=0.5, label='Baseline')
    ax2.plot(dates, df_res['sales'], 'b-', lw=2, label='Forecasted Sales with Promos')
    ax2.fill_between(dates, df_res['sales_no_promo'], df_res['sales'], 
                     where=(df_res['sales'] > df_res['sales_no_promo']), 
                     color='green', alpha=0.2, label='Incremental Lift')
    ax2.set_title('Daily Sales Forecast', fontsize=13)
    ax2.set_ylabel('Sales')
    ax2.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print(f"Summary:")
    print(f"Promo 1: {df_res['promo'].sum()} days")
    print(f"Promo 2: {df_res['promo2'].sum()} days")
    print(f"Both Promos will run for : {overlap.sum()} days")
    print(f"Forecasted Total Sales: {df_res['sales'].sum():,.0f}")
    print(f"Forecasted Total Sales (no promo): {df_res['sales_no_promo'].sum():,.0f}")
    print(f"Forecasted Avg Daily Sales: {df_res['sales'].mean():,.0f}")


print("Bayesian module loaded successfully!")