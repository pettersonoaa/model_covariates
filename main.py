"""
Refactored Prophet forecasting pipeline with enhanced readability and structure.

Includes:
- Data loading and preparation (log transform, lags, holiday/date dummies)
- Covariate evaluation (correlation, Granger causality)
- Prophet model training and prediction
- Comprehensive evaluation metrics (MAPE) and diagnostics
- Detailed visualizations (forecast, components, fit, covariate analysis)
- Artifact exporting (model, forecast, plots)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
import pickle
import os
import re
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
import logging
from dateutil.easter import easter # For holiday calculation
from matplotlib.gridspec import GridSpec

# --- Configuration ---

# Suppress unnecessary warnings and logs for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# --- Constants ---
DEFAULT_DATE_COL = 'date'
DEFAULT_TARGET_COL = 'sales'
DEFAULT_OUTPUT_DIR = 'exports'
DEFAULT_PLOTS_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'plots')
DEFAULT_FORECASTS_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'forecasts')
DEFAULT_MODELS_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'models')

DEFAULT_COLOR_COMPONENT = 'tab:blue'
DEFAULT_COLOR_STANDARD = 'slategrey'
DEFAULT_COLOR_HIGHLIGHT = 'tab:orange'
DEFAULT_COLOR_ACTUAL = 'slategrey'
DEFAULT_COLOR_PREDICTED = 'tab:red'
DEFAULT_COLOR_FORECAST = 'tab:orange'
DEFAULT_COLOR_PASSED_TEST = 'tab:blue'
DEFAULT_COLOR_FAILED_TEST = 'tab:red'

# --- Data Handling ---

def load_data(target_path, covariate_path=None, date_col=DEFAULT_DATE_COL,
              target_col=DEFAULT_TARGET_COL, covariate_cols=None):
    """
    Load target and optional covariate data from CSV files.

    Args:
        target_path (str): Path to the target CSV file.
        covariate_path (str, optional): Path to the covariate CSV file.
        date_col (str): Name of the date column.
        target_col (str): Name of the target column.
        covariate_cols (list, optional): List of covariate column names to load.

    Returns:
        tuple: (pd.Series, pd.DataFrame or None) Target series and Covariate DataFrame.
    """
    print(f"Loading target data from: {target_path}")
    try:
        target_df = pd.read_csv(target_path, parse_dates=[date_col])
        target_series = pd.Series(
            target_df[target_col].values,
            index=pd.DatetimeIndex(target_df[date_col]),
            name=target_col
        )
        print(f"  Loaded target: {len(target_series)} points from {target_series.index.min().date()} to {target_series.index.max().date()}")
    except FileNotFoundError:
        print(f"ERROR: Target file not found at {target_path}")
        return None, None
    except KeyError as e:
        print(f"ERROR: Column '{e}' not found in target file.")
        return None, None

    covariate_df = None
    if covariate_path and covariate_cols:
        print(f"Loading covariates from: {covariate_path}")
        try:
            temp_df = pd.read_csv(covariate_path, parse_dates=[date_col])
            available_cols = [col for col in covariate_cols if col in temp_df.columns]
            missing_cols = [col for col in covariate_cols if col not in temp_df.columns]

            if missing_cols:
                print(f"  WARNING: Missing requested covariates: {missing_cols}")
            if not available_cols:
                print(f"  ERROR: None of the requested covariates found: {covariate_cols}")
            else:
                covariate_df = pd.DataFrame(index=pd.DatetimeIndex(temp_df[date_col]))
                for col in available_cols:
                    covariate_df[col] = temp_df[col].values
                print(f"  Loaded covariates: {', '.join(available_cols)} ({len(covariate_df)} points)")
        except FileNotFoundError:
            print(f"  ERROR: Covariate file not found at {covariate_path}")
        except KeyError as e:
            print(f"  ERROR: Column '{e}' not found in covariate file.")

    return target_series, covariate_df

# --- Feature Engineering ---

def apply_log_transform(data, epsilon=1e-9):
    """
    Apply log1p transformation, handling zeros/negatives by replacing with epsilon.

    Args:
        data (pd.Series or pd.DataFrame): Input data.
        epsilon (float): Small value to replace non-positive numbers.

    Returns:
        pd.Series or pd.DataFrame: Log-transformed data.
    """

    transformed = data.copy()
    if isinstance(data, pd.Series):
        zero_mask = transformed <= 0
        if zero_mask.any():
            print(f"  Replacing {zero_mask.sum()} non-positive values in '{data.name}' with {epsilon}")
            transformed[zero_mask] = epsilon
        return np.log1p(transformed)
    elif isinstance(data, pd.DataFrame):
        for col in transformed.columns:
            zero_mask = transformed[col] <= 0
            if zero_mask.any():
                print(f"  Replacing {zero_mask.sum()} non-positive values in '{col}' with {epsilon}")
                transformed.loc[zero_mask, col] = epsilon
        return np.log1p(transformed)
    else:
        raise TypeError("Input must be a pandas Series or DataFrame")

def add_lags(data, lags):
    """
    Create lagged features for a Series or DataFrame.

    Args:
        data (pd.Series or pd.DataFrame): Input time series data.
        lags (list): List of lag periods (integers).

    Returns:
        pd.DataFrame: DataFrame with original and lagged features.
    """
    if isinstance(data, pd.Series):
        df = data.to_frame()
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("Input must be a pandas Series or DataFrame")

    original_cols = df.columns.tolist()
    for col in original_cols:
        if 'dummy' in col or 'calendar' in col:
            continue # Skip dummy and calendar columns
        for lag in lags:
            if lag >= 0:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # Fill NaNs resulting from shifts (use backfill then forward fill)
    df = df.bfill().ffill()
    print(f"  Added lags: {lags} for columns: {original_cols}")
    
    # original_cols = [col for col in original_cols if 'dummy' not in col and 'calendar' not in col]
    # df = df.drop(columns=original_cols, axis=1) # Drop original columns to avoid redundancy
    
    return df

def create_brazil_holiday_dummies_spec(start_date, end_date):
    """
    Create specifications for Brazilian holiday dummy variables.

    Args:
        start_date (datetime): Start date for the holiday range.
        end_date (datetime): End date for the holiday range.

    Returns:
        list: List of dictionaries, each specifying a holiday dummy.
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    holiday_specs = []
    years = range(start_date.year - 1, end_date.year + 2) # Buffer years

    holiday_keys = [
        # Natal window
        'natal_pre5', 'natal_pre4', 'natal_pre3', 'natal_pre2', 'natal_pre1',
        'natal_dia0', 'natal_pos1', 'natal_pos2', 'natal_pos3',
        # Ano Novo window
        'anonovo_pre3', 'anonovo_pre2', 'anonovo_pre1', 'anonovo_dia0',
        'anonovo_pos1', 'anonovo_pos2', 'anonovo_pos3', 'anonovo_pos4',
        'anonovo_pos5', 'anonovo_pos6',
        # Fixed holidays
        'tiradentes', 'diadotrab', 'independ', 'aparecida', 'finados', 'republica',
        # Carnival window
        'carnival_pre7', 'carnival_pre6', 'carnival_pre5', 'carnival_pre4',
        'carnival_pre3', 'carnival_pre2', 'carnival_pre1', 'carnival_dia0',
        'carnival_pos1', 'carnival_pos2', 'carnival_pos3', 'carnival_pos4',
        'carnival_pos5', 'carnival_pos6', 'carnival_pos7',
        # Easter-related
        'sextasanta', 'corpuschristi',
        # Black Friday window
        'blackfriday_pre9', 'blackfriday_pre8', 'blackfriday_pre7', 'blackfriday_pre6',
        'blackfriday_pre5', 'blackfriday_pre4', 'blackfriday_pre3', 'blackfriday_pre2',
        'blackfriday_pre1', 'blackfriday_dia0', 'blackfriday_pos1', 'blackfriday_pos2',
        'blackfriday_pos3', 'blackfriday_pos4', 'blackfriday_pos5', 'blackfriday_pos6',
        'blackfriday_pos7', 'blackfriday_pos8', 'blackfriday_pos9'
    ]

    holiday_dates = {name: [] for name in holiday_keys}

    for year in years:
        # Fixed dates
        holiday_dates['natal_pre5'].append(f"{year}-12-20")
        holiday_dates['natal_pre4'].append(f"{year}-12-21")
        holiday_dates['natal_pre3'].append(f"{year}-12-22")
        holiday_dates['natal_pre2'].append(f"{year}-12-23")
        holiday_dates['natal_pre1'].append(f"{year}-12-24")
        holiday_dates['natal_dia0'].append(f"{year}-12-25")
        holiday_dates['natal_pos1'].append(f"{year}-12-26")
        holiday_dates['natal_pos2'].append(f"{year}-12-27")
        holiday_dates['natal_pos3'].append(f"{year}-12-28")
        holiday_dates['anonovo_pre3'].append(f"{year}-12-29")
        holiday_dates['anonovo_pre2'].append(f"{year}-12-30")
        holiday_dates['anonovo_pre1'].append(f"{year}-12-31")
        holiday_dates['anonovo_dia0'].append(f"{year}-01-01")
        holiday_dates['anonovo_pos1'].append(f"{year}-01-02")
        holiday_dates['anonovo_pos2'].append(f"{year}-01-03")
        holiday_dates['anonovo_pos3'].append(f"{year}-01-04")
        holiday_dates['anonovo_pos4'].append(f"{year}-01-05")
        holiday_dates['anonovo_pos5'].append(f"{year}-01-06")
        holiday_dates['anonovo_pos6'].append(f"{year}-01-07")
        holiday_dates['tiradentes'].append(f"{year}-04-21")
        holiday_dates['diadotrab'].append(f"{year}-05-01")
        holiday_dates['independ'].append(f"{year}-09-07")
        holiday_dates['aparecida'].append(f"{year}-10-12")
        holiday_dates['finados'].append(f"{year}-11-02")
        holiday_dates['republica'].append(f"{year}-11-15")

        # Easter-based
        try:
            easter_date = easter(year)
            holiday_dates['carnival_pre7'].append((easter_date - timedelta(days=54)).strftime("%Y-%m-%d"))
            holiday_dates['carnival_pre6'].append((easter_date - timedelta(days=53)).strftime("%Y-%m-%d"))
            holiday_dates['carnival_pre5'].append((easter_date - timedelta(days=52)).strftime("%Y-%m-%d"))
            holiday_dates['carnival_pre4'].append((easter_date - timedelta(days=51)).strftime("%Y-%m-%d"))
            holiday_dates['carnival_pre3'].append((easter_date - timedelta(days=50)).strftime("%Y-%m-%d"))
            holiday_dates['carnival_pre2'].append((easter_date - timedelta(days=49)).strftime("%Y-%m-%d"))
            holiday_dates['carnival_pre1'].append((easter_date - timedelta(days=48)).strftime("%Y-%m-%d"))
            holiday_dates['carnival_dia0'].append((easter_date - timedelta(days=47)).strftime("%Y-%m-%d"))
            holiday_dates['carnival_pos1'].append((easter_date - timedelta(days=46)).strftime("%Y-%m-%d"))
            holiday_dates['carnival_pos2'].append((easter_date - timedelta(days=45)).strftime("%Y-%m-%d"))
            holiday_dates['carnival_pos3'].append((easter_date - timedelta(days=44)).strftime("%Y-%m-%d"))
            holiday_dates['carnival_pos4'].append((easter_date - timedelta(days=43)).strftime("%Y-%m-%d"))
            holiday_dates['carnival_pos5'].append((easter_date - timedelta(days=42)).strftime("%Y-%m-%d"))
            holiday_dates['carnival_pos6'].append((easter_date - timedelta(days=41)).strftime("%Y-%m-%d"))
            holiday_dates['carnival_pos7'].append((easter_date - timedelta(days=40)).strftime("%Y-%m-%d"))
            holiday_dates['sextasanta'].append((easter_date - timedelta(days=2)).strftime("%Y-%m-%d"))
            holiday_dates['corpuschristi'].append((easter_date + timedelta(days=60)).strftime("%Y-%m-%d"))
        except Exception as e:
            print(f"  WARNING: Error calculating Easter holidays for {year}: {e}")

        # Black Friday (4th Friday in Nov)
        try: # Add try-except for robustness
            thanksgiving = pd.Timestamp(f'{year}-11-01')
            # Find the first Thursday
            while thanksgiving.dayofweek != 3: thanksgiving += timedelta(days=1)
            # Move to the fourth Thursday
            thanksgiving += timedelta(days=21)
            holiday_dates['blackfriday_pre9'].append((thanksgiving + timedelta(days=-8)).strftime("%Y-%m-%d"))
            holiday_dates['blackfriday_pre8'].append((thanksgiving + timedelta(days=-7)).strftime("%Y-%m-%d"))
            holiday_dates['blackfriday_pre7'].append((thanksgiving + timedelta(days=-6)).strftime("%Y-%m-%d"))
            holiday_dates['blackfriday_pre6'].append((thanksgiving + timedelta(days=-5)).strftime("%Y-%m-%d"))
            holiday_dates['blackfriday_pre5'].append((thanksgiving + timedelta(days=-4)).strftime("%Y-%m-%d"))
            holiday_dates['blackfriday_pre4'].append((thanksgiving + timedelta(days=-3)).strftime("%Y-%m-%d"))
            holiday_dates['blackfriday_pre3'].append((thanksgiving + timedelta(days=-2)).strftime("%Y-%m-%d"))
            holiday_dates['blackfriday_pre2'].append((thanksgiving + timedelta(days=-1)).strftime("%Y-%m-%d"))
            holiday_dates['blackfriday_pre1'].append((thanksgiving + timedelta(days=0)).strftime("%Y-%m-%d"))
            holiday_dates['blackfriday_dia0'].append((thanksgiving + timedelta(days=1)).strftime("%Y-%m-%d"))
            holiday_dates['blackfriday_pos1'].append((thanksgiving + timedelta(days=2)).strftime("%Y-%m-%d"))
            holiday_dates['blackfriday_pos2'].append((thanksgiving + timedelta(days=3)).strftime("%Y-%m-%d"))
            holiday_dates['blackfriday_pos3'].append((thanksgiving + timedelta(days=4)).strftime("%Y-%m-%d"))
            holiday_dates['blackfriday_pos4'].append((thanksgiving + timedelta(days=5)).strftime("%Y-%m-%d"))
            holiday_dates['blackfriday_pos5'].append((thanksgiving + timedelta(days=6)).strftime("%Y-%m-%d"))
            holiday_dates['blackfriday_pos6'].append((thanksgiving + timedelta(days=7)).strftime("%Y-%m-%d"))
            holiday_dates['blackfriday_pos7'].append((thanksgiving + timedelta(days=8)).strftime("%Y-%m-%d"))
            holiday_dates['blackfriday_pos8'].append((thanksgiving + timedelta(days=9)).strftime("%Y-%m-%d"))
            holiday_dates['blackfriday_pos9'].append((thanksgiving + timedelta(days=10)).strftime("%Y-%m-%d"))
        except Exception as e:
             print(f"  WARNING: Error calculating Black Friday for {year}: {e}")

    # Define windows and create specs
    for name, dates in holiday_dates.items():
        dummy_name = f"{name}_dummy"
        # --- FIX: Simplified window logic based on name patterns ---
        window_before, window_after = 0, 0
        if '_pre_nivel_' in name:
            try:
                num = int(name.split('_pre_nivel_')[-1])
                window_before = num
            except ValueError: pass # Keep 0 if parsing fails
        elif '_pos_nivel_' in name:
            try:
                num = int(name.split('_pos_nivel_')[-1])
                window_after = num
            except ValueError: pass # Keep 0 if parsing fails
        elif name in ['carnival', 'natal', 'anonovo', 'blackfriday']: # Base names (if used)
             window_before, window_after = 3, 2 # Default window for broader events
        elif name in ['sextasanta', 'diadotrab']:
             window_before, window_after = 1, 1
        else:
            window_before, window_after = 0, 0

        holiday_specs.append({
            'name': dummy_name,
            'specific_dates': dates,
            'window_before': window_before,
            'window_after': window_after
        })

    print(f"  Created {len(holiday_specs)} Brazilian holiday dummy specifications.")
    return holiday_specs

def add_date_dummies(df, date_patterns):
    """
    Add dummy variables to a DataFrame based on date patterns.

    Args:
        df (pd.DataFrame): DataFrame with a 'ds' column (datetime).
        date_patterns (list): List of dictionaries specifying dummy patterns.
                                Each dict needs 'name', and optionally 'month',
                                'day', 'specific_dates', 'window_before', 'window_after'.

    Returns:
        pd.DataFrame: DataFrame with added dummy columns.
    """
    result_df = df.copy()
    dates = pd.to_datetime(result_df['ds'])
    date_to_idx = {date: i for i, date in enumerate(dates)} # For efficient window lookup

    for pattern in date_patterns:
        name = pattern['name']
        month = pattern.get('month')
        day = pattern.get('day')
        specific_dates = pattern.get('specific_dates', [])
        window_before = pattern.get('window_before', 0)
        window_after = pattern.get('window_after', 0)

        result_df[name] = 0.0 # Initialize as float

        # --- Identify base dates ---
        base_match_indices = pd.Series(False, index=result_df.index)
        # Monthly/Day pattern
        if month is not None and day is not None:
            base_match_indices |= (dates.dt.month == month) & (dates.dt.day == day)
        # Specific dates
        if specific_dates:
            specific_dates_dt = pd.to_datetime(specific_dates)
            base_match_indices |= dates.isin(specific_dates_dt)

        # Set base value to 1.0 for matched dates
        result_df.loc[base_match_indices, name] = 1.0

        # --- Apply window effect ---
        if (window_before > 0 or window_after > 0) and base_match_indices.any():
            match_dates = dates[base_match_indices]
            temp_values = result_df[name].copy() # Work on a copy for efficiency

            for current_date, current_idx in date_to_idx.items():
                # Check proximity to each base match date
                for match_date in match_dates:
                    days_diff = (current_date - match_date).days
                    # Apply effect before the date
                    if -window_before <= days_diff < 0:
                        effect = 1.0 - (abs(days_diff) / (window_before + 1))
                        temp_values.iloc[current_idx] = max(temp_values.iloc[current_idx], effect)
                    # Apply effect after the date
                    elif 0 < days_diff <= window_after:
                        effect = 1.0 - (days_diff / (window_after + 1))
                        temp_values.iloc[current_idx] = max(temp_values.iloc[current_idx], effect)

            result_df[name] = temp_values # Update column with window effects

    print(f"  Added/updated date dummy columns: {[p['name'] for p in date_patterns]}")
    return result_df

def prepare_prophet_input_X(X=None, lags=None, log_transform=True, date_dummies=None, covariates=None):
    """
    Prepare the input DataFrame for Prophet, including transformations and features.

    Args:
        y (pd.Series): Target time series.
        X (pd.DataFrame, optional): Covariate DataFrame.
        lags (list, optional): List of lag periods for covariates.
        log_transform (bool): Whether to apply log transform.
        date_dummies (list, optional): Specifications for date dummy variables.

    Returns:
        pd.DataFrame: DataFrame ready for Prophet training/prediction.
    """
    print("Preparing data for Prophet...")
    df = pd.DataFrame({'ds': X.index})
    all_regressor_cols = []

    # 2. Process Covariates (Log Transform, Lags)
    if X is not None:
        # X_aligned = X.reindex(df['ds']) # Align covariates with target dates
        X_aligned = X
        print(f"  Processing {len(X_aligned.columns)} covariates.")

        # Log Transform (Covariates)
        if log_transform:
            # Identify columns to transform and columns to skip
            transform_cols = [col for col in X_aligned.columns 
                             if 'dummy' not in col.lower() and 'calendar' not in col.lower()]
            skip_cols = [col for col in X_aligned.columns 
                        if 'dummy' in col.lower() or 'calendar' in col.lower()]
            
            X_processed = X_aligned.copy()
            
            if transform_cols:
                print(f"  Applying log transform to {len(transform_cols)} covariates")
                X_processed[transform_cols] = apply_log_transform(X_aligned[transform_cols])
            
            if skip_cols:
                print(f"  Skipping log transform for {len(skip_cols)} calendar/dummy covariates: {skip_cols}")
        else:
            X_processed = X_aligned.copy()

        # Add Lags (Covariates)
        if lags:
            X_processed = add_lags(X_processed, lags)

        # Add processed covariate columns to the main DataFrame
        for col in X_processed.columns:
            if col not in df.columns: 
                df[col] = X_processed[col].values
                all_regressor_cols.append(col)

    # 3. Add Date Dummies
    if date_dummies:
        df = add_date_dummies(df, date_dummies)
        dummy_names = [p['name'] for p in date_dummies]
        all_regressor_cols.extend(dummy_names)

    print(f"  Prepared DataFrame shape: {df.shape}")
    print(f"  Regressor columns added: {all_regressor_cols}")
    return df, all_regressor_cols

def prepare_prophet_input_y(y, log_transform=True):
    """
    Prepare the input DataFrame for Prophet, including transformations and features.

    Args:
        y (pd.Series): Target time series.
        X (pd.DataFrame, optional): Covariate DataFrame.
        lags (list, optional): List of lag periods for covariates.
        log_transform (bool): Whether to apply log transform.
        date_dummies (list, optional): Specifications for date dummy variables.

    Returns:
        pd.DataFrame: DataFrame ready for Prophet training/prediction.
    """
    print("Preparing data for Prophet...")
    df = pd.DataFrame({'ds': y.index, 'y': y.values})

    # 1. Log Transform (Target)
    if log_transform:
        print("  Applying log transform to target 'y'")
        df['y'] = apply_log_transform(df['y']).values

    return df

def split_data(df, test_size):
    """Split DataFrame into training and testing sets."""
    if test_size >= len(df):
        raise ValueError("test_size must be smaller than the total number of data points.")
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    print(f"Splitting data: Train ({len(train_df)} rows), Test ({len(test_df)} rows)")
    return train_df, test_df

# --- Modeling ---

def train_model(train_df, regressor_cols=None, yearly_seasonality=True,
                weekly_seasonality=True, daily_seasonality=False):
    """
    Initialize and train a Prophet model.

    Args:
        train_df (pd.DataFrame): Training data (must include 'ds' and 'y').
        regressor_cols (list, optional): Names of regressor columns.
        yearly_seasonality (bool): Enable yearly seasonality.
        weekly_seasonality (bool): Enable weekly seasonality.
        daily_seasonality (bool): Enable daily seasonality.

    Returns:
        Prophet: Trained Prophet model instance.
    """
    print("Initializing Prophet model...")
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        # holidays=holidays_df, # Using dummies as regressors instead
        # holidays_prior_scale=10.0 # Not needed when using dummies
    )

    if regressor_cols:
        print(f"  Adding {len(regressor_cols)} regressors...")
        for col in regressor_cols:
            if col in train_df.columns:
                model.add_regressor(col)
            else:
                print(f"  WARNING: Regressor column '{col}' not found in training data.")

    print("Fitting Prophet model...")
    model.fit(train_df)
    print("Model fitting complete.")
    return model

def make_predictions(model, future_df):
    """
    Make predictions using a trained Prophet model.

    Args:
        model (Prophet): Trained Prophet model.
        future_df (pd.DataFrame): DataFrame with 'ds' and future regressor values.

    Returns:
        pd.DataFrame: Forecast DataFrame with predictions ('yhat', etc.).
    """
    print(f"Making predictions for {len(future_df)} periods...")
    forecast = model.predict(future_df)
    print("Prediction complete.")
    return forecast

def generate_future_df(last_hist_date, periods, regressor_cols=None, hist_data=None, date_dummies=None):
    """
    Create a DataFrame for future predictions, including generating future regressor values.

    Args:
        last_hist_date (datetime): The last date in the historical data.
        periods (int): Number of future periods to generate.
        regressor_cols (list, optional): Names of all regressor columns needed.
        hist_data (pd.DataFrame, optional): Historical data (including regressors)
                                            needed for calculating future lags.
        date_dummies (list, optional): Specifications for date dummy variables.

    Returns:
        pd.DataFrame: DataFrame ready for model.predict().
    """
    if periods <= 0:
        return pd.DataFrame({'ds': []})

    print(f"Generating future DataFrame for {periods} periods...")
    future_dates = pd.date_range(
        start=last_hist_date + timedelta(days=1),
        periods=periods,
        freq='D'
    )
    future_df = pd.DataFrame({'ds': future_dates})

    if not regressor_cols:
        return future_df

    print("  Generating future regressor values...")
    # Combine historical and future data temporarily for lag calculation
    combined_df = pd.concat([hist_data, future_df], ignore_index=True) if hist_data is not None else future_df.copy()
    combined_df.sort_values('ds', inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    # Generate date dummies on combined_df
    if date_dummies:
        dummy_names = [p['name'] for p in date_dummies if p['name'] in regressor_cols]
        if dummy_names:
            relevant_patterns = [p for p in date_dummies if p['name'] in dummy_names]
            combined_df = add_date_dummies(combined_df, relevant_patterns)

    # Handle continuous and lagged features
    lag_pattern = '_lag_'
    processed_lags = set()
    for col in regressor_cols:
        if col in future_df.columns: # Already generated (e.g., date-based)
            continue

        if lag_pattern in col:
            base_col, lag_str = col.split(lag_pattern)
            try:
                lag = int(lag_str)
                if base_col not in combined_df.columns:
                    print(f"  WARNING: Base column '{base_col}' for lag '{col}' not found. Assuming 0.")
                    combined_df[base_col] = 0 # Fallback
                        
                # Calculate the lag on the combined dataframe
                combined_df[col] = combined_df[base_col].shift(lag)
                processed_lags.add(col)
            except ValueError:
                print(f"  WARNING: Could not parse lag from '{col}'. Assuming 0.")
                combined_df[col] = 0
        elif col not in combined_df.columns: # Continuous variable not generated yet
            # Use last known value from historical data if available
            if hist_data is not None and col in hist_data.columns:
                last_value = hist_data[col].iloc[-1]
                print(f"  Using last known value ({last_value:.4f}) for future '{col}'")
                combined_df[col] = last_value # Forward fill
            else:
                print(f"  WARNING: Cannot determine future value for '{col}'. Assuming 0.")
                combined_df[col] = 0 # Fallback

    # Fill NaNs created by shifts, especially at the start of the future period
    # Use forward fill from the end of historical data
    if hist_data is not None:
        combined_df.ffill(inplace=True)

    # Select only the future part and required columns
    future_df_final = combined_df[combined_df['ds'] > last_hist_date].copy()

    # Ensure all required regressor columns exist, fill with 0 if missing after all steps
    missing_cols = [col for col in regressor_cols if col not in future_df_final.columns]
    if missing_cols:
        print(f"  WARNING: Regressors missing after generation: {missing_cols}. Filling with 0.")
        for col in missing_cols:
            future_df_final[col] = 0

    return future_df_final[['ds'] + regressor_cols]


# --- Evaluation ---

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error, handling zeros."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_covariates(y, X, max_lag=70, alpha=0.05):
    """
    Evaluate covariates using correlation and Granger causality.

    Args:
        y (pd.Series): Target time series.
        X (pd.DataFrame): Covariate DataFrame.
        max_lag (int): Maximum lag for Granger causality.
        alpha (float): Significance level for Granger causality.

    Returns:
        dict: Evaluation results including correlations, Granger tests, and recommendations.
    """
    print(f"\nEvaluating {len(X.columns)} covariates (max_lag={max_lag}, alpha={alpha})...")
    results = {'correlation': {}, 'granger_causality': {}, 'recommended_vars': []}
    common_idx = y.index.intersection(X.index)
    if len(common_idx) == 0:
        print("  ERROR: No common dates between target and covariates.")
        return results

    y_aligned = y.loc[common_idx]
    X_aligned = X.loc[common_idx]
    print(f"  Using {len(common_idx)} aligned data points.")

    # Correlation
    for col in X_aligned.columns:
        results['correlation'][col] = {
            'pearson': X_aligned[col].corr(y_aligned, method='pearson'),
            'spearman': X_aligned[col].corr(y_aligned, method='spearman')
        }

    # Granger Causality
    combined_data = pd.concat([y_aligned, X_aligned], axis=1).dropna()
    lags_to_test = [lag for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70] if lag <= max_lag]

    for col in X_aligned.columns:
        test_data = combined_data[[y.name, col]].values
        gc_results = {}
        significant_lags = []
        for lag in lags_to_test:
            try:
                with warnings.catch_warnings(): # Suppress statsmodels warnings
                    warnings.simplefilter("ignore")
                    gc_test = grangercausalitytests(test_data, maxlag=[lag], verbose=False)
                # Extract p-value from the F-test result
                pval = gc_test[lag][0]['ssr_ftest'][1]
                is_significant = pval < alpha
                gc_results[lag] = {'pval': pval, 'significant': is_significant}
                if is_significant:
                    significant_lags.append(lag)
            except Exception as e:
                # print(f"  Granger test failed for '{col}' at lag {lag}: {e}")
                gc_results[lag] = {'pval': np.nan, 'significant': False, 'error': str(e)}

        results['granger_causality'][col] = {
            'results': gc_results,
            'any_significant': bool(significant_lags),
            'best_lag': min(significant_lags) if significant_lags else None
        }

    # Recommendations
    recommendations = []
    for col in X_aligned.columns:
        corr_score = abs(results['correlation'][col]['pearson'])
        gc_info = results['granger_causality'][col]
        gc_significant = gc_info['any_significant']
        best_gc_lag = gc_info['best_lag']

        if corr_score > 0.3:
            recommendations.append({'variable': col, 'reason': f'Strong Correlation (r={corr_score:.2f})', 'score': corr_score})
        if gc_significant:
            gc_score = 1 - gc_info['results'][best_gc_lag]['pval'] # Score based on p-value
            recommendations.append({'variable': col, 'reason': f'Granger Causes (lag {best_gc_lag})', 'score': gc_score})

    # Deduplicate and sort recommendations
    seen = set()
    unique_recommendations = []
    for rec in sorted(recommendations, key=lambda x: x['score'], reverse=True):
        if rec['variable'] not in seen:
            unique_recommendations.append(rec)
            seen.add(rec['variable'])
    results['recommended_vars'] = unique_recommendations

    if unique_recommendations:
        print("  Recommended variables:")
        for i, rec in enumerate(unique_recommendations):
            print(f"    {i+1}. {rec['variable']} ({rec['reason']})")
    else:
        print("  No variables strongly recommended based on tests.")

    return results

# --- Visualization ---

def plot_covariate_evaluation(eval_results, y, X, show=True):
    """Plot detailed evaluation for each covariate."""
    variables = list(eval_results.get('correlation', {}).keys())
    if not variables: return None

    common_idx = y.index.intersection(X.index)
    if len(common_idx) == 0: return None
    y_aligned = y.loc[common_idx]
    X_aligned = X.loc[common_idx]

    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 3, figsize=(18, 5 * n_vars), squeeze=False)
    fig.suptitle("Covariate Evaluation Analysis", fontsize=16, y=1.0)

    for i, var_name in enumerate(variables):
        ax_gc, ax_scatter, ax_monthly = axes[i, 0], axes[i, 1], axes[i, 2]

        # Granger Causality Plot
        gc_data = eval_results.get('granger_causality', {}).get(var_name, {}).get('results', {})
        if gc_data:
            lags = sorted([lag for lag in gc_data if isinstance(lag, int)])
            pvals = [gc_data[lag]['pval'] for lag in lags]
            significant = [gc_data[lag]['significant'] for lag in lags]
            colors = [DEFAULT_COLOR_PASSED_TEST if sig else DEFAULT_COLOR_FAILED_TEST for sig in significant]
            ax_gc.bar(lags, pvals, color=colors, alpha=0.7)
            ax_gc.axhline(0.05, color='black', linestyle='--', alpha=0.5, label='p=0.05')
            best_lag = eval_results['granger_causality'][var_name].get('best_lag')
            if best_lag is not None:
                 best_idx = lags.index(best_lag)
                 ax_gc.bar([best_lag], [pvals[best_idx]], color=DEFAULT_COLOR_HIGHLIGHT) # Highlight best lag
                 ax_gc.text(0.05, 0.95, f"Best Lag: {best_lag}", transform=ax_gc.transAxes, va='top', bbox=dict(facecolor='lightgreen', alpha=0.8))
            ax_gc.set_yscale('log')
            ax_gc.set_ylim(top=1.0)
            ax_gc.set_xticks(lags)
            ax_gc.set_xticklabels([str(lag) for lag in lags])
            ax_gc.set_title(f'Granger Causality: {var_name} -> Target')
            ax_gc.set_xlabel('Lag')
            ax_gc.set_ylabel('p-value (log)')
            ax_gc.legend()
        else:
            ax_gc.text(0.5, 0.5, "No Granger Results", ha='center', va='center')
        ax_gc.grid(True, axis='y', alpha=0.3)

        # Scatter Plot
        var_data = X_aligned[var_name].values
        target_data = y_aligned.values
        mask = ~np.isnan(var_data) & ~np.isnan(target_data)
        if mask.sum() > 1:
            ax_scatter.scatter(var_data[mask], target_data[mask], alpha=0.5, color=DEFAULT_COLOR_STANDARD, edgecolors='w', linewidth=0.5)
            z = np.polyfit(var_data[mask], target_data[mask], 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(var_data[mask]), max(var_data[mask]), 100)
            ax_scatter.plot(x_range, p(x_range), color=DEFAULT_COLOR_HIGHLIGHT, linewidth=2)
            r_squared = stats.pearsonr(var_data[mask], target_data[mask])[0]**2
            ax_scatter.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax_scatter.transAxes, va='top', bbox=dict(facecolor='white', alpha=0.8))
        ax_scatter.set_title(f'Target vs. {var_name}')
        ax_scatter.set_xlabel(var_name)
        ax_scatter.set_ylabel('Target')
        ax_scatter.grid(True, alpha=0.3)

        # Monthly Time Series Plot
        monthly_target = y_aligned.resample('ME').mean()
        monthly_covar = X_aligned[var_name].resample('ME').mean()
        ax_monthly_twin = ax_monthly.twinx()
        ln1 = ax_monthly.plot(monthly_target.index, monthly_target.values, color=DEFAULT_COLOR_ACTUAL, label='Target (Monthly Avg)')
        ln2 = ax_monthly_twin.plot(monthly_covar.index, monthly_covar.values, color=DEFAULT_COLOR_HIGHLIGHT, alpha=0.7, label=f'{var_name} (Monthly Avg)')
        ax_monthly.set_ylabel('Target Value', color=DEFAULT_COLOR_ACTUAL)
        ax_monthly_twin.set_ylabel(f'{var_name} Value', color=DEFAULT_COLOR_PREDICTED)
        ax_monthly.tick_params(axis='y', labelcolor=DEFAULT_COLOR_ACTUAL)
        ax_monthly_twin.tick_params(axis='y', labelcolor=DEFAULT_COLOR_PREDICTED)
        monthly_corr = monthly_target.corr(monthly_covar)
        ax_monthly.text(0.05, 0.95, f'Monthly Corr: {monthly_corr:.3f}', transform=ax_monthly.transAxes, va='top', bbox=dict(facecolor='white', alpha=0.8))
        ax_monthly.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_monthly.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(monthly_target)//6))) # Adjust interval
        plt.setp(ax_monthly.get_xticklabels(), rotation=45, ha='right')
        ax_monthly.set_title(f'Monthly {var_name} vs. Target')
        ax_monthly.grid(True, alpha=0.3)
        # Combine legends
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax_monthly.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)

    plt.tight_layout(rect=[0, 0.05, 1, 0.98]) # Adjust layout
    if show: plt.show()
    return fig

def plot_model_fit(train_forecast, y_full, test_size, log_transform=True, title="Model Fit Diagnostics", show=True):
    """
    Plot fitted values against actual values to evaluate model fit on training data (6 subplots).
    Adapted from plot_fitted_vs_actuals in the original script.

    Args:
        train_forecast (pd.DataFrame): Forecast results on the training period.
        y_full (pd.Series): The complete original target time series.
        test_size (int): The size of the test set (to identify training data).
        log_transform (bool): Whether log transform was applied (for inverse transform).
        title (str): Plot title.
        show (bool): Whether to display the plot immediately (controlled by pipeline).

    Returns:
        matplotlib.figure.Figure: The figure object containing the plots, or None if error.
    """
    print("Generating detailed model fit plot (6 subplots - original style)...")
    # Get training data
    y_train = y_full.iloc[:-test_size]

    # Get fitted values (predictions for the training period)
    train_dates = y_train.index

    # More reliable date matching approach
    train_dates_str = [d.strftime('%Y-%m-%d') for d in train_dates]
    forecast_dates_str = [d.strftime('%Y-%m-%d') for d in pd.to_datetime(train_forecast['ds'])]

    # Create a mapping between forecast indices and training indices
    matched_indices = []
    forecast_date_map = {date_str: i for i, date_str in enumerate(forecast_dates_str)}
    train_date_map = {date_str: i for i, date_str in enumerate(train_dates_str)}

    common_dates_str = set(forecast_date_map.keys()) & set(train_date_map.keys())

    if not common_dates_str:
        print("ERROR: No matching dates found between forecast and training data for plot_model_fit")
        return None

    for date_str in sorted(list(common_dates_str)):
        forecast_idx = forecast_date_map[date_str]
        train_idx = train_date_map[date_str]
        matched_indices.append((forecast_idx, train_idx))

    print(f"Found {len(matched_indices)} matching dates between forecast and training data for plot_model_fit")

    # Extract matched values
    forecast_indices, train_indices = zip(*matched_indices)
    # Ensure indices are lists for proper slicing
    forecast_indices = list(forecast_indices)
    train_indices = list(train_indices)

    fitted_values = train_forecast.loc[forecast_indices, 'yhat'].values
    actuals = y_train.iloc[train_indices].values

    # Extract matched dates for time series plot
    matched_dates = y_train.index[train_indices] # Use index directly

    # Apply inverse transform if needed
    if log_transform:
        # Inverse transform fitted values
        fitted_values = np.expm1(fitted_values)
        # Actuals are already on the original scale as loaded by load_data

    # Check for and handle NaN/Inf values AFTER potential inverse transform
    valid_mask = ~np.isnan(fitted_values) & ~np.isnan(actuals) & ~np.isinf(fitted_values) & ~np.isinf(actuals)
    if not valid_mask.all():
        print(f"WARNING: Found {len(valid_mask) - valid_mask.sum()} NaN/Inf values. Removing them from fit plot analysis.")
        fitted_values = fitted_values[valid_mask]
        actuals = actuals[valid_mask]
        matched_dates = matched_dates[valid_mask]
        # Also update the indices for trend extraction later
        forecast_indices = [idx for i, idx in enumerate(forecast_indices) if valid_mask[i]]

    # Verify arrays are the same length and not empty
    print(f"Final array lengths for fit plot - Actuals: {len(actuals)}, Fitted: {len(fitted_values)}")
    if len(actuals) == 0 or len(fitted_values) == 0:
        print("ERROR: No valid data points remain after filtering for plot_model_fit")
        return None

    # Calculate residuals and statistics BEFORE creating the plots
    residuals = actuals - fitted_values
    mean = np.mean(residuals)
    std = np.std(residuals)
    skew = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)

    # Create a 2x5 grid of subplots - daily plot will span the entire bottom row
    fig = plt.figure(figsize=(24, 12)) # Keep original size

    # Create a gridspec with 2 rows and 5 columns
    gs = GridSpec(2, 5, height_ratios=[1, 1.5], figure=fig) # Use GridSpec directly

    # Create the six subplots in the specified layout
    ax1 = fig.add_subplot(gs[0, 0])  # Scatter plot (top-left)
    ax4 = fig.add_subplot(gs[0, 1])  # Residuals histogram (top-middle-left)
    ax3 = fig.add_subplot(gs[0, 2])  # Monthly plot (top-middle-right)
    ax5 = fig.add_subplot(gs[0, 3])  # Weekly seasonality (top-right-1)
    ax6 = fig.add_subplot(gs[0, 4])  # Monthly seasonality (top-right-2)
    ax2 = fig.add_subplot(gs[1, :])  # Daily plot (full bottom row)

    fig.suptitle(title, fontsize=16, y=1.02) # Add overall title

    # --- SUBPLOT 1: Scatter plot ---
    ax1.scatter(actuals, fitted_values, s=8, alpha=0.5, color=DEFAULT_COLOR_STANDARD)
    min_val = min(np.min(actuals), np.min(fitted_values)) * 0.98
    max_val = max(np.max(actuals), np.max(fitted_values)) * 1.02
    ax1.plot([min_val, max_val], [min_val, max_val], color=DEFAULT_COLOR_HIGHLIGHT, linewidth=2.5 , alpha=0.8, label='Perfect Fit')
    outlier_threshold = 3 * std
    ax1.plot([min_val, max_val], [min_val + outlier_threshold, max_val + outlier_threshold],
            color=DEFAULT_COLOR_HIGHLIGHT, linestyle='dashed', alpha=0.5, linewidth=0.7, label=f'±{outlier_threshold:.1f} Threshold') # Updated label
    ax1.plot([min_val, max_val], [min_val - outlier_threshold, max_val - outlier_threshold],
            color=DEFAULT_COLOR_HIGHLIGHT, linestyle='dashed', alpha=0.5, linewidth=0.7)
    outlier_mask = np.abs(residuals) > outlier_threshold
    if outlier_mask.any():
        ax1.scatter(actuals[outlier_mask], fitted_values[outlier_mask],
                    s=16, color=DEFAULT_COLOR_HIGHLIGHT, alpha=1.0, marker='o', label=f'Outliers ({sum(outlier_mask)})')
    try:
        correlation = np.corrcoef(actuals, fitted_values)[0, 1]
        r_squared = correlation ** 2 if not np.isnan(correlation) else 0
        mape = calculate_mape(actuals, fitted_values) # Assumes calculate_mape exists
        stats_text = f"R²: {r_squared:.4f}\nMAPE: {mape:.2f}%" if not np.isnan(mape) else f"R²: {r_squared:.4f}"
        ax1.text(0.05, 0.95, stats_text,
                transform=ax1.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    except Exception as e:
        print(f"ERROR calculating scatter plot statistics: {e}")
        ax1.text(0.05, 0.95, "Stats failed", transform=ax1.transAxes, fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Fitted Values')
    ax1.set_title("Actual vs. Fitted (Train)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='lower right')
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)

    # --- SUBPLOT 4: Residuals histogram ---
    n_bins = min(50, max(10, int(len(residuals) / 10)))
    ax4.hist(residuals, bins=n_bins, alpha=0.6, color=DEFAULT_COLOR_STANDARD, edgecolor='black', density=True) # Use density=True
    try:
        x_norm = np.linspace(min(residuals), max(residuals), 100)
        y_norm = stats.norm.pdf(x_norm, mean, std)
        ax4.plot(x_norm, y_norm, color=DEFAULT_COLOR_HIGHLIGHT, alpha=0.8, linewidth=2.5, label='Normal Dist.')
    except Exception as e:
        print(f"Could not plot normal curve overlay: {e}")
    # Diagnostics
    mean_acceptable = abs(mean) < 0.01 * np.mean(np.abs(actuals)) if np.mean(np.abs(actuals)) > 0 else True
    is_normal = (abs(skew) < 0.5) and (abs(kurtosis) < 1)
    try:
        homoscedasticity_corr = np.corrcoef(np.abs(residuals), fitted_values)[0, 1] if len(residuals) > 1 else 0
    except ValueError: # Handle constant arrays
        homoscedasticity_corr = 0
    is_homoscedastic = abs(homoscedasticity_corr) < 0.3
    dw_stat = durbin_watson(residuals) if len(residuals) > 1 else np.nan
    no_autocorr = 1.5 < dw_stat < 2.5 if not np.isnan(dw_stat) else False
    outlier_count = np.sum(outlier_mask)
    no_outliers = outlier_count < 0.01 * len(residuals)
    diagnostics_text = (
        f"RESIDUAL DIAGNOSTICS:\n"
        f"Zero mean? {'✓' if mean_acceptable else '✗'}\n"
        f"Normal distr.? {'✓' if is_normal else '✗'} (sk={skew:.2f}, ku={kurtosis:.2f})\n"
        f"Homoscedastic? {'✓' if is_homoscedastic else '✗'} (corr={homoscedasticity_corr:.2f})\n"
        f"No autocorrel.? {'✓' if no_autocorr else '✗'} (DW={dw_stat:.2f})\n"
        f"No outliers? {'✓' if no_outliers else '✗'} ({outlier_count} pts)"
    )
    ax4.text(0.02, 0.97, diagnostics_text,
            transform=ax4.transAxes, fontsize=8, verticalalignment='top', # Smaller font
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax4.set_xlabel('Residual (Actual - Fitted)')
    ax4.set_ylabel('Density')
    ax4.set_title('Residuals Distribution')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axvline(x=0, color=DEFAULT_COLOR_HIGHLIGHT, linestyle='--', alpha=0.7)
    ax4.legend(fontsize=8)

    # --- SUBPLOT 3: Monthly aggregation ---
    ts_df = pd.DataFrame({'actual': actuals, 'fitted': fitted_values}, index=matched_dates)
    # Add trend if available in train_forecast
    if 'trend' in train_forecast.columns:
        trend_values = train_forecast.loc[forecast_indices, 'trend'].values
        if log_transform:
            trend_values = np.expm1(trend_values)
        # Align trend with potentially filtered actuals/fitted
        if len(trend_values) == len(ts_df):
             ts_df['trend'] = trend_values
        else: # If trend length doesn't match after filtering NaNs
             print("Warning: Trend length mismatch after filtering, cannot plot trend in monthly aggregation.")


    monthly_df = ts_df.resample('ME').mean()
    ax3.plot(monthly_df.index, monthly_df['actual'], color=DEFAULT_COLOR_ACTUAL, linewidth=2, alpha=1.0, label='Actual (Avg)')
    ax3.plot(monthly_df.index, monthly_df['fitted'], color=DEFAULT_COLOR_PREDICTED, linewidth=2, alpha=0.7, label='Fitted (Avg)')
    if 'trend' in monthly_df.columns:
        ax3.plot(monthly_df.index, monthly_df['trend'], color=DEFAULT_COLOR_COMPONENT, linewidth=3, alpha=1.0, linestyle='-', label='Trend (Avg)')
    try:
        monthly_corr = np.corrcoef(monthly_df['actual'].dropna().values, monthly_df['fitted'].dropna().values)[0, 1]
        monthly_r2 = monthly_corr ** 2 if not np.isnan(monthly_corr) else 0
        monthly_mape = calculate_mape(monthly_df['actual'].dropna().values, monthly_df['fitted'].dropna().values)
        monthly_stats = f"Monthly R²: {monthly_r2:.4f}\nMonthly MAPE: {monthly_mape:.2f}%"
        ax3.text(0.05, 0.95, monthly_stats,
                transform=ax3.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    except Exception as e:
        print(f"Error calculating monthly statistics: {e}")
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Monthly Average Value')
    ax3.set_title("Monthly Aggregation (Train)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) # Format date axis
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7)) # Auto ticks

    # --- SUBPLOT 5: Weekly seasonality ---
    seasonal_df = pd.DataFrame({'date': matched_dates, 'actual': actuals})
    seasonal_df['day_of_week'] = seasonal_df['date'].dt.dayofweek
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    overall_mean_actual = np.mean(actuals) if len(actuals) > 0 else 1.0
    if overall_mean_actual == 0: overall_mean_actual = 1.0 # Avoid division by zero

    dow_grouped = seasonal_df.groupby('day_of_week')['actual'].agg(['mean', 'std']).reindex(range(7)) # Ensure all days are present
    dow_grouped['normalized_mean'] = dow_grouped['mean'] / overall_mean_actual
    dow_grouped['normalized_std'] = dow_grouped['std'] / overall_mean_actual

    x_dow = np.arange(len(day_names))
    ax5.plot(x_dow, dow_grouped['normalized_mean'], 'o-', color=DEFAULT_COLOR_COMPONENT, linewidth=3, markersize=8, label='Actual (% of avg)')
    upper_bound_dow = dow_grouped['normalized_mean'] + dow_grouped['normalized_std']
    lower_bound_dow = dow_grouped['normalized_mean'] - dow_grouped['normalized_std']
    ax5.fill_between(x_dow, lower_bound_dow.fillna(dow_grouped['normalized_mean']), upper_bound_dow.fillna(dow_grouped['normalized_mean']), color=DEFAULT_COLOR_COMPONENT, alpha=0.2, label='±1σ per day') # Handle NaNs in std
    ax5.axhline(y=1.0, color=DEFAULT_COLOR_STANDARD, linestyle='--', alpha=0.7, label='Overall Average')
    ax5.set_xticks(x_dow)
    ax5.set_xticklabels(day_names, rotation=45)
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax5.set_xlabel('Day of Week')
    ax5.set_ylabel('Relative to Average')
    ax5.set_title('Weekly Pattern (Actuals)')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.legend(fontsize=8)
    ax5.set_ylim(bottom=max(0, lower_bound_dow.min() * 0.9 if not lower_bound_dow.isnull().all() else 0)) # Adjust y-lim

    # --- SUBPLOT 6: Monthly seasonality ---
    seasonal_df['month'] = seasonal_df['date'].dt.month
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_grouped = seasonal_df.groupby('month')['actual'].agg(['mean', 'std']).reindex(range(1, 13)) # Ensure all months
    month_grouped['normalized_mean'] = month_grouped['mean'] / overall_mean_actual
    month_grouped['normalized_std'] = month_grouped['std'] / overall_mean_actual

    x_month = np.arange(len(month_names))
    ax6.plot(x_month, month_grouped['normalized_mean'], 'o-', color=DEFAULT_COLOR_COMPONENT, linewidth=3, markersize=8, label='Actual (% of avg)')
    upper_bound_month = month_grouped['normalized_mean'] + month_grouped['normalized_std']
    lower_bound_month = month_grouped['normalized_mean'] - month_grouped['normalized_std']
    ax6.fill_between(x_month, lower_bound_month.fillna(month_grouped['normalized_mean']), upper_bound_month.fillna(month_grouped['normalized_mean']), color=DEFAULT_COLOR_COMPONENT, alpha=0.2, label='±1σ per month') # Handle NaNs
    ax6.axhline(y=1.0, color=DEFAULT_COLOR_STANDARD, linestyle='--', alpha=0.7, label='Overall Average')
    ax6.set_xticks(x_month)
    ax6.set_xticklabels(month_names, rotation=45)
    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax6.set_xlabel('Month')
    ax6.set_ylabel('Relative to Average')
    ax6.set_title('Monthly Pattern (Actuals)')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.legend(fontsize=8)
    ax6.set_ylim(bottom=max(0, lower_bound_month.min() * 0.9 if not lower_bound_month.isnull().all() else 0)) # Adjust y-lim

    # --- SUBPLOT 2: Daily time series ---
    ax2.plot(matched_dates, actuals, color=DEFAULT_COLOR_ACTUAL, alpha=1.0, label='Actual Values', linewidth=1.0)
    ax2.plot(matched_dates, fitted_values, color=DEFAULT_COLOR_PREDICTED, alpha=0.7, label='Fitted Values', linewidth=1.5)
    # Plot trend if available and aligned
    if 'trend' in ts_df.columns:
        ax2.plot(matched_dates, ts_df['trend'].values, color=DEFAULT_COLOR_COMPONENT, linewidth=2, alpha=0.7, linestyle='--', label='Trend')

    if outlier_mask.any():
        outlier_dates = matched_dates[outlier_mask]
        outlier_actuals = actuals[outlier_mask]
        ax2.scatter(outlier_dates, outlier_actuals, s=200, color=DEFAULT_COLOR_HIGHLIGHT, alpha=1.0,
                    marker='o', edgecolors='white', linewidths=1.0,
                    label=f'Outliers ({sum(outlier_mask)})')
        # Add annotations (optional, can be slow for many outliers)
        for date, value in zip(outlier_dates, outlier_actuals):
            date_str = date.strftime('%Y-%m-%d')
            y_pos = value * 1.04
            ax2.annotate(date_str, (date, y_pos), textcoords="offset points", xytext=(0, 5),
                         ha='center', fontsize=12, rotation=45,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="orange", alpha=0.7))

    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value')
    ax2.set_title("Daily Actual vs. Fitted (Train)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    # fig.autofmt_xdate(ax=ax2) # Incorrect: Figure.autofmt_xdate doesn't take 'ax'
    # Apply rotation directly to the specific axis' labels
    plt.setp(ax2.get_xticklabels(), rotation=30, ha='right') # Rotate labels for ax2

    # --- Final Adjustments ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title/label overlap

    if show:
        plt.show() # Keep show for direct calls, pipeline will use show=False

    return fig

def plot_forecast_results(forecast, y_full, test_size, log_transform=True,
                          future_forecast=None, history_days=70, title="Prophet Forecast", show=True):
    """Plot historical data, test actuals, predictions, and future forecast."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Data slicing
    train_dates = y_full.index[:-test_size]
    test_dates = y_full.index[-test_size:]
    train_values = y_full.values[:-test_size]
    test_values = y_full.values[-test_size:]

    # Limit history shown
    history_cutoff = train_dates[-1] - timedelta(days=history_days) if len(train_dates) > 0 else None
    hist_mask = train_dates >= history_cutoff if history_cutoff else slice(None)
    hist_dates, hist_values = train_dates[hist_mask], train_values[hist_mask]

    # Ensure proper inverse transform for all forecast values
    fcst_dates = pd.to_datetime(forecast['ds'])
    
    # Find which indices in the forecast correspond to test dates
    test_dates_str = [d.strftime('%Y-%m-%d') for d in test_dates]
    forecast_dates_str = [d.strftime('%Y-%m-%d') for d in fcst_dates]
    test_mask = [d in test_dates_str for d in forecast_dates_str]
    
    # Apply inverse transform for test period forecast
    yhat = np.expm1(forecast['yhat'].values) if log_transform else forecast['yhat'].values
    
    # Plotting
    ax.plot(hist_dates, hist_values, color=DEFAULT_COLOR_ACTUAL, lw=0.7, label='Trainning Actuals')
    ax.plot(test_dates, test_values, color=DEFAULT_COLOR_ACTUAL, lw=2, label='Test Actuals')
    
    # Plot test period predictions
    if any(test_mask):
        test_fcst_dates = fcst_dates[test_mask]
        test_yhat = yhat[test_mask]
        ax.plot(test_fcst_dates, test_yhat, color=DEFAULT_COLOR_PREDICTED, lw=2, alpha=0.7, label='Test Prediction')

    # Plot future forecast if available - handle the inverse transform carefully
    if future_forecast is not None:
        future_dates = pd.to_datetime(future_forecast['ds'])
        
        # IMPORTANT: Apply inverse transform to future predictions 
        future_yhat = np.expm1(future_forecast['yhat'].values) if log_transform else future_forecast['yhat'].values
        future_lower = np.expm1(future_forecast['yhat_lower'].values) if log_transform else future_forecast['yhat_lower'].values
        future_upper = np.expm1(future_forecast['yhat_upper'].values) if log_transform else future_forecast['yhat_upper'].values
        
        # Only plot future dates that come after the test period
        last_test_date = test_dates[-1]
        future_mask = [d > last_test_date for d in future_dates]
        
        if any(future_mask):
            future_plot_dates = future_dates[future_mask]
            future_plot_yhat = future_yhat[future_mask]
            future_plot_lower = future_lower[future_mask]
            future_plot_upper = future_upper[future_mask]
            
            ax.plot(future_plot_dates, future_plot_yhat, color=DEFAULT_COLOR_FORECAST, lw=2, label='Forecast')
            ax.fill_between(future_plot_dates, future_plot_lower, future_plot_upper, color=DEFAULT_COLOR_FORECAST, alpha=0.15)

    # Formatting
    ax.axvline(train_dates[-1], color=DEFAULT_COLOR_PREDICTED, linestyle=':', lw=1, label='Train/Test Split')
    if future_forecast is not None:
        ax.axvline(test_dates[-1], color=DEFAULT_COLOR_FORECAST, linestyle=':', lw=1, label='Forecast Start')

    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if history_cutoff: ax.set_xlim(left=history_cutoff)
    fig.autofmt_xdate()
    plt.tight_layout()
    if show: plt.show()
    return fig


def plot_model_components(model, forecast, show=True):
    """Plot Prophet model components."""
    print("Plotting model components...")
    try:
        fig = model.plot_components(forecast)
        plt.tight_layout()
        if show: plt.show()
        return fig
    except Exception as e:
        print(f"  ERROR plotting components: {e}")
        return None

# --- Exporting ---

def export_artifacts(results, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Export model, forecast data, and plots.

    Args:
        results (dict): Dictionary containing pipeline results.
        output_dir (str): Directory to save artifacts.

    Returns:
        dict: Dictionary of paths to the exported files.
    """
    print(f"\nExporting artifacts to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(DEFAULT_PLOTS_DIR, exist_ok=True)
    os.makedirs(DEFAULT_FORECASTS_DIR, exist_ok=True)
    os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {}

    # Model
    try:
        model_path = os.path.join(DEFAULT_MODELS_DIR, f'prophet_model_{timestamp}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(results['model'], f)
        paths['model'] = model_path
        print(f"  Model saved: {model_path}")
    except Exception as e:
        print(f"  ERROR saving model: {e}")

    # Forecast Data
    try:
        forecast_path = os.path.join(DEFAULT_FORECASTS_DIR, f'forecast_test_{timestamp}.csv')
        results['forecast_test'].to_csv(forecast_path, index=False)
        paths['forecast_test'] = forecast_path
        print(f"  Test forecast saved: {forecast_path}")
        forecast_path = os.path.join(DEFAULT_FORECASTS_DIR, f'forecast_train_{timestamp}.csv')
        results['forecast_train'].to_csv(forecast_path, index=False)
        paths['forecast_train'] = forecast_path
        print(f"  Train forecast saved: {forecast_path}")
        if 'forecast_future' in results and results['forecast_future'] is not None:
             future_forecast_path = os.path.join(DEFAULT_FORECASTS_DIR, f'forecast_future_{timestamp}.csv')
             results['forecast_future'].to_csv(future_forecast_path, index=False)
             paths['forecast_future'] = future_forecast_path
             print(f"  Future forecast saved: {future_forecast_path}")
    except Exception as e:
        print(f"  ERROR saving forecast data: {e}")

    # Plots
    plot_keys = {
        'fig_model_fit': 'model_fit',
        'fig_components': 'components',
        'fig_forecast': 'forecast',
        'fig_regressor_importance': 'regressor_importance',
        'fig_covariate_evaluation': 'covariate_evaluation'
    }
    for key, name in plot_keys.items():
        if key in results and results[key] is not None:
            try:
                plot_path = os.path.join(DEFAULT_PLOTS_DIR, f'{name}_{timestamp}.png')
                results[key].savefig(plot_path)
                plt.close(results[key]) # Close figure after saving
                paths[key] = plot_path
                print(f"  Plot saved: {plot_path}")
            except Exception as e:
                print(f"  ERROR saving plot '{name}': {e}")

    print("Artifact export complete.")
    return paths

# --- Main Pipeline Orchestrator ---

def run_prophet_pipeline(
    target_path,
    covariate_path=None,
    covariate_cols=None,
    date_col=DEFAULT_DATE_COL,
    target_col=DEFAULT_TARGET_COL,
    test_size=35,
    future_periods=35,
    lags=None, # Default to None, add if needed
    log_transform=True,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    country_code_holidays=None, # Use None to disable Prophet holidays, 'BR' for Brazil dummies
    custom_date_dummies=None,
    perform_covariate_evaluation=True,
    max_lag_for_evaluation=70,
    output_dir=DEFAULT_OUTPUT_DIR
):
    """
    Execute the full Prophet forecasting pipeline.

    Args:
        target_path (str): Path to target data CSV.
        covariate_path (str, optional): Path to covariate data CSV.
        covariate_cols (list, optional): List of covariate columns to use.
        date_col (str): Name of the date column.
        target_col (str): Name of the target column.
        test_size (int): Number of periods for the test set.
        future_periods (int): Number of periods to forecast into the future.
        lags (list, optional): Lag values to generate for covariates.
        log_transform (bool): Apply log transform to target and covariates.
        yearly_seasonality (bool): Enable yearly seasonality.
        weekly_seasonality (bool): Enable weekly seasonality.
        daily_seasonality (bool): Enable daily seasonality.
        country_code_holidays (str, optional): Set to 'BR' to add Brazilian holiday dummies.
                                              Set to other country codes to use Prophet's built-in (not recommended with dummies).
                                              Set to None to disable holiday features.
        custom_date_dummies (list, optional): List of custom date dummy specifications.
        perform_covariate_evaluation (bool): Perform covariate evaluation. # Updated docstring
        max_lag_for_evaluation (int): Max lag for Granger causality tests.
        output_dir (str): Directory to save results.

    Returns:
        dict: Dictionary containing results (model, forecasts, metrics, figures).
    """
    print("\n--- Starting Prophet Forecasting Pipeline ---")
    results = {}
    lags = lags or [] # Ensure lags is a list

    # 1. Load Data
    y, X = load_data(target_path, covariate_path, date_col, target_col, covariate_cols)
    if y is None: return {"error": "Failed to load target data."}
    results['y_full'] = y
    transformable_cols = [col for col in X.columns if 'dummy' not in col.lower() and 'calendar' not in col.lower()]

    
    # 2. Evaluate Covariates (Optional)
    results['fig_covariate_evaluation'] = None
    # Use the renamed parameter in the condition
    if perform_covariate_evaluation and X is not None:
        # Call the actual function evaluate_covariates
        results['covariate_evaluation'] = evaluate_covariates(y, X, max_lag=max_lag_for_evaluation)
        # Check if evaluation produced results before plotting
        if results['covariate_evaluation'] and results['covariate_evaluation'].get('correlation'):
             results['fig_covariate_evaluation'] = plot_covariate_evaluation(results['covariate_evaluation'], y, X, show=False)
        else:
             print("  Skipping covariate evaluation plot due to lack of results.")

    # 3. Prepare Date Dummies (Holidays + Custom)
    all_date_dummies = custom_date_dummies or []
    if country_code_holidays == 'BR':
        print("\nGenerating Brazilian holiday dummies...")
        br_holiday_specs = create_brazil_holiday_dummies_spec(y.index.min(), y.index.max())
        all_date_dummies.extend(br_holiday_specs)
    results['date_dummies'] = all_date_dummies

    # 4. Prepare Prophet Input Data (Transformations, Features)
    transformed_y_df = prepare_prophet_input_y(y, log_transform)
    transformed_X_df, transformed_X_cols = prepare_prophet_input_X(X, lags, log_transform, all_date_dummies)
    #  transformed_df, transformed_cols = prepare_prophet_input(y, X, lags, log_transform, all_date_dummies)
    original_covariates_cols = [col for col in X.columns if 'dummy' not in col and 'calendar' not in col]
    regressor_cols = [col for col in transformed_X_cols if col not in original_covariates_cols]

    prophet_df = transformed_X_df.copy()
    prophet_df['y'] = transformed_y_df['y']
    prophet_df = prophet_df.reset_index(drop=False) # Reset index to avoid issues with Prophet
    prophet_df = prophet_df[['ds', 'y'] + regressor_cols]
    prophet_df = prophet_df.dropna() # Drop rows with NaN values
    results['regressor_cols'] = regressor_cols

    # 5. Split Data
    train_df, test_df = split_data(prophet_df, test_size)

    # 6. Train Model
    model = train_model(train_df, regressor_cols, yearly_seasonality, weekly_seasonality, daily_seasonality)
    results['model'] = model
    
    
    # 7. Make Predictions (Test Period)
    # Ensure test_df has all necessary regressor columns
    test_future_df = test_df[['ds'] + regressor_cols].copy()
    forecast_test = make_predictions(model, test_future_df)
    results['forecast_test'] = forecast_test

    # 8. Make Predictions (Training Period - for fit plot)
    train_future_df = train_df[['ds'] + regressor_cols].copy()
    forecast_train = make_predictions(model, train_future_df)
    results['forecast_train'] = forecast_train

    # 9. Make Predictions (Future Period)
    results['forecast_future'] = None
    if future_periods > 0:
        last_hist_date = prophet_df['ds'].iloc[-1]
        # Pass historical data including regressors for lag calculation
        hist_data_for_future = transformed_X_df[['ds'] + transformed_X_cols].copy()
        future_df = generate_future_df(last_hist_date, future_periods, regressor_cols, hist_data_for_future, all_date_dummies)
        if not future_df.empty:
             results['forecast_future'] = make_predictions(model, future_df)

    # 10. Evaluate Model
    # Inverse transform predictions for evaluation
    y_test_actual_orig = y.iloc[-test_size:] # Original scale
    y_pred_test = forecast_test['yhat'].values
    if log_transform:
        y_pred_test = np.expm1(y_pred_test)

    results['mape'] = calculate_mape(y_test_actual_orig.values, y_pred_test)
    print(f"\nEvaluation on Test Set: MAPE = {results['mape']:.2f}%")
    
    # 11. Generate Plots
    print("\nGenerating plots...")
    results['fig_model_fit'] = plot_model_fit(
        train_forecast=results['forecast_train'], # Use forecast on training data
        y_full=results['y_full'], # Full actuals needed for train/test split inside
        test_size=test_size,
        log_transform=log_transform,
        title=f"Model Fit Diagnostics (Train Set)", # Add specific title
        show=False # Pipeline controls showing/saving
    )
    results['fig_components'] = plot_model_components(model, forecast_test, show=False) # Components based on test forecast
    results['fig_forecast'] = plot_forecast_results(
        forecast_test, y, test_size, log_transform,
        future_forecast=results['forecast_future'],
        title=f"Prophet Forecast (Test MAPE: {results['mape']:.2f}%)",
        show=False
    )
    
    # 12. Export Artifacts
    export_artifacts(results)

    print("\n--- Prophet Forecasting Pipeline Finished ---")
    return results


# --- Example Usage ---
if __name__ == "__main__":
    # Define file paths and parameters
    TARGET_PATH = 'data/groupby_train.csv'
    COVARIATE_PATH = 'data/groupby_transactions_2.csv'

    COVARIATE_COLS_TO_USE = [
        'transactions',
        # 'calendar_is_weekend',
        # Add other covariates as needed
    ]

    LAGS_TO_GENERATE = [0, 1, 7, 14, 28]

    CUSTOM_DATE_DUMMIES = [
        {
            'name': 'july_15_promo', # Descriptive name
            'month': 7,
            'day': 15,
            'window_before': 2, # Effect starts 2 days before
            'window_after': 1  # Effect lasts 1 day after
        },
        # Add more custom events if needed
    ]

    # Run the pipeline
    pipeline_results = run_prophet_pipeline(
        target_path=TARGET_PATH,
        covariate_path=COVARIATE_PATH,
        covariate_cols=COVARIATE_COLS_TO_USE,
        test_size=35,
        future_periods=35,
        lags=LAGS_TO_GENERATE,
        log_transform=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False, # Usually False for daily data unless strong intra-day pattern
        country_code_holidays='BR', # Use 'BR' for Brazilian holiday dummies, None to disable
        custom_date_dummies=CUSTOM_DATE_DUMMIES,
        perform_covariate_evaluation=True,
        max_lag_for_evaluation=70
    )

    # Optional: Access specific results if needed
    # model = pipeline_results.get('model')
    # mape = pipeline_results.get('mape')
    # print(f"Final MAPE: {mape}") 