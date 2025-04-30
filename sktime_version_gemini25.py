"""
Core implementation for time series forecasting using Prophet with sktime
for hyperparameter tuning and direct Prophet for final modeling and forecasting.
Focuses on the essential logic without plotting.
"""

import logging
import sys
import pandas as pd
import numpy as np
from holidays import country_holidays

import matplotlib.pyplot as plt

# sktime components
from sktime.forecasting.fbprophet import Prophet as SktimeProphet
from sktime.transformations.series.boxcox import LogTransformer
from sktime.split import SlidingWindowSplitter, temporal_train_test_split
from sktime.transformations.compose import OptionalPassthrough
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import TransformedTargetForecaster

# Direct Prophet
from prophet import Prophet as ProphetDirect


# --- Utility Functions ---

def load_series(csv_path, time_col='date', value_col='sales', name='y', freq='D', epsilon=1e-6):
    """
    Loads and prepares time series data from a CSV file.
    Interpolates missing values and replaces zeros with a small epsilon.
    """
    try:
        df = pd.read_csv(csv_path, usecols=[time_col, value_col], index_col=time_col, parse_dates=[time_col])
        # Ensure unique index by grouping and summing, then set frequency
        df = df.groupby(level=0).sum()
        series = df[value_col].asfreq(freq)
        series.name = name

        # --- Handle Missing Values ---
        original_na_count = series.isna().sum()
        if original_na_count > 0:
            print(f"Found {original_na_count} missing values in '{name}' after initial load. Interpolating...")
            series = series.interpolate(method='time')
            remaining_na = series.isna().sum()
            if remaining_na > 0:
                 print(f"Warning: {remaining_na} missing values remain in '{name}' after interpolation. Filling with 0.")
                 series = series.fillna(0) # Fill any remaining NAs (e.g., at start/end)

        # --- Handle Zero Values ---
        zero_mask = (series == 0)
        num_zeros = zero_mask.sum()
        if num_zeros > 0:
            print(f"Found {num_zeros} zero values in '{name}'. Replacing with epsilon ({epsilon}).")
            # Replace zeros with the small epsilon value
            series = series.mask(zero_mask, epsilon)
            # Alternative using np.where: series = pd.Series(np.where(series == 0, epsilon, series), index=series.index, name=series.name)

        print(f"Loaded series '{name}' from {csv_path}. Shape: {series.shape}")
        # Verify no zeros remain after replacement
        final_zero_count = (series == 0).sum()
        if final_zero_count > 0:
             print(f"ERROR: Zeros still present in '{name}' after replacement attempt!")
        # Verify no non-positive values remain if epsilon is positive
        non_positive_count = (series <= 0).sum()
        if non_positive_count > 0:
             print(f"Warning: {non_positive_count} non-positive values found in '{name}' after zero replacement (check epsilon).")


        return series
    except Exception as e:
        print(f"Error loading series from {csv_path}: {e}")
        raise

def holidays_features(data_index, country='BR', future_years=5):
    """Creates a DataFrame of holidays for Prophet."""
    min_year = data_index.year.min()
    max_year = data_index.year.max()
    years_range = list(range(min_year, max_year + future_years))
    
    try:
        country_specific_holidays = country_holidays(country, years=years_range)
        if not country_specific_holidays:
            print(f"No holidays found for country '{country}'. Returning empty DataFrame.")
            return pd.DataFrame(columns=['ds', 'holiday', 'lower_window', 'upper_window'])
            
        holidays_df = pd.DataFrame({
            'ds': pd.to_datetime(list(country_specific_holidays.keys())),
            'holiday': list(country_specific_holidays.values())
        })
        holidays_df['lower_window'] = 0 # Adjust window as needed, e.g., -1 for day before
        holidays_df['upper_window'] = 0 # Adjust window as needed, e.g., 1 for day after
        return holidays_df
    except KeyError:
        print(f"Warning: Country code '{country}' not recognized by holidays library. No holidays added.")
        return pd.DataFrame(columns=['ds', 'holiday', 'lower_window', 'upper_window'])
    except Exception as e:
        print(f"Error generating holidays: {e}")
        return pd.DataFrame(columns=['ds', 'holiday', 'lower_window', 'upper_window'])

def mape_metric(y_true, y_pred, month_transform=True):
    if month_transform:
        y_true = y_true.groupby(y_true.index.month).sum()
        y_pred = y_pred.groupby(y_pred.index.month).sum()
    return np.mean(np.abs((y_true - y_pred) / y_true) * 100)

# --- Feature Engineering ---

def preprocess_covariates(X_train, X_test, lags=[1, 7, 14], deseasonalize=True):
    """Preprocesses covariates by adding lags and optionally deseasonalizing."""
    if X_train is None or X_test is None:
        return None, None
        
    max_lag = max(lags) if lags else 0
    if len(X_train) <= max_lag:
        raise ValueError(f"Training data length ({len(X_train)}) must be greater than max lag ({max_lag})")

    X_train_df = pd.DataFrame(X_train).copy()
    X_test_df = pd.DataFrame(X_test).copy()
    col_name = X_train.name if X_train.name else 'covariate' # Handle unnamed series

    # Combine for consistent processing
    all_data = pd.concat([X_train_df, X_test_df])
    
    processed_features = pd.DataFrame(index=all_data.index)

    if deseasonalize:
        try:
            # Simple deseasonalization: subtract weekly moving average
            # Use min_periods to handle start/end NAs from rolling
            weekly_ma = all_data[col_name].rolling(window=7, center=True, min_periods=1).mean()
            deseasonalized_col = all_data[col_name] - weekly_ma
            processed_features[f'{col_name}_deseasonalized'] = deseasonalized_col
        except Exception as e:
            print(f"Warning: Could not deseasonalize covariate. Error: {e}")
            # Fallback: use original column if deseasonalization fails
            processed_features[col_name] = all_data[col_name]
    else:
         processed_features[col_name] = all_data[col_name] # Use original if not deseasonalizing

    # Add lagged features based on the original column
    for lag in lags:
        processed_features[f'{col_name}_lag_{lag}'] = all_data[col_name].shift(lag)

    # Split back and handle NaNs introduced by shifting/rolling
    # Use backward fill first, then forward fill for robustness
    X_train_processed = processed_features.loc[X_train.index].bfill().ffill()
    X_test_processed = processed_features.loc[X_test.index].bfill().ffill() # Apply to test set too

    # Ensure no NaNs remain in train set after filling
    if X_train_processed.isnull().any().any():
         print("Warning: NaNs remain in processed training covariates after filling.")
         # Option: fill remaining with 0 or mean, or raise error
         X_train_processed = X_train_processed.fillna(0) 
         
    # Ensure test set has same columns (important if preprocessing created different features)
    X_test_processed = X_test_processed.reindex(columns=X_train_processed.columns).bfill().ffill().fillna(0)


    return X_train_processed, X_test_processed

# --- sktime Grid Search for Parameter Finding ---

def find_best_prophet_params(y_train, X_train, cv_splitter, param_grid, country_code='BR'):
    """Uses sktime's GridSearchCV to find the best Prophet parameters."""
    print("Starting sktime grid search for Prophet parameters...")
    
    # Base forecaster with holidays
    base_prophet = SktimeProphet(
        holidays=holidays_features(y_train.index, country=country_code),
        add_country_holidays=False # Holidays are provided manually
    )

    # Pipeline with optional log transform and the Prophet forecaster
    pipe = TransformedTargetForecaster(
        [
            ("ln", OptionalPassthrough(LogTransformer(), passthrough=True)), # Default to not applying log
            ("forecaster", base_prophet),
        ]
    )

    # Grid search setup
    gscv = ForecastingGridSearchCV(
        forecaster=pipe,
        param_grid=param_grid,
        cv=cv_splitter,
        scoring=mape_metric, # Use custom MAPE
        # n_jobs=-1, # Removed: n_jobs is deprecated in sktime >= 0.27.0
        backend="loky", # Use loky backend for parallel processing (requires joblib)
        backend_params={"n_jobs": -1}, # Pass n_jobs=-1 to the backend
        verbose=1 # Show progress
    )

    try:
        gscv.fit(y=y_train, X=X_train)
        print(f"Grid search completed. Best score (MAPE): {gscv.best_score_:.2f}%")
        print(f"Best parameters found: {gscv.best_params_}")
        return gscv.best_params_
    except Exception as e:
        print(f"Error during grid search: {e}")
        # Return default parameters or raise error
        return { # Sensible defaults
             'ln__passthrough': True, 
             'forecaster__seasonality_mode': 'additive'
        } 


# --- Direct Prophet Training ---

def train_direct_prophet(y_train, X_train, best_params,
                         covariate_lags=[1, 7, 14], deseasonalize_covariates=True,
                         country_code='BR'):
    """Trains a direct Prophet model using best parameters and processed covariates."""
    print("Training direct Prophet model with best parameters...")

    # 1. Preprocess Training Covariates ONLY
    X_train_proc = None
    if X_train is not None:
        print("Processing training covariates...")
        try:
            # --- Start inline preprocessing for training data ---
            # This section replicates the logic of preprocess_covariates but only for X_train
            X_train_df = pd.DataFrame(X_train).copy()
            col_name = X_train.name if X_train.name else 'covariate'
            processed_features = pd.DataFrame(index=X_train_df.index)
            base_col_to_process = X_train_df[col_name] # Start with the original column

            if deseasonalize_covariates:
                try:
                    # Calculate rolling mean on the training data only
                    weekly_ma = base_col_to_process.rolling(window=7, center=True, min_periods=1).mean()
                    deseasonalized_col = base_col_to_process - weekly_ma
                    # Add the deseasonalized feature if calculation succeeded
                    processed_features[f'{col_name}_deseasonalized'] = deseasonalized_col
                    # Use the deseasonalized column for lag creation if successful
                    base_col_to_process = deseasonalized_col 
                except Exception as e:
                    print(f"Warning: Could not deseasonalize training covariate. Error: {e}")
                    # If deseasonalization fails, add the original column
                    processed_features[col_name] = base_col_to_process
            else:
                 # If not deseasonalizing, add the original column name
                 processed_features[col_name] = base_col_to_process

            # Add lagged features based on the original training column values
            for lag in covariate_lags:
                processed_features[f'{col_name}_lag_{lag}'] = X_train_df[col_name].shift(lag)

            # Handle NaNs introduced by shifting/rolling using bfill then ffill
            X_train_proc = processed_features.bfill().ffill()
            
            # Final check for any remaining NaNs (e.g., at the very beginning)
            if X_train_proc.isnull().any().any():
                print("Warning: NaNs remain in processed training covariates after filling. Filling with 0.")
                X_train_proc = X_train_proc.fillna(0) # Fill remaining NaNs with 0
            # --- End inline preprocessing ---
            
            print(f"Processed training covariates. Shape: {X_train_proc.shape}")
            
            # --- Crucial Check ---
            if len(X_train_proc) != len(y_train):
                 raise ValueError(f"Length mismatch after processing covariates: X_train_proc ({len(X_train_proc)}) vs y_train ({len(y_train)})")

        except Exception as e:
            print(f"Error processing training covariates: {e}")
            X_train_proc = None # Ensure it's None if processing fails

    # 2. Prepare Training Data for Prophet
    df_train = pd.DataFrame({'ds': y_train.index, 'y': y_train.values})

    apply_log_transform = not best_params.get('ln__passthrough', True) # passthrough=True means NO log
    if apply_log_transform:
        print("Applying log transform to training data.")
        df_train['y'] = np.log1p(df_train['y'])

    # 3. Initialize Prophet Model
    prophet_params = {
        key.replace('forecaster__', ''): value
        for key, value in best_params.items()
        if key.startswith('forecaster__')
    }
    prophet_params.setdefault('seasonality_mode', 'additive')

    model = ProphetDirect(
        holidays=holidays_features(y_train.index, country=country_code),
        **prophet_params
    )
    model.log_transform_applied = apply_log_transform

    # 4. Add Regressors
    added_regressors = []
    if X_train_proc is not None:
        print("Adding regressors to the model...")
        for col in X_train_proc.columns:
            # Check if column contains non-numeric or infinite values before adding
            if pd.api.types.is_numeric_dtype(X_train_proc[col]) and \
               np.all(np.isfinite(X_train_proc[col])):
                try:
                    model.add_regressor(col)
                    df_train[col] = X_train_proc[col].values # Assign values to df_train
                    added_regressors.append(col)
                except Exception as e:
                     print(f"Warning: Could not add regressor '{col}'. Error: {e}")
            else:
                print(f"Warning: Skipping regressor '{col}' due to non-numeric or infinite values.")
        
        if not added_regressors:
             print("Warning: No valid regressors were added to the model.")


    # 5. Fit Model
    try:
        # Ensure df_train contains all columns needed (y and added regressors)
        model.fit(df_train)
        print("Direct Prophet model fitted successfully.")
        # Return the model and the actual columns used as regressors
        return model, added_regressors 
    except Exception as e:
        # Provide more context if fit fails due to missing regressors
        if "missing from dataframe" in str(e):
             print(f"Error fitting direct Prophet model: {e}. This often happens if regressors were not added correctly.")
             print(f"Columns in df_train: {df_train.columns.tolist()}")
             print(f"Regressors expected by model: {model.extra_regressors.keys()}")
        else:
             print(f"Error fitting direct Prophet model: {e}")
        return None, [] # Return None model and empty list of regressors

# --- Future Covariate Forecasting & Processing ---
# Modify this function to accept the list of regressors actually used in training
def forecast_process_future_covariates(X_history, future_dates, trained_regressors,
                                       lags=[1, 7, 14], deseasonalize=True):
    """Forecasts future covariate values and applies consistent preprocessing,
       ensuring output columns match the trained regressors."""
    if X_history is None or not trained_regressors: # If no regressors were used, skip
        return None

    print("Forecasting and processing future covariates...")
    col_name = X_history.name if X_history.name else 'covariate'

    # 1. Forecast Future Covariate Values (Simple Seasonal Naive)
    future_X_forecast = pd.Series(index=future_dates, name=col_name, dtype=float)
    for date in future_dates:
        day_of_week = date.dayofweek
        past_similar_days = X_history[X_history.index.dayofweek == day_of_week].iloc[-4:]
        if not past_similar_days.empty:
            future_X_forecast[date] = past_similar_days.mean()
        else:
            future_X_forecast[date] = X_history.mean()

    # 2. Combine History and Forecast for Consistent Processing
    full_X = pd.concat([X_history, future_X_forecast])
    all_data_df = pd.DataFrame(full_X) # Use DataFrame for processing

    # 3. Apply the SAME preprocessing steps as used in training
    processed_features = pd.DataFrame(index=full_X.index)
    base_col_to_process = all_data_df[col_name]

    if deseasonalize:
        try:
            weekly_ma = base_col_to_process.rolling(window=7, center=True, min_periods=1).mean()
            deseasonalized_col = base_col_to_process - weekly_ma
            processed_features[f'{col_name}_deseasonalized'] = deseasonalized_col
            # Check if deseasonalized column was actually used in training
            if f'{col_name}_deseasonalized' in trained_regressors:
                 base_col_to_process = deseasonalized_col
        except Exception as e:
            print(f"Warning: Could not deseasonalize future covariate. Error: {e}")
            if col_name in trained_regressors: # Check if original column was used
                 processed_features[col_name] = base_col_to_process
    elif col_name in trained_regressors: # If not deseasonalizing, add original if used
         processed_features[col_name] = base_col_to_process

    # Add lagged features based on the original combined column values
    for lag in lags:
         lag_col_name = f'{col_name}_lag_{lag}'
         if lag_col_name in trained_regressors: # Only create lags that were used
              processed_features[lag_col_name] = all_data_df[col_name].shift(lag)

    # Select only the future dates and fill NaNs
    X_future_processed = processed_features.loc[future_dates].bfill().ffill()

    # Ensure only columns used during training are present and fill any remaining NaNs
    X_future_processed = X_future_processed.reindex(columns=trained_regressors).fillna(0)

    print(f"Processed future covariates. Shape: {X_future_processed.shape}")
    return X_future_processed


# --- Future Forecasting with Direct Prophet ---
# Modify this function to accept the list of trained regressors
def make_future_prophet_forecast(model, y_history, X_history, periods, trained_regressors,
                                 covariate_lags=[1, 7, 14], deseasonalize_covariates=True):
    """Generates future forecasts using the trained direct Prophet model."""
    if model is None:
        print("Error: Cannot make forecast, the model was not trained successfully.")
        return None

    print(f"Generating forecast for {periods} periods...")

    # 1. Create Future Dates
    last_date = y_history.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
    df_future = pd.DataFrame({'ds': future_dates})

    # 2. Forecast and Process Future Covariates using the list of trained regressors
    X_future_proc = forecast_process_future_covariates(
        X_history, future_dates, trained_regressors, # Pass the list here
        lags=covariate_lags,
        deseasonalize=deseasonalize_covariates
    )

    # 3. Add Processed Covariates to Future DataFrame
    if X_future_proc is not None:
        # Add columns based on the trained_regressors list
        for col in trained_regressors:
             if col in X_future_proc.columns:
                  df_future[col] = X_future_proc[col].values
             else:
                  # This case should be less likely now due to reindex in forecast_process_future_covariates
                  print(f"Warning: Regressor '{col}' used in training but missing from processed future covariates. Filling with 0.")
                  df_future[col] = 0 

    # 4. Predict
    try:
        # Ensure df_future has all columns the model expects (ds + trained_regressors)
        forecast_df = model.predict(df_future)
    except Exception as e:
        print(f"Error during prediction: {e}")
        # Add more debug info if prediction fails
        print(f"Columns in df_future for prediction: {df_future.columns.tolist()}")
        print(f"Regressors expected by model: {model.extra_regressors.keys()}")
        return None

    # 5. Inverse Transform if Necessary
    if hasattr(model, 'log_transform_applied') and model.log_transform_applied:
        print("Applying inverse log transform to forecast.")
        forecast_df['yhat'] = np.expm1(forecast_df['yhat'])
        forecast_df['yhat_lower'] = np.expm1(forecast_df['yhat_lower'])
        forecast_df['yhat_upper'] = np.expm1(forecast_df['yhat_upper'])

    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds')


# --- Main Execution ---
# Modify the main function to handle the list of trained regressors
def main():
    """Main workflow execution."""
    # ... (Keep existing configuration setup) ...
    DATA_PATH = 'data/' # Configure your data path
    TARGET_FILE = DATA_PATH + 'groupby_train.csv'
    COVARIATE_FILE = DATA_PATH + 'groupby_transactions.csv'
    TARGET_COL = 'sales'
    COVARIATE_COL = 'transactions'
    COUNTRY_CODE = 'BR'
    TEST_SIZE = 35 # e.g., 5 weeks
    FUTURE_PERIODS = 35 # e.g., 5 weeks
    ZERO_REPLACEMENT_EPSILON = 1e-6 # Define the small number for zero replacement
    
    # Covariate processing settings
    COVARIATE_LAGS = [1, 7, 14, 28]
    DESEASONALIZE_COVARIATES = True

    # sktime Grid Search Parameters
    PARAM_GRID = {
        # Control log transform (passthrough=True means NO log transform)
        "ln__passthrough": [True, False],  #, False 
        # Prophet specific parameters (note the 'forecaster__' prefix)
        # "forecaster__seasonality_mode": ['additive', 'multiplicative'],
        # "forecaster__changepoint_prior_scale": [0.01, 0.05, 0.1],
        # Add other Prophet params as needed
        # "forecaster__holidays_prior_scale": [5.0, 10.0], 
        # "forecaster__seasonality_prior_scale": [5.0, 10.0],
    }
    
    # sktime CV Splitter Configuration
    # Adjust window sizes based on your data frequency and seasonality
    CV_INITIAL_WINDOW = 28 * 12 * 3 # ~3 years initial train
    CV_WINDOW_LENGTH = 28 * 12 * 1 # ~1 year validation window
    CV_STEP_LENGTH = 28 * 3        # Step forward by ~3 months

    try:
        # 1. Load Data - Pass epsilon for zero replacement
        print("\n--- Loading Data ---")
        y = load_series(TARGET_FILE, value_col=TARGET_COL, name='y', epsilon=ZERO_REPLACEMENT_EPSILON)
        X = load_series(COVARIATE_FILE, value_col=COVARIATE_COL, name='transactions', epsilon=ZERO_REPLACEMENT_EPSILON)

        # --- Data Summary ---
        print("\n--- Data Summary (Post Loading & Preprocessing) ---")
        print(f"Target Series ('{y.name}'):")
        print(f"  Time range: {y.index.min()} to {y.index.max()}")
        print(f"  Number of observations: {len(y)}")
        y_na_count = y.isna().sum()
        y_zero_count = (y == 0).sum() # Should be 0 now
        y_nonpos_count = (y <= 0).sum() # Check for non-positive after replacement
        print(f"  Missing values (NaNs): {y_na_count}")
        print(f"  Zero values (should be 0): {y_zero_count}")
        print(f"  Non-positive values (<= 0): {y_nonpos_count}") # Includes epsilon values
        if y_na_count > 0:
            print("  WARNING: Missing values detected in target series post-load!")
        if y_nonpos_count > 0 and y_nonpos_count != (y == ZERO_REPLACEMENT_EPSILON).sum():
             print("  WARNING: Non-positive values other than epsilon detected in target series!")


        if X is not None:
            print(f"\nCovariate Series ('{X.name}'):")
            # Align X to y's index before summarizing
            X = X.reindex(y.index)
            print(f"  Time range: {X.index.min()} to {X.index.max()}")
            print(f"  Number of observations: {len(X)}")
            X_na_count = X.isna().sum()
            X_zero_count = (X == 0).sum() # Should be 0 now
            X_nonpos_count = (X <= 0).sum()
            print(f"  Missing values (NaNs): {X_na_count}")
            print(f"  Zero values (should be 0): {X_zero_count}")
            print(f"  Non-positive values (<= 0): {X_nonpos_count}")
            if X_na_count > 0:
                print("  WARNING: Missing values detected in covariate series post-load! Attempting to fill...")
                # Fill NAs in covariate after alignment (e.g., using interpolation or ffill/bfill)
                X = X.interpolate(method='time').bfill().ffill()
                if X.isna().any(): # Check again after filling
                     print("  ERROR: Could not fill all missing values in covariate series.")
                     X = X.fillna(0) # Example: fill remaining with 0 (might reintroduce zeros!)
                     # Consider filling with epsilon instead if zeros are problematic:
                     # X = X.fillna(ZERO_REPLACEMENT_EPSILON)
            if X_nonpos_count > 0 and X_nonpos_count != (X == ZERO_REPLACEMENT_EPSILON).sum():
                 print("  WARNING: Non-positive values other than epsilon detected in covariate series!")


            # Check correlation after alignment and potential filling
            try:
                 correlation = y.corr(X)
                 print(f"  Correlation with target ('{y.name}'): {correlation:.3f}")
            except Exception as corr_err:
                 print(f"  Could not calculate correlation with target: {corr_err}")
        else:
            print("\nNo covariate series loaded.")
        # --- End Data Summary ---

        # 2. Split Data
        print("\n--- Splitting Data ---")
        y_train, y_test, X_train, X_test = temporal_train_test_split(
            y, X, test_size=TEST_SIZE
        )
        print(f"Data split: Train size={len(y_train)}, Test size={len(y_test)}")

        # 3. Define CV Strategy for Grid Search
        print("\n--- Setting up Cross-Validation ---")
        cv_fh = ForecastingHorizon(np.arange(1, len(y_test) + 1))
        cv_splitter = SlidingWindowSplitter(
            initial_window=CV_INITIAL_WINDOW,
            window_length=CV_WINDOW_LENGTH,
            step_length=CV_STEP_LENGTH,
            fh=cv_fh
        )

        # 4. Find Best Parameters via Grid Search
        print("\n--- Finding Best Parameters (Grid Search) ---")
        best_params = find_best_prophet_params(
            y_train, X_train, cv_splitter, PARAM_GRID, COUNTRY_CODE
        )

        # 5. Train Final Model with Direct Prophet
        # Now returns the model and the list of regressors actually added
        print("\n--- Training Final Model ---")
        fitted_model, trained_regressors = train_direct_prophet( 
            y_train, X_train, best_params,
            covariate_lags=COVARIATE_LAGS,
            deseasonalize_covariates=DESEASONALIZE_COVARIATES,
            country_code=COUNTRY_CODE
        )
        print(f"Regressors successfully trained in the model: {trained_regressors}")

        # 6. Make Future Forecast
        print("\n--- Generating Future Forecast ---")
        if fitted_model:
            future_forecast = make_future_prophet_forecast(
                fitted_model, y, X, # Use full history for context
                periods=FUTURE_PERIODS,
                trained_regressors=trained_regressors, # Pass the list of used regressors
                covariate_lags=COVARIATE_LAGS,
                deseasonalize_covariates=DESEASONALIZE_COVARIATES
            )

            if future_forecast is not None:
                print("\n--- Future Forecast ---")
                print(future_forecast.head())
                try:
                     forecast_filename = DATA_PATH + 'future_forecast_prophet.csv'
                     future_forecast.to_csv(forecast_filename)
                     print(f"Future forecast saved to {forecast_filename}")
                except Exception as e:
                     print(f"Error saving forecast: {e}")
            else:
                 print("Future forecast generation failed.")
        else:
            print("Skipping future forecast as model training failed.")

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An error occurred in the main workflow: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()