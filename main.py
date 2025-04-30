"""
Simple and clean Prophet forecasting pipeline with:
- Log transformation of target and covariates
- Lagged features from covariates
- Basic visualization of results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays
from datetime import timedelta
from prophet import Prophet
import logging

# Configure logging to suppress Prophet and cmdstanpy output
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)


def load_data(target_path, covariate_path=None, date_col='date', target_col='sales', covariate_col='transactions'):
    """Load target and optional covariate data."""
    # Load target data
    target_df = pd.read_csv(target_path, parse_dates=[date_col])
    target_series = pd.Series(target_df[target_col].values, index=pd.DatetimeIndex(target_df[date_col]), name=target_col)
    
    # Load covariate data if provided
    covariate_series = None
    if covariate_path:
        covariate_df = pd.read_csv(covariate_path, parse_dates=[date_col])
        covariate_series = pd.Series(covariate_df[covariate_col].values, 
                                    index=pd.DatetimeIndex(covariate_df[date_col]), 
                                    name=covariate_col)
    
    print(f"Loaded target data: {len(target_series)} points, spanning {target_series.index.min()} to {target_series.index.max()}")
    if covariate_series is not None:
        print(f"Loaded covariate data: {len(covariate_series)} points")
        
    return target_series, covariate_series

def get_country_holidays(country_code, start_date, end_date, pre_post_days=1):
    """
    Generate holiday features suitable for Prophet from the holidays package.
    
    Parameters:
    -----------
    country_code : str
        Country code (e.g., 'US', 'BR', 'UK')
    start_date : datetime
        Start date for the holiday range
    end_date : datetime
        End date for the holiday range
    pre_post_days : int, default=1
        Days to include before and after holidays to capture proximity effects
        
    Returns:
    --------
    pd.DataFrame
        Holiday DataFrame in format required by Prophet
    """
    # Get holidays for specified country and date range
    country_holidays_dict = holidays.country_holidays(country_code, years=range(start_date.year-1, end_date.year+2))
    
    holiday_list = []
    for date, name in country_holidays_dict.items():
        # Add the main holiday
        holiday_list.append({
            'holiday': f"{name}",
            'ds': pd.Timestamp(date),
            'lower_window': -pre_post_days,
            'upper_window': pre_post_days
        })
    
    # Convert to DataFrame
    holidays_df = pd.DataFrame(holiday_list)
    
    print(f"Generated {len(holidays_df)} holiday features for {country_code} between {start_date.date()} and {end_date.date()}")
    return holidays_df

def add_date_dummies(df, date_patterns):
    """
    Add dummy variables for specific dates or date patterns.
    Works for both historical data and future predictions.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with 'ds' column containing dates
    date_patterns : list of dict
        List of dictionaries specifying date patterns
        Each dict should have:
            - 'name': name for the dummy column
            - 'month': month (1-12) or None for any month
            - 'day': day (1-31) or None for any day
            - 'specific_dates': list of specific dates (optional)
            - 'window_before': days before to include (default 0)
            - 'window_after': days after to include (default 0)
    
    Returns:
    --------
    DataFrame
        Original DataFrame with added dummy columns
    """
    result_df = df.copy()
    dates = pd.to_datetime(df['ds'])
    
    for pattern in date_patterns:
        name = pattern['name']
        month = pattern.get('month', None)
        day = pattern.get('day', None)
        specific_dates = pattern.get('specific_dates', [])
        window_before = pattern.get('window_before', 0)
        window_after = pattern.get('window_after', 0)
        
        # Initialize dummy column with zeros as float type to avoid dtype warnings
        result_df[name] = 0.0  # Use 0.0 instead of 0 to ensure float dtype
        
        # Check for pattern matches
        if month is not None and day is not None:
            # Monthly-day pattern (e.g., every July 15)
            pattern_matches = (dates.dt.month == month) & (dates.dt.day == day)
            result_df.loc[pattern_matches, name] = 1.0  # Use 1.0 instead of 1
            
            # Print for debugging
            if pattern_matches.any():
                print(f"Added {pattern_matches.sum()} {name} dates for month {month}, day {day}")
        
        # Add specific dates if provided
        if specific_dates:
            for date_str in specific_dates:
                specific_date = pd.to_datetime(date_str)
                result_df.loc[dates == specific_date, name] = 1.0  # Use 1.0 instead of 1
        
        # Apply window effects if requested
        if window_before > 0 or window_after > 0:
            # Get indices of all 1s - MORE ROBUST METHOD:
            match_dates = dates[result_df[name] > 0.9]  # Use > 0.9 instead of == 1 for float comparison
            
            if not match_dates.empty:
                # Loop through all dates in the dataset
                for i, current_date in enumerate(dates):
                    # Check if this date is within window_before days before any match date
                    for match_date in match_dates:
                        days_before = (match_date - current_date).days
                        if 0 < days_before <= window_before:
                            # Gradually decreasing effect
                            effect = 1.0 - (days_before / (window_before + 1))
                            result_df.loc[i, name] = max(result_df.loc[i, name], effect)
                        
                        # Check if this date is within window_after days after any match date
                        days_after = (current_date - match_date).days
                        if 0 < days_after <= window_after:
                            # Gradually decreasing effect
                            effect = 1.0 - (days_after / (window_after + 1))
                            result_df.loc[i, name] = max(result_df.loc[i, name], effect)
    
    return result_df

def apply_log_transform(series, epsilon=1e-9):
    """Apply log transformation to a series, handling zeros and negative values."""
    # Create a copy to avoid modifying the original
    transformed = series.copy()
    
    # Replace zeros and negative values with epsilon
    zero_mask = transformed <= 0
    if zero_mask.any():
        n_zeros = zero_mask.sum()
        print(f"Found {n_zeros} zero/negative values in {series.name}. Replacing with {epsilon} before log transform.")
        transformed[zero_mask] = epsilon
    
    # Apply log transform
    log_transformed = np.log1p(transformed)  # log1p(x) = log(1 + x)
    log_transformed.name = f"log_{series.name}"
    
    return log_transformed


def add_lags(series, lags=[1, 7, 14]):
    """Create lag features from a time series."""
    df = pd.DataFrame(index=series.index)
    df[series.name] = series
    
    for lag in lags:
        df[f"{series.name}_lag_{lag}"] = series.shift(lag)
    
    # Fill NAs created by shifting - using recommended methods instead of deprecated fillna(method='...')
    df = df.bfill().ffill()
    
    return df


def prepare_prophet_data(y, X=None, lags=[1, 7, 14], log_transform=True, date_dummies=None):
    """Prepare data for Prophet with transformations."""
    # Start with Prophet's required format
    df = pd.DataFrame({'ds': y.index, 'y': y.values})
    
    # Apply log transform to y if requested
    if log_transform:
        y_transformed = apply_log_transform(y)
        df['y'] = y_transformed.values
    
    # Process covariates if provided
    if X is not None:
        # Apply log transform to X if requested
        if log_transform:
            X_transformed = apply_log_transform(X)
        else:
            X_transformed = X.copy()
        
        # Create lag features
        X_with_lags = add_lags(X_transformed, lags)
        
        # Add each feature to the Prophet dataframe
        for col in X_with_lags.columns:
            df[col] = X_with_lags[col].values
    
    # Add date dummies if specified
    if date_dummies:
        df = add_date_dummies(df, date_dummies)
    
    print(f"Prepared data for Prophet with {df.shape[1]} columns")
    return df


def split_data(df, test_size=30):
    """Split data into train and test sets."""
    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()
    print(f"Train set: {len(train)} rows, Test set: {len(test)} rows")
    return train, test


def train_prophet_model(train_df, X_cols=None, yearly_seasonality=True, weekly_seasonality=True, 
                        daily_seasonality=False, country_code=None):
    """Train a Prophet model with optional regressors and holidays."""
    # Initialize model
    model_kwargs = {
        'yearly_seasonality': yearly_seasonality,
        'weekly_seasonality': weekly_seasonality,
        'daily_seasonality': daily_seasonality
    }
    
    # Add holidays if country code provided
    if country_code:
        # Get min and max dates from training data
        min_date = pd.to_datetime(train_df['ds'].min())
        max_date = pd.to_datetime(train_df['ds'].max())
        
        # Generate holiday DataFrame
        holidays_df = get_country_holidays(
            country_code=country_code, 
            start_date=min_date - timedelta(days=365),  # Include previous year
            end_date=max_date + timedelta(days=365)     # Include next year
        )
        
        # Add holidays to model
        model_kwargs['holidays'] = holidays_df
        print(f"Added {len(holidays_df)} holidays from {country_code} to the model")
    
    # Create model with compiled parameters
    model = Prophet(**model_kwargs)
    
    # Add regressors
    if X_cols:
        for col in X_cols:
            model.add_regressor(col)
    
    # Fit the model
    print("Fitting Prophet model...")
    model.fit(train_df)
    print("Model fitted successfully")
    
    return model


def make_predictions(model, periods, train_df, test_df=None, X_cols=None):
    """Make predictions with the trained model."""
    # Create future dataframe
    if test_df is not None:
        future = test_df[['ds']].copy()  # Use test dates for prediction
    else:
        future = model.make_future_dataframe(periods=periods)
    
    # Add regressor values if available
    if X_cols and test_df is not None:
        for col in X_cols:
            future[col] = test_df[col].values
    
    # Make prediction
    forecast = model.predict(future)
    
    return forecast


def inverse_transform(series, log_transformed=True):
    """Apply inverse transformation to predictions."""
    if log_transformed:
        return np.expm1(series)  # expm1(x) = exp(x) - 1
    return series


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    # Handle zeros in y_true
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan  # Can't calculate MAPE if all y_true are zero
    
    # Calculate MAPE only for non-zero y_true values
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def plot_results(y_true, y_pred, title="Model Performance"):
    """Plot actual vs predicted values and residuals."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Actual vs Predicted
    ax1.plot(y_true.index, y_true.values, label='Actual', color='blue')
    ax1.plot(y_pred.index, y_pred.values, label='Predicted', color='red')
    ax1.set_title(title)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = y_true - y_pred
    ax2.plot(residuals.index, residuals.values, color='green')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('Residuals')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Residual')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_forecast(model, forecast, y_full, test_size, X_cols=None, log_transform=True, title="Prophet Forecast", future_periods=0, history_days=365):
    """
    Plot the complete forecast including historical data, test data, and future predictions.
    
    Parameters:
    -----------
    model : Prophet model
        The trained Prophet model
    forecast : DataFrame
        The forecast DataFrame from model.predict()
    y_full : Series
        The complete original time series
    test_size : int
        Number of data points in the test set
    X_cols : list
        List of regressor column names
    log_transform : bool
        Whether log transform was applied
    title : str
        Plot title
    future_periods : int
        Number of periods to forecast beyond the test set
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare index dates for the entire dataset and forecast
    dates = y_full.index
    forecast_dates = pd.to_datetime(forecast['ds'])
    
    # Split dates into train and test
    train_dates = dates[:-test_size]
    test_dates = dates[-test_size:]
    
    # Get the original values
    y_orig = y_full.values
    train_values = y_orig[:-test_size]
    test_values = y_orig[-test_size:]

    # ---- NEW CODE: Limit historical data to last history_days ----
    # Calculate cutoff date (default: 365 days before end of training)
    if len(train_dates) > 0:
        end_train_date = train_dates[-1]
        cutoff_date = end_train_date - pd.Timedelta(days=history_days)
        
        # Filter train data to only show the specified history period
        history_mask = train_dates >= cutoff_date
        history_dates = train_dates[history_mask]
        history_values = train_values[history_mask]
        
        print(f"Showing {len(history_dates)} days of history (from {history_dates[0].date()} to {history_dates[-1].date()})")
    else:
        history_dates = train_dates
        history_values = train_values
    # ---------------------------------------------------------
    
    # Get predicted values and apply inverse transform if needed
    yhat = forecast['yhat'].values
    if log_transform:
        yhat = np.expm1(yhat)
    
    # Confidence intervals
    yhat_lower = forecast['yhat_lower'].values
    yhat_upper = forecast['yhat_upper'].values
    if log_transform:
        yhat_lower = np.expm1(yhat_lower)
        yhat_upper = np.expm1(yhat_upper)
    
    # Plot historical data
    ax.plot(history_dates, history_values, 'k-', label='Historical Actuals')
    
    # Plot test data
    ax.plot(test_dates, test_values, 'b-', label='Test Actuals')
    
    # Plot predictions over the entire range
    ax.plot(forecast_dates, yhat, 'r-', label='Forecast')
    
    # Plot confidence intervals
    ax.fill_between(forecast_dates, yhat_lower, yhat_upper, color='red', alpha=0.2, label='95% Confidence Interval')
    
    # # Add future forecast if requested
    # if future_periods > 0:
    #     # Create future dataframe
    #     future = model.make_future_dataframe(periods=test_size + future_periods)
        
    #     # CRITICAL FIX: Add regressor values for future prediction
    #     if X_cols:
    #         print(f"Adding regressor columns for future forecast: {X_cols}")
    #         # Get the existing forecast dataframe which already contains the regressor values
    #         # for the historical and test periods
    #         full_forecast_df = pd.DataFrame({'ds': forecast['ds']})
    #         for col in X_cols:
    #             # First get all existing regressor values from the original forecast
    #             if col in forecast.columns:
    #                 full_forecast_df[col] = forecast[col].values
    #             else:
    #                 print(f"Warning: Regressor '{col}' not found in forecast dataframe")
            
    #         # Create the future dataframe with the same columns
    #         future_df = pd.DataFrame({'ds': future['ds']})
            
    #         # Merge the existing values for past dates
    #         future_df = future_df.merge(full_forecast_df, on='ds', how='left')
            
    #         # For future dates beyond the original forecast, forward fill the last values
    #         future_df = future_df.ffill()
            
    #         # Make sure we're using the properly populated dataframe
    #         future = future_df
        
    #     # Now predict with the properly populated future dataframe
    #     try:
    #         future_forecast = model.predict(future)
    #         future_dates = pd.to_datetime(future_forecast['ds'])[-future_periods:]
            
    #         # Get future predicted values and apply inverse transform if needed
    #         future_yhat = future_forecast['yhat'].values[-future_periods:]
    #         future_yhat_lower = future_forecast['yhat_lower'].values[-future_periods:]
    #         future_yhat_upper = future_forecast['yhat_upper'].values[-future_periods:]
            
    #         if log_transform:
    #             future_yhat = np.expm1(future_yhat)
    #             future_yhat_lower = np.expm1(future_yhat_lower)
    #             future_yhat_upper = np.expm1(future_yhat_upper)
            
    #         # Plot future forecast with different style
    #         ax.plot(future_dates, future_yhat, 'r--', linewidth=2, label='Future Forecast')
    #         ax.fill_between(future_dates, future_yhat_lower, future_yhat_upper, color='red', alpha=0.1)
    #     except Exception as e:
    #         print(f"Error generating future forecast: {e}")




    # Add future forecast if requested
    if future_periods > 0:
        try:
            # Create future dataframe with dates only
            future = model.make_future_dataframe(periods=test_size + future_periods)
            
            if X_cols:
                print(f"Processing future regressor values...")
                future_df = future.copy()
                
                # Split regressor columns into categories
                dummy_cols = [col for col in X_cols if '_dummy' in col]
                continuous_cols = [col for col in X_cols if col not in dummy_cols]
                
                # 1. Handle continuous regressors (log_transactions and its lags)
                for col in continuous_cols:
                    if col in forecast.columns:
                        # Get the last known value for this regressor
                        last_known_value = forecast[col].iloc[-1]
                        print(f"Using last known value {last_known_value:.4f} for {col} in future periods")
                        
                        # Copy existing values from forecast for historical periods
                        common_dates = pd.merge(
                            pd.DataFrame({'ds': future_df['ds']}),
                            pd.DataFrame({'ds': forecast['ds'], 'match': True}),
                            on='ds', how='left'
                        )['match'].notna()
                        
                        # Create column initialized with the last known value
                        future_df[col] = last_known_value
                        
                        # For dates that exist in the forecast, use those values
                        if common_dates.any():
                            common_indices = future_df.index[common_dates]
                            forecast_index = forecast.index[:len(common_indices)]
                            future_df.loc[common_indices, col] = forecast.loc[forecast_index, col].values
                    else:
                        print(f"Warning: {col} not found in forecast columns. Using zeros.")
                        future_df[col] = 0
                
                # 2. Handle date dummy variables by recalculating
                if dummy_cols and date_dummies:
                    print(f"Recalculating date dummies for future periods: {dummy_cols}")
                    # Only keep the date dummies that are in X_cols
                    relevant_patterns = []
                    for pattern in date_dummies:
                        if pattern['name'] in dummy_cols:
                            relevant_patterns.append(pattern)
                    
                    # Apply date patterns to the future dataframe
                    if relevant_patterns:
                        future_df = add_date_dummies(future_df, relevant_patterns)
                        print(f"Successfully added date patterns to future data")
                
                future = future_df
            
            # Final verification - no NaNs should be present
            na_cols = future.columns[future.isna().any()].tolist()
            if na_cols:
                print(f"WARNING: Still found NaNs in columns {na_cols}. Using zeros as last resort.")
                future = future.fillna(0)
            
            # Now predict with the properly populated future dataframe
            future_forecast = model.predict(future)
            future_dates = pd.to_datetime(future_forecast['ds'])[-future_periods:]

            # Get future predicted values and apply inverse transform if needed
            future_yhat = future_forecast['yhat'].values[-future_periods:]
            future_yhat_lower = future_forecast['yhat_lower'].values[-future_periods:]
            future_yhat_upper = future_forecast['yhat_upper'].values[-future_periods:]
            
            if log_transform:
                future_yhat = np.expm1(future_yhat)
                future_yhat_lower = np.expm1(future_yhat_lower)
                future_yhat_upper = np.expm1(future_yhat_upper)
            
            # Plot future forecast with different style
            ax.plot(future_dates, future_yhat, 'r--', linewidth=2, label='Future Forecast')
            ax.fill_between(future_dates, future_yhat_lower, future_yhat_upper, color='red', alpha=0.1)
        
        except Exception as e:
            print(f"Error generating future forecast: {e}")
            import traceback
            traceback.print_exc()  # Print the full stack trace for better debugging








    
    # Add vertical line to separate train and test data
    ax.axvline(x=train_dates[-1], color='gray', linestyle='--')

    # Set x-axis limits to show only the desired history period
    if len(history_dates) > 0:
        ax.set_xlim(left=history_dates[0], right=None)  # Start from first history date
    
    # Add labels and legend
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def plot_components(model, forecast, log_transform=True):
    """
    Plot the individual components of the Prophet model such as trend,
    seasonality, holidays, and regressors.
    
    Parameters:
    -----------
    model : Prophet model
        The trained Prophet model
    forecast : DataFrame
        The forecast DataFrame from model.predict()
    log_transform : bool
        Whether log transform was applied
    """
    print("Plotting model components...")
    
    # Get fresh components plot from Prophet
    fig = model.plot_components(forecast)
    
    # If log transform was applied, we need to note this in the title
    if log_transform:
        # Add a text annotation explaining the log transform 
        plt.figtext(0.01, 0.01, "Note: Components are on log scale due to log transformation", 
                   fontsize=10, ha='left')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def run_prophet_pipeline(
    target_path, 
    covariate_path=None, 
    test_size=35,
    future_periods=35,
    lags=[1, 7, 14], 
    log_transform=True,
    yearly_seasonality=True,
    weekly_seasonality=True,
    country_code=None,
    date_dummies=None
):
    """
    Run the complete Prophet forecasting pipeline.
    
    Parameters:
    -----------
    target_path : str
        Path to the target CSV file
    covariate_path : str, optional
        Path to the covariate CSV file
    test_size : int, default=35
        Number of data points to use for testing
    future_periods : int, default=35
        Number of periods to forecast into the future
    lags : list, default=[1, 7, 14]
        List of lag values to create for covariates
    log_transform : bool, default=True
        Whether to apply log transformation to target and covariates
    yearly_seasonality : bool, default=True
        Whether to include yearly seasonality in the Prophet model
    weekly_seasonality : bool, default=True
        Whether to include weekly seasonality in the Prophet model
    country_code : str, optional
        Country code for holidays (e.g., 'US', 'BR', 'UK')
    date_dummies : list of dict, optional
        Custom date patterns to add as dummy variables
    """
    print("\n--- Starting Prophet Pipeline ---\n")
    
    # 1. Load data
    print("Loading data...")
    y, X = load_data(target_path, covariate_path)
    
    # 2. Prepare data for Prophet
    print("\nPreparing data for Prophet...")
    prophet_df = prepare_prophet_data(y, X, lags, log_transform, date_dummies)
    
    # 3. Split data
    print("\nSplitting data into train and test sets...")
    train_df, test_df = split_data(prophet_df, test_size)
    
    # 4. Determine regressor columns
    X_cols = None
    if X is not None:
        X_base_name = X.name if X.name else 'covariate'
        # Get all columns except 'ds' and 'y'
        X_cols = [col for col in prophet_df.columns if col not in ['ds', 'y']]
    
    # 5. Train model
    print("\nTraining Prophet model...")
    model = train_prophet_model(
        train_df, 
        X_cols=X_cols, 
        yearly_seasonality=yearly_seasonality, 
        weekly_seasonality=weekly_seasonality,
        country_code=country_code  # Pass country_code to the function
    )
    
    # 6. Make predictions
    print("\nMaking predictions...")
    forecast = make_predictions(model, test_size, train_df, test_df, X_cols)
    
    # 7. Prepare actual and predicted series for comparison
    y_test = pd.Series(test_df['y'].values, index=pd.DatetimeIndex(test_df['ds']), name='y_test')
    y_pred = pd.Series(forecast['yhat'].values, index=pd.DatetimeIndex(forecast['ds']), name='y_pred')
    
    # 8. Apply inverse transform if log transform was used
    if log_transform:
        print("\nApplying inverse log transformation...")
        y_test = pd.Series(inverse_transform(y_test, log_transform), index=y_test.index, name='y_test')
        y_pred = pd.Series(inverse_transform(y_pred, log_transform), index=y_pred.index, name='y_pred')
    
    # 9. Calculate metrics
    print("\nCalculating performance metrics...")
    mape = calculate_mape(y_test.values, y_pred.values)
    print(f"MAPE: {mape:.2f}%")
    
    # 10. Plot results
    print("\nPlotting model components...")
    plot_components(model, forecast, log_transform)

    print("\nPlotting results...")
    plot_results(y_test, y_pred, f"Prophet Forecast (MAPE: {mape:.2f}%)")
    
    print("\nPlotting complete forecast with historical data...")
    plot_forecast(
        model, 
        forecast, 
        y, 
        test_size,
        X_cols=X_cols,  # Pass X_cols to the function
        log_transform=log_transform,
        title=f"Prophet Complete Forecast (MAPE: {mape:.2f}%)",
        future_periods=future_periods
    )

    # 11. Return results for further analysis if needed
    return {
        'model': model,
        'forecast': forecast,
        'y_test': y_test,
        'y_pred': y_pred,
        'mape': mape
    }


# Example usage
if __name__ == "__main__":
    TARGET_PATH = 'data/groupby_train.csv'
    COVARIATE_PATH = 'data/groupby_transactions.csv'

    # Define dummy variables for specific dates
    date_dummies = [
        # July 15th every year with a 1-day effect before and after
        {
            'name': 'july_15_dummy',
            'month': 7,
            'day': 15,
            'window_before': 1,
            'window_after': 1
        },
        # Christmas dummy (December 25th)
        # {
        #     'name': 'christmas_dummy',
        #     'month': 12,
        #     'day': 25,
        #     'window_before': 3,  # Effect starts 3 days before
        #     'window_after': 1    # Effect continues 1 day after
        # },
        # Example of a specific one-time event
        # {
        #     'name': 'special_event',
        #     'specific_dates': ['2016-08-01', '2017-03-15'],
        # }
    ]
    
    results = run_prophet_pipeline(
        target_path=TARGET_PATH,
        covariate_path=COVARIATE_PATH,
        test_size=35,
        future_periods=35,
        lags=[1, 7, 14, 28],
        log_transform=True,
        country_code='BR',
        date_dummies=date_dummies
    )