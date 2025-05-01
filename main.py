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
from datetime import datetime, timedelta
from prophet import Prophet
import logging
import pickle
import os
from scipy import stats
from statsmodels.stats.stattools import durbin_watson

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


def analyze_regressor_importance(model, forecast):
    """
    Analyze the importance of each regressor in the Prophet model.
    
    Parameters:
    -----------
    model : Prophet model
        The trained Prophet model
    forecast : DataFrame
        The forecast DataFrame from model.predict()
        
    Returns:
    --------
    dict
        Dictionary with regressor names as keys and their importance metrics as values,
        sorted by importance (descending)
    """
    print("\nAnalyzing regressor importance...")
    
    # Get list of regressors from the model
    regressors = []
    if hasattr(model, 'extra_regressors'):
        regressors = list(model.extra_regressors.keys())
    
    if not regressors:
        print("No regressors found in the model.")
        return {}
    
    # Initialize results dictionary
    importance = {}
    
    # Calculate importance for each regressor
    for reg in regressors:
        # Check if this regressor has a direct component in the forecast
        if reg in forecast.columns:
            # Calculate absolute contribution to the model
            abs_contribution = np.abs(forecast[reg]).mean()
            importance[reg] = {
                'abs_mean': abs_contribution,
                'std': forecast[reg].std(),
                'min': forecast[reg].min(),
                'max': forecast[reg].max(),
                'range': forecast[reg].max() - forecast[reg].min()
            }
    
    # Sort by absolute contribution
    sorted_importance = dict(sorted(
        importance.items(), 
        key=lambda x: x[1]['abs_mean'], 
        reverse=True
    ))
    
    # Print the results in a nice format
    print("\nRegressor importance (sorted by absolute contribution):")
    print("-" * 80)
    print(f"{'Regressor':<25} | {'Abs Mean':>10} | {'Range':>10} | {'Min':>10} | {'Max':>10}")
    print("-" * 80)
    
    for reg, metrics in sorted_importance.items():
        print(f"{reg:<25} | {metrics['abs_mean']:>10.4f} | {metrics['range']:>10.4f} | {metrics['min']:>10.4f} | {metrics['max']:>10.4f}")
    
    return sorted_importance


def plot_regressor_importance(importance, top_n=None, show=True):
    """
    Plot the importance of regressors as a bar chart.
    
    Parameters:
    -----------
    importance : dict
        Dictionary of regressor importance metrics from analyze_regressor_importance
    top_n : int, optional
        Number of top regressors to show. If None, shows all.
    show : bool, default=True
        Whether to show the plot immediately
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the plot
    """
    if not importance:
        print("No regressors to plot.")
        return None
    
    # Get the data for plotting
    regressors = list(importance.keys())
    values = [metrics['abs_mean'] for metrics in importance.values()]
    
    # Limit to top_n if specified
    if top_n and top_n < len(regressors):
        regressors = regressors[:top_n]
        values = values[:top_n]
    
    # Create bar chart
    fig = plt.figure(figsize=(12, 6))
    bars = plt.barh(regressors, values, color='skyblue')
    
    # Add labels and formatting
    plt.xlabel('Absolute Mean Contribution')
    plt.title('Regressor Importance')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.001,  # Small offset
            bar.get_y() + bar.get_height()/2,
            f'{width:.4f}',
            va='center'
        )
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_fitted_vs_actuals(model, forecast, y_full, test_size, log_transform=True, title="Fitted vs Actuals"):
    """
    Plot fitted values against actual values to evaluate model fit on training data.
    
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
    log_transform : bool
        Whether log transform was applied
    title : str
        Plot title
    """
    # Get training data
    y_train = y_full.iloc[:-test_size]
    
    # Get fitted values (predictions for the training period)
    train_dates = y_train.index
    
    # DEBUG: Print array lengths before processing
    print(f"Training dates length: {len(train_dates)}")
    print(f"Forecast dates length: {len(forecast['ds'])}")
    
    # More reliable date matching approach
    # Convert dates to strings for comparison to avoid timezone issues
    train_dates_str = [d.strftime('%Y-%m-%d') for d in train_dates]
    forecast_dates_str = [d.strftime('%Y-%m-%d') for d in pd.to_datetime(forecast['ds'])]
    
    # Create a mapping between forecast indices and training indices
    matched_indices = []
    for i, date_str in enumerate(forecast_dates_str):
        if date_str in train_dates_str:
            train_idx = train_dates_str.index(date_str)
            matched_indices.append((i, train_idx))
    
    if not matched_indices:
        print("ERROR: No matching dates found between forecast and training data")
        return None
    
    print(f"Found {len(matched_indices)} matching dates between forecast and training data")
    
    # Extract matched values
    forecast_indices, train_indices = zip(*matched_indices)
    fitted_values = forecast.loc[list(forecast_indices), 'yhat'].values
    actuals = y_train.iloc[list(train_indices)].values
    
    # Extract matched dates for time series plot
    matched_dates = [train_dates[idx] for idx in train_indices]
    
    # Apply inverse transform if needed
    if log_transform:
        fitted_values = np.expm1(fitted_values)
    
    # Check for and handle NaN values
    nan_mask = np.isnan(fitted_values) | np.isnan(actuals)
    if nan_mask.any():
        print(f"WARNING: Found {nan_mask.sum()} NaN values. Removing them from analysis.")
        fitted_values = fitted_values[~nan_mask]
        actuals = actuals[~nan_mask]
        matched_dates = [d for i, d in enumerate(matched_dates) if not nan_mask[i]]
        # Also update the indices for trend extraction later
        forecast_indices = [idx for i, idx in enumerate(forecast_indices) if not nan_mask[i]]
        
    # Check for and handle infinite values
    inf_mask = np.isinf(fitted_values) | np.isinf(actuals)
    if inf_mask.any():
        print(f"WARNING: Found {inf_mask.sum()} infinite values. Removing them from analysis.")
        fitted_values = fitted_values[~inf_mask]
        actuals = actuals[~inf_mask]
        matched_dates = [d for i, d in enumerate(matched_dates) if not inf_mask[i]]
        # Also update the indices for trend extraction later
        forecast_indices = [idx for i, idx in enumerate(forecast_indices) if not inf_mask[i]]
    
    # Verify arrays are the same length and not empty
    print(f"Final array lengths - Actuals: {len(actuals)}, Fitted: {len(fitted_values)}")
    if len(actuals) == 0 or len(fitted_values) == 0:
        print("ERROR: No valid data points remain after filtering")
        return None
    

    # Calculate residuals and statistics BEFORE creating the plots
    residuals = actuals - fitted_values
    mean = np.mean(residuals)
    std = np.std(residuals)
    skew = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)


    # Create a 2x4 grid of subplots - daily plot will still span the entire bottom row
    fig = plt.figure(figsize=(24, 12))
    
    # Create a gridspec with 2 rows and 4 columns
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.5])
    
    # Create the five subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Scatter plot (top-left)
    ax4 = fig.add_subplot(gs[0, 1])  # Residuals histogram (top-middle-left)
    ax3 = fig.add_subplot(gs[0, 2])  # Monthly plot (top-middle-right)
    ax5 = fig.add_subplot(gs[0, 3])  # NEW: Seasonal patterns (top-right)
    ax2 = fig.add_subplot(gs[1, :])  # Daily plot (full bottom row)
    

    # SUBPLOT 1: Scatter plot of actual vs fitted values (same as before)
    ax1.scatter(actuals, fitted_values, s=8, alpha=0.5, color='slategrey')
    
    # Add perfect prediction line (y=x)
    min_val = min(np.min(actuals), np.min(fitted_values))
    max_val = max(np.max(actuals), np.max(fitted_values))
    ax1.plot([min_val, max_val], [min_val, max_val], color='tab:orange', linewidth=2.5 , alpha=0.8, label='Perfect Fit') #, linestyle='dashed'
    
    # Add outlier bands (3-sigma threshold)
    outlier_threshold = 3 * std
    ax1.plot([min_val, max_val], [min_val + outlier_threshold, max_val + outlier_threshold], 
            color='tab:orange', linestyle='dashed', alpha=0.5, linewidth=0.7, label='+3σ Threshold')
    ax1.plot([min_val, max_val], [min_val - outlier_threshold, max_val - outlier_threshold], 
            color='tab:orange', linestyle='dashed', alpha=0.5, linewidth=0.7, label='-3σ Threshold')

    # Highlight outlier points
    outlier_mask = np.abs(residuals) > outlier_threshold
    if outlier_mask.any():
        ax1.scatter(actuals[outlier_mask], fitted_values[outlier_mask], 
                    s=16, color='tab:orange', alpha=1.0, marker='o', label=f'Outliers ({sum(outlier_mask)})')
    
    # Calculate and add stats to scatter plot
    try:
        correlation = np.corrcoef(actuals, fitted_values)[0, 1]
        r_squared = correlation ** 2 if not np.isnan(correlation) else 0
        
        try:
            mape = calculate_mape(actuals, fitted_values)
        except:
            mape = np.nan
            print("WARNING: Error calculating MAPE")
        
        stats_text = f"R²: {r_squared:.4f}\nMAPE: {mape:.2f}%" if not np.isnan(mape) else f"R²: {r_squared:.4f}"
        ax1.text(0.05, 0.95, stats_text, 
                transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    except Exception as e:
        print(f"ERROR calculating statistics: {e}")
        ax1.text(0.05, 0.95, "Stats calculation failed", 
                transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add labels to scatter plot
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Fitted Values')
    ax1.set_title(f"{title} (Correlation)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()


    # NEW SUBPLOT 4: Residuals histogram
    # Calculate residuals for the histogram
    n_bins = min(50, int(len(residuals) / 10))  # Rule of thumb for number of bins
    ax4.hist(residuals, bins=n_bins, alpha=0.6, color='slategray', edgecolor='black')

    # Determine if distribution is approximately normal
    # A normal distribution has skewness close to 0 and kurtosis close to 0
    is_normal = (abs(skew) < 0.5) and (abs(kurtosis) < 1)
    
    # Generate points for normal curve
    x = np.linspace(min(residuals), max(residuals), 100)
    y = stats.norm.pdf(x, mean, std) * len(residuals) * (max(residuals) - min(residuals)) / n_bins
    
    # Plot the normal curve
    ax4.plot(x, y, color='tab:orange', alpha=0.8, linewidth=2.5)
    
     # Calculate additional residual diagnostics for comprehensive checks
    # 1. Zero mean - already calculated 
    mean_acceptable = abs(mean) < 0.01 * np.mean(np.abs(actuals))  # Mean should be < 1% of average value

    # 2. Normal distribution - already calculated with skew and kurtosis
    is_normal = (abs(skew) < 0.5) and (abs(kurtosis) < 1)

    # 3. Homoscedasticity - check if variance is consistent across predictions
    # Calculate correlation between absolute residuals and fitted values
    homoscedasticity_corr = np.corrcoef(np.abs(residuals), fitted_values)[0, 1]
    is_homoscedastic = abs(homoscedasticity_corr) < 0.3  # Common threshold

    # 4. Independence/No autocorrelation - Durbin-Watson test
    dw_stat = durbin_watson(residuals)
    no_autocorr = 1.5 < dw_stat < 2.5  # DW should be close to 2

    # 5. No outliers - check for extreme values
    outlier_threshold = 3 * std
    outlier_count = np.sum(np.abs(residuals) > outlier_threshold)
    no_outliers = outlier_count < 0.01 * len(residuals)  # Less than 1% outliers

    # Create comprehensive diagnostic text with pass/fail indicators
    diagnostics_text = (
        f"RESIDUAL DIAGNOSTICS:\n\n"
        f"1. Zero mean? {'✓ PASS' if mean_acceptable else '✗ FAIL'}\n"
        f"2. Normal distr.? {'✓ PASS' if is_normal else '✗ FAIL'}\n"
        f"   • |skew|<0.5: {skew:.2f}\n"
        f"   • |kurtosis|<1.0: {kurtosis:.2f}\n"
        f"3. Homoscedastic? {'✓ PASS' if is_homoscedastic else '✗ FAIL'}\n"
        f"   • var. correl.: {homoscedasticity_corr:.2f}\n"
        f"4. No autocorrel.? {'✓ PASS' if no_autocorr else '✗ FAIL'}\n"
        f"   • D.Watson: {dw_stat:.2f} (~2.0)\n" # DurbinWatson
        f"5. No outliers? {'✓ PASS' if no_outliers else '✗ FAIL'}\n"
        f"   • {outlier_count} outliers ({100*outlier_count/len(residuals):.1f}%)"
    )

    # Add the diagnostic text box to the right side of residual plot
    ax4.text(0.02, 0.97, diagnostics_text,
            transform=ax4.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add labels
    ax4.set_xlabel('Residual (Actual - Fitted)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Residuals Distribution')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Draw vertical line at zero
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.7)


    # SUBPLOT 5: Seasonal patterns
    # Create a DataFrame with dates and residuals for seasonal analysis
    seasonal_df = pd.DataFrame({
        'date': matched_dates,
        'residual': residuals,
        'fitted': fitted_values,
        'actual': actuals
    })
    
    # Add time components
    seasonal_df['year'] = [d.year for d in matched_dates]
    seasonal_df['month'] = [d.month for d in matched_dates]
    seasonal_df['day_of_week'] = [d.dayofweek for d in matched_dates]
    seasonal_df['week_of_year'] = [d.isocalendar()[1] for d in matched_dates]
    
    # Choose what to display - day of week seasonality
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Group by day of week and calculate mean and std (instead of min/max)
    dow_grouped = seasonal_df.groupby('day_of_week').agg({
        'actual': ['mean', 'std'], 
        'fitted': ['mean', 'std'],
        'residual': ['mean', 'std']
    }).reset_index()

    # Set up x positions for plotting
    x = np.arange(len(day_names))

    # Plot mean lines
    ax5.plot(x, dow_grouped['actual']['mean'], 'o-', color='slategray', linewidth=2, markersize=8, label='Actual (Mean)')
    # ax5.plot(x, dow_grouped['fitted']['mean'], 'o-', color='tab:red', linewidth=2, markersize=8, label='Fitted (Mean)')

    # Plot ±1 std ranges as shaded areas (instead of min-max)
    ax5.fill_between(x, 
                    dow_grouped['actual']['mean'] - dow_grouped['actual']['std'], 
                    dow_grouped['actual']['mean'] + dow_grouped['actual']['std'], 
                    color='slategray', alpha=0.2, label='Actual (±1σ)')
    # ax5.fill_between(x, 
    #                 dow_grouped['fitted']['mean'] - dow_grouped['fitted']['std'], 
    #                 dow_grouped['fitted']['mean'] + dow_grouped['fitted']['std'], 
    #                 color='tab:red', alpha=0.2, label='Fitted (±1σ)')

    # Set x-axis labels to day names
    ax5.set_xticks(x)
    ax5.set_xticklabels(day_names, rotation=45)
    
    # Add labels and title
    ax5.set_xlabel('Day of Week')
    ax5.set_ylabel('Average Value')
    ax5.set_title('Seasonal Pattern by Day of Week')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.legend()

    
    # SUBPLOT 2: Daily time series of actuals and fitted values (same as before)
    ax2.plot(matched_dates, actuals, color='slategray', alpha=1.0, label='Actual Values')
    ax2.plot(matched_dates, fitted_values, color='tab:red', alpha=0.7, label='Fitted Values')

    # Extract trend values AFTER filtering - now forecast_indices will match the filtered data
    trend_values = forecast.loc[list(forecast_indices), 'trend'].values
    if log_transform:
        trend_values = np.expm1(trend_values)

    # Plot trend as a thicker line
    ax2.plot(matched_dates, trend_values, color='darkslategrey', linewidth=1, alpha=1.0, 
            linestyle='--', label='Trend')

    # Highlight outlier points on the time series
    if outlier_mask.any():
        # Extract outlier dates, actual values and predictions
        outlier_dates = [matched_dates[i] for i, is_outlier in enumerate(outlier_mask) if is_outlier]
        outlier_actuals = actuals[outlier_mask]
        
        # Add outlier points to the daily plot
        ax2.scatter(outlier_dates, outlier_actuals, s=80, color='tab:orange', alpha=0.8, 
                    marker='o', edgecolors='tab:orange', linewidths=1.0, 
                    label=f'Outliers ({sum(outlier_mask)})')
        
        # Optional: Add vertical lines at outlier dates to make them more visible
        for date in outlier_dates:
            ax2.axvline(x=date, color='tab:orange', linestyle=':', alpha=0.6, linewidth=1)
        
        # NEW: Add date labels to each outlier point
        for i, (date, value) in enumerate(zip(outlier_dates, outlier_actuals)):
            # Format date as string - short format to avoid clutter
            date_str = date.strftime('%Y-%m-%d')
            
            # Calculate vertical position for annotation (slightly above the point)
            y_pos = value * 1.04  # 4% above the point
            
            # Add text annotation with the date
            ax2.annotate(date_str, 
                        (date, y_pos),
                        textcoords="offset points", 
                        xytext=(0, 5),   # Offset text slightly above
                        ha='center',     # Horizontally center text
                        fontsize=6,      # Smaller font size
                        rotation=45,     # Rotate text for better readability
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="orange", alpha=0.7))


    
    # Add labels to time series plot
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value')
    ax2.set_title(f"{title} (Daily Time Series)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # SUBPLOT 3: NEW - Monthly aggregation
    # Create DataFrame for monthly resampling
    ts_df = pd.DataFrame({
        'date': matched_dates,
        'actual': actuals,
        'fitted': fitted_values,
        'trend': trend_values 
    })
    ts_df.set_index('date', inplace=True)
    
    # Resample to monthly frequency and take the mean
    monthly_df = ts_df.resample('M').mean()
    
    # Plot monthly data
    ax3.plot(monthly_df.index, monthly_df['actual'], color='slategray', linewidth=2, alpha=1.0, label='Actual Values')
    ax3.plot(monthly_df.index, monthly_df['fitted'], color='tab:red', linewidth=2, alpha=0.7, label='Fitted Values')
    ax3.plot(monthly_df.index, monthly_df['trend'], color='darkslategrey', linewidth=0.5, alpha=0.8, 
        linestyle='--', label='Trend')

    # Calculate monthly correlation and MAPE
    try:
        monthly_corr = np.corrcoef(monthly_df['actual'].values, monthly_df['fitted'].values)[0, 1]
        monthly_r2 = monthly_corr ** 2 if not np.isnan(monthly_corr) else 0
        monthly_mape = calculate_mape(monthly_df['actual'].values, monthly_df['fitted'].values)
        
        monthly_stats = f"Monthly R²: {monthly_r2:.4f}\nMonthly MAPE: {monthly_mape:.2f}%"
        ax3.text(0.05, 0.95, monthly_stats, 
                transform=ax3.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    except Exception as e:
        print(f"Error calculating monthly statistics: {e}")
    
    # Add labels to monthly plot
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Monthly Average Value')
    ax3.set_title(f"{title} (Monthly Aggregation)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Format x-axis dates to prevent overcrowding on all subplots
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_forecast(model, forecast, y_full, test_size, X_cols=None, date_dummies=None, log_transform=True, title="Prophet Forecast", future_periods=0, history_days=70):
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
    ax.plot(history_dates, history_values, color='slategray', linewidth=1, label='Historical Actuals')
    
    # Plot test data
    # ax.plot(test_dates, test_values, 'b-', label='Test Actuals')
    ax.plot(test_dates, test_values, color='tab:blue', linewidth=2, label='Test Actuals')
    
    # Plot predictions over the entire range
    ax.plot(forecast_dates, yhat, color='tab:red', linewidth=2,  label='Predicted') # linestyle='dashed',
    
    # Plot confidence intervals
    # ax.fill_between(forecast_dates, yhat_lower, yhat_upper, color='orange', alpha=0.2, label='95% Confidence Interval')
    
    # Add future forecast if requested
    if future_periods > 0:
        try:
            print("\nGenerating future forecast...")
            
            # CRITICAL FIX: Create future dates directly instead of using make_future_dataframe + filtering
            # Get the most recent date in the original forecast
            last_date = pd.to_datetime(forecast['ds'].iloc[-1])
            
            # Create future dataframe with dates starting from the day after last_date
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=future_periods,
                freq='D'
            )
            
            # Create dataframe with these dates
            future = pd.DataFrame({'ds': future_dates})
            print(f"Created future dataframe with {len(future)} rows from {future['ds'].min().date()} to {future['ds'].max().date()}")
            
            # If we have regressors, we need to add them to the future dataframe
            if X_cols:
                print(f"Adding regressor values for future prediction...")
                dummy_cols = [col for col in X_cols if '_dummy' in col]
                continuous_cols = [col for col in X_cols if col not in dummy_cols]
                
                # For continuous values, use the most recent actual value and repeat it
                for col in continuous_cols:
                    # Get the last known value from the original forecast
                    if col in forecast.columns:
                        last_value = forecast[col].iloc[-1]
                        print(f"Using last known value {last_value:.4f} for {col}")
                        future[col] = last_value
                    else:
                        print(f"Warning: {col} not found in forecast. Using zero.")
                        future[col] = 0
                
                # For dummy variables, use the date patterns
                if dummy_cols and date_dummies:
                    print(f"Recalculating date dummies: {dummy_cols}")
                    relevant_patterns = [p for p in date_dummies if p['name'] in dummy_cols]
                    if relevant_patterns:
                        future = add_date_dummies(future, relevant_patterns)
            
            print(f"Future dataframe shape: {future.shape}, dates from {future['ds'].min()} to {future['ds'].max()}")
            
            # Make the future prediction
            future_forecast = model.predict(future)
            
            # Debug raw predictions
            print(f"Raw future prediction range: [{future_forecast['yhat'].min():.4f} to {future_forecast['yhat'].max():.4f}]")
            
            # Apply inverse transform
            future_yhat = future_forecast['yhat'].values
            future_yhat_lower = future_forecast['yhat_lower'].values 
            future_yhat_upper = future_forecast['yhat_upper'].values
            
            if log_transform:
                future_yhat = np.expm1(future_yhat)
                future_yhat_lower = np.expm1(future_yhat_lower)
                future_yhat_upper = np.expm1(future_yhat_upper)
            
            # Debug transformed values
            print(f"Transformed future range: [{np.min(future_yhat):.2f} to {np.max(future_yhat):.2f}], mean: {np.mean(future_yhat):.2f}")
            
            # Check if values seem reasonable compared to test data
            test_mean = np.mean(yhat[-test_size:])
            future_mean = np.mean(future_yhat)
            ratio = future_mean / test_mean
            
            if ratio < 0.1 or ratio > 10:
                print(f"WARNING: Future forecast mean ({future_mean:.2f}) is very different from test data mean ({test_mean:.2f})")
                print("This suggests potential scaling issues. Attempting to fix...")
                
                # If values are extremely small, apply a correction
                if ratio < 0.1:
                    scaling_factor = test_mean / future_mean if future_mean > 0 else test_mean
                    future_yhat = future_yhat * scaling_factor
                    future_yhat_lower = future_yhat_lower * scaling_factor
                    future_yhat_upper = future_yhat_upper * scaling_factor
                    print(f"Applied scaling factor of {scaling_factor:.2f} to make values visible")
            
            # Plot the future forecast
            future_dates = pd.to_datetime(future_forecast['ds']) #linestyle='dashed',
            ax.plot(future_dates, future_yhat, color='tab:orange', linewidth=2,  label='Forecast')
            ax.fill_between(future_dates, future_yhat_lower, future_yhat_upper, color='tab:orange', alpha=0.1)
            
        except Exception as e:
            print(f"Error generating future forecast: {e}")
            import traceback
            traceback.print_exc()

    
    # Add vertical line to separate train and test data
    ax.axvline(x=train_dates[-1], color='tab:blue', linewidth=1, linestyle='--')
    ax.axvline(x=test_dates[-1], color='tab:blue', linewidth=1, linestyle='--')

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


def export_model_artifacts(results, output_dir='exports', include_plots=True):
    """
    Export Prophet model artifacts including model, forecast, and components.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model, forecast and other results from run_prophet_pipeline
    output_dir : str, default='exports'
        Directory where exports should be saved
    include_plots : bool, default=True
        Whether to include plot exports as PNG files
    """
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Extract objects to export
    model = results['model']
    forecast = results['forecast']
    
    # 1. Export model using pickle
    model_path = os.path.join(output_dir, f'prophet_model_{timestamp}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Exported model to: {model_path}")
    
    # 2. Export forecast DataFrame to CSV
    forecast_path = os.path.join(output_dir, f'forecast_{timestamp}.csv')
    forecast.to_csv(forecast_path, index=False)
    print(f"Exported forecast to: {forecast_path}")
    
    # 3. Export model components
    components_path = os.path.join(output_dir, f'components_{timestamp}.csv')
    
    # Extract components from forecast
    components_cols = ['ds', 'trend', 'yhat']
    
    # Add seasonality components if they exist
    for col in forecast.columns:
        if col.startswith(('yearly', 'weekly', 'daily')) or 'holiday' in col:
            components_cols.append(col)
    
    # Add regressor components if they exist
    regressor_cols = [col for col in forecast.columns 
                      if col in results.get('regressor_importance', {})]
    components_cols.extend(regressor_cols)
    
    # Create components DataFrame with available columns
    available_cols = [col for col in components_cols if col in forecast.columns]
    components_df = forecast[available_cols]
    components_df.to_csv(components_path, index=False)
    print(f"Exported model components to: {components_path}")
    
    # 4. Export regressor importance if available
    if 'regressor_importance' in results and results['regressor_importance']:
        reg_imp_path = os.path.join(output_dir, f'regressor_importance_{timestamp}.csv')
        reg_imp_df = pd.DataFrame([
            {
                'regressor': reg,
                'abs_mean': metrics['abs_mean'],
                'std': metrics['std'],
                'min': metrics['min'],
                'max': metrics['max'],
                'range': metrics['range']
            }
            for reg, metrics in results['regressor_importance'].items()
        ])
        reg_imp_df.to_csv(reg_imp_path, index=False)
        print(f"Exported regressor importance to: {reg_imp_path}")
    
    # 5. Export plots if requested
    if include_plots:
        # Create plots directory
        plots_dir = os.path.join(output_dir, 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # Export complete forecast plot using stored figure
        if 'forecast_fig' in results and results['forecast_fig'] is not None:
            forecast_fig = results['forecast_fig']
            forecast_complete_path = os.path.join(plots_dir, f'forecast_complete_plot_{timestamp}.png')
            forecast_fig.savefig(forecast_complete_path)
            plt.close(forecast_fig)
            print(f"Exported complete forecast plot to: {forecast_complete_path}")
        
        # Export fitted vs actuals plot using stored figure
        if 'fitted_vs_actuals_fig' in results and results['fitted_vs_actuals_fig'] is not None:
            fitted_vs_actuals_fig = results['fitted_vs_actuals_fig']
            fitted_vs_actuals_path = os.path.join(plots_dir, f'fitted_vs_actuals_plot_{timestamp}.png')
            fitted_vs_actuals_fig.savefig(fitted_vs_actuals_path)
            plt.close(fitted_vs_actuals_fig)
            print(f"Exported fitted vs actuals plot to: {fitted_vs_actuals_path}")
        
        # Export components plot using stored figure
        if 'components_fig' in results and results['components_fig'] is not None:
            components_fig = results['components_fig']
            components_plot_path = os.path.join(plots_dir, f'components_plot_{timestamp}.png')
            components_fig.savefig(components_plot_path)
            plt.close(components_fig)
            print(f"Exported components plot to: {components_plot_path}")
        
        # Export regressor importance plot using stored figure
        if 'reg_importance_fig' in results and results['reg_importance_fig'] is not None:
            reg_imp_fig = results['reg_importance_fig']
            reg_imp_plot_path = os.path.join(plots_dir, f'regressor_importance_plot_{timestamp}.png')
            reg_imp_fig.savefig(reg_imp_plot_path)
            plt.close(reg_imp_fig)
            print(f"Exported regressor importance plot to: {reg_imp_plot_path}")
    
    return {
        'model_path': model_path,
        'forecast_path': forecast_path,
        'components_path': components_path
    }


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

    print("\nMaking training predictions for fitted vs actual plot...")
    train_forecast = make_predictions(model, 0, train_df, train_df, X_cols)
    print(f"Generated {len(train_forecast)} predictions for training data")

    if X_cols:
        print("\nAnalyzing regressor importance...")
        reg_importance = analyze_regressor_importance(model, forecast)
        plot_regressor_importance(reg_importance)
    
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
    print("\nPlotting fitted vs actual values for training data...")
    fitted_vs_actuals_fig = plot_fitted_vs_actuals(
        model, 
        train_forecast, 
        y, 
        test_size,
        log_transform=log_transform,
        title="Prophet Model Fit"
    )

    print("\nPlotting model components...")
    components_fig = plot_components(model, forecast, log_transform)

    print("\nPlotting complete forecast with historical data...")
    forecast_fig, _ = plot_forecast(
        model, 
        forecast, 
        y, 
        test_size,
        X_cols=X_cols,
        date_dummies=date_dummies,  # Add this line to pass date_dummies parameter
        log_transform=log_transform,
        title=f"Prophet Complete Forecast (MAPE: {mape:.2f}%)",
        future_periods=future_periods
    )

    reg_importance_fig = None
    if X_cols:
        print("\nAnalyzing regressor importance...")
        reg_importance = analyze_regressor_importance(model, forecast)
        reg_importance_fig = plot_regressor_importance(reg_importance, show=False)

    # 11. Return results for further analysis if needed
    return {
        'model': model,
        'forecast': forecast,
        'train_forecast': train_forecast,
        'y_test': y_test,
        'y_pred': y_pred,
        'mape': mape,
        'regressor_importance': reg_importance if X_cols else {},
        'y_full': y,
        'future_periods': future_periods, 
        'X_cols': X_cols,  
        'date_dummies': date_dummies,  
        'log_transform': log_transform,
        'fitted_vs_actuals_fig': fitted_vs_actuals_fig,
        'components_fig': components_fig,
        'forecast_fig': forecast_fig,
        'reg_importance_fig': reg_importance_fig
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
        {
            'name': 'christmas_dummy',
            'month': 12,
            'day': 25,
            'window_before': 4,  # Effect starts 3 days before
            'window_after': 2    # Effect continues 1 day after
        },
        # # Example of a specific one-time event
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

    # Export model artifacts
    print("\nExporting model artifacts...")
    export_paths = export_model_artifacts(results)
    print("\nExport complete. Files saved to:")
    for key, path in export_paths.items():
        print(f"- {key}: {path}")