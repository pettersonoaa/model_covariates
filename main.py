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
from statsmodels.tsa.stattools import grangercausalitytests
from matplotlib.gridspec import GridSpec
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import warnings

# Configure logging to suppress Prophet and cmdstanpy output
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)


def load_data(target_path, covariate_path=None, date_col='date', target_col='sales', covariate_cols=None):
    """
    Load target and optional covariate data with support for multiple covariates.
    
    Parameters:
    -----------
    target_path : str
        Path to the target CSV file
    covariate_path : str, optional
        Path to the covariate CSV file
    date_col : str, default='date'
        Name of the date column
    target_col : str, default='sales'
        Name of the target column
    covariate_cols : list, optional
        List of covariate column names to load. If None, defaults to ['transactions']
    """
    # Load target data
    target_df = pd.read_csv(target_path, parse_dates=[date_col])
    target_series = pd.Series(target_df[target_col].values, index=pd.DatetimeIndex(target_df[date_col]), name=target_col)
    
    # Load covariate data if provided
    covariate_df = None
    if covariate_path:
        # Default to 'transactions' if no columns specified
        if covariate_cols is None:
            covariate_cols = ['transactions']
            
        # Read the full covariate file
        temp_df = pd.read_csv(covariate_path, parse_dates=[date_col])
        
        # Show available columns for reference
        print(f"Available columns in covariate file: {', '.join(temp_df.columns)}")
        
        # Check which columns actually exist in the file
        available_cols = [col for col in covariate_cols if col in temp_df.columns]
        missing_cols = [col for col in covariate_cols if col not in temp_df.columns]
        
        if missing_cols:
            print(f"WARNING: The following requested covariates are not in the file: {missing_cols}")
        
        if not available_cols:
            print(f"ERROR: None of the requested covariates {covariate_cols} are in the file!")
            return target_series, None
            
        # Extract just the columns we need (with date index)
        covariate_df = pd.DataFrame(index=pd.DatetimeIndex(temp_df[date_col]))
        for col in available_cols:
            covariate_df[col] = temp_df[col].values
    
    print(f"Loaded target data: {len(target_series)} points, spanning {target_series.index.min()} to {target_series.index.max()}")
    if covariate_df is not None:
        print(f"Loaded covariate data: {len(covariate_df)} points with {len(covariate_df.columns)} variables")
        print(f"Covariates: {', '.join(covariate_df.columns)}")
        
    return target_series, covariate_df

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
            'ds': pd.Timestamp(date),
            'holiday': f"{name}",
            'lower_window': -pre_post_days,
            'upper_window': pre_post_days,
            'prior_scale': 20.0  # Increase effect size for individual holidays
        })
    
    # Convert to DataFrame
    holidays_df = pd.DataFrame(holiday_list)

    # Add additional Brazilian specific holidays with stronger effects
    if country_code.upper() == 'BR':
        # Add extra emphasis to major Brazilian holidays
        carnival_holidays = []
        for year in range(start_date.year-1, end_date.year+2):
            # Carnival - extremely important in Brazil
            carnival_date = calculate_carnival_date(year)
            if carnival_date:
                carnival_holidays.append({
                    'ds': carnival_date,
                    'holiday': 'Carnival',
                    'lower_window': -2,  # Effects start earlier
                    'upper_window': 2,   # Effects last longer
                    'prior_scale': 30.0  # Much stronger effect
                })
        
        # If we found any carnival dates, add them using concat
        if carnival_holidays:
            carnival_df = pd.DataFrame(carnival_holidays)
            holidays_df = pd.concat([holidays_df, carnival_df], ignore_index=True)
    
    print(f"Generated {len(holidays_df)} holiday features for {country_code} between {start_date.date()} and {end_date.date()}")
    return holidays_df


def create_enhanced_holidays(country_code, start_date, end_date):
    """
    Create enhanced holiday features with stronger prior scales.
    
    Parameters:
    -----------
    country_code : str
        Country code (e.g., 'BR', 'US')
    start_date : datetime or Timestamp
        Start date for holiday range
    end_date : datetime or Timestamp
        End date for holiday range
        
    Returns:
    --------
    pd.DataFrame
        Holiday DataFrame ready for Prophet with enhanced effects
    """
    # Ensure dates are pandas Timestamps
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Initialize empty DataFrame for holidays
    holidays_list = []
    
    if country_code and country_code.upper() == 'BR':
        # Create manually defined Brazilian holidays with high prior scales
        years = range(start_date.year - 1, end_date.year + 2)
        
        # Add key Brazilian holidays for each year
        for year in years:
            # New Year's Day
            holidays_list.append({
                'holiday': 'New_Year',
                'ds': pd.Timestamp(f'{year}-01-01'),
                'lower_window': -1,
                'upper_window': 1,
                'prior_scale': 50.0
            })
            
            # Carnival (calculate based on Easter)
            from dateutil.easter import easter
            easter_date = easter(year)
            carnival_date = easter_date - pd.Timedelta(days=47)
            
            holidays_list.append({
                'holiday': 'Carnival',
                'ds': carnival_date,
                'lower_window': -2,
                'upper_window': 2,
                'prior_scale': 100.0  # Very high prior scale for Carnival
            })
            
            # Good Friday
            good_friday = easter_date - pd.Timedelta(days=2)
            holidays_list.append({
                'holiday': 'Good_Friday',
                'ds': good_friday,
                'lower_window': -1,
                'upper_window': 1,
                'prior_scale': 50.0
            })
            
            # Tiradentes Day
            holidays_list.append({
                'holiday': 'Tiradentes_Day',
                'ds': pd.Timestamp(f'{year}-04-21'),
                'lower_window': 0,
                'upper_window': 0,
                'prior_scale': 30.0
            })
            
            # Labor Day
            holidays_list.append({
                'holiday': 'Labor_Day',
                'ds': pd.Timestamp(f'{year}-05-01'),
                'lower_window': -1,
                'upper_window': 1,
                'prior_scale': 40.0
            })
            
            # Independence Day
            holidays_list.append({
                'holiday': 'Independence_Day',
                'ds': pd.Timestamp(f'{year}-09-07'),
                'lower_window': -1,
                'upper_window': 1,
                'prior_scale': 40.0
            })
            
            # Our Lady of Aparecida
            holidays_list.append({
                'holiday': 'Our_Lady_Aparecida',
                'ds': pd.Timestamp(f'{year}-10-12'),
                'lower_window': -1,
                'upper_window': 1,
                'prior_scale': 40.0
            })
            
            # All Souls' Day
            holidays_list.append({
                'holiday': 'All_Souls_Day',
                'ds': pd.Timestamp(f'{year}-11-02'),
                'lower_window': -1,
                'upper_window': 1,
                'prior_scale': 40.0
            })
            
            # Republic Proclamation Day
            holidays_list.append({
                'holiday': 'Republic_Day',
                'ds': pd.Timestamp(f'{year}-11-15'),
                'lower_window': -1,
                'upper_window': 1,
                'prior_scale': 40.0
            })

            # Black Friday - add as the 4th Friday of November
            # Calculate the date of the 4th Thursday of November
            thanksgiving = pd.Timestamp(f'{year}-11-01')
            while thanksgiving.dayofweek != 3:  # 3 = Thursday (Mon=0, Sun=6)
                thanksgiving += pd.Timedelta(days=1)
            # Find the 4th Thursday
            thanksgiving += pd.Timedelta(days=7 * 3)
            # Black Friday is the day after
            black_friday = thanksgiving + pd.Timedelta(days=1)
            
            holidays_list.append({
                'holiday': 'Black_Friday',
                'ds': black_friday,
                'lower_window': -3,  # Effect starts a few days before (pre-sales)
                'upper_window': 3,   # Effect continues for a few days after
                'prior_scale': 90.0  # Very high impact for sales data
            })
            
            # Christmas
            holidays_list.append({
                'holiday': 'Christmas',
                'ds': pd.Timestamp(f'{year}-12-25'),
                'lower_window': -3,
                'upper_window': 1,
                'prior_scale': 80.0
            })
    else:
        # For non-BR countries, use the holidays package with enhanced prior scales
        try:
            country_holidays_dict = holidays.country_holidays(
                country_code, 
                years=range(start_date.year-1, end_date.year+2)
            )
            
            for date, name in country_holidays_dict.items():
                holidays_list.append({
                    'holiday': f"{name}",
                    'ds': pd.Timestamp(date),
                    'lower_window': -1,
                    'upper_window': 1,
                    'prior_scale': 30.0  # Higher default prior scale
                })
        except Exception as e:
            print(f"Error getting holidays for {country_code}: {e}")
    
    # Convert to DataFrame
    if holidays_list:
        holidays_df = pd.DataFrame(holidays_list)
    else:
        # Return empty DataFrame with the right columns if no holidays
        holidays_df = pd.DataFrame(columns=['holiday', 'ds', 'lower_window', 'upper_window', 'prior_scale'])
    
    print(f"Created {len(holidays_df)} holidays for {country_code} with enhanced effects")
    
    return holidays_df


def calculate_carnival_date(year):
    """Calculate the date of Carnival (47 days before Easter)"""
    try:
        from dateutil.easter import easter
        easter_date = easter(year)
        carnival_date = easter_date - pd.Timedelta(days=47)
        return carnival_date
    except:
        return None


def create_brazil_holiday_dummies(start_date, end_date):
    """
    Create dummy variable specifications for Brazilian holidays.
    
    Parameters:
    -----------
    start_date : datetime or string
        Start date for the holiday range
    end_date : datetime or string
        End date for the holiday range
    
    Returns:
    --------
    list of dict
        List of holiday dummy specifications ready for add_date_dummies function
    """
    # Convert to pandas datetime if needed
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Initialize empty list for holiday dummy specifications
    holiday_dummies = []
    
    # Get all years in the date range plus buffer years
    years = range(start_date.year - 1, end_date.year + 2)
    
    # List to collect all specific holiday dates
    holiday_dates = {
        'new_year_dummy': [],
        'carnival_dummy': [],
        'good_friday_dummy': [],
        'tiradentes_dummy': [],
        'labor_day_dummy': [],
        'corpus_christi_dummy': [],
        'independence_day_dummy': [],
        'our_lady_aparecida_dummy': [],
        'all_souls_day_dummy': [],
        'republic_day_dummy': [],
        'black_friday_dummy': [],
        'christmas_dummy': []
    }
    
    # Calculate holidays for each year
    for year in years:
        # Fixed date holidays
        holiday_dates['new_year_dummy'].append(f"{year}-01-01")  # New Year's Day
        holiday_dates['tiradentes_dummy'].append(f"{year}-04-21")  # Tiradentes Day
        holiday_dates['labor_day_dummy'].append(f"{year}-05-01")  # Labor Day
        holiday_dates['independence_day_dummy'].append(f"{year}-09-07")  # Independence Day
        holiday_dates['our_lady_aparecida_dummy'].append(f"{year}-10-12")  # Our Lady Aparecida
        holiday_dates['all_souls_day_dummy'].append(f"{year}-11-02")  # All Souls' Day
        holiday_dates['republic_day_dummy'].append(f"{year}-11-15")  # Republic Day
        holiday_dates['christmas_dummy'].append(f"{year}-12-25")  # Christmas
        
        # Easter-based holidays
        try:
            from dateutil.easter import easter
            easter_date = easter(year)
            
            # Carnival (47 days before Easter)
            carnival_date = easter_date - pd.Timedelta(days=47)
            holiday_dates['carnival_dummy'].append(carnival_date.strftime("%Y-%m-%d"))
            
            # Good Friday (2 days before Easter)
            good_friday = easter_date - pd.Timedelta(days=2)
            holiday_dates['good_friday_dummy'].append(good_friday.strftime("%Y-%m-%d"))
            
            # Corpus Christi (60 days after Easter)
            corpus_christi = easter_date + pd.Timedelta(days=60)
            holiday_dates['corpus_christi_dummy'].append(corpus_christi.strftime("%Y-%m-%d"))
        except Exception as e:
            print(f"Error calculating Easter-based holidays for {year}: {e}")
        
        # Black Friday (4th Friday of November)
        thanksgiving = pd.Timestamp(f'{year}-11-01')
        while thanksgiving.dayofweek != 3:  # 3 = Thursday
            thanksgiving += pd.Timedelta(days=1)
        # Find the 4th Thursday
        thanksgiving += pd.Timedelta(days=7 * 3)
        # Black Friday is the day after
        black_friday = thanksgiving + pd.Timedelta(days=1)
        holiday_dates['black_friday_dummy'].append(black_friday.strftime("%Y-%m-%d"))
    
    # Create dummy specifications for each holiday with appropriate windows
    for holiday_name, dates in holiday_dates.items():
        # Customize window based on holiday importance
        if holiday_name in ['carnival_dummy', 'christmas_dummy', 'new_year_dummy', 'black_friday_dummy']:
            # Major holidays with extended effects
            window_before = 3
            window_after = 2
        elif holiday_name in ['good_friday_dummy', 'labor_day_dummy']:
            # Medium holidays
            window_before = 1
            window_after = 1
        else:
            # Standard holidays
            window_before = 1
            window_after = 0
        
        # Create the dummy specification
        holiday_dummies.append({
            'name': holiday_name,
            'specific_dates': dates,
            'window_before': window_before,
            'window_after': window_after
        })
    
    print(f"Created {len(holiday_dummies)} holiday dummy specifications covering {len(years)} years")
    return holiday_dummies


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
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    dates = pd.to_datetime(df['ds'])
    
    # Create a mapping from date to index position for efficient lookup
    date_to_idx = {date: i for i, date in enumerate(dates)}  # ADD THIS LINE
    
    for pattern in date_patterns:
        name = pattern['name']
        month = pattern.get('month', None)
        day = pattern.get('day', None)
        specific_dates = pattern.get('specific_dates', [])
        window_before = pattern.get('window_before', 0)
        window_after = pattern.get('window_after', 0)
        
        # Initialize dummy column with zeros as float type
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
                # Find indices where dates match
                matching_indices = dates == specific_date
                result_df.loc[matching_indices, name] = 1.0
        
        # Apply window effects if requested
        if window_before > 0 or window_after > 0:
            # Get indices of all 1s - MORE ROBUST METHOD:
            match_indices = result_df[name] > 0.9
            match_dates = dates[match_indices]

            if not match_dates.empty:
                # Create temporary Series for efficient updates
                temp_values = result_df[name].copy()
                
                # Loop through all dates in the dataset
                for current_date, current_idx in date_to_idx.items():
                    # Check if this date is within window of any match date
                    for match_date in match_dates:
                        days_before = (match_date - current_date).days
                        if 0 < days_before <= window_before:
                            # Gradually decreasing effect
                            effect = 1.0 - (days_before / (window_before + 1))
                            temp_values.iloc[current_idx] = max(temp_values.iloc[current_idx], effect)
                        
                        # Check if this date is within window_after days after
                        days_after = (current_date - match_date).days
                        if 0 < days_after <= window_after:
                            # Gradually decreasing effect
                            effect = 1.0 - (days_after / (window_after + 1))
                            temp_values.iloc[current_idx] = max(temp_values.iloc[current_idx], effect)
                
                # Update the column with all changes at once
                result_df[name] = temp_values
    
    return result_df

def apply_log_transform(series, epsilon=1e-9):
    """
    Apply log transformation to a series or DataFrame, handling zeros and negative values.
    """
    # Handle Series vs DataFrame
    if isinstance(series, pd.Series):
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
        
    else:  # DataFrame
        # Create a copy to avoid modifying the original
        log_transformed = pd.DataFrame(index=series.index)
        
        # Process each column separately
        for col in series.columns:
            col_series = series[col]
            
            # Replace zeros and negative values with epsilon
            zero_mask = col_series <= 0
            if zero_mask.any():
                n_zeros = zero_mask.sum()
                print(f"Found {n_zeros} zero/negative values in {col}. Replacing with {epsilon} before log transform.")
                col_series.loc[zero_mask] = epsilon
            
            # Apply log transform
            log_transformed[col] = np.log1p(col_series)  # log1p(x) = log(1 + x)
            
    return log_transformed


def add_lags(data, lags=[1, 7, 14]):
    """
    Create lag features from a time series or multiple time series.
    """
    df = pd.DataFrame(index=data.index)
    
    # Handle Series or DataFrame
    if isinstance(data, pd.Series):
        # Convert Series to DataFrame
        df[data.name] = data
        columns = [data.name]
    else:
        # Copy DataFrame columns
        for col in data.columns:
            df[col] = data[col]
        columns = list(data.columns)
    
    # Create lag features for each column
    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    
    # Fill NAs created by shifting
    df = df.bfill().ffill()
    
    return df


def prepare_prophet_data(y, X=None, X_df=None, lags=[1, 7, 14], log_transform=True, date_dummies=None):
    """Prepare data for Prophet with transformations."""
    # Start with Prophet's required format
    df = pd.DataFrame({'ds': y.index, 'y': y.values})
    # Create a copy of X_df if provided
    X_df_desaligned = None
    if X_df is not None:
        # CRITICAL FIX: Add 'ds' column to X_df_desaligned
        X_df_desaligned = X_df.copy()
        X_df_desaligned['ds'] = X_df.index  # Add date column using the index
    
    # Apply log transform to y if requested
    if log_transform:
        y_transformed = apply_log_transform(y)
        df['y'] = y_transformed.values
    
    # Process covariates if provided
    if X is not None:
        # CRITICAL FIX: Align X with y's index to ensure same dates
        X_desaligned = X.copy()
        if isinstance(X, pd.Series):
            X_aligned = X.reindex(y.index)
            print(f"Aligned covariates: original length {len(X)}, aligned length {len(X_aligned)}")
        else:  # DataFrame
            X_aligned = X.reindex(y.index)
            print(f"Aligned covariates: original shape {X.shape}, aligned shape {X_aligned.shape}")
            
        # Apply log transform to X if requested
        if log_transform:
            X_transformed = apply_log_transform(X_aligned)
            X_transformed_desaligned = apply_log_transform(X_desaligned)
        else:
            X_transformed = X_aligned.copy()
            X_transformed_desaligned = X_desaligned.copy()
        
        # Create lag features
        X_with_lags = add_lags(X_transformed, lags)
        X_with_lags_desaligned = add_lags(X_transformed_desaligned, lags)
        
        # Add each feature to the Prophet dataframe
        for col in X_with_lags.columns:
            df[col] = X_with_lags[col].values
            if X_df_desaligned is not None:
                X_df_desaligned[col] = X_with_lags_desaligned[col].values
    
    # Add date dummies if specified
    if date_dummies:
        df = add_date_dummies(df, date_dummies)
        X_df_desaligned = add_date_dummies(X_df_desaligned, date_dummies)
    
    print(f"Prepared data for Prophet with {df.shape[1]} columns")
    return df, X_df_desaligned


def split_data(df, test_size=30):
    """Split data into train and test sets."""
    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()
    print(f"Train set: {len(train)} rows, Test set: {len(test)} rows")
    return train, test


def train_prophet_model(train_df, X_cols=None, yearly_seasonality=True, weekly_seasonality=True, 
                       daily_seasonality=False, country_code=None):
    """Train a Prophet model with optional regressors and enhanced holidays."""
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
        
        # Use the new function to create enhanced holidays
        holidays_df = create_enhanced_holidays(
            country_code=country_code,
            start_date=min_date - timedelta(days=365),
            end_date=max_date + timedelta(days=365)
        )
        
        # Add holidays to model
        model_kwargs['holidays'] = holidays_df
        model_kwargs['holidays_prior_scale'] = 50.0  # Global scale for all holidays
        print(f"Added {len(holidays_df)} enhanced holidays from {country_code}")
        
        # Print holidays in training period for debugging
        if len(holidays_df) > 0:
            holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
            
            in_range_holidays = holidays_df[
                (holidays_df['ds'] >= min_date) & 
                (holidays_df['ds'] <= max_date)
            ]
            
            if len(in_range_holidays) == 0:
                print("  WARNING: No holidays found within the training date range!")
            else:
                print(f"  Found {len(in_range_holidays)} holidays in training range")
                for _, row in in_range_holidays.head(10).iterrows():
                    holiday_date = row['ds'].strftime('%Y-%m-%d')
                    holiday_name = row['holiday']
                    prior_scale = row.get('prior_scale', model_kwargs['holidays_prior_scale'])
                    print(f"  • {holiday_date}: {holiday_name} (prior_scale={prior_scale})")
                if len(in_range_holidays) > 10:
                    print(f"  • ... and {len(in_range_holidays) - 10} more holidays")

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


def evaluate_exogenous_variables(y, X, max_lag=70, correlation_types=['pearson', 'spearman'], alpha=0.05):
    """
    Evaluate potential exogenous variables through correlation analysis and Granger causality tests.
    
    Parameters:
    -----------
    y : pd.Series
        Target time series
    X : pd.DataFrame or pd.Series
        DataFrame containing potential exogenous variables
    max_lag : int, default=28
        Maximum lag to consider for Granger causality tests
    correlation_types : list, default=['pearson', 'spearman']
        Types of correlation coefficients to calculate
    alpha : float, default=0.05
        Significance level for Granger causality tests
    
    Returns:
    --------
    dict
        Dictionary containing evaluation results
    """
    results = {
        'correlation': {},
        'granger_causality': {},
        'recommended_vars': []
    }
    
    # Ensure X is a DataFrame
    if isinstance(X, pd.Series):
        X = pd.DataFrame({X.name: X})
        
    print(f"Evaluating {len(X.columns)} potential exogenous variables...")
    
    # CRITICAL FIX: Align the series to ensure they cover the same date range
    common_idx = y.index.intersection(X.index)
    if len(common_idx) == 0:
        print("ERROR: No common dates found between target and covariates!")
        return results
    
    y_aligned = y.loc[common_idx]
    X_aligned = X.loc[common_idx]
    
    print(f"Aligned data: {len(y_aligned)} points (target: {len(y)}, covariates: {len(X)})")
    
    # 1. Calculate correlation for each variable using aligned data
    for col in X_aligned.columns:
        results['correlation'][col] = {}
        
        # Calculate contemporaneous correlation
        for corr_type in correlation_types:
            corr = X_aligned[col].corr(y_aligned, method=corr_type)
            results['correlation'][col]['contemporaneous_' + corr_type] = corr
        
        # Calculate lagged correlation
        lag_corrs = []
        for lag in range(1, min(max_lag + 1, len(X_aligned) // 4)):
            for corr_type in correlation_types:
                # Ensure both arrays have the same length
                y_lagged = y_aligned[lag:].values
                x_lagged = X_aligned[col][:-lag].values
                
                if len(y_lagged) == len(x_lagged):
                    if corr_type == 'pearson':
                        corr = np.corrcoef(y_lagged, x_lagged)[0, 1]
                    else:  # spearman
                        corr = stats.spearmanr(y_lagged, x_lagged)[0]
                    
                    if not np.isnan(corr):
                        lag_corrs.append((lag, corr_type, corr))
                else:
                    print(f"Warning: Length mismatch for lag {lag}, y_shape={len(y_lagged)}, x_shape={len(x_lagged)}")
        
        # Store the max lagged correlation
        if lag_corrs:
            max_lag_corr = max(lag_corrs, key=lambda x: abs(x[2]))
            results['correlation'][col]['max_lagged'] = {
                'lag': max_lag_corr[0],
                'type': max_lag_corr[1],
                'value': max_lag_corr[2]
            }
    
    # 2. Perform Granger causality tests with aligned data
    # Create DataFrame with aligned data
    combined_data = pd.DataFrame({'y': y_aligned})
    for col in X_aligned.columns:
        combined_data[col] = X_aligned[col]
    
    # Remove any NaN values
    combined_data = combined_data.dropna()
    
    # Test for each variable
    for col in X.columns:
        data = combined_data[['y', col]].values
        granger_results = {}
        
        # Test for lags 1, 2, 4, 8, etc. up to max_lag
        lags_to_test = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84]
        lags_to_test = [lag for lag in lags_to_test if lag <= max_lag]
        
        for lag in lags_to_test:
            try:
                # Suppress FutureWarning about verbose parameter
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=FutureWarning)
                    gc_res = grangercausalitytests(data, maxlag=lag, verbose=False)

                # Extract p-values for different test statistics
                ssr_ftest_pval = gc_res[lag][0]['ssr_ftest'][1]
                granger_results[lag] = {
                    'ssr_ftest_pval': ssr_ftest_pval,
                    'significant': ssr_ftest_pval < alpha
                }
            except Exception as e:
                print(f"Granger test failed for {col} at lag {lag}: {e}")
                granger_results[lag] = {'error': 'Test failed'}
        
        results['granger_causality'][col] = granger_results
        
        # Check if any lag is significant
        any_significant = any(info.get('significant', False) for lag, info in granger_results.items())
        
        # Get best lag (smallest significant lag)
        if any_significant:
            best_lag = min([lag for lag, info in granger_results.items() 
                           if info.get('significant', False)], default=None)
        else:
            best_lag = None
        
        results['granger_causality'][col]['any_significant'] = any_significant
        results['granger_causality'][col]['best_lag'] = best_lag
    
    # 3. Recommend variables based on criteria
    for col in X.columns:
        # Strong contemporaneous correlation (absolute value > 0.3)
        if abs(results['correlation'][col].get('contemporaneous_pearson', 0)) > 0.3:
            results['recommended_vars'].append({
                'variable': col,
                'reason': 'Strong contemporaneous correlation',
                'value': results['correlation'][col].get('contemporaneous_pearson'),
                'score': abs(results['correlation'][col].get('contemporaneous_pearson', 0))
            })
        
        # Strong lagged correlation (absolute value > 0.3)
        if 'max_lagged' in results['correlation'][col]:
            if abs(results['correlation'][col]['max_lagged']['value']) > 0.3:
                results['recommended_vars'].append({
                    'variable': col,
                    'reason': f"Strong lagged correlation at lag {results['correlation'][col]['max_lagged']['lag']}",
                    'value': results['correlation'][col]['max_lagged']['value'],
                    'score': abs(results['correlation'][col]['max_lagged']['value'])
                })
        
        # Significant Granger causality
        if results['granger_causality'][col].get('any_significant', False):
            best_lag = results['granger_causality'][col].get('best_lag')
            pval = results['granger_causality'][col][best_lag]['ssr_ftest_pval']
            results['recommended_vars'].append({
                'variable': col,
                'reason': f"Granger causes target at lag {best_lag} (p={pval:.4f})",
                'value': 1 - pval,  # Convert p-value to a "strength" score
                'score': 1 - pval
            })
    
    # Remove duplicates and sort by strength
    seen = set()
    unique_recommendations = []
    for rec in sorted(results['recommended_vars'], key=lambda x: x['score'], reverse=True):
        if rec['variable'] not in seen:
            unique_recommendations.append(rec)
            seen.add(rec['variable'])
    
    results['recommended_vars'] = unique_recommendations
    
    return results


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
    bars = plt.barh(regressors, values, color='slategrey')
    
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



def generate_future_forecast(model, forecast, future_periods, X_cols=None, X_df=None, 
                          date_dummies=None, log_transform=True):
    """
    Generate forecast for future periods beyond the existing forecast.
    
    Parameters:
    -----------
    model : Prophet model
        The trained Prophet model
    forecast : DataFrame
        The existing forecast DataFrame from model.predict()
    future_periods : int
        Number of periods to forecast into the future
    X_cols : list, optional
        List of regressor column names
    X_df : DataFrame, optional
        Original covariate DataFrame with potential future values
    date_dummies : list of dict, optional
        Custom date patterns to add as dummy variables
    log_transform : bool, default=True
        Whether log transform was applied
        
    Returns:
    --------
    tuple
        (future_forecast, future_dates, future_values, future_lower, future_upper)
        - future_forecast: DataFrame with full Prophet forecast results
        - future_dates: DatetimeIndex of future dates
        - future_values: array of predicted values (inverse transformed if needed)
        - future_lower: array of lower bounds (inverse transformed if needed)
        - future_upper: array of upper bounds (inverse transformed if needed)
    """
    if future_periods <= 0:
        return None, None, None, None, None
        
    try:
        print("\nGenerating future forecast...")
        
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
            
            # Categorize columns by type
            date_pattern_cols = {
                'weekday': [],      # Features based on day of week (0-6)
                'weekend': [],      # Weekend indicators
                'month': [],        # Month-based features
                'day': [],          # Day of month features
                'dummy_dates': [],  # Holiday or specific date dummies
                'continuous': []    # Regular continuous variables
            }
            
            # Classify each column based on its name pattern
            for col in X_cols:
                if col == 'is_weekend':
                    date_pattern_cols['weekend'].append(col)
                elif col.startswith('is_dayofweek_'):
                    date_pattern_cols['weekday'].append(col)
                elif col.startswith('is_month_'):
                    date_pattern_cols['month'].append(col)
                elif col.startswith('is_day_'):
                    date_pattern_cols['day'].append(col)
                elif col.endswith('_dummy'):
                    date_pattern_cols['dummy_dates'].append(col)
                else:
                    date_pattern_cols['continuous'].append(col)
            
            # 1. Generate weekend features
            for col in date_pattern_cols['weekend']:
                # Change this line:
                # future[col] = (future_dates.dt.dayofweek >= 5).astype(int)
                # To:
                future[col] = (future_dates.dayofweek >= 5).astype(int)
                print(f"Calculated {col} based on weekend days (Sat/Sun)")
            
            # 2. Generate day of week features
            for col in date_pattern_cols['weekday']:
                try:
                    # Extract day number from column name (is_dayofweek_1 -> 1)
                    day_num = int(col.split('_')[-1])
                    # Change this line:
                    # future[col] = (future_dates.dt.dayofweek == (day_num - 1) % 7).astype(int)
                    # To:
                    future[col] = (future_dates.dayofweek == (day_num - 1) % 7).astype(int)
                    print(f"Calculated {col} based on day of week {day_num}")
                except (ValueError, IndexError):
                    print(f"Warning: Couldn't parse day number from {col}, using zero")
                    future[col] = 0
            
            # 3. Generate month-based features
            for col in date_pattern_cols['month']:
                try:
                    # Extract month number from column name (is_month_1 -> 1)
                    month_num = int(col.split('_')[-1])
                    # Change this line:
                    # future[col] = (future_dates.dt.month == month_num).astype(int)
                    # To:
                    future[col] = (future_dates.month == month_num).astype(int)
                    print(f"Calculated {col} based on month {month_num}")
                except (ValueError, IndexError):
                    print(f"Warning: Couldn't parse month number from {col}, using zero")
                    future[col] = 0
            
            # 4. Generate day of month features
            for col in date_pattern_cols['day']:
                try:
                    # Extract day number from column name (is_day_1 -> 1)
                    day_num = int(col.split('_')[-1])
                    # Change this line:
                    # future[col] = (future_dates.dt.day == day_num).astype(int)
                    # To:
                    future[col] = (future_dates.day == day_num).astype(int)
                    print(f"Calculated {col} based on day of month {day_num}")
                except (ValueError, IndexError):
                    print(f"Warning: Couldn't parse day number from {col}, using zero")
                    future[col] = 0
                    
            # 5. Handle dummy date variables through the existing mechanism
            if date_pattern_cols['dummy_dates'] and date_dummies:
                relevant_patterns = [p for p in date_dummies if p['name'] in date_pattern_cols['dummy_dates']]
                if relevant_patterns:
                    future = add_date_dummies(future, relevant_patterns)
            
            # 6. For continuous values, try to use actual data from X_df
            for col in date_pattern_cols['continuous']:
                if col in forecast.columns:
                    # Get the original covariate data loaded at the beginning of the pipeline
                    original_covariate = X_df[col] if isinstance(X_df, pd.DataFrame) and col in X_df else None
                    
                    if original_covariate is not None:
                        # Try to map future dates to actual values in the original dataset
                        future_filled = False
                        future_values = []
                        
                        for future_date in future_dates:
                            if future_date in original_covariate.index:
                                # Use actual value from dataset for this date
                                future_values.append(original_covariate.loc[future_date])
                                future_filled = True
                            else:
                                # If date not found, use last known value
                                future_values.append(X_df[col].iloc[-1])
                        
                        if future_filled:
                            # At least some dates were found in the original dataset
                            print(f"Using actual values from original dataset for {col} where available")
                            future[col] = future_values
                        else:
                            # No dates found, fall back to last known value
                            last_value = forecast[col].iloc[-1]
                            print(f"No future data found in original dataset. Using last known value {last_value:.4f} for {col}")
                            future[col] = last_value
                    else:
                        # Original covariate not available, use last known value
                        last_value = forecast[col].iloc[-1]
                        print(f"Using last known value {last_value:.4f} for {col}")
                        future[col] = last_value
                else:
                    print(f"Warning: {col} not found in forecast. Using zero.")
                    future[col] = 0
        
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
        
        # Verify forecast looks reasonable compared to historical data
        test_mean = np.mean(forecast['yhat'].values[-future_periods:])
        if log_transform:
            test_mean = np.expm1(test_mean)
            
        future_mean = np.mean(future_yhat)
        ratio = future_mean / test_mean if test_mean > 0 else float('inf')
        
        if ratio < 0.1 or ratio > 10:
            print(f"WARNING: Future forecast mean ({future_mean:.2f}) is very different from recent historical mean ({test_mean:.2f})")
        
        return future_forecast, future_dates, future_yhat, future_yhat_lower, future_yhat_upper
        
    except Exception as e:
        print(f"Error generating future forecast: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None



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
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.5])
    
    # Create the six subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Scatter plot (top-left)
    ax4 = fig.add_subplot(gs[0, 1])  # Residuals histogram (top-middle-left)
    ax3 = fig.add_subplot(gs[0, 2])  # Monthly plot (top-middle-right)
    ax5 = fig.add_subplot(gs[0, 3])  # Weekly seasonality (top-right-1)
    ax6 = fig.add_subplot(gs[0, 4])  # Monthly seasonality (top-right-2)
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
                transform=ax1.transAxes, fontsize=8, verticalalignment='top',
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
    ax1.legend(fontsize=8)


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
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Calculate overall average of actual values
    overall_mean_actual = np.mean(actuals)
    
    # Group by day of week and calculate mean and std
    dow_grouped = seasonal_df.groupby('day_of_week').agg({
        'actual': ['mean', 'std'], 
        'fitted': ['mean', 'std'],
        'residual': ['mean', 'std']
    }).reset_index()

    # Normalize means around 1.0 (percentage of overall average)
    dow_grouped['actual', 'normalized_mean'] = dow_grouped['actual', 'mean'] / overall_mean_actual

    # Set up x positions for plotting
    x = np.arange(len(day_names))

    # Plot normalized mean lines
    ax5.plot(x, dow_grouped['actual', 'normalized_mean'], 'o-', color='slategray', 
            linewidth=2, markersize=8, label='Actual (% of avg)')

    # Plot per-day std ranges as shaded areas - using actual std values for each day
    upper_bound = dow_grouped['actual', 'normalized_mean'] + dow_grouped['actual', 'std'] / overall_mean_actual
    lower_bound = dow_grouped['actual', 'normalized_mean'] - dow_grouped['actual', 'std'] / overall_mean_actual
    ax5.fill_between(x, lower_bound, upper_bound, color='slategray', alpha=0.2, label='±1σ per day')

    # Add horizontal line at 1.0 to represent the average
    ax5.axhline(y=1.0, color='tab:orange', linestyle='--', alpha=0.7, label='Overall Average')
    
    # Set x-axis labels to day names
    ax5.set_xticks(x)
    ax5.set_xticklabels(day_names, rotation=45)

    # Format y-axis as percentage
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

    # Add labels and title
    ax5.set_xlabel('Day of Week')
    ax5.set_ylabel('Relative to Average (1.0 = Average)')
    ax5.set_title('Weekly Seasonal Pattern')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.legend(fontsize=8)

    
    # SUBPLOT 6: MONTHLY SEASONALITY
    # Month names for x-axis
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Group by month and calculate statistics
    month_grouped = seasonal_df.groupby('month').agg({
        'actual': ['mean', 'std', 'count'], 
        'fitted': ['mean', 'std'],
        'residual': ['mean', 'std']
    }).reset_index()

    # Normalize means around 1.0 (percentage of overall average)
    month_grouped['actual', 'normalized_mean'] = month_grouped['actual', 'mean'] / overall_mean_actual

    # Valid months and normalized values
    valid_months = month_grouped['month'].values - 1  # 0-based indexing for plotting
    normalized_means = month_grouped['actual', 'normalized_mean'].values

    # Calculate per-month upper and lower bounds using each month's actual std
    upper_bounds = normalized_means + month_grouped['actual', 'std'].values / overall_mean_actual
    lower_bounds = normalized_means - month_grouped['actual', 'std'].values / overall_mean_actual

    # Plot normalized monthly pattern
    ax6.plot(valid_months, normalized_means, 'o-', color='slategray', linewidth=2, markersize=8, label='Actual (% of avg)')

    # Plot per-month std ranges
    ax6.fill_between(valid_months, lower_bounds, upper_bounds, color='slategray', alpha=0.2, label='±1σ per month')

    # Add horizontal line at 1.0 to represent the average
    ax6.axhline(y=1.0, color='tab:orange', linestyle='--', alpha=0.7, label='Overall Average')

    # Format y-axis as percentage
    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    
    # Set x-axis labels to month names for valid months
    ax6.set_xticks(valid_months)
    ax6.set_xticklabels([month_names[i] for i in valid_months], rotation=45)

    # Add labels and title for monthly seasonality
    ax6.set_xlabel('Month')
    ax6.set_ylabel('Relative to Average (1.0 = Average)')
    ax6.set_title('Monthly Seasonal Pattern')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.legend(fontsize=8)


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
    ax2.legend(fontsize=8)
    
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
    monthly_df = ts_df.resample('ME').mean()
    
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
                transform=ax3.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    except Exception as e:
        print(f"Error calculating monthly statistics: {e}")
    
    # Add labels to monthly plot
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Monthly Average Value')
    ax3.set_title(f"{title} (Monthly Aggregation)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    
    # Format x-axis dates to prevent overcrowding on all subplots
    # fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_forecast(model, forecast, y_full, test_size, X_cols=None, date_dummies=None, log_transform=True, title="Prophet Forecast", future_periods=0, history_days=70, X_df=None):
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
    ax.plot(test_dates, test_values, color='slategray', linewidth=2, label='Test Actuals')
    
    # Plot predictions over the entire range
    ax.plot(forecast_dates, yhat, color='tab:red', linewidth=2, alpha=0.7, label='Predicted') # linestyle='dashed',
    
    # Plot confidence intervals
    # ax.fill_between(forecast_dates, yhat_lower, yhat_upper, color='orange', alpha=0.2, label='95% Confidence Interval')
    
    # Add future forecast if requested
    if future_periods > 0:
        # Use the new function
        future_forecast, future_dates, future_yhat, future_yhat_lower, future_yhat_upper = generate_future_forecast(
            model, forecast, future_periods, X_cols, X_df, date_dummies, log_transform
        )
        
        if future_forecast is not None:
            # Plot the future forecast
            ax.plot(future_dates, future_yhat, color='tab:orange', linewidth=2, label='Forecast')
            ax.fill_between(future_dates, future_yhat_lower, future_yhat_upper, color='tab:orange', alpha=0.1)
    
    # Add vertical line to separate train and test data
    ax.axvline(x=train_dates[-1], color='slategray', linewidth=1, linestyle='--')
    ax.axvline(x=test_dates[-1], color='slategray', linewidth=1, linestyle='--')

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


def plot_variable_evaluation_enhanced(eval_results, y, X):
    """
    Simplified visualization of covariate evaluation results.
    Shows Granger causality, scatter plots, and monthly time series for each covariate.
    """
    # Get variable names
    variables = list(eval_results['correlation'].keys())
    if not variables:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No variables to evaluate", ha='center', va='center')
        return fig
    
    # Align the series to ensure they cover the same date range
    common_idx = y.index.intersection(X.index)
    if len(common_idx) == 0:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "ERROR: No common dates found", ha='center', va='center')
        return fig
    
    y_aligned = y.loc[common_idx]
    X_aligned = X.loc[common_idx]
    
    print(f"Visualization: Using {len(common_idx)} aligned data points for {len(variables)} covariates")
    
    # Create figure with appropriate size (3 columns per variable)
    fig = plt.figure(figsize=(20, 5 * len(variables)))
    
    # For each variable, create a row of plots
    for var_idx, var_name in enumerate(variables):
        print(f"Plotting analysis for covariate: {var_name}")
        
        # === GRANGER CAUSALITY PLOT (left) ===
        ax_granger = fig.add_subplot(len(variables), 3, var_idx*3 + 1)
        
        if 'granger_causality' in eval_results and var_name in eval_results['granger_causality']:
            granger_results = eval_results['granger_causality'][var_name]
            
            # Extract lags and p-values
            lags = []
            pvals = []
            significant = []
            
            for lag, lag_results in granger_results.items():
                if isinstance(lag, int) and 'ssr_ftest_pval' in lag_results:
                    lags.append(lag)
                    pval = lag_results['ssr_ftest_pval']
                    pvals.append(pval)
                    significant.append(lag_results.get('significant', False))
            
            if lags:
                # Plot p-values for each lag
                bar_colors = ['slategrey' if sig else 'tab:red' for sig in significant]
                ax_granger.bar(lags, pvals, color=bar_colors, alpha=0.7)
                
                # Add significance threshold line
                alpha = 0.05  # Common significance threshold
                ax_granger.axhline(y=alpha, color='black', linestyle='--', alpha=0.5)
                ax_granger.text(max(lags), alpha, f'p={alpha} threshold', va='bottom', ha='right', 
                        bbox=dict(facecolor='white', alpha=0.8))
                
                # Determine the best (smallest significant) lag
                best_lag = None
                if any(significant):
                    best_lag = min([lag for i, lag in enumerate(lags) if significant[i]])
                    
                if best_lag is not None:
                    best_idx = lags.index(best_lag)
                    # Highlight the best lag
                    ax_granger.bar([best_lag], [pvals[best_idx]], color='tab:orange', alpha=1.0)
                    ax_granger.annotate(f"Best lag: {best_lag}\np={pvals[best_idx]:.4f}", 
                                xy=(best_lag, pvals[best_idx]), 
                                xytext=(best_lag, pvals[best_idx] + 0.1),
                                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                                ha='center', va='bottom',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="k"))
                    
                    # Add visual indicator for Granger causality at this lag
                    ax_granger.text(0.02, 0.02, 
                            f"{var_name} Granger causes Target\nat lag {best_lag} (p={pvals[best_idx]:.4f})",
                            transform=ax_granger.transAxes, 
                            fontsize=10, ha='left', va='bottom',
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                else:
                    ax_granger.text(0.5, 0.5, f"No significant Granger causality found", 
                            transform=ax_granger.transAxes, ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                    
                ax_granger.set_xticks(lags)
                ax_granger.set_xticklabels([str(lag) for lag in lags])
                ax_granger.set_yscale('log')  # Log scale for better visualization
                ax_granger.set_ylim(top=1.0)  # Maximum p-value is 1.0
                
                ax_granger.set_title(f'Granger Causality: {var_name} → Target')
                ax_granger.set_xlabel('Lag')
                ax_granger.set_ylabel('p-value (log scale)')
                ax_granger.grid(True, axis='y', alpha=0.3)
            else:
                ax_granger.text(0.5, 0.5, f"No Granger causality results", 
                        transform=ax_granger.transAxes, ha='center', va='center')
        
        # === SCATTER PLOT (middle) ===
        ax_scatter = fig.add_subplot(len(variables), 3, var_idx*3 + 2)
        
        # Get data for this variable
        var_data = X_aligned[var_name].values
        target_data = y_aligned.values
        
        # Plot scatter
        ax_scatter.scatter(var_data, target_data, alpha=0.5, color='slategrey', edgecolors='w', linewidth=0.5)
        
        # Add trend line
        try:
            mask = ~np.isnan(var_data) & ~np.isnan(target_data)
            if mask.sum() > 1:  # Need at least 2 points for regression
                z = np.polyfit(var_data[mask], target_data[mask], 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(var_data[mask]), max(var_data[mask]), 100)
                ax_scatter.plot(x_range, p(x_range), color='tab:orange', linewidth=2.5, alpha=0.7)
                
                # Add R²
                from scipy import stats
                r_squared = stats.pearsonr(var_data[mask], target_data[mask])[0]**2
                ax_scatter.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax_scatter.transAxes,
                        va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8))
        except Exception as e:
            print(f"Error fitting trend line for {var_name}: {e}")
        
        ax_scatter.set_title(f'Target vs. {var_name}')
        ax_scatter.set_xlabel(var_name)
        ax_scatter.set_ylabel('Target')
        ax_scatter.grid(True, alpha=0.3)
        
        # === NEW: MONTHLY TIME SERIES PLOT (right) ===
        ax_monthly = fig.add_subplot(len(variables), 3, var_idx*3 + 3)
        
        # Create DataFrames for resampling
        target_series = pd.Series(target_data, index=y_aligned.index, name='Target')
        covar_series = pd.Series(var_data, index=X_aligned.index, name=var_name)
        
        # Resample to monthly frequency and take the mean
        monthly_target = target_series.resample('ME').mean()
        monthly_covar = covar_series.resample('ME').mean()
        
        # Plot monthly target data on left y-axis
        color1 = 'slategrey'
        ax_monthly.plot(monthly_target.index, monthly_target.values, 
                      color=color1, linewidth=2, 
                      label='Target')
        ax_monthly.set_ylabel('Target Value', color=color1)
        ax_monthly.tick_params(axis='y', labelcolor=color1)
        
        # Create twin axis for covariate
        ax_monthly_twin = ax_monthly.twinx()
        
        # Plot monthly covariate data on right y-axis
        color2 = 'tab:red'
        ax_monthly_twin.plot(monthly_covar.index, monthly_covar.values, 
                           color=color2, linewidth=2, alpha=0.7,
                           label=var_name)
        ax_monthly_twin.set_ylabel(f'{var_name} Value', color=color2)
        ax_monthly_twin.tick_params(axis='y', labelcolor=color2)
        
        # Calculate correlation at monthly level
        monthly_df = pd.DataFrame({
            'target': monthly_target,
            'covariate': monthly_covar
        })
        monthly_df = monthly_df.dropna()  # Remove any NaN values
        
        if len(monthly_df) > 1:
            monthly_corr = monthly_df['target'].corr(monthly_df['covariate'])
            ax_monthly.text(0.05, 0.95, f'Monthly Correlation: {monthly_corr:.3f}', 
                          transform=ax_monthly.transAxes, fontsize=10,
                          va='top', ha='left', 
                          bbox=dict(facecolor='white', alpha=0.8))
        
        # Format x-axis dates
        import matplotlib.dates as mdates
        ax_monthly.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_monthly.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Every 6 months
        plt.setp(ax_monthly.xaxis.get_majorticklabels(), rotation=45)
        
        # Add grid
        ax_monthly.grid(True, alpha=0.3)
        
        # Add combined legend
        lines1, labels1 = ax_monthly.get_legend_handles_labels()
        lines2, labels2 = ax_monthly_twin.get_legend_handles_labels()
        ax_monthly.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
                        bbox_to_anchor=(0.5, -0.15), ncol=2)
        
        ax_monthly.set_title(f'Monthly {var_name} vs. Target')
    
    # Add overall title
    fig.suptitle("Variable Evaluation Analysis", fontsize=16, y=0.99)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.4, bottom=0.1)
    
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

        
        # Export regressor importance plot using stored figure
        if 'reg_importance_fig' in results and results['reg_importance_fig'] is not None:
            reg_imp_fig = results['reg_importance_fig']
            reg_imp_plot_path = os.path.join(plots_dir, f'regressor_importance_plot_{timestamp}.png')
            reg_imp_fig.savefig(reg_imp_plot_path)
            plt.close(reg_imp_fig)
            print(f"Exported regressor importance plot to: {reg_imp_plot_path}")
    
        # Export covariate evaluation plot if available
        if 'covariate_eval_fig' in results and results['covariate_eval_fig'] is not None:
            eval_fig = results['covariate_eval_fig']
            eval_fig_path = os.path.join(plots_dir, f'covariate_evaluation_plot_{timestamp}.png')
            eval_fig.savefig(eval_fig_path)
            plt.close(eval_fig)
            print(f"Exported covariate evaluation plot to: {eval_fig_path}")


    return {
        'model_path': model_path,
        'forecast_path': forecast_path,
        'components_path': components_path,
        'covariate_eval_path': eval_fig_path,
    }


def run_prophet_pipeline(
    target_path, 
    covariate_path=None, 
    covariate_cols=None,
    test_size=35,
    future_periods=35,
    lags=[1, 7, 14], 
    log_transform=True,
    yearly_seasonality=True,
    weekly_seasonality=True,
    country_code=None,
    date_dummies=None,
    evaluate_covariates=True,
    max_lag_for_evaluation=70,
    use_prophet_holidays=False
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
    evaluate_covariates : bool, default=True
        Whether to perform correlation and Granger causality tests on covariates
    max_lag_for_evaluation : int, default=28
        Maximum lag to consider for covariate evaluation
    """
    print("\n--- Starting Prophet Pipeline ---\n")
    
    # 1. Load data
    print("Loading data...")
    y, X = load_data(target_path, covariate_path, covariate_cols=covariate_cols)

    # Convert X to DataFrame if it's a Series
    X_df = X.to_frame() if isinstance(X, pd.Series) else X

    # Evaluate covariates if requested and if covariates exist
    eval_results = None
    eval_fig = None
    if evaluate_covariates and X is not None:
        print("\nEvaluating potential exogenous variables...")
        
        # Run evaluation
        eval_results = evaluate_exogenous_variables(y, X_df, max_lag=max_lag_for_evaluation)

        # Create enhanced visualization
        eval_fig = plot_variable_evaluation_enhanced(eval_results, y, X_df)
        
        # Print recommendations
        if eval_results['recommended_vars']:
            print("\nRecommended variables based on statistical tests:")
            for i, rec in enumerate(eval_results['recommended_vars']):
                print(f"{i+1}. {rec['variable']} - {rec['reason']}")
        else:
            print("\nNo variables recommended based on statistical tests.")
    
    # 2. Prepare data for Prophet
    print("\nPreparing data for Prophet...")
    prophet_df, prophet_X_df = prepare_prophet_data(y, X, X_df, lags, log_transform, date_dummies)

    # Create holiday dummy variables if country_code is specified but we're not using Prophet's holidays
    if country_code and not use_prophet_holidays:
        print(f"\nGenerating holiday dummies for {country_code}...")
        # Initialize date_dummies if needed
        if date_dummies is None:
            date_dummies = []
            
        # Get min and max dates from the data
        min_date = y.index.min()
        max_date = y.index.max()
        
        # Generate holiday dummies based on country code
        if country_code.upper() == 'BR':
            holiday_dummies = create_brazil_holiday_dummies(min_date, max_date)
            date_dummies.extend(holiday_dummies)
            print(f"Added {len(holiday_dummies)} Brazilian holiday dummies to date_dummies")
    
    # 3. Split data
    print("\nSplitting data into train and test sets...")
    train_df, test_df = split_data(prophet_df, test_size)
    
    # 4. Determine regressor columns
    X_cols = None
    if X is not None:
        # Get all columns except 'ds' and 'y'
        X_cols = [col for col in prophet_df.columns if col not in ['ds', 'y']]
    
    # 5. Train model
    print("\nTraining Prophet model...")
    model = train_prophet_model(
        train_df, 
        X_cols=X_cols, 
        yearly_seasonality=yearly_seasonality, 
        weekly_seasonality=weekly_seasonality,
        country_code=country_code if use_prophet_holidays else None
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
        future_periods=future_periods,
        X_df = prophet_X_df
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
        # 'components_fig': components_fig,
        'forecast_fig': forecast_fig,
        'reg_importance_fig': reg_importance_fig,
        'covariate_evaluation': eval_results,
        'covariate_eval_fig': eval_fig
    }


# Example usage
if __name__ == "__main__":
    TARGET_PATH = 'data/groupby_train.csv'
    COVARIATE_PATH = 'data/groupby_transactions_2.csv'

    # List all covariates to include
    COVARIATE_COLS = [
        'transactions',     
        'is_weekend',      
        # 'is_dayofweek_3',
        # 'is_dayofweek_4',
    ]

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
        covariate_cols=COVARIATE_COLS,
        test_size=35,
        future_periods=35,
        lags=[1, 7, 14, 28],
        log_transform=True,
        country_code='BR',
        date_dummies=date_dummies,
        use_prophet_holidays=False
    )

    # # Export model artifacts
    # print("\nExporting model artifacts...")
    # export_paths = export_model_artifacts(results)
    # print("\nExport complete. Files saved to:")
    # for key, path in export_paths.items():
    #     print(f"- {key}: {path}")