import os
import warnings
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, date
from scipy import stats
from prophet import Prophet
from prophet.utilities import regressor_coefficients
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import grangercausalitytests
from dateutil.easter import easter # For holiday calculation
from matplotlib.gridspec import GridSpec
from pandas.tseries.offsets import MonthEnd, MonthBegin
from tqdm import tqdm # Import tqdm for progress bar

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

def calendar_events_features(start_date=None,
                             end_date=None, 
                             weekdays: bool = True, 
                             workingdays: bool = False,
                             lags: int = None, 
                             leads: int = None,
                             slope_coef: int = 0,
                             drift_date=None,
                             event_names: list = None):
    
    # Initialize columns dictionary
    columns_by_events_types = {}
    columns_by_events = {}

    # Set start and end date range 
    start_date = date.today() - timedelta(days=10*365) if start_date is None else start_date
    end_date = date.today() + timedelta(days=10*365) if end_date is None else end_date

    start_date = datetime.strptime(start_date, "%Y-%m-%d").date() if isinstance(start_date, str) else start_date
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date() if isinstance(end_date, str) else end_date
    drift_date = datetime.strptime(drift_date, "%Y-%m-%d").date() if isinstance(drift_date, str) else drift_date

    # Extend the range by 1 year on each side for lag/lead features
    start = (start_date - timedelta(days=365))
    end = (end_date + timedelta(days=365))

    # Create a daily date range DataFrame
    date_range = pd.date_range(start=start, end=end, freq='D')
    data = pd.DataFrame(date_range, columns=['date'])

    # calculate the pulse value if the slope_coef is provided
    def pulse_value(x, y=start.year, coef=slope_coef):
        return (x - y) * coef + 1.0

    # List of fixed-date holidays
    EVENTS = {
        'fixed_date_events': [
            {'name': 'anonovo','month': 1,'day': 1},
            {'name': 'aniversariosp','month': 1,'day': 25},
            {'name': 'aniversariorj','month': 3,'day': 1},
            {'name': 'tiradentes','month': 3,'day': 21},
            {'name': 'diadotrabalho','month': 5,'day': 1},
            {'name': 'diadosnamorados','month': 6,'day': 12},
            {'name': 'saojoao','month': 6,'day': 24},
            {'name': 'revolucaoconst','month': 7,'day': 9},
            {'name': 'independencia','month': 9,'day': 7},
            {'name': 'aparecida','month': 10,'day': 12},
            {'name': 'servidorpublico','month': 10,'day': 28},
            {'name': 'finados','month': 11,'day': 2},
            {'name': 'proclamacao','month': 11,'day': 15},
            {'name': 'consciencianegra','month': 11,'day': 20},
            {'name': 'natal','month': 12,'day': 25},
        ],
        'easter_based_events': [
            {'name': 'carnaval','delta_days': -47},
            {'name': 'paixaocristo','delta_days': -2},
            {'name': 'pascoa','delta_days': 0},
            {'name': 'corpuschristi','delta_days': 60},
        ],
        'special_events': [
            {'name': 'blackfriday','month': 11,'weekday': 3,'delta_days': 22}, # day after US Thanksgiving, witch is the 4th Thursday of November
            {'name': 'diadospais','month': 8,'weekday': 6,'delta_days': 7}, # 2nd Sunday of August
            {'name': 'diadasmaes','month': 5,'weekday': 6,'delta_days': 7}, # 2nd Sunday of May
        ]
    }
    fixed_events = [event for event in EVENTS['fixed_date_events'] if event['name'] in event_names] if event_names else EVENTS['fixed_date_events']

    # Add fixed-date holidays to DataFrame
    for event in fixed_events:
        data['calendar_fixed_event_' + event['name']] = data['date'].apply(
            lambda x: pulse_value(x.year) if (x.month == event['month'] and x.day == event['day']) else 0.0
        )
        
    columns_by_events_types['fixed_events'] = [col for col in data.columns.to_list() if col != 'date']
    
    # Add Easter-based holidays to DataFrame
    easter_events = [event for event in EVENTS['easter_based_events'] if event['name'] in event_names] if event_names else EVENTS['easter_based_events']
    for holiday in easter_events:
        data['calendar_moving_event_' + holiday['name']] = data['date'].apply(
            lambda x: pulse_value(x.year) if (x.date() == (easter(x.year) + timedelta(days=holiday['delta_days']))) else 0.0
        )
    
    # Add special holidays to DataFrame
    special_events = [event for event in EVENTS['special_events'] if event['name'] in event_names] if event_names else EVENTS['special_events']
    def day_of_special_event(year,month,weekday,delta_days):
        fist_day_of_month = date(year, month, 1)
        days_to_weekday = (weekday - fist_day_of_month.weekday()) % 7
        first_weekday = fist_day_of_month + timedelta(days=days_to_weekday)
        return first_weekday + timedelta(days=delta_days)   
    for event in special_events:
        data['calendar_moving_event_' + event['name']] = data['date'].apply(
            lambda x: pulse_value(x.year) if (x.date() == (day_of_special_event(x.year, event['month'], event['weekday'], event['delta_days']))) else 0.0
        )

    columns_by_events_types['moving_events'] = [col for col in data.columns.to_list() if col not in columns_by_events_types['fixed_events'] and col != 'date']

    # Set date as index for easier filtering and joining
    data = data.set_index('date')

    # Add weekday-specific holiday columns if requested
    columns_by_events_types['fixed_events_per_weekday'] = []
    if weekdays:
        for weekday in range(7):
            weekday_name = '_wd' + str(weekday)
            mul_data_columns = [event for event in columns_by_events_types['fixed_events']]
            mul_data = data[mul_data_columns].copy()

            # create a column for the weekday
            mul_data[weekday_name] = 0.0
            mask = mul_data.index.weekday == weekday
            mul_data.loc[mask, weekday_name] = 1.0

            # multiply all holiday columns by the weekday column, so it will be 0.0 if the date's weekday is not that weekday
            mul_data = mul_data[mul_data_columns].mul(mul_data[weekday_name], axis=0)
            mul_data.columns = [col + weekday_name for col in mul_data_columns]

            # merge the weekday columns with the main data
            data = data.merge(mul_data, on='date', how='left')

        columns_by_events_types['fixed_events_per_weekday'] = [col for col in data.columns.to_list() if '_wd' in col]

    # Add working day column (1.0 for working days, 0.0 for weekends/holidays)
    columns_by_events_types['workingdays'] = []
    if workingdays:
        data['calendar_workingday'] = 1.0
        data.loc[data.index.weekday >= 5, 'calendar_workingday'] = 0.0
        for event in columns_by_events_types['fixed_events'] + columns_by_events_types['moving_events']:
            if 'blackfriday' not in event:
                data.loc[data[event] == 1.0, 'calendar_workingday'] = 0.0
        columns_by_events_types['workingdays'] = ['calendar_workingday']        

    # add a column for drift_date if provided: 1.0 if date >= drift_date, else 0.0
    if drift_date:
        data['calendar_drift'] = 0.0
        mask = data.index.date >= drift_date
        data.loc[mask, 'calendar_drift'] = 1.0

        # multiply all holiday columns by the drift column, so it will be 0.0 if the date is before drift_date
        data = data.drop(columns=['calendar_drift']).mul(data['calendar_drift'], axis=0)

    # Add lagged versions of holiday/weekday/workingday columns
    columns_by_events_types['events_lags'] = []
    if lags:
        for lag in range(1, lags + 1):
            laged_data_columns = columns_by_events_types['fixed_events'] + columns_by_events_types['moving_events'] + columns_by_events_types['fixed_events_per_weekday'] + columns_by_events_types['workingdays']
            laged_data = data[laged_data_columns].copy()
            laged_data = laged_data.shift(lag)
            laged_data.columns = [f"{col}_pre{lag}" for col in laged_data.columns]
            data = data.merge(laged_data, on='date', how='left')
            columns_by_events_types['events_lags'] += laged_data.columns.to_list()
    
    # Add lead versions of holiday/weekday/workingday columns
    columns_by_events_types['events_leads'] = []
    if leads:
        for lead in range(1, leads + 1):
            leaded_data_columns = columns_by_events_types['fixed_events'] + columns_by_events_types['moving_events'] + columns_by_events_types['fixed_events_per_weekday'] + columns_by_events_types['workingdays']
            leaded_data = data[leaded_data_columns].copy()
            leaded_data = leaded_data.shift(-lead)
            leaded_data.columns = [f"{col}_pos{lead}" for col in leaded_data.columns]
            data = data.merge(leaded_data, on='date', how='left')
            columns_by_events_types['events_leads'] += leaded_data.columns.to_list()
    
    # Filter data to the requested date range
    data = data[(data.index.date>=start_date) & (data.index.date<=end_date)]

    # list of events
    for event_type in EVENTS:
        for event in EVENTS[event_type]:
            columns_by_events.update({event['name']: [col for col in data.columns if event['name'] in col]})
    if workingdays:
        columns_by_events.update({'workingdays': [col for col in data.columns if '_workingday' in col]})

    return data, columns_by_events, columns_by_events_types


def calendar_seasonal_features(start_date=None, 
                               end_date=None, 
                               dayofweek: bool = True, 
                               dayofweekpermonth: bool = True, 
                               dayofmonth: bool = True, 
                               monthofyear: bool = True, 
                               drift_date=None,
                               drop_first: bool = False):

    # Set start and end date range 
    start_date = date.today() - timedelta(days=10*365) if start_date is None else start_date
    end_date = date.today() + timedelta(days=10*365) if end_date is None else end_date

    start_date = datetime.strptime(start_date, "%Y-%m-%d").date() if isinstance(start_date, str) else start_date
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date() if isinstance(end_date, str) else end_date
    drift_date = datetime.strptime(drift_date, "%Y-%m-%d").date() if isinstance(drift_date, str) else drift_date

    # Extend the range by 1 year on each side for context
    start = (start_date - timedelta(days=365))
    end = (end_date + timedelta(days=365))

    # Create a daily date range DataFrame
    date_range = pd.date_range(start=start, end=end, freq='D')
    data = pd.DataFrame(date_range, columns=['date'])

    # Add day of week feature
    if dayofweek:
        data['calendar_seas_dayofweek'] = data['date'].dt.strftime('%A') # e.g., Monday, Tuesday
        
    # Add day of week per month feature
    if dayofweekpermonth:
        data['calendar_seas_dayofweekpermonth'] = data['date'].dt.strftime('%a_%b') # e.g., Mon_Jan

    # Add day of month feature
    if dayofmonth:
        data['calendar_seas_dayofmonth'] = data['date'].dt.strftime('%d') # e.g., 01, 31

    # Add month of year feature
    if monthofyear:
        data['calendar_seas_monthofyear'] = data['date'].dt.strftime('%B') # e.g., January
    
    # Set date as index
    data = data.set_index('date')   

    # One-hot encode categorical features
    data = pd.get_dummies(data=data, drop_first=drop_first).astype(np.float32)

    # add a column for drift_date if provided: 1.0 if date >= drift_date, else 0.0
    if drift_date:
        data['calendar_drift'] = 0.0
        mask = data.index.date >= drift_date
        data.loc[mask, 'calendar_drift'] = 1.0

        # multiply all holiday columns by the drift column, so it will be 0.0 if the date is before drift_date
        data = data.drop(columns=['calendar_drift']).mul(data['calendar_drift'], axis=0)


    # Filter data to the requested date range and lowercase column names
    data = data[(data.index.date>=start_date) & (data.index.date<=end_date)]
    data.columns = data.columns.str.lower()

    # Build columns dictionary for reference
    columns_by_type = {
        'dayofweek': [col for col in data.columns if '_dayofweek_' in col],
        'dayofweekpermonth': [col for col in data.columns if '_dayofweekpermonth_' in col],
        'dayofmonth': [col for col in data.columns if '_dayofmonth_' in col],
        'monthofyear': [col for col in data.columns if '_monthofyear_' in col]
    }

    return data, columns_by_type

def feature_engeneering(X, 
                        log_transform, 
                        covariate_forecasting=True, 
                        future_periods=0,
                        lags=None, 
                        date_col=DEFAULT_DATE_COL, 
                        custom_date_dummies=None):

    # separate covariates from X to apply transformations (log and lags)
    transformable_cols = [col for col in X.columns if 'dummy' not in col.lower() and 'calendar' not in col.lower()]
    covariates = X[transformable_cols].copy()

    # apply log transformation if needed
    if log_transform:
        covariates = apply_log_transform(covariates)
    
    # forecast covariates if needed
    merged_covariates = covariates.copy()
    if covariate_forecasting:
        for col in covariates.columns.to_list():
            cov_df = covariates[col]\
                    .reset_index()\
                    .rename(columns={date_col: 'ds', col: 'y'})\
                    .copy()
            cov_model = Prophet().fit(cov_df)
            cov_future_dataframe = cov_model.make_future_dataframe(periods=future_periods)
            cov_forecast = cov_model\
                        .predict(cov_future_dataframe)\
                        .rename(columns={'ds': date_col})\
                        .set_index(date_col)
            merged_covariates = merged_covariates\
                                .merge(cov_forecast['yhat'], on=date_col, how='outer')\
                                .drop(columns=[col])\
                                .rename(columns={'yhat': col})
        covariates = merged_covariates

    # Apply lags to covariates if specified
    merged_covariates = covariates.copy()
    if lags:
        for lag in lags:
            lagged_covariates = covariates.copy().shift(lag)
            lagged_covariates.columns = [f"{col}_lag{lag}" for col in lagged_covariates.columns]
            merged_covariates = merged_covariates.merge(lagged_covariates, on=date_col, how='left')
        covariates = merged_covariates.bfill()
    
    # Merge non-transformable covariates back into the transformed DataFrame
    non_transformable_cols = [col for col in X.columns if col not in transformable_cols]
    covariates = covariates.merge(X[non_transformable_cols], on=date_col, how='left')

    # merge custom date dummy variables
    if custom_date_dummies:
        for dummy in custom_date_dummies:
            if isinstance(dummy, dict) and 'name' in dummy and 'dates' in dummy:
                dummy_dates = pd.to_datetime(dummy['dates'])
                covariates[dummy['name']] = np.where(covariates[date_col].isin(dummy_dates), 1, 0)
    
    # Merge calendar features
    calendar_events, cols_by_event_name, _ = calendar_events_features(
        start_date=str(covariates.index[0].date()), 
        end_date=str(covariates.index[-1].date()),
        lags=1, 
        leads=1, 
        workingdays=True
    )
    calendar_events = calendar_events.drop(columns=cols_by_event_name['carnaval']+cols_by_event_name['blackfriday'])
    other_events, _, _ = calendar_events_features(
        start_date=str(covariates.index[0].date()), 
        end_date=str(covariates.index[-1].date()),
        lags=2, 
        leads=2, 
        event_names=['carnaval', 'blackfriday']
    )
    calendar_events = calendar_events.merge(other_events, on=other_events.index.name ,how='left')
    calendar_events.index.rename(date_col, inplace=True)
    covariates = covariates.merge(calendar_events, on=date_col, how='left')

    # Merge seasonality features
    seasonality_features, cols_by_seas_type = calendar_seasonal_features(
        start_date=str(covariates.index[0].date()), 
        end_date=str(covariates.index[-1].date()),
    )
    seasonality_features.index.rename(date_col, inplace=True)
    covariates = covariates.merge(seasonality_features, on=date_col, how='left')

    covariates.index.rename('ds', inplace=True)
    covariates = covariates.drop(columns=transformable_cols, axis=1)

    return covariates

# --- Evaluation ---

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_covariates(y, X, max_lag=70, alpha=0.05):
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
                pval = gc_test[lag][0]['ssr_ftest'][1]
                is_significant = pval < alpha
                gc_results[lag] = {'pval': pval, 'significant': is_significant}
                if is_significant:
                    significant_lags.append(lag)
            except Exception as e:
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
        ln2 = ax_monthly_twin.plot(monthly_covar.index, monthly_covar.values, color=DEFAULT_COLOR_PREDICTED, alpha=0.7, label=f'{var_name} (Monthly Avg)')
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

def plot_model_fit(train_forecast, y_full, test_size, model, log_transform=True, title="Model Fit Diagnostics", show=True):
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

    # --- BIC Calculation ---
    n = len(actuals) # Number of data points
    k = 0 # Number of parameters (approximation)
    bic = np.nan

    if n > 0:
        # Estimate k (number of parameters) - This is an approximation for Prophet
        k += 1 # Noise variance sigma_obs
        if model.growth == 'linear':
            k += 1 # growth rate k
            k += 1 # offset m
            k += len(model.changepoints_t) # Number of changepoint deltas
        # Add seasonality parameters (Fourier terms * 2 for sin/cos)
        for name, props in model.seasonalities.items():
            k += 2 * props['fourier_order']
        # Add regressor parameters
        k += len(model.extra_regressors)
        # Add holiday parameters (if using built-in, but we use dummies)
        # if model.holidays is not None: k += len(model.holidays.columns) -1 # Exclude ds

        # Calculate Residual Sum of Squares (RSS)
        rss = np.sum(residuals**2)

        # Calculate Log-Likelihood (assuming normal errors)
        # LL = -n/2 * log(2*pi) - n/2 * log(RSS/n) - n/2
        if rss > 0: # Avoid log(0)
            log_likelihood = -n / 2.0 * np.log(2 * np.pi) - n / 2.0 * np.log(rss / n) - n / 2.0
            # Calculate BIC: k * log(n) - 2 * LL
            bic = k * np.log(n) - 2 * log_likelihood
        else: # Handle case of perfect fit (RSS=0), BIC is -inf technically
            bic = -np.inf

    print(f"  Estimated BIC: {bic:.2f} (n={n}, approx k={k})")
    # --- End BIC Calculation ---



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
        stats_text = f"R²: {r_squared:.4f}\nMAPE: {mape:.2f}%\nBIC: {bic:,.1f}" if not np.isnan(mape) else f"R²: {r_squared:.4f}\nBIC: {bic:,.1f}"
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
        # Recalculate monthly residuals for monthly BIC
        monthly_actuals = monthly_df['actual'].dropna().values
        monthly_fitted = monthly_df['fitted'].dropna().values
        monthly_residuals = monthly_actuals - monthly_fitted
        n_monthly = len(monthly_actuals)
        bic_monthly = np.nan
        if n_monthly > 0:
            rss_monthly = np.sum(monthly_residuals**2)
            if rss_monthly > 0:
                ll_monthly = -n_monthly / 2.0 * np.log(2 * np.pi) - n_monthly / 2.0 * np.log(rss_monthly / n_monthly) - n_monthly / 2.0
                # Use the same k as before, as it's model complexity, not data frequency dependent
                bic_monthly = k * np.log(n_monthly) - 2 * ll_monthly
            else:
                bic_monthly = -np.inf

        monthly_corr = np.corrcoef(monthly_actuals, monthly_fitted)[0, 1]
        monthly_r2 = monthly_corr ** 2 if not np.isnan(monthly_corr) else 0
        monthly_mape = calculate_mape(monthly_actuals, monthly_fitted)
        # --- MODIFIED: Add Monthly BIC to stats text ---
        monthly_stats = f"Monthly R²: {monthly_r2:.4f}\nMonthly MAPE: {monthly_mape:.2f}%\nMonthly BIC: {bic_monthly:,.1f}"
        # --- END MODIFIED ---
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

def plot_regressor_coefficients(model, show=True):
    print("Plotting regressor coefficients with uncertainty intervals...")
    if not model.extra_regressors:
        print("  No extra regressors found in the model.")
        return None

    try:
        # Use Prophet's utility to get coefficients including uncertainty intervals
        coef_df = regressor_coefficients(model)

        if coef_df.empty:
            print("  Could not extract regressor coefficients using Prophet utility.")
            return None

        # Sort by absolute coefficient value for better visualization
        coef_df = coef_df.reindex(coef_df['coef'].abs().sort_values(ascending=True).index)

        fig, ax = plt.subplots(figsize=(12, max(6, len(coef_df) * 0.4)))

        colors = []
        for _, row in coef_df.iterrows():
            if row['coef_lower'] < 0 < row['coef_upper']:
                colors.append(DEFAULT_COLOR_STANDARD) # Grey if interval contains zero
            elif row['coef'] > 0:
                colors.append(DEFAULT_COLOR_HIGHLIGHT) # Orange if positive and interval doesn't contain zero
            else:
                colors.append(DEFAULT_COLOR_COMPONENT) # Blue if negative and interval doesn't contain zero

        y_pos = np.arange(len(coef_df))

        # --- Plot each error bar and marker individually ---
        non_zero_errors_found = False

        for i, regressor_index in enumerate(coef_df.index):
            row = coef_df.loc[regressor_index]
            current_y = y_pos[i]
            current_coef = row['coef']
            current_color = colors[i]

            # Calculate asymmetric error (original calculation)
            lower_bound = row['coef_lower']
            upper_bound = row['coef_upper']
            current_lower_err = current_coef - lower_bound # Reverted max(0,...) just in case
            current_upper_err = upper_bound - current_coef # Reverted max(0,...) just in case
            current_xerr = [[current_lower_err], [current_upper_err]]

            is_error_visible = (abs(current_lower_err) > 1e-9 or abs(current_upper_err) > 1e-9) # Check absolute value
            if is_error_visible:
                non_zero_errors_found = True
                ax.errorbar(x=current_coef, y=current_y, xerr=current_xerr, fmt='none',
                            ecolor=current_color, elinewidth=2, capsize=4)

            ax.plot(current_coef, current_y, 'o', color='black', markersize=5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(coef_df['regressor'])

        ax.axvline(0, color='black', linestyle='--', alpha=0.7)

        # --- Explicitly set X-limits ---
        min_lim = coef_df['coef_lower'].min()
        max_lim = coef_df['coef_upper'].max()
        padding = (max_lim - min_lim) * 0.15 # Add 15% padding
        ax.set_xlim(min_lim - padding, max_lim + padding)
        # --- End set X-limits ---

        # --- Add coefficient value labels (optional, can get crowded) ---
        for i, row in coef_df.iterrows():
            width = row['coef']
            # Adjust label position slightly based on error bar extent
            label_x_pos = row['coef_upper'] + abs(row['coef_upper'])*0.10 if width >= 0 else row['coef_lower'] - abs(row['coef_lower'])*0.10
            ha = 'left' if width >= 0 else 'right'
            ax.text(label_x_pos, y_pos[coef_df.index.get_loc(i)],
                      f'{width:.3f}',
                      va='center', ha=ha, fontsize=7)
        # --- End Add labels ---

        ax.set_xlabel('Estimated Coefficient (Posterior Mean with 95% Interval)')
        ax.set_ylabel('Regressor')
        ax.set_title('Extra Regressor Coefficients and Uncertainty')
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()

        if show:
            plt.show()
        return fig

    except Exception as e:
        print(f"  ERROR plotting regressor coefficients: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging this plot
        return None

def simulate_rolling_forecast(
    prophet_df,
    y_full,
    regressor_cols,
    initial_train_size,
    simulation_days=30,
    forecast_horizon=35,
    log_transform=True,
    model_params=None
):
    print(f"\n--- Starting Rolling Forecast Simulation ({simulation_days} days, {forecast_horizon}-day horizon) ---")

    # Ensure output directory exists
    os.makedirs(DEFAULT_FORECASTS_DIR, exist_ok=True)

    all_forecasts_list = [] # List to store individual forecast DataFrames
    forecast_sums = {} # Store forecast sums keyed by the date they were made *on*
    monthly_actual_plus_forecast_sums = {} # Dictionary for monthly sums

    # Determine the date range for the simulation
    initial_train_end_date = prophet_df['ds'].iloc[initial_train_size - 1]
    simulation_start_date = initial_train_end_date + timedelta(days=1)
    simulation_end_date = initial_train_end_date + timedelta(days=simulation_days)

    print(f"Initial train end date: {initial_train_end_date.date()}")
    print(f"Simulation period: {simulation_start_date.date()} to {simulation_end_date.date()}")

    # Use tqdm for progress bar
    progress_bar = tqdm(range(simulation_days), desc="Rolling Forecast")

    for i in progress_bar:
        current_train_end_date = initial_train_end_date + timedelta(days=i)
        print(f"\nSimulating Day {i+1}/{simulation_days} (Training up to {current_train_end_date.date()})")
        # progress_bar.set_description(f"Rolling Forecast (Day {i+1}/{simulation_days}, Train End: {current_train_end_date.date()})")

        # 1. Prepare current training data
        current_train_df = prophet_df[prophet_df['ds'] <= current_train_end_date].copy()
        if len(current_train_df) < 2: # Need at least 2 data points for Prophet
            print(f"  Skipping day {i+1}: Not enough training data ({len(current_train_df)} points).")
            continue

        # 2. Train Model
        model = Prophet(**model_params)
        if regressor_cols:
            for col in regressor_cols:
                if col in current_train_df.columns:
                    model.add_regressor(col)
        try:
            model.fit(current_train_df[['ds', 'y'] + regressor_cols])
        except Exception as e:
            print(f"  ERROR fitting model on day {i+1}: {e}")
            continue # Skip to next day if model fails

        # 3. Generate Future DataFrame for the next forecast_horizon days
        future_df = prophet_df[prophet_df['ds'] > current_train_end_date].copy()
        future_df = future_df.head(forecast_horizon).drop(columns=['y'], errors='ignore') # Drop y if it exists

        # 4. Make Prediction
        try:
            forecast = model.predict(future_df)
            # Keep relevant columns
            forecast_subset = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            # Add column indicating when the forecast was made
            forecast_subset['forecast_made_on'] = current_train_end_date
            # Store the forecast DataFrame in the list
            all_forecasts_list.append(forecast_subset)
            print(f"  Forecast generated for {forecast_subset['ds'].min().date()} to {forecast_subset['ds'].max().date()}")

            # 5. Calculate and store forecast sum (for plotting)
            yhat_values = forecast_subset['yhat'].values
            if log_transform:
                yhat_values = np.expm1(yhat_values)
            forecast_sums[current_train_end_date] = np.sum(yhat_values)

            # --- Calculate Monthly Actual + Forecast Sum ---
            try:
                # Get the month start and end for the forecast_date (current_train_end_date)
                current_month_start = current_train_end_date - MonthBegin(1)
                current_month_end = current_train_end_date + MonthEnd(0)
                forecast_start_date = current_train_end_date + timedelta(days=1) # Forecast starts day after train end

                # Get actuals from month start up to current_train_end_date (inclusive)
                actuals_in_month_mask = (y_full.index >= current_month_start) & (y_full.index <= current_train_end_date)
                actual_sum_part = y_full[actuals_in_month_mask].sum()

                # Get forecast from forecast_start_date to month end
                forecast_in_month_mask = (forecast_subset['ds'] >= forecast_start_date) & (forecast_subset['ds'] <= current_month_end)
                forecast_sum_part_df = forecast_subset[forecast_in_month_mask]

                # Inverse transform forecast if needed before summing
                forecast_values_to_sum = forecast_sum_part_df['yhat'].values
                if log_transform:
                     # Ensure non-negative before expm1 if log transform was used
                     forecast_values_to_sum = np.expm1(np.maximum(0, forecast_values_to_sum))

                forecast_sum_part = np.sum(forecast_values_to_sum)

                # Combine and store
                monthly_actual_plus_forecast_sums[current_train_end_date] = actual_sum_part + forecast_sum_part

            except Exception as e_month:
                print(f"\n  WARNING: Could not calculate monthly sum for {current_train_end_date.date()}: {e_month}")
            # --- END ADDED ---

        except Exception as e:
            print(f"  ERROR making prediction or calculating sum on day {i+1}: {e}")

    # --- Combine and Export All Forecasts ---
    combined_forecast_path = None
    if all_forecasts_list:
        print("\nCombining all rolling forecasts...")
        combined_forecasts_df = pd.concat(all_forecasts_list, ignore_index=True)
        # Reorder columns for clarity
        cols_order = ['forecast_made_on', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']
        # Ensure all columns exist, add if missing before reordering
        for col in cols_order:
            if col not in combined_forecasts_df.columns:
                 combined_forecasts_df[col] = np.nan # Or some default value
        combined_forecasts_df = combined_forecasts_df[cols_order]
        combined_forecasts_df.sort_values(by=['forecast_made_on', 'ds'], inplace=True)

        # --- Apply inverse transform before saving ---
        if log_transform:
            print("  Applying inverse transform (expm1) to combined rolling forecasts before saving...")
            # Only transform columns that exist in the combined df
            cols_to_transform_rolling = ['yhat', 'yhat_lower', 'yhat_upper']
            for col in cols_to_transform_rolling:
                if col in combined_forecasts_df.columns:
                    combined_forecasts_df[col] = np.expm1(combined_forecasts_df[col])
                else:
                    print(f"    Warning: Column '{col}' not found in combined rolling forecast, skipping inverse transform for it.")
        # --- End inverse transform ---

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f'rolling_forecasts_combined_{timestamp}.csv'
            combined_forecast_path = os.path.join(DEFAULT_FORECASTS_DIR, csv_filename)
            combined_forecasts_df.to_csv(combined_forecast_path, index=False)
            print(f"  Combined rolling forecasts saved to: {combined_forecast_path}")
        except Exception as export_e:
            print(f"  ERROR exporting combined rolling forecasts: {export_e}")
    else:
        print("\nNo rolling forecasts were generated to combine or export.")


    # --- Plotting the Rolling Forecast ---
    print("\nGenerating rolling forecast plot...")
    if not all_forecasts_list: # Check the list instead of the old dict
        print("No forecasts were generated, cannot create plot.")
        return None, combined_forecast_path # Return None for fig, but path might exist if export failed mid-way

    # Pass the monthly sums to the plotting function
    fig = plot_rolling_forecast_results(
        all_forecasts_list, # Pass the list of forecast dfs
        y_full,
        initial_train_end_date,
        simulation_end_date,
        forecast_horizon,
        forecast_sums,
        monthly_actual_plus_forecast_sums, # Pass the new dictionary
        log_transform=log_transform,
        show=False # Control showing externally if needed
    )

    # --- Save the plot ---
    plot_path = os.path.join(DEFAULT_PLOTS_DIR, f'rolling_forecast_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    try:
        if fig: # Check if fig was created
            fig.savefig(plot_path)
            print(f"Rolling forecast plot saved to: {plot_path}")
            plt.close(fig) # Close the figure after saving
        else:
            print("Skipping saving rolling forecast plot as it was not generated.")
    except Exception as e:
        print(f"ERROR saving rolling forecast plot: {e}")


    # Return the figure object (or None) and the path to the combined CSV
    return fig, combined_forecast_path # Return fig object


def plot_rolling_forecast_results(
    all_forecasts_list, # Changed from forecast_results dict to list
    y_full,
    initial_train_end_date,
    simulation_end_date,
    forecast_horizon,
    forecast_sums=None,
    monthly_sums=None, # ADDED: Accept monthly sums
    log_transform=True,
    show=True
):
    if not all_forecasts_list:
        print("  No forecast results to plot.")
        return None

    # Determine plot range
    plot_start_date = initial_train_end_date - timedelta(days=60) # Show some history
    # Find the latest date across all forecasts
    max_forecast_ds = max(df['ds'].max() for df in all_forecasts_list) if all_forecasts_list else simulation_end_date
    actuals_plot_end_date = max(simulation_end_date + timedelta(days=1), max_forecast_ds) # Ensure plot covers full range of actuals/forecasts

    plot_end_date_limit = simulation_end_date + MonthEnd(0)

    # Create figure with 3 subplots, sharing the x-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]}) # Adjusted figsize and ratios

    # --- Subplot 1: Individual Forecasts vs Actuals ---
    # Plot actuals up to the maximum forecast date or simulation end
    actuals_to_plot = y_full[(y_full.index >= plot_start_date) & (y_full.index <= actuals_plot_end_date)]
    ax1.plot(actuals_to_plot.index, actuals_to_plot.values, label='Actual Sales', color=DEFAULT_COLOR_ACTUAL, linewidth=1.0, zorder=5) # Ensure actuals are on top

    # Plot individual forecasts
    num_forecasts = len(all_forecasts_list)
    color_map = plt.get_cmap('Oranges') # Or another sequential colormap
    # Sort by 'forecast_made_on' date before plotting for consistent color gradient
    all_forecasts_list.sort(key=lambda df: df['forecast_made_on'].iloc[0])

    for i, forecast_df in enumerate(all_forecasts_list):
        forecast_date = forecast_df['forecast_made_on'].iloc[0]
        dates = pd.to_datetime(forecast_df['ds'])
        yhat = forecast_df['yhat'].values

        # Inverse transform if necessary
        if log_transform:
            yhat = np.expm1(np.maximum(0, yhat)) # Ensure non-negative

        # Normalize index for color map (use 0.2 to 1.0 range for better visibility)
        color_intensity = 0.2 + 0.8 * (i / max(1, num_forecasts - 1))
        color = color_map(color_intensity)
        # Label first and last, and maybe one in the middle
        label = None
        if i == 0 or i == num_forecasts - 1 or i == num_forecasts // 2:
             label = f'Forecast from {forecast_date.date()}'

        ax1.plot(dates, yhat, label=label, color=color, alpha=0.8, linewidth=1.0, zorder=3)

    # Formatting for Subplot 1
    ax1.axvline(initial_train_end_date, color=DEFAULT_COLOR_STANDARD, linestyle='--', label='Initial Train End', zorder=4)
    ax1.axvline(simulation_end_date, color=DEFAULT_COLOR_STANDARD, linestyle=':', label='Simulation End', zorder=4)
    ax1.set_title('Individual Forecasts Over Time vs Actuals')
    ax1.set_ylabel('Sales')
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Sum of Forecasts (Horizon Sum) ---
    if forecast_sums:
        sum_dates = sorted(forecast_sums.keys())
        sums = np.array([forecast_sums[d] for d in sum_dates])
        ax2.plot(sum_dates, sums, marker='o', markersize=5, linestyle='-', linewidth=2.0, color=DEFAULT_COLOR_HIGHLIGHT, label=f'Sum of {forecast_horizon}-day Forecast')

        # Calculate statistics for the text
        mean_sum = np.mean(sums) if len(sums) > 0 else 0
        std_sum = np.std(sums) if len(sums) > 0 else 0
        std_pct = (std_sum / mean_sum) * 100 if mean_sum != 0 else 0
        stats_text = f"Mean: {mean_sum:,.0f}\nStd: {std_sum:,.0f} ({std_pct:.1f}%)" # Added Mean
        ax2.text(0.98, 0.95, stats_text, transform=ax2.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Formatting for Subplot 2
        ax2.set_title(f'Sum of {forecast_horizon}-Day Forecast vs. Forecast Date')
        ax2.set_ylabel(f'Total Forecasted Sales\n(next {forecast_horizon} days)')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No forecast horizon sums calculated.', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title(f'Sum of {forecast_horizon}-Day Forecast vs. Forecast Date')


    # --- Subplot 3: Monthly Actual + Forecast Sum ---
    if monthly_sums:
        month_sum_dates = sorted(monthly_sums.keys())
        month_sums = np.array([monthly_sums[d] for d in month_sum_dates])
        ax3.plot(month_sum_dates, month_sums, marker='s', markersize=5, linestyle='-', linewidth=2.0, color=DEFAULT_COLOR_COMPONENT, label='Monthly Total (Actual+Forecast)') # Square marker, blue color

        # Calculate statistics for the text
        mean_m_sum = np.mean(month_sums) if len(month_sums) > 0 else 0
        std_m_sum = np.std(month_sums) if len(month_sums) > 0 else 0
        std_m_pct = (std_m_sum / mean_m_sum) * 100 if mean_m_sum != 0 else 0
        m_stats_text = f"Mean: {mean_m_sum:,.0f}\nStd: {std_m_sum:,.0f} ({std_m_pct:.1f}%)"
        ax3.text(0.98, 0.95, m_stats_text, transform=ax3.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Formatting for Subplot 3
        ax3.set_title('Estimated Monthly Total (Actual + Forecast) vs. Forecast Date')
        ax3.set_ylabel('Estimated Total Sales\n(Current Month)')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No monthly sums calculated.', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
        ax3.set_title('Estimated Monthly Total (Actual + Forecast) vs. Forecast Date')

    # --- Final Formatting ---
    # Set shared x-axis limits and label (only on bottom plot)
    ax3.set_xlim(plot_start_date, plot_end_date_limit)
    ax3.set_xlabel('Date')

    # Improve legend handling (place on ax1)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    # Combine legends - adjust ncol if needed
    ax1.legend(handles=lines1 + lines2 + lines3, labels=labels1 + labels2 + labels3, loc='upper left', fontsize=8, ncol=3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout slightly for suptitle
    fig.suptitle(f'Rolling Forecast Simulation ({len(all_forecasts_list)} forecasts, {forecast_horizon}-day horizon)', fontsize=16) # Updated suptitle

    if show:
        plt.show()

    return fig

def plot_combined_holiday_impact(forecast, holiday_regressor_names, y_full=None, log_transform=False, title="Holiday Impact Components Over Time (Last Year)", show=True):
    print("Plotting individual and combined holiday impacts over time (last year)...")
    if forecast is None or forecast.empty:
        print("  Forecast DataFrame is empty or None.")
        return None

    # --- Filter forecast to the last year of data ---
    original_forecast_dates = pd.to_datetime(forecast['ds'])
    if original_forecast_dates.empty:
        print("  Forecast DataFrame has no dates.")
        return None
    
    last_date_in_forecast = original_forecast_dates.max()
    first_date_for_last_year = last_date_in_forecast - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    
    forecast_last_year = forecast[original_forecast_dates >= first_date_for_last_year].copy()
    
    if forecast_last_year.empty:
        print(f"  No forecast data found within the last year (from {first_date_for_last_year.date()} to {last_date_in_forecast.date()}).")
        return None
    print(f"  Filtered forecast to last year: {forecast_last_year['ds'].min()} to {forecast_last_year['ds'].max()}")
    # --- End filter ---

    # Use the filtered forecast DataFrame from now on
    current_forecast_df = forecast_last_year

    relevant_holiday_cols = [col for col in holiday_regressor_names if col in current_forecast_df.columns]

    if not relevant_holiday_cols:
        print("  No specified holiday regressor components found in the filtered forecast DataFrame.")
        return None

    print(f"  Plotting effects for {len(relevant_holiday_cols)} holiday regressors.")
    combined_holiday_effect = current_forecast_df[relevant_holiday_cols].sum(axis=1)

    fig, ax1 = plt.subplots(figsize=(15, 8))
    dates_for_ax1 = pd.to_datetime(current_forecast_df['ds']) # Dates are now from the last year

    plot_ylabel = "Holiday Impact Component Value"
    if log_transform:
        plot_ylabel += " (on Log Scale of y)"

    num_holidays = len(relevant_holiday_cols)
    color_map_individual = plt.colormaps.get_cmap('turbo')

    for i, holiday_col in enumerate(relevant_holiday_cols):
        clean_label = holiday_col.replace('_dummy', '').replace('recent_', '').replace('_', ' ').title()
        color_sample_point = i / max(1, num_holidays - 1) if num_holidays > 1 else 0.5
        ax1.plot(dates_for_ax1, current_forecast_df[holiday_col].values,
                 color=color_map_individual(color_sample_point),
                 label=clean_label,
                 linewidth=0.9, alpha=0.75)

    ax1.plot(dates_for_ax1, combined_holiday_effect.values, color='black', label='Combined Holiday Impact', linewidth=1.0, linestyle='-')

    ax1.set_xlabel('Date')
    ax1.set_ylabel(plot_ylabel)
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3, linestyle='-')
    ax1.axhline(0, color='dimgray', linestyle='-', alpha=0.7, linewidth=1.0)

    ax2 = None
    lines, labels = ax1.get_legend_handles_labels()
    all_lines = lines
    all_labels = labels

    if y_full is not None and not dates_for_ax1.empty: # dates_for_ax1 is already last year
        ax2 = ax1.twinx()
        
        # Filter y_full to the same "last year" window as dates_for_ax1
        # (first_date_for_last_year and last_date_in_forecast are from the initial forecast filtering)
        y_full_last_year_segment_original_index = y_full[
            (y_full.index >= first_date_for_last_year) &
            (y_full.index <= last_date_in_forecast) # Use the overall last date from original forecast
        ]
        
        # Reindex this segment to the specific dates present in dates_for_ax1 (which is already last year)
        target_reindex_dtindex = pd.DatetimeIndex(dates_for_ax1.unique())
        
        y_full_aligned_for_plot = y_full_last_year_segment_original_index.reindex(target_reindex_dtindex).interpolate(method='time')
        
        valid_y_values_in_segment_mask = ~y_full_aligned_for_plot.isna()
        dates_to_plot_for_y_full = y_full_aligned_for_plot.index[valid_y_values_in_segment_mask]
        values_to_plot_for_y_full = y_full_aligned_for_plot[valid_y_values_in_segment_mask].values

        if not dates_to_plot_for_y_full.empty:
            ax2.plot(dates_to_plot_for_y_full, values_to_plot_for_y_full,
                     color=DEFAULT_COLOR_ACTUAL, alpha=0.5,
                     label='Actual Sales (Last Year, Context)', linestyle='-', linewidth=1.0)
            ax2.set_ylabel('Target Value (Context)', color=DEFAULT_COLOR_ACTUAL, alpha=0.7)
            ax2.tick_params(axis='y', labelcolor=DEFAULT_COLOR_ACTUAL)
            
            lines2, labels2 = ax2.get_legend_handles_labels()
            all_lines.extend(lines2) # Use extend for lists
            all_labels.extend(labels2)
        else:
            print("  Warning: No y_full data available for the last year of the displayed forecast period after alignment.")
            if ax2:
                ax2.set_frame_on(False)
                ax2.set_yticks([])
                ax2.set_yticklabels([])

    ax1.set_title(title, fontsize=14) # Title updated to reflect "Last Year"
    fig.autofmt_xdate() 

    if not (len(all_labels) > 15):
        plt.tight_layout()

    if show:
        plt.show()

    return fig


# --- Exporting ---

def export_artifacts(results, output_dir=DEFAULT_OUTPUT_DIR, log_transform=True):
    print(f"\nExporting artifacts to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(DEFAULT_PLOTS_DIR, exist_ok=True)
    os.makedirs(DEFAULT_FORECASTS_DIR, exist_ok=True)
    os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {}

    # Columns typically affected by log transform of y in Prophet output
    cols_to_inverse_transform = [
        'yhat', 
        'yhat_lower', 
        'yhat_upper', 
        'trend', 
        'trend_lower', 
        'trend_upper'
    ]

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
        # Process forecast_test
        df_test = results['forecast_test'].copy()
        if log_transform:
            print("  Applying inverse transform (expm1) to forecast_test before saving...")
            for col in cols_to_inverse_transform:
                if col in df_test.columns:
                    df_test[col] = np.expm1(df_test[col])
        forecast_path_test = os.path.join(DEFAULT_FORECASTS_DIR, f'forecast_test_{timestamp}.csv')
        df_test.to_csv(forecast_path_test, index=False)
        paths['forecast_test'] = forecast_path_test
        print(f"  Test forecast saved: {forecast_path_test}")

        # Process forecast_train
        df_train = results['forecast_train'].copy()
        if log_transform:
            print("  Applying inverse transform (expm1) to forecast_train before saving...")
            for col in cols_to_inverse_transform:
                if col in df_train.columns:
                    df_train[col] = np.expm1(df_train[col])
        forecast_path_train = os.path.join(DEFAULT_FORECASTS_DIR, f'forecast_train_{timestamp}.csv')
        df_train.to_csv(forecast_path_train, index=False)
        paths['forecast_train'] = forecast_path_train
        print(f"  Train forecast saved: {forecast_path_train}")

        # Process forecast_future
        if 'forecast_future' in results and results['forecast_future'] is not None:
             df_future = results['forecast_future'].copy()
             if log_transform:
                 print("  Applying inverse transform (expm1) to forecast_future before saving...")
                 for col in cols_to_inverse_transform:
                     if col in df_future.columns:
                         df_future[col] = np.expm1(df_future[col])
             future_forecast_path = os.path.join(DEFAULT_FORECASTS_DIR, f'forecast_future_{timestamp}.csv')
             df_future.to_csv(future_forecast_path, index=False)
             paths['forecast_future'] = future_forecast_path
             print(f"  Future forecast saved: {future_forecast_path}")

    except Exception as e:
        print(f"  ERROR saving forecast data: {e}")
        import traceback
        traceback.print_exc() # Print traceback for debugging saving errors

    # Plots
    plot_keys = {
        'fig_model_fit': 'model_fit',
        'fig_forecast': 'forecast',
        'fig_regressor_coefficients': 'regressor_coefficients', 
        'fig_covariate_evaluation': 'covariate_evaluation',
        'fig_combined_holiday_impact': 'combined_holiday_impact',







        'fig_monthly_forecast': 'monthly_forecast',










        
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















def plot_monthly_actuals_and_forecast(
    actuals_daily_series: pd.Series, 
    forecast_daily_df: pd.DataFrame, 
    log_transform: bool = True, 
    history_days_to_show: int = 365*3, 
    title: str = "Actuals and forecast for long time period", 
    show: bool = True
):
    # Create figure with two subplots (monthly on left, annual on right)
    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1])
    ax = fig.add_subplot(gs[0, 0])      # Monthly subplot (left)
    ax_annual = fig.add_subplot(gs[0, 1])  # Annual subplot (right)

    # --- 1. Prepare Data ---
    last_actual_date_month_end = None
    actuals_monthly_history_to_plot = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    
    # Prophet forecast components (daily)
    fcst_ds = pd.to_datetime(forecast_daily_df['ds'])
    yhat_daily_values = forecast_daily_df['yhat'].copy()
    
    if log_transform:
        yhat_daily_values = np.expm1(yhat_daily_values)
        
    yhat_daily_series_full = pd.Series(yhat_daily_values.values, index=fcst_ds)
    
    # Monthly Actuals
    if actuals_daily_series is not None and not actuals_daily_series.empty:
        if not isinstance(actuals_daily_series.index, pd.DatetimeIndex):
            print("ERROR: actuals_daily_series must have a DatetimeIndex.")
            return fig
        
        actuals_monthly_all = actuals_daily_series.resample('ME').sum()
        if not actuals_monthly_all.empty:
            last_actual_date_month_end = actuals_monthly_all.index[-1]
            
            num_history_months = max(1, int(history_days_to_show / 30.44))
            if len(actuals_monthly_all) > num_history_months:
                actuals_monthly_history_to_plot = actuals_monthly_all.iloc[-num_history_months:]
            else:
                actuals_monthly_history_to_plot = actuals_monthly_all
    else:
        print("Warning: Actuals data is empty or None. Plotting forecast and trend only.")

    # --- 2. Create Combined Actuals/Forecast Line (Monthly) ---
    combined_line_monthly = pd.Series(dtype=float)
    if not actuals_monthly_history_to_plot.empty:
        combined_line_monthly = actuals_monthly_history_to_plot.copy()

    # Forecast part of the combined line
    yhat_future_monthly_for_combined_line = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    if last_actual_date_month_end:
        yhat_future_daily_for_combined = yhat_daily_series_full[yhat_daily_series_full.index > last_actual_date_month_end]
        if not yhat_future_daily_for_combined.empty:
            yhat_future_monthly_for_combined_line = yhat_future_daily_for_combined.resample('ME').sum()
            # Append to combined_line_monthly, ensuring no overlap and correct order
            combined_line_monthly = pd.concat([
                combined_line_monthly[combined_line_monthly.index < yhat_future_monthly_for_combined_line.index.min()], 
                yhat_future_monthly_for_combined_line
            ]).sort_index()
    elif not yhat_daily_series_full.empty: # No actuals, plot full yhat as the main line
        combined_line_monthly = yhat_daily_series_full.resample('ME').sum()
        # Apply history cutoff if applicable to the forecast itself
        num_history_months = max(1, int(history_days_to_show / 30.44))
        if len(combined_line_monthly) > num_history_months:
             combined_line_monthly = combined_line_monthly.iloc[-num_history_months:]

    # --- 3. Prepare Annual Data (for the new subplot) ---
    # Get annual aggregates from the same data
    actuals_annual = None
    forecast_annual = None
    
    if actuals_daily_series is not None and not actuals_daily_series.empty:
        actuals_annual = actuals_daily_series.resample('YE').sum()
    
    if not yhat_daily_series_full.empty:
        forecast_annual = yhat_daily_series_full.resample('YE').sum()
    
    # Determine which years to show for actuals and forecast
    annual_years = set()
    if actuals_annual is not None and not actuals_annual.empty:
        annual_years.update([date.year for date in actuals_annual.index])
    
    if forecast_annual is not None and not forecast_annual.empty:
        forecast_years = [date.year for date in forecast_annual.index]
        annual_years.update(forecast_years)
    
    # --- 4. Plotting Monthly Subplot (left) ---
    split_actuals = combined_line_monthly[combined_line_monthly.index <= last_actual_date_month_end] if last_actual_date_month_end else None
    split_forecast = combined_line_monthly[combined_line_monthly.index >= last_actual_date_month_end] if last_actual_date_month_end else None
    if not combined_line_monthly.empty:
        ax.plot(split_actuals.index, split_actuals.values, 
                label='Actuals (Monthly Sum)', color=DEFAULT_COLOR_ACTUAL, linestyle='-', linewidth=2.5)
        ax.plot(split_forecast.index, split_forecast.values, 
                label='Forecast (Monthly Sum)', color=DEFAULT_COLOR_FORECAST, linestyle='-', linewidth=2.5)
        
    # Vertical separator line
    if last_actual_date_month_end:
        ax.axvline(last_actual_date_month_end, color=DEFAULT_COLOR_ACTUAL, linestyle='--', linewidth=1.0, alpha=0.5, label='Actuals/Forecast Split')

    # --- 5. Monthly Subplot Formatting ---
    ax.set_xlabel('Date (Month End)')
    ax.set_ylabel('Value (Monthly Sum)')
    ax.set_title('Monthly Aggregation')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.4, linestyle='--')
    
    # Determine overall date range for x-axis ticks and limits
    all_plotted_dates_for_ticks = []
    if not combined_line_monthly.empty: all_plotted_dates_for_ticks.extend(combined_line_monthly.index)

    if all_plotted_dates_for_ticks:
        # Ensure unique, sorted dates
        unique_dates = sorted(list(set(pd.Timestamp(date) for date in all_plotted_dates_for_ticks)))
        if unique_dates:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(unique_dates) // 12)))
            
            # Set x-limits to the start of the first month and end of the last month
            min_date_overall = unique_dates[0]
            max_date_overall = unique_dates[-1]
            plot_xlim_start = min_date_overall.to_period('M').start_time - pd.Timedelta(days=1)
            plot_xlim_end = max_date_overall.to_period('M').end_time + pd.Timedelta(days=1)
            ax.set_xlim(plot_xlim_start, plot_xlim_end)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    # --- 6. Plotting Annual Subplot (right) ---
    # Check if the last forecast year is a full year
    has_partial_year = False
    last_year = None
    
    if forecast_annual is not None and not forecast_annual.empty:
        forecast_end_date = yhat_daily_series_full.index.max()
        last_year = forecast_end_date.year
        year_end_date = pd.Timestamp(f"{last_year}-12-31")
        
        # If the forecast doesn't go through December 31st of the last year, it's partial
        if forecast_end_date < year_end_date:
            has_partial_year = True
            print(f"Last year ({last_year}) is partial, ending on {forecast_end_date.date()} instead of {year_end_date.date()}. Excluding from annual plot.")
    
    # Sort years for consistent ordering and filter out partial year if needed
    sorted_years = sorted(annual_years)
    if has_partial_year and last_year in sorted_years:
        sorted_years.remove(last_year)
    
    annual_values = []
    annual_colors = []
    annual_sources = []  # To track the source of data for each year (actuals, forecast, or combined)
    
    # For each year, sum actuals and forecast values together
    for year in sorted_years:
        year_total = 0
        has_actuals = False
        has_forecast = False
        
        # Add actuals for this year if available
        if actuals_annual is not None and not actuals_annual.empty:
            year_matches = [i for i, date in enumerate(actuals_annual.index) if date.year == year]
            if year_matches:
                year_total += actuals_annual.iloc[year_matches[0]]
                has_actuals = True
        
        # Add forecasts for this year if available
        if forecast_annual is not None and not forecast_annual.empty:
            year_matches = [i for i, date in enumerate(forecast_annual.index) if date.year == year]
            if year_matches:
                # For years with actuals, we only want to add forecast values not covered by actuals
                if has_actuals and last_actual_date_month_end:
                    # Get the last day of actuals for this year
                    year_last_actual = pd.Timestamp(year=year, month=12, day=31)
                    if year_last_actual > last_actual_date_month_end:
                        # Get daily forecast values after the last actual date
                        forecast_days = yhat_daily_series_full[
                            (yhat_daily_series_full.index > last_actual_date_month_end) & 
                            (yhat_daily_series_full.index.year == year)
                        ]
                        # Sum these forecast values and add to the total
                        if not forecast_days.empty:
                            year_total += forecast_days.sum()
                            has_forecast = True
                else:
                    # If no actuals for this year, use the full forecast
                    year_total += forecast_annual.iloc[year_matches[0]]
                    has_forecast = True
        
        annual_values.append(year_total)
        
        # Determine color based on data sources
        if has_actuals and has_forecast:
            annual_colors.append(DEFAULT_COLOR_PREDICTED) 
            annual_sources.append('combined')
        elif has_actuals:
            annual_colors.append(DEFAULT_COLOR_ACTUAL)
            annual_sources.append('actual')
        elif has_forecast:
            annual_colors.append(DEFAULT_COLOR_FORECAST)
            annual_sources.append('forecast')
        else:
            annual_colors.append('lightgray')
            annual_sources.append('none')
    
    # Create the barplot
    if sorted_years:
        bars = ax_annual.bar(sorted_years, annual_values, color=annual_colors, alpha=0.8)
        
        # Add value labels on top of each bar with YoY variation
        for i, (bar, value) in enumerate(zip(bars, annual_values)):
            height = bar.get_height()
            
            # Calculate year-over-year variation
            yoy_text = ""
            if i > 0:  # Skip for the first year as there's no previous year to compare
                prev_value = annual_values[i-1]
                if prev_value > 0:  # Avoid division by zero
                    yoy_pct = ((value - prev_value) / prev_value) * 100
                    sign = "+" if yoy_pct >= 0 else ""
                    yoy_text = f"\n{sign}{yoy_pct:.1f}% YoY"
            
            # Display value and YoY variation
            ax_annual.text(bar.get_x() + bar.get_width()/2., height + (max(annual_values) * 0.02),
                    f'{int(value):,}{yoy_text}', ha='center', va='bottom', rotation=0, fontsize=9)
        
        # Formatting for annual subplot
        ax_annual.set_xlabel('Year')
        ax_annual.set_ylabel('Annual Total')
        ax_annual.set_title('Annual Aggregation')
        ax_annual.grid(True, alpha=0.3, axis='y')
        
        # Create a custom legend for the annual subplot
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=DEFAULT_COLOR_ACTUAL, label='Actual Only'),
            Patch(facecolor=DEFAULT_COLOR_FORECAST, label='Forecast Only'),
            Patch(facecolor=DEFAULT_COLOR_PREDICTED, label='Actual + Forecast')
        ]
        ax_annual.legend(handles=legend_elements, loc='upper left')
        
        # Ensure integer ticks for years and proper spacing
        ax_annual.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Adjust y-axis limits to leave room for value labels (increased to accommodate YoY labels)
        ax_annual.set_ylim(0, max(annual_values) * 1.25)
        
    else:
        ax_annual.text(0.5, 0.5, "No annual data available", 
                      horizontalalignment='center', verticalalignment='center', 
                      transform=ax_annual.transAxes)
    
    # --- 7. Finalize plot ---
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if show:
        plt.show()
    return fig























# --- Main Pipeline Orchestrator ---

def run_prophet_pipeline(
    target_path,
    covariate_path=None,
    covariate_cols=None,
    date_col=DEFAULT_DATE_COL,
    target_col=DEFAULT_TARGET_COL,
    test_size=35,
    future_periods=35,
    lags=None, 
    log_transform=True,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    custom_date_dummies=None,
    perform_covariate_evaluation=True,
    max_lag_for_evaluation=70,
    output_dir=DEFAULT_OUTPUT_DIR,
    covariate_forecasting=True
):
    print("\n--- Starting Prophet Forecasting Pipeline ---")
    results = {}
    lags = lags or [] # Ensure lags is a list

    # 1. Load Data
    y, X = load_data(target_path, covariate_path, date_col, target_col, covariate_cols)
    if y is None: return {"error": "Failed to load target data."}
    results['y_full'] = y

    # 2. Evaluate Covariates (Optional)
    results['fig_covariate_evaluation'] = None
    # Use the renamed parameter in the condition
    if perform_covariate_evaluation and X is not None:
        # Call the actual function evaluate_covariates
        results['covariate_evaluation'] = evaluate_covariates(y, X, max_lag=max_lag_for_evaluation)
        # Check if evaluation produced results before plotting
        if results['covariate_evaluation'] and results['covariate_evaluation'].get('correlation'):
             results['fig_covariate_evaluation'] = plot_covariate_evaluation(results['covariate_evaluation'], y, X, show=False)


    # 3. Feature Engineering
    transformed_X_df = feature_engeneering(
        X=X, 
        log_transform=log_transform, 
        covariate_forecasting=covariate_forecasting, 
        future_periods=future_periods,
        lags=lags, 
        date_col=date_col, 
        custom_date_dummies=custom_date_dummies)


    # 4. Prepare Prophet Input Data (Transformations, Features)
    print("Preparing data for Prophet...")
    transformed_y_df = pd.DataFrame({'ds': y.index, 'y': y.values})
    if log_transform:
        transformed_y_df['y'] = apply_log_transform(transformed_y_df['y']).values
    transformed_y_df = transformed_y_df.set_index('ds')
    
    transformed_X_df = transformed_X_df[(transformed_X_df.index >= transformed_y_df.index.min()) & (transformed_X_df.index <= transformed_y_df.index.max() + timedelta(days=future_periods))]
    regressor_cols = transformed_X_df.columns.tolist()
    results['regressor_cols'] = regressor_cols

    prophet_df = transformed_X_df.merge(transformed_y_df, on='ds', how='left').reset_index()
    results['prophet_df'] = prophet_df

    # 5. Split Data
    train_df = prophet_df[:-test_size-future_periods][['ds', 'y'] + regressor_cols].copy() 
    test_df = prophet_df[-test_size-future_periods:-future_periods][['ds', 'y'] + regressor_cols].copy() 
    forecast_df = prophet_df[-future_periods:][['ds'] + regressor_cols].copy() 

    # 6. Train Model

    print("Initializing Prophet model...")
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        uncertainty_samples=10000
    )

    if regressor_cols:
        print(f"  Adding {len(regressor_cols)} regressors...")
        for col in regressor_cols:
            model.add_regressor(col)

    print("Fitting Prophet model...")
    model.fit(train_df)
    print("Model fitting complete.")
    results['model'] = model
    results['fig_regressor_coefficients'] = plot_regressor_coefficients(model, show=False) if regressor_cols else None

    # 7. Make Predictions
    results['forecast_test'] = model.predict(test_df)
    results['forecast_train'] = model.predict(train_df[['ds'] + regressor_cols])
    results['forecast_future'] = model.predict(forecast_df)

    # 10. Evaluate Model
    y_actual_test = y.iloc[-test_size:].values
    y_pred_test = results['forecast_test']['yhat'].values
    if log_transform:
        y_pred_test = np.expm1(y_pred_test)
    results['mape'] = calculate_mape(y_actual_test, y_pred_test)
    print(f"\nEvaluation on Test Set: MAPE = {results['mape']:.2f}%")
    
    # 11. Generate Plots
    print("\nGenerating plots...")
    results['fig_model_fit'] = plot_model_fit(
        train_forecast=results['forecast_train'], # Use forecast on training data
        y_full=results['y_full'], # Full actuals needed for train/test split inside
        test_size=test_size,
        model=results['model'],
        log_transform=log_transform,
        title=f"Model Fit Diagnostics (Train Set)", # Add specific title
        show=False # Pipeline controls showing/saving
    )
    results['fig_forecast'] = plot_forecast_results(
        results['forecast_test'], y, test_size, log_transform,
        future_forecast=results['forecast_future'],
        title=f"Prophet Forecast (Test MAPE: {results['mape']:.2f}%)",
        show=False
    )
    results['fig_combined_holiday_impact'] = plot_combined_holiday_impact(
        forecast=results['forecast_train'], # Use train forecast as it covers historical period
        holiday_regressor_names=[name for name in model.extra_regressors.keys() if '_event_' in name],
        y_full=results['y_full'], # Original scale y for context
        log_transform=log_transform, # Pass the pipeline's log_transform flag
        title="Combined Holiday Dummies Impact (Train Period)",
        show=False # Pipeline controls showing/saving
    )









    results['fig_monthly_forecast'] = plot_monthly_actuals_and_forecast(
        actuals_daily_series=results['y_full'], # Original scale y for context
        forecast_daily_df=results['forecast_future'], # Use future forecast for monthly plot
        log_transform=log_transform, # Pass the pipeline's log_transform flag
        show=False # Pipeline controls showing/saving
    )











    
    # 12. Export Artifacts
    export_artifacts(results, output_dir=output_dir, log_transform=log_transform)

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

    TEST_SIZE = 180
    FUTURE_PERIODS = 365 * 3
    SIMULATION_DAYS = 3

    # Run the pipeline
    pipeline_results = run_prophet_pipeline(
        target_path=TARGET_PATH,
        covariate_path=COVARIATE_PATH,
        covariate_cols=COVARIATE_COLS_TO_USE,
        test_size=TEST_SIZE,
        future_periods=FUTURE_PERIODS,
        lags=LAGS_TO_GENERATE,
        log_transform=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False, # Usually False for daily data unless strong intra-day pattern
        custom_date_dummies=CUSTOM_DATE_DUMMIES,
        perform_covariate_evaluation=True,
        max_lag_for_evaluation=70
    )


        # --- Run the Rolling Forecast Simulation ---
    if pipeline_results and 'error' not in pipeline_results and SIMULATION_DAYS > 1:
        print("\nRe-preparing data for rolling simulation...")
        try:
            
            # --- Call the Simulation ---
            rolling_fig, combined_csv_path = simulate_rolling_forecast(
                prophet_df=pipeline_results['prophet_df'],
                y_full=pipeline_results['y_full'],
                regressor_cols=pipeline_results['regressor_cols'],
                initial_train_size=len(pipeline_results['prophet_df']) - TEST_SIZE,
                simulation_days=SIMULATION_DAYS,
                forecast_horizon=FUTURE_PERIODS,
                log_transform=True,
                model_params={
                    'yearly_seasonality': True,
                    'weekly_seasonality': True,
                    'daily_seasonality': False
                }
            )
            if combined_csv_path:
                print(f"\nCombined rolling forecast CSV exported to: {combined_csv_path}")

        except Exception as e:
                print(f"ERROR during data preparation for rolling simulation: {e}")
                import traceback
                traceback.print_exc()
