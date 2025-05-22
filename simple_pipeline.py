import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from dateutil.easter import easter
from prophet import Prophet
from prophet.utilities import regressor_coefficients
from pandas.errors import PerformanceWarning

# --- Configuration ---
warnings.filterwarnings('ignore', category=PerformanceWarning)

# --- Constants ---
DEFAULT_OUTPUT_DIR = 'exports'
DEFAULT_PLOTS_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'plots')
DEFAULT_FORECASTS_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'forecasts')
DEFAULT_MODELS_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'models')

TARGET_PATH = 'data/groupby_train.csv'
COVARIATE_PATH = 'data/groupby_transactions_2.csv'
DEFAULT_DATE_COL = 'date'
DEFAULT_TARGET_COL = 'sales'
DEFAULT_COVARIATE_COLS = ['transactions']

DEFAULT_LOG_TRANSFORM = True
DEFAULT_DIFF_TRANSFORM = False
DEFAULT_COVARIATE_FORECASTING = True
DEFAULT_COVARIATES_LAGS = [0, 1, 7, 30]
DEFAULT_EVENT_LAGS = 1
DEFAULT_EVENT_LEADS = 1
DEFAULT_THRESHOLD_FEATURE_SELECTION = 0.05

TEST_SIZE = 30
FORECAST_SIZE = 180

# --- Color Constants ---
TEXT_RED = '\033[91m'
TEXT_GREEN = '\033[92m'
TEXT_YELLOW = '\033[93m'
TEXT_BLUE = '\033[94m'
TEXT_ORANGE = '\033[38;5;208m'  # Specific code for orange
TEXT_COLOR_END = '\033[0m'  # Reset to default color




def load_data(target_path: str = TARGET_PATH, 
              covariate_path: str = COVARIATE_PATH, 
              date_col: str = DEFAULT_DATE_COL,
              target_col: str = DEFAULT_TARGET_COL, 
              covariate_cols: list = DEFAULT_COVARIATE_COLS):
    
    print(f"\nLoading target data from: {target_path}")
    try:
        loaded_target_df = pd.read_csv(target_path, parse_dates=[date_col])
        target_df = pd.DataFrame(
            data=loaded_target_df[target_col].values,
            index=pd.DatetimeIndex(loaded_target_df[date_col]),
            columns=[target_col]
        )
        print(f"  Loaded target: {len(target_df)} points from {target_df.index.min().date()} to {target_df.index.max().date()}")
    except KeyError as e:
        print(f"ERROR: Column '{e}' not found in target file.")
        return None, None

    covariate_df = None
    if covariate_path and covariate_cols:
        print(f"\nLoading covariates from: {covariate_path}")
        try:
            temp_df = pd.read_csv(covariate_path, parse_dates=[date_col])
            available_cols = [col for col in covariate_cols if col in temp_df.columns]

            covariate_df = pd.DataFrame(index=pd.DatetimeIndex(temp_df[date_col]))
            for col in available_cols:
                covariate_df[col] = temp_df[col].values
            print(f"  Loaded covariates: {', '.join(available_cols)} ({len(covariate_df)} points)")
        except KeyError as e:
            print(f"  ERROR: Column '{e}' not found in covariate file.")

    return pd.concat([target_df, covariate_df], axis=1) 





def calendar_events_features(start_date: str|date = None,
                             end_date: str|date = None, 
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





def calendar_seasonal_features(start_date: str|date = None, 
                               end_date: str|date = None, 
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




def feature_engeneering(df = None, 
                        date_col: str = DEFAULT_DATE_COL,
                        target_col: str = DEFAULT_TARGET_COL,
                        log_transform: bool = DEFAULT_LOG_TRANSFORM, 
                        diff_transform: bool = DEFAULT_DIFF_TRANSFORM, 
                        covariate_forecasting: bool = DEFAULT_COVARIATE_FORECASTING, 
                        future_periods: int = FORECAST_SIZE,
                        covariates_lags: list = DEFAULT_COVARIATES_LAGS, 
                        custom_date_dummies: dict = None,
                        event_lags: int = DEFAULT_EVENT_LAGS,
                        event_leads: int = DEFAULT_EVENT_LEADS,
                        event_weekdays: bool = True,
                        event_workingdays: bool = True
                        ) -> dict:
    
    print(f"\n-- Feature Engeneering: \n")
    
    results = {}

    # separate covariates from df to apply transformations (log and lags)
    #transformable_cols = [col for col in df.columns if 'dummy' not in col.lower() and 'calendar' not in col.lower()]
    patterns = ['dummy', 'dum_', 'calendar', 'cal_', '_event', '_seas', 'seasonality' '_workingday', '_lag', '_lead', '_pre', '_pos']
    transformable_cols = [col for col in df.columns if any(pattern not in col.lower() for pattern in patterns)]
    transformable_df = df[transformable_cols].copy()

    # apply log transformation if needed
    if log_transform:
        print(f"   Log transformation... {TEXT_GREEN} Ok!{TEXT_COLOR_END}")
        epsilon = 1e-9 # very low number to avoid log(0)
        for col in transformable_df.columns:
            zero_mask = transformable_df[col] <= 0
            if zero_mask.any():
                transformable_df.loc[zero_mask, col] = epsilon
        transformable_df = np.log1p(transformable_df)
        transformable_df.columns = ['log_' + col for col in transformable_df.columns]
    else:
        print(f"   Log transformation... {TEXT_RED} Passed!{TEXT_COLOR_END}")

    # apply diff transformation if needed
    if diff_transform:
        print(f"   Diff transformation...{TEXT_GREEN} Ok!{TEXT_COLOR_END}")
        transformable_df = transformable_df.astype(np.float32)
        transformable_df = transformable_df.diff()
        transformable_df.columns = ['diff_' + col for col in transformable_df.columns]
    else:
        print(f"   Diff transformation... {TEXT_RED} Passed!{TEXT_COLOR_END}")

    # forecast covariates if needed
    covariate_cols = [col for col in transformable_df.columns if target_col not in col]
    covariates = merged_covariates = transformable_df[covariate_cols].copy()
    if covariate_forecasting:
        print(f"   Forecasting covariates...{TEXT_GREEN} Ok!{TEXT_COLOR_END}")
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
    else:
        print(f"   Forecasting covariates... {TEXT_RED} Passed!{TEXT_COLOR_END}")

    # Apply lags to covariates if specified
    merged_covariates = covariates.copy()
    if covariates_lags:
        print(f"   Adding lags to covariates...{TEXT_GREEN} Ok!{TEXT_COLOR_END}")
        for lag in covariates_lags:
            lagged_covariates = covariates.copy().shift(lag)
            lagged_covariates.columns = [col + f'_lag{lag}' for col in lagged_covariates.columns]
            merged_covariates = merged_covariates.merge(lagged_covariates, on=date_col, how='outer')
        covariates = merged_covariates
    else:
        print(f"   Adding lags to covariates... {TEXT_RED} Passed!{TEXT_COLOR_END}")

    if 0 in covariates_lags:
        covariates = covariates.drop(columns=covariate_cols)
    
    target_cols = [col for col in transformable_df.columns if target_col in col]
    data = pd.concat([transformable_df[target_cols], covariates], axis=1)

    results['target_cols'] = target_cols
    results['covariate_cols'] = covariates.columns.to_list()
    
    # Merge non-transformable covariates back into the transformed DataFrame
    non_transformable_cols = results['dummy_cols'] = [col for col in df.columns if col not in transformable_cols]
    data = pd.concat([data, df[non_transformable_cols]], axis=1)

    # merge custom date dummy variables
    if custom_date_dummies:
        print(f"   Adding custom dummies...{TEXT_GREEN} Ok!{TEXT_COLOR_END}")
        for dummy in custom_date_dummies:
            if isinstance(dummy, dict) and 'name' in dummy and 'dates' in dummy:
                dummy_dates = pd.to_datetime(dummy['dates'])
                data[dummy['name']] = np.where(data.index.isin(dummy_dates), 1, 0)
                results['dummy_cols'].append(dummy['name'])
    else:
        print(f"   Adding custom dummies... {TEXT_RED} Passed!{TEXT_COLOR_END}")

    # Merge calendar features
    print(f"   Adding calendar events...{TEXT_GREEN} Ok!{TEXT_COLOR_END}")
    events_features, results['event_names'], _ = calendar_events_features(
        start_date=data.index[0].date(), 
        end_date=data.index[-1].date(),
        lags=event_lags, 
        leads=event_leads,
        weekdays=event_weekdays, 
        workingdays=event_workingdays
    )
    events_features.index.rename(date_col, inplace=True)
    mask = events_features.index.isin(data.index)
    data = pd.concat([data, events_features[mask]], axis=1)

    # Merge seasonality features
    print(f"   Adding calendar seasonalities...{TEXT_GREEN} Ok!{TEXT_COLOR_END}")
    seasonal_features, results['seasonal_names'] = calendar_seasonal_features(
        start_date=str(covariates.index[0].date()), 
        end_date=str(covariates.index[-1].date()),
    )
    seasonal_features.index.rename(date_col, inplace=True)
    mask = seasonal_features.index.isin(data.index)
    data = pd.concat([data, seasonal_features[mask]], axis=1)

    results['target_cutoff_dates'] = {
        'start': data[target_cols].dropna().index[0].date(),
        'end': data[target_cols].dropna().index[-1].date()
    }
    results['covariates_cutoff_dates'] = {
        'start': data.drop(columns=target_cols).dropna().index[0].date(),
        'end': data.drop(columns=target_cols).dropna().index[-1].date()
    }
    results['data'] = data.copy()

    return results





def split_data(data: pd.DataFrame, 
             start_training_date: str|date = None, 
             end_training_date: str|date = None,
             end_test_date: str|date = None, 
             end_forecast_date: str|date = None, 
             test_size: int|None = TEST_SIZE,
             forecast_size: int|None = FORECAST_SIZE) -> dict:

    end_forecast_date = data.index[-1].date() if end_forecast_date is None else end_forecast_date
    if isinstance(end_forecast_date, str):
        end_forecast_date = datetime.strptime(end_forecast_date, "%Y-%m-%d").date()

    end_test_date = data.dropna().index[-1].date() if end_test_date is None else end_test_date
    if isinstance(end_test_date, str):
        end_test_date = datetime.strptime(end_test_date, "%Y-%m-%d").date()
    
    end_training_date = end_test_date - timedelta(days=test_size) if end_training_date is None else end_training_date
    if isinstance(end_training_date, str):
        end_training_date = datetime.strptime(end_training_date, "%Y-%m-%d").date()
    
    start_training_date = data.dropna().index[0].date() if start_training_date is None else start_training_date
    if isinstance(start_training_date, str):
        start_training_date = datetime.strptime(start_training_date, "%Y-%m-%d").date()

    forecast_mask = (data.index.date > end_test_date) & (data.index.date <= end_forecast_date)
    test_data_mask = (data.index.date > end_training_date) & (data.index.date <= end_test_date)
    train_data_mask = (data.index.date >= start_date) & (data.index.date <= end_training_date)

    split_data_results = {
        'train': data[train_data_mask],
        'test': data[test_data_mask],
        'forecast': data[forecast_mask]
    }

    text = f"\n-- Split Data:  {split_data_results['train'].index[0].date()}" 
    text += f"{TEXT_BLUE} >  TRAIN  < {TEXT_COLOR_END}{split_data_results['train'].index[-1].date()}"
    if test_size > 0:
        text += f"{TEXT_RED} >  TEST  < {TEXT_COLOR_END}{split_data_results['test'].index[-1].date()}"
    if forecast_size > 0:
        text += f"{TEXT_ORANGE} >  FORECAST  < {TEXT_COLOR_END}{split_data_results['forecast'].index[-1].date()}"
    text += f"\n"
    print(text)
    
    return split_data_results





def train_prophet_model(train_data: pd.DataFrame, 
                        test_data: pd.DataFrame = None,
                        forecast_data: pd.DataFrame = None,
                        date_col: str = DEFAULT_DATE_COL, 
                        target_col: str = None,
                        covariate_cols: list = None
                        ) -> dict:

    # Prepare the data for Prophet
    dataframes = {}
    dataframes['train'] = train_data.copy() if train_data is not None else None
    dataframes['test'] = test_data.copy() if test_data is not None else None
    dataframes['forecast'] = forecast_data.copy() if forecast_data is not None else None
    for key, df in dataframes.items():
        if df is None:
            continue
        df = df.reset_index()
        df = df.rename(columns={date_col: 'ds', target_col: 'y'})
        df = df[['ds', 'y'] + covariate_cols]
        df['ds'] = pd.to_datetime(df['ds'])
        dataframes[key] = df

    # Train Prophet model
    model = Prophet()
    for col in covariate_cols:
        if col != 'y' and col != 'ds':
            model.add_regressor(col)
    model.fit(dataframes['train'])

    coefficients = regressor_coefficients(model)

    # Make predictions 
    predictions = {}
    for key, df in dataframes.items():
        if df is None or len(df) == 0:
            continue
        predictions[key] = model.predict(df) 
        predictions[key] = predictions[key].rename(columns={'ds': date_col}).set_index(date_col)

    return {
        'model': model,
        'coefficients': coefficients,
        'predictions': predictions
    }

def check_regressor_under_threshold(train_data: pd.DataFrame, 
                                    target_col: str, 
                                    covariate_cols: list[str] | None,
                                    checking_cols: list[str],
                                    threshold: float = DEFAULT_THRESHOLD_FEATURE_SELECTION
                                    ) -> list[str] | None: 
        
        covariate_cols = [] if covariate_cols is None else covariate_cols
        results = train_prophet_model(
            train_data=train_data, 
            target_col=target_col, 
            covariate_cols=covariate_cols + checking_cols
        )
        coefficients = results['coefficients']
        under_threshold_cols = coefficients[coefficients['coef'].abs() <= threshold]['regressor'].tolist()
        under_threshold_checked_cols = [col for col in under_threshold_cols if col in checking_cols]
        
        print(f"   Regressors under threshold ({TEXT_YELLOW}{threshold}{TEXT_COLOR_END}): {len(under_threshold_checked_cols)} / {len(checking_cols)} ({TEXT_YELLOW}{len(under_threshold_checked_cols) / len(checking_cols) *100:.0f}%{TEXT_COLOR_END})\n")
        return under_threshold_checked_cols




def feature_selection(split_data_results: dict, 
                      feature_engeneering_results: dict,
                      threshold: float = DEFAULT_THRESHOLD_FEATURE_SELECTION
                      ) -> list[str] | None:
    
    seas_cols: list[str] = []
    for _, cols in feature_engeneering_results['seasonal_names'].items():
        seas_cols += cols
    
    events_cols: list[str] = []
    for _, cols in feature_engeneering_results['event_names'].items():
        events_cols += cols

    main_regression_cols: list[str] = feature_engeneering_results['covariate_cols']\
        + feature_engeneering_results['dummy_cols']

    print(f"\n-- Feature Selection: \n")
    under_threshold_regressors: list[str] = []
    if len(main_regression_cols + seas_cols + events_cols) / len(split_data_results['train']) > 0.5:
 
        for event, event_cols in feature_engeneering_results['event_names'].items():
            print(f"   Check regressors coefficients for event: {TEXT_BLUE}{event}{TEXT_COLOR_END}")
            under_threshold_regressors += check_regressor_under_threshold(
                train_data=split_data_results['train'], 
                target_col=feature_engeneering_results['target_cols'][0], 
                covariate_cols=main_regression_cols + seas_cols,
                checking_cols=event_cols,
                threshold=threshold
            )
        checked_events_cols: list[str] = [col for col in events_cols if col not in under_threshold_regressors]

        for seas, seas_cols in feature_engeneering_results['seasonal_names'].items():
            print(f"   Check regressors coefficients for seas: {TEXT_ORANGE}{seas}{TEXT_COLOR_END}")
            under_threshold_regressors += check_regressor_under_threshold(
                train_data=split_data_results['train'], 
                target_col=feature_engeneering_results['target_cols'][0], 
                covariate_cols=main_regression_cols + checked_events_cols,
                checking_cols=seas_cols,
                threshold=threshold
            )
        checked_seas_cols: list[str] = [col for col in seas_cols if col not in under_threshold_regressors]

    else :
        checked_events_cols: list[str] = events_cols
        checked_seas_cols: list[str] = seas_cols

    print(f"   Check regressors coefficients")
    regression_cols: list[str] = main_regression_cols + checked_events_cols + checked_seas_cols
    under_threshold_regressors += check_regressor_under_threshold(
        train_data=split_data_results['train'], 
        covariate_cols=None,
        target_col=feature_engeneering_results['target_cols'][0], 
        checking_cols=regression_cols,
        threshold=threshold
    )
    return [col for col in regression_cols if col not in under_threshold_regressors]




def reverse_transformations(model_results: dict, 
                            log_transform: bool = DEFAULT_LOG_TRANSFORM,
                            diff_transform: bool = DEFAULT_DIFF_TRANSFORM,
                            target_col: str = DEFAULT_TARGET_COL
                            ) -> dict:
    
    # Reverse log transformation
    results: dict = {}
    cols: list = [target_col, 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper'] 
    for df in model_results['predictions']:
        results[df] = pd.DataFrame(index=model_results['predictions'][df].index)
        for col in cols:
            if col in model_results['predictions'][df].columns:
                if diff_transform:
                    results[df][col] = model_results['predictions'][df][col].cumsum()
                    if log_transform:
                        results[df][col] = np.expm1(results[df][col])
                
                elif log_transform:
                    results[df][col] = np.expm1(model_results['predictions'][df][col])
    return results




def evaluation_metrics(split_data_results: pd.DataFrame,
                       prediction_results: pd.DataFrame,
                       target_col: str = DEFAULT_TARGET_COL,
                       n_params: int = 0,
                       monthly: bool = True,
                       yearly: bool = True
                       ) -> dict:
        
    def mape(actual, pred) -> float:
        """
        Mean Absolute Percentage Error
        less is better (less error)
        """
        return np.mean(np.abs((actual - pred) / actual)) * 100
    
    def smape(actual, pred) -> float:
        """
        Symmetric Mean Absolute Percentage Error
        less is better (less error)
        """
        return np.mean(np.abs(actual - pred) / (np.abs(actual) + np.abs(pred))) * 100
    
    def aicc(actual, pred, n_params) -> float:
        """
        Entropy of the model (complexity)
        less is better (less complex)
        """
        n = len(actual)
        residuals = actual - pred
        sse = np.sum(residuals**2)
        return n * np.log(sse / n) + 2 * n_params + (2 * n_params * (n_params + 1)) / (n - n_params - 1)

    def durbin_watson(actual, pred) -> float:
        """ 
        Autocorrelation of residuals
         ≈ 2: No autocorrelation (ideal)
         < 1.5: Positive autocorrelation (residual errors tend to be followed by residuals with similar sign)
         > 2.5: Negative autocorrelation (residuals tend to alternate signs)
        """
        residuals = actual - pred
        return np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    
    def homocedasticity(actual, pred) -> float:
        """
        Variance of residuals
         ≈ 0: Homoscedasticity (constant variance)
         < 0: Heteroscdasticity (increasing variance)
         > 0: Heterosceedasticity (decreasing variance)
        """
        return np.mean((actual - pred)**2) / np.mean(actual**2)
    
    def skewness(actual, pred) -> float:
        """
        Asymmetry of residuals
         ≈ 0: Normal distribution (symetrical)
         < 0: Left-skewed (long tail on the left)
         > 0: Right-skewed (long tail on the right)
        """
        return pd.Series(actual - pred).skew()
    
    def kurtosis(actual, pred) -> float:
        """ 
        Peakedness of residuals
         ≈ 3: Normal distribution 
         < 3: Flat distribution 
         > 3: Sharp peak (heavy outliers)
        """
        return pd.Series(actual - pred).kurtosis()
    
    def r_squared(actual, pred) -> float:
        """
        R-squared
         ≈ 1: Perfect fit
         < 0: Poor fit
        """
        ss_res = np.sum((actual - pred)**2)
        ss_tot = np.sum((actual - np.mean(actual))**2)
        return 1 - (ss_res / ss_tot)

    def overfitting(train, test) -> float:
        """
        Overfitting
         ≈ 0: No overfitting
         < 0: Overfitting
         > 0: Underfitting
        """
        return (train - test) / train

    if n_params <= 0:
        n_params = len(split_data_results['train'].columns) - 1 # -1 for the target column


    train_predictions = prediction_results['train']['yhat']
    train_actuals = split_data_results['train'][target_col].fillna(train_predictions)

    results = {
        'daily': {},
        'monthly': {},
        'yearly': {}
    }    
    results['daily']['mape'] = {
        'train': mape(train_actuals.values, train_predictions.values),
    }
    results['daily']['smape'] = {
        'train': smape(train_actuals.values, train_predictions.values),
    }
    results['daily']['aicc'] = aicc(train_actuals.values, train_predictions.values, n_params)
    results['daily']['durbin-watson ~2'] = durbin_watson(train_actuals.values, train_predictions.values)
    results['daily']['homocedasticity ~0'] = homocedasticity(train_actuals.values, train_predictions.values)
    results['daily']['skewness ~0'] = skewness(train_actuals.values, train_predictions.values)
    results['daily']['kurtosis ~3'] = kurtosis(train_actuals.values, train_predictions.values)
    results['daily']['r-squared ~1'] = r_squared(train_actuals.values, train_predictions.values)
    
    if 'test' in prediction_results:
        test_predictions = prediction_results['test']['yhat']
        test_actuals = split_data_results['test'][target_col].fillna(test_predictions)

        results['daily']['mape']['test'] = mape(test_actuals.values, test_predictions.values)
        results['daily']['smape']['test'] = smape(test_actuals.values, test_predictions.values)

        results['daily']['overfitting ~0'] = overfitting(results['daily']['smape']['train'], results['daily']['smape']['test'])

    if monthly:
        results['monthly']['mape'] = {
            'train': mape(train_actuals.resample('ME').sum().values, train_predictions.resample('ME').sum().values)
        }
        results['monthly']['smape'] = {
            'train': smape(train_actuals.resample('ME').sum().values, train_predictions.resample('ME').sum().values)
        }
        if 'test' in prediction_results:
            results['monthly']['mape']['test'] = mape(test_actuals.resample('ME').sum().values, test_predictions.resample('ME').sum().values)
            results['monthly']['smape']['test'] = smape(test_actuals.resample('ME').sum().values, test_predictions.resample('ME').sum().values)


    if yearly:
        results['yearly']['mape'] = {
            'train': mape(train_actuals.resample('YE').sum().values, train_predictions.resample('YE').sum().values)
        }
        results['yearly']['smape'] = {
            'train': smape(train_actuals.resample('YE').sum().values, train_predictions.resample('YE').sum().values)
        }
        if 'test' in prediction_results:
            results['yearly']['mape']['test'] = mape(test_actuals.resample('YE').sum().values, test_predictions.resample('YE').sum().values)
            results['yearly']['smape']['test'] = smape(test_actuals.resample('YE').sum().values, test_predictions.resample('YE').sum().values)

    print(f"\n-- Model Evaluation:")    
    for key, value in results.items():
        if len(value) == 0:
            continue
        print(f"\n   {key.capitalize()} metrics:")
        for metric, val in value.items():
            if isinstance(val, dict):
                text = f"      {metric}:"
                for sub_metric, sub_val in val.items():
                    text += f" {sub_metric}: {TEXT_YELLOW}{sub_val:.2f}%{TEXT_COLOR_END}"
                print(text)
            else:
                print(f"      {metric}: {TEXT_YELLOW}{val:.2f}{TEXT_COLOR_END}")

    return results
        




# pipeline
if __name__ == "__main__":
    # Load data
    loaded_data = load_data()

    # Feature engineering
    feature_engeneering_results: dict = feature_engeneering(
        df = loaded_data, 
        date_col = DEFAULT_DATE_COL,
        target_col = DEFAULT_TARGET_COL,
        log_transform = DEFAULT_LOG_TRANSFORM, 
        diff_transform = DEFAULT_DIFF_TRANSFORM, 
        covariate_forecasting = DEFAULT_COVARIATE_FORECASTING, 
        future_periods = FORECAST_SIZE,
        covariates_lags = DEFAULT_COVARIATES_LAGS, 
        custom_date_dummies = None,
        event_lags = DEFAULT_EVENT_LAGS,
        event_leads = DEFAULT_EVENT_LEADS,
        event_weekdays = False,
        event_workingdays = True
    )

    # split data into train, test and forecast sets
    start_date = max(feature_engeneering_results['target_cutoff_dates']['start'], feature_engeneering_results['covariates_cutoff_dates']['start'])
    end_date = min(feature_engeneering_results['target_cutoff_dates']['end'], feature_engeneering_results['covariates_cutoff_dates']['end'])
    split_data_results: dict = split_data(
        data=feature_engeneering_results['data'].copy(),
        start_training_date=start_date,
        end_test_date=end_date
    )

    # Feature selection
    regressor_cols: list[str] = feature_selection(
        split_data_results=split_data_results, 
        feature_engeneering_results=feature_engeneering_results,
        threshold=DEFAULT_THRESHOLD_FEATURE_SELECTION
    )

    # Train Model 
    model_results: dict = train_prophet_model(
        train_data=split_data_results['train'], 
        test_data=split_data_results['test'],
        forecast_data=split_data_results['forecast'],
        target_col=feature_engeneering_results['target_cols'][0],
        covariate_cols=regressor_cols
    )

    # reverse transformations
    reverse_transform_results: dict = reverse_transformations(
        model_results=model_results,
        log_transform=DEFAULT_LOG_TRANSFORM,
        diff_transform=DEFAULT_DIFF_TRANSFORM,
        target_col=feature_engeneering_results['target_cols'][0]
    )

    # Model evaluation
    model_evaluation_results: dict = evaluation_metrics(
        split_data_results=split_data_results,
        prediction_results=model_results['predictions'],
        target_col=feature_engeneering_results['target_cols'][0],
        n_params=len(regressor_cols) - 1
    )
        


    