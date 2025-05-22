import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from datetime import datetime, timedelta, date
from dateutil.easter import easter
from scipy import stats
from prophet import Prophet
from prophet.utilities import regressor_coefficients
from pandas.errors import PerformanceWarning

# --- Configuration ---
warnings.filterwarnings('ignore', category=PerformanceWarning)

# --- Especification Constants ---
DEFAULT_DATE_COL = 'date'
DEFAULT_TARGET_COL = 'sales'
DEFAULT_COVARIATE_COLS = ['transactions']
DEFAULT_CUSTOM_DATE_DUMMIES = [
        {
            'name': 'blackfriday_dummy',
            'dates': [
                '2014-11-28', '2015-11-27', '2016-11-25', '2017-11-24', 
                '2018-11-23', '2019-11-29', '2020-11-27', '2021-11-26',
                '2022-11-25', '2023-11-24'
            ]
        }
    ]

DEFAULT_LOG_TRANSFORM = True
DEFAULT_DIFF_TRANSFORM = False
DEFAULT_COVARIATE_FORECASTING = True
DEFAULT_COVARIATES_LAGS = [0, 1, 7, 30]
DEFAULT_EVENT_LAGS = 1
DEFAULT_EVENT_LEADS = 1
DEFAULT_THRESHOLD_FEATURE_SELECTION = 0.01

TEST_SIZE = 180
FORECAST_SIZE = 365 * 3

# --- Directory Constants ---
TARGET_PATH = os.path.join(os.getcwd(), 'data/groupby_train.csv')
COVARIATE_PATH = os.path.join(os.getcwd(), 'data/groupby_transactions_2.csv')


# --- Color Constants ---
DEFAULT_COLOR_COMPONENT = 'tab:blue'
DEFAULT_COLOR_STANDARD = 'slategrey'
DEFAULT_COLOR_HIGHLIGHT = 'tab:orange'
DEFAULT_COLOR_ACTUAL = 'slategrey'
DEFAULT_COLOR_PREDICTED = 'tab:red'
DEFAULT_COLOR_FORECAST = 'tab:orange'
DEFAULT_COLOR_PASSED_TEST = 'tab:blue'
DEFAULT_COLOR_FAILED_TEST = 'tab:red'

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
                        covariates_lags: list[int] = DEFAULT_COVARIATES_LAGS, 
                        custom_date_dummies: list[dict] = DEFAULT_CUSTOM_DATE_DUMMIES,
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
    
    results['target_col'] = target_col = [col for col in transformable_df.columns if target_col in col][0]
    data = pd.concat([transformable_df[target_col], covariates], axis=1)
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
        'start': data[target_col].dropna().index[0].date(),
        'end': data[target_col].dropna().index[-1].date()
    }
    results['covariates_cutoff_dates'] = {
        'start': data.drop(columns=target_col).dropna().index[0].date(),
        'end': data.drop(columns=target_col).dropna().index[-1].date()
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
                target_col=feature_engeneering_results['target_col'], 
                covariate_cols=main_regression_cols + seas_cols,
                checking_cols=event_cols,
                threshold=threshold
            )
        checked_events_cols: list[str] = [col for col in events_cols if col not in under_threshold_regressors]

        for seas, seas_cols in feature_engeneering_results['seasonal_names'].items():
            print(f"   Check regressors coefficients for seas: {TEXT_ORANGE}{seas}{TEXT_COLOR_END}")
            under_threshold_regressors += check_regressor_under_threshold(
                train_data=split_data_results['train'], 
                target_col=feature_engeneering_results['target_col'], 
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
        target_col=feature_engeneering_results['target_col'], 
        checking_cols=regression_cols,
        threshold=threshold
    )
    return [col for col in regression_cols if col not in under_threshold_regressors]




def reverse_transformations(target_col: str,
                            loaded_data: pd.DataFrame,
                            model_results: dict, 
                            log_transform: bool = DEFAULT_LOG_TRANSFORM,
                            diff_transform: bool = DEFAULT_DIFF_TRANSFORM
                            ) -> dict:
    
    # Reverse log transformation
    results: dict = {}
    cols: list = ['yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper'] 
    for df in model_results['predictions']:
        results[df] = pd.DataFrame(index=model_results['predictions'][df].index)\
            .merge(
                loaded_data[target_col.replace('diff_','').replace('log_','')], 
                how='left', 
                left_index=True, 
                right_index=True
            )
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
    print(f"\n")  

    return results



def plot_fitted_model(target_col: str,
                      model_results: dict,
                      split_data_results: dict,
                      evaluation_results: dict,
                      ) -> dict:       

    # data
    fitted = model_results['predictions']['train']['yhat'].values
    actuals = split_data_results['train'][target_col].fillna(model_results['predictions']['train']['yhat']).values
    trend = model_results['predictions']['train']['trend'].values
    matched_dates = split_data_results['train'].index
    monthly_df = pd.DataFrame({'actual': actuals, 'fitted': fitted, 'trend': trend},index=matched_dates).resample('ME').mean()
    seasonal_df = pd.DataFrame({'date': matched_dates, 'fitted': fitted})
    seasonal_df['day_of_week'] = seasonal_df['date'].dt.dayofweek
    seasonal_df['month'] = seasonal_df['date'].dt.month
    mean_fitted = 1.0 if np.mean(fitted) == 0 else np.mean(fitted) # Avoid division by zero

    dow_grouped = seasonal_df.groupby('day_of_week')['fitted'].agg(['mean', 'std']).reindex(range(7)) # Ensure all days are present
    dow_grouped['normalized_mean'] = dow_grouped['mean'] / mean_fitted
    dow_grouped['normalized_std'] = dow_grouped['std'] / mean_fitted
    upper_bound_dow = dow_grouped['normalized_mean'] + dow_grouped['normalized_std']
    lower_bound_dow = dow_grouped['normalized_mean'] - dow_grouped['normalized_std']
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    x_dow = np.arange(len(day_names))

    month_grouped = seasonal_df.groupby('month')['fitted'].agg(['mean', 'std']).reindex(range(1, 13)) # Ensure all months
    month_grouped['normalized_mean'] = month_grouped['mean'] / mean_fitted
    month_grouped['normalized_std'] = month_grouped['std'] / mean_fitted
    upper_bound_month = month_grouped['normalized_mean'] + month_grouped['normalized_std']
    lower_bound_month = month_grouped['normalized_mean'] - month_grouped['normalized_std']
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x_month = np.arange(len(month_names))

    # Metrics
    min_val = min(np.min(actuals), np.min(fitted)) * 0.98
    max_val = max(np.max(actuals), np.max(fitted)) * 1.02
    residuals = actuals - fitted
    mean = np.mean(residuals)
    std = np.std(residuals)
    overfitting = evaluation_results['daily']['overfitting ~0']
    r_squared = evaluation_results['daily']['r-squared ~1']
    mape = evaluation_results['daily']['mape']['train']
    aicc = evaluation_results['daily']['aicc']
    skew = evaluation_results['daily']['skewness ~0']
    kurtosis = evaluation_results['daily']['kurtosis ~3']
    homos = evaluation_results['daily']['homocedasticity ~0']
    dw = evaluation_results['daily']['durbin-watson ~2']
    monthly_mape = evaluation_results['monthly']['mape']['train']

    # Diagnostics
    x_norm = np.linspace(min(residuals), max(residuals), 100)
    y_norm = stats.norm.pdf(x_norm, mean, std)
    mean_acceptable = abs(mean) < np.mean(np.abs(actuals)) * 0.01
    is_normal = (abs(skew) < 0.5) and (2.5 < abs(kurtosis) < 3.5)
    is_homoscedastic = abs(homos) < 0.3
    no_autocorrel = 1.5 < dw < 2.5
    outlier_threshold = 3 * std
    outlier_mask = np.abs(residuals) > outlier_threshold
    outlier_count = np.sum(outlier_mask)
    no_outliers = outlier_count < len(residuals) * 0.03
    diagnostics_text = (
        f"RESIDUAL DIAGNOSTICS:\n"
        f"Zero mean? {'✓' if mean_acceptable else '✗'}\n"
        f"Normal distr.? {'✓' if is_normal else '✗'} (sk={skew:.2f}, ku={kurtosis:.2f})\n"
        f"Homoscedastic? {'✓' if is_homoscedastic else '✗'} (corr={homos:.2f})\n"
        f"No autocorrel.? {'✓' if no_autocorrel else '✗'} (DW={dw:.2f})\n"
        f"No outliers? {'✓' if no_outliers else '✗'} ({outlier_count} pts)"
    )

    # Create a 2x5 grid of subplots
    fig = plt.figure(figsize=(24, 12)) 
    gs = GridSpec(2, 5, height_ratios=[1, 1.5], figure=fig) 
    ax1 = fig.add_subplot(gs[0, 0])  # Scatter plot (top-left)
    ax4 = fig.add_subplot(gs[0, 1])  # Residuals histogram (top-middle-left)
    ax3 = fig.add_subplot(gs[0, 2])  # Monthly plot (top-middle-right)
    ax5 = fig.add_subplot(gs[0, 3])  # Weekly seasonality (top-right-1)
    ax6 = fig.add_subplot(gs[0, 4])  # Monthly seasonality (top-right-2)
    ax2 = fig.add_subplot(gs[1, :])  # Daily plot (full bottom row)    

    # --- SUBPLOT 1: Scatter plot ---
    ax1.scatter(actuals, fitted, s=8, alpha=0.5, color=DEFAULT_COLOR_STANDARD)
    ax1.plot([min_val, max_val], [min_val, max_val], color=DEFAULT_COLOR_HIGHLIGHT, linewidth=2.5 , alpha=0.8, label='Perfect Fit')
    ax1.plot([min_val, max_val], [min_val + outlier_threshold, max_val + outlier_threshold],
            color=DEFAULT_COLOR_HIGHLIGHT, linestyle='dashed', alpha=0.5, linewidth=0.7, label=f'±{outlier_threshold:.1f} Threshold') # Updated label
    ax1.plot([min_val, max_val], [min_val - outlier_threshold, max_val - outlier_threshold],
            color=DEFAULT_COLOR_HIGHLIGHT, linestyle='dashed', alpha=0.5, linewidth=0.7)  
    if outlier_mask.any():
        ax1.scatter(actuals[outlier_mask], fitted[outlier_mask],
                    s=16, color=DEFAULT_COLOR_HIGHLIGHT, alpha=1.0, 
                    marker='o', label=f'Outliers ({sum(outlier_mask)})')
    stats_text = f"R²: {r_squared:.2f}\nMAPE: {mape:.2f}%\nAICc: {aicc:,.0f}"
    ax1.text(0.05, 0.95, stats_text,
            transform=ax1.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
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
    ax4.plot(x_norm, y_norm, color=DEFAULT_COLOR_HIGHLIGHT, alpha=0.8, linewidth=2.5, label='Normal Dist.')
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
    ax3.plot(monthly_df.index, monthly_df['actual'], color=DEFAULT_COLOR_ACTUAL, linewidth=2, alpha=1.0, label='Actual (Avg)')
    ax3.plot(monthly_df.index, monthly_df['fitted'], color=DEFAULT_COLOR_PREDICTED, linewidth=2, alpha=0.7, label='Fitted (Avg)')
    ax3.plot(monthly_df.index, monthly_df['trend'], color=DEFAULT_COLOR_COMPONENT, linewidth=3, alpha=1.0, linestyle='-', label='Trend (Avg)')
    monthly_stats = f"Monthly MAPE: {monthly_mape:.2f}%"
    ax3.text(0.05, 0.95, monthly_stats,
            transform=ax3.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Monthly Average Value')
    ax3.set_title("Monthly Aggregation (Train)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) # Format date axis
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7)) # Auto ticks

    # --- SUBPLOT 5: Weekly seasonality ---
    ax5.plot(x_dow, dow_grouped['normalized_mean'], 'o-', color=DEFAULT_COLOR_COMPONENT, linewidth=3, markersize=8, label='Actual (% of avg)')
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
    ax6.plot(x_month, month_grouped['normalized_mean'], 'o-', color=DEFAULT_COLOR_COMPONENT, linewidth=3, markersize=8, label='Actual (% of avg)')
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
    ax2.plot(matched_dates, fitted, color=DEFAULT_COLOR_PREDICTED, alpha=0.7, label='Fitted Values', linewidth=1.5)
    ax2.plot(matched_dates, trend, color=DEFAULT_COLOR_COMPONENT, linewidth=2, alpha=0.7, linestyle='--', label='Trend')
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

    plt.setp(ax2.get_xticklabels(), rotation=30, ha='right') # Rotate labels for ax2
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title/label overlap

    return fig



def plot_forecast(target_col: str,
                  evaluation_results: dict,
                  reverse_transform_results: dict
                  ) -> dict:   
    
    # data
    target_col = target_col.replace('diff_','').replace('log_','')
    daily_train_date = reverse_transform_results['train'].tail(60).index
    daily_train_actual = reverse_transform_results['train'][target_col].tail(60).values
    daily_test_date = reverse_transform_results['test'].index
    daily_test_actual = reverse_transform_results['test'][target_col].values
    daily_test_pred = reverse_transform_results['test']['yhat'].values
    daily_forecast_date = reverse_transform_results['forecast'].index
    daily_forecast_pred = reverse_transform_results['forecast']['yhat'].values
    daily_forecast_pred_lower = reverse_transform_results['forecast']['yhat_lower'].values
    daily_forecast_pred_upper = reverse_transform_results['forecast']['yhat_upper'].values

    train_date_cutoff = pd.to_datetime(reverse_transform_results['train'].index[-1]).to_period('M').start_time
    test_date_cutoff_end = pd.to_datetime(reverse_transform_results['test'].index[-1]).to_period('M').end_time
    test_date_cutoff_start = pd.to_datetime(reverse_transform_results['test'].index[-1]).to_period('M').start_time
    forecast_date_cutoff = pd.to_datetime(reverse_transform_results['forecast'].index[-1]).to_period('M').start_time - timedelta(days=1)

    daily_data = pd.concat([reverse_transform_results['train'], reverse_transform_results['test'], reverse_transform_results['forecast']])
    daily_data['combined'] = np.where(
        daily_data.index <= train_date_cutoff,
        daily_data[target_col],
        daily_data['yhat']
    )
    monthly_data = daily_data.resample('ME').sum()
    yearly_data = daily_data.resample('YE').sum()
    # remove last year if it is not full
    yearly_data = yearly_data[yearly_data.index.year < pd.to_datetime(reverse_transform_results['forecast'].index[-1]).year]

    monthly_train_date = monthly_data.index[monthly_data.index <= train_date_cutoff]
    monthly_train_actual = monthly_data.loc[monthly_train_date, target_col].values
    monthly_train_test_date = monthly_data.index[monthly_data.index <= test_date_cutoff_end]
    monthly_train_test_actual = monthly_data.loc[monthly_train_test_date, target_col].values
    monthly_test_date = monthly_data.index[(monthly_data.index >= train_date_cutoff) & (monthly_data.index <= test_date_cutoff_end)]
    monthly_test_actual = monthly_data.loc[monthly_test_date, target_col].values
    monthly_test_pred = monthly_data.loc[monthly_test_date, 'yhat'].values
    monthly_forecast_date = monthly_data.index[(monthly_data.index >= test_date_cutoff_start) & (monthly_data.index <= forecast_date_cutoff)]
    monthly_forecast_pred = monthly_data.loc[monthly_forecast_date, 'yhat'].values

    annual_colors = []
    for year in yearly_data.index.year:
        if year < train_date_cutoff.year:
            annual_colors.append(DEFAULT_COLOR_ACTUAL)
        elif year > test_date_cutoff_end.year:
            annual_colors.append(DEFAULT_COLOR_FORECAST)
        else:
            annual_colors.append(DEFAULT_COLOR_PREDICTED)


    # Create a 2x5 grid of subplots
    fig = plt.figure(figsize=(24, 12)) 
    gs = GridSpec(2, 2, height_ratios=[1, 1.5], figure=fig) 
    ax3 = fig.add_subplot(gs[0, 0])  # Monthly (top-left)
    ax2 = fig.add_subplot(gs[0, 1])  # Yearly (top-right)
    ax1 = fig.add_subplot(gs[1, :])  # Daily (full bottom row)  

    # --- SUBPLOT 1: Daily ---
    ax1.plot(daily_train_date, daily_train_actual, color=DEFAULT_COLOR_ACTUAL, lw=0.7, label='Trainning Actuals')
    ax1.plot(daily_test_date, daily_test_actual, color=DEFAULT_COLOR_ACTUAL, lw=2, label='Test Actuals')
    ax1.plot(daily_test_date, daily_test_pred, color=DEFAULT_COLOR_PREDICTED, lw=2, alpha=0.7, label='Test Prediction')
    ax1.plot(daily_forecast_date, daily_forecast_pred, color=DEFAULT_COLOR_FORECAST, lw=2, label='Forecast')
    if len(daily_forecast_date) < 180:
        ax1.fill_between(daily_forecast_date, daily_forecast_pred_lower, daily_forecast_pred_upper, color=DEFAULT_COLOR_FORECAST, alpha=0.15)
    ax1.axvline(daily_test_date[0], color=DEFAULT_COLOR_PREDICTED, linestyle=':', lw=1, label='Train/Test Split')
    ax1.axvline(daily_forecast_date[0], color=DEFAULT_COLOR_FORECAST, linestyle=':', lw=1, label='Forecast Start')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- SUBPLOT 2: Monthly ---
    ax2.plot(monthly_train_test_date, monthly_train_test_actual, label='Trainning Actuals', color=DEFAULT_COLOR_ACTUAL, linestyle='-', linewidth=1.0)
    ax2.plot(monthly_test_date, monthly_test_actual, label='Test Actuals', color=DEFAULT_COLOR_ACTUAL, linestyle='-', linewidth=2.5)
    ax2.plot(monthly_test_date, monthly_test_pred, label='Test Prediction', color=DEFAULT_COLOR_PREDICTED, linestyle='-', linewidth=2.5 , alpha=0.7)
    ax2.plot(monthly_forecast_date, monthly_forecast_pred, label='Forecast', color=DEFAULT_COLOR_FORECAST, linestyle='-', linewidth=2.5)
    ax2.axvline(monthly_test_date[0], color=DEFAULT_COLOR_PREDICTED, linestyle=':', linewidth=1.0, alpha=0.5, label='Train/Test Split')
    ax2.axvline(monthly_forecast_date[0], color=DEFAULT_COLOR_FORECAST, linestyle=':', linewidth=1.0, alpha=0.5, label='Forecast Start')
    ax2.set_ylabel('Value (Monthly Sum)')
    ax2.set_title('Monthly Aggregation')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # --- SUBPLOT 2: Yearly ---
    bars = ax3.bar(yearly_data.index.year, yearly_data['combined'].values, color=annual_colors, alpha=0.9)
    ax3.set_ylabel('Annual Total')
    ax3.set_title('Annual Aggregation')
    ax3.grid(True, alpha=0.3) #, axis='y'
    ax3.legend(handles=[
            Patch(facecolor=DEFAULT_COLOR_ACTUAL, label='Actual Only'),
            Patch(facecolor=DEFAULT_COLOR_FORECAST, label='Forecast Only'),
            Patch(facecolor=DEFAULT_COLOR_PREDICTED, label='Actual + Forecast')
        ], loc='upper left') 
    
    # Add value labels on top of each bar with YoY variation
    for i, (bar, value) in enumerate(zip(bars, yearly_data['combined'].values)):
        height = bar.get_height()
        
        # Calculate year-over-year variation
        yoy_text = ""
        if i > 0:  # Skip for the first year as there's no previous year to compare
            prev_value = yearly_data['combined'].values[i-1]
            if prev_value > 0:  # Avoid division by zero
                yoy_pct = ((value - prev_value) / prev_value) * 100
                sign = "+" if yoy_pct >= 0 else ""
                yoy_text = f"\n{sign}{yoy_pct:.1f}%"
        
        # Display value in milions and YoY variation
        ax3.text(bar.get_x() + bar.get_width()/2., height + (max(yearly_data['combined'].values) * 0.02),
                f'{int(value/1_000_000):,}M{yoy_text}', ha='center', va='bottom', rotation=0, fontsize=16)
    ax3.set_ylim(0, max(yearly_data['combined'].values) * 1.25)

    plt.tight_layout()

    return fig



def export_output(target_col: str,
                  feature_engeneering_results: dict,
                  split_data_results: dict, 
                  model_results: dict, 
                  model_evaluation_results: dict,
                  reverse_transform_results: dict,
                  output_path: str|None,
                  export_models: bool = False,
                  export_dataframes: bool = False,
                  export_plots: bool = False,
                  ) -> None:

    output_path = os.path.join(os.getcwd(), 'output') if output_path is None else output_path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if export_models and model_results and 'model' in model_results:
        models_path = os.path.join(output_path, target_col, 'models')
        os.makedirs(models_path, exist_ok=True)
        with open(os.path.join(models_path, f"model_{timestamp}.pkl"), 'wb') as f:
            pickle.dump(model_results['model'], f)
        print(f"   Model saved to {TEXT_BLUE}{os.path.join(models_path, f'model_{timestamp}.pkl')}{TEXT_COLOR_END}")

    if export_dataframes:
        dataframes_path = os.path.join(output_path, target_col, 'dataframes')
        os.makedirs(dataframes_path, exist_ok=True)
        if feature_engeneering_results and 'data' in feature_engeneering_results:
            feature_engeneering_results['data'].to_csv(os.path.join(dataframes_path, f'feature_engeneering_data_{timestamp}.csv'))
        print(f"   Feature engeneering data saved to {TEXT_BLUE}{os.path.join(dataframes_path, f'feature_engeneering_data_{timestamp}.csv')}{TEXT_COLOR_END}")

        if split_data_results and export_dataframes:
            for key, df in split_data_results.items():
                df.to_csv(os.path.join(dataframes_path, f"split_data_{key}_{timestamp}.csv"))
        print(f"   Split data saved to {TEXT_BLUE}{os.path.join(dataframes_path, f'split_data_{key}_{timestamp}.csv')}{TEXT_COLOR_END}")

        if model_results and 'coefficients' in model_results:
            model_results['coefficients'].to_csv(os.path.join(dataframes_path, f"model_coefficients_{timestamp}.csv"))
        print(f"   Model coefficients saved to {TEXT_BLUE}{os.path.join(dataframes_path, f'model_coefficients_{timestamp}.csv')}{TEXT_COLOR_END}")

        if model_results and 'predictions' in model_results:
            for key, df in model_results['predictions'].items():
                df['yhat'].to_csv(os.path.join(dataframes_path, f"predictions_{key}_{timestamp}.csv"))
        print(f"   Predictions saved to {TEXT_BLUE}{os.path.join(dataframes_path, f'predictions_{key}_{timestamp}.csv')}{TEXT_COLOR_END}")

        if reverse_transform_results:
            for key, df in reverse_transform_results.items():
                df.to_csv(os.path.join(dataframes_path, f"forecast_{key}_{timestamp}.csv"))
        print(f"   Forecast data saved to {TEXT_BLUE}{os.path.join(dataframes_path, f'forecast_{key}_{timestamp}.csv')}{TEXT_COLOR_END}")

    if export_plots:
        plots_path = os.path.join(output_path, target_col, 'plots')
        os.makedirs(plots_path, exist_ok=True)

        plot_fitted_model(
            target_col=target_col,
            model_results=model_results,
            split_data_results=split_data_results,
            evaluation_results=model_evaluation_results
        ).savefig(os.path.join(plots_path, f"fitted_model_{timestamp}.png"), dpi=300)
        print(f"   Fitted model plot saved to {TEXT_BLUE}{os.path.join(plots_path, f'fitted_model_{timestamp}.png')}{TEXT_COLOR_END}")

        plot_forecast(
            target_col=target_col,
            evaluation_results=model_evaluation_results,
            reverse_transform_results=reverse_transform_results
        ).savefig(os.path.join(plots_path, f"forecast_{timestamp}.png"), dpi=300)
        print(f"   Forecast plot saved to {TEXT_BLUE}{os.path.join(plots_path, f'forecast_{timestamp}.png')}{TEXT_COLOR_END}")

    




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
        custom_date_dummies = DEFAULT_CUSTOM_DATE_DUMMIES,
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
        target_col=feature_engeneering_results['target_col'],
        covariate_cols=regressor_cols
    )

    # Model evaluation
    model_evaluation_results: dict = evaluation_metrics(
        split_data_results=split_data_results,
        prediction_results=model_results['predictions'],
        target_col=feature_engeneering_results['target_col'],
        n_params=len(regressor_cols) - 1
    )

    # reverse transformations
    reverse_transform_results: dict = reverse_transformations(
        target_col=feature_engeneering_results['target_col'],
        loaded_data=loaded_data,
        model_results=model_results,
        log_transform=DEFAULT_LOG_TRANSFORM,
        diff_transform=DEFAULT_DIFF_TRANSFORM
    )
 
    # Export output
    export_output(
        target_col=feature_engeneering_results['target_col'],
        feature_engeneering_results=feature_engeneering_results,
        split_data_results=split_data_results,
        model_results=model_results,
        model_evaluation_results=model_evaluation_results,
        reverse_transform_results=reverse_transform_results,
        output_path=None,
        export_models=True,
        export_dataframes=False,
        export_plots=True
    )
    

    