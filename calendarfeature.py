"""
calendarfeature.py

This module provides functions to generate calendar-based features for time series and machine learning models.

Functions:
-----------
- holiday_feature(...): 
- seasonal_features(...):

"""

from datetime import date, timedelta, datetime
from dateutil.easter import easter
import pandas as pd
import numpy as np

def calendar_events_features(start_date=None,
                             end_date=None, 
                             weekdays: bool = True, 
                             workingdays: bool = False,
                             lags: int = None, 
                             leads: int = None,
                             slope_coef: int = 0,
                             drift_date=None,
                             event_names: list = None):
    """
    holiday_feature(...): 
        Generates a DataFrame with binary columns for Brazilian holidays, including fixed-date, Easter-based, and special holidays (e.g., Black Friday, Mother's Day, Father's Day). 
        Optionally adds weekday-specific, working day, lagged, and lead features for each holiday.

        Parameters:
            start_date (date): Start of the date range (default: 10 years ago).
            end_date (date): End of the date range (default: 10 years ahead).
            weekdays (bool): If True, adds columns for each holiday by weekday.
            workingdays (bool): If True, adds a column indicating working days.
            lags (int): Number of lagged versions of holiday/weekday/workingday columns to add.
            leads (int): Number of lead versions of holiday/weekday/workingday columns to add.
            slope_coef (int): Coefficient for pulse value calculation.
            drift_date (str/date): If provided, features before this date are zeroed out.
            event_names (list): Optional list of specific event names to generate features for.


        Returns:
            data (pd.DataFrame): DataFrame indexed by date with holiday and calendar features.
            columns_by_events (dict): Dictionary with lists of column names grouped by original event name.
            columns_by_events_types (dict): Dictionary with lists of column names by feature type (fixed, moving, weekday, etc.).
    """
    
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
            lambda x: pulse_value(x.year) if (x == (easter(x.year) + timedelta(days=holiday['delta_days']))) else 0.0
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
            lambda x: pulse_value(x.year) if (x == (day_of_special_event(x.year, event['month'], event['weekday'], event['delta_days']))) else 0.0
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
            leaded_data = leaded_data.shift(lead)
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
    """
    seasonal_features(...):
        Generates seasonal features such as day of week, day of month, and month of year, 
        with optional one-hot encoding and drift logic.

        Parameters:
            start_date (str/date): Start of the date range (default: 10 years ago).
            end_date (str/date): End of the date range (default: 10 years ahead).
            dayofweek (bool): If True, adds day of week feature.
            dayofweekpermonth (bool): If True, adds combined day of week and month feature.
            dayofmonth (bool): If True, adds day of month feature.
            monthofyear (bool): If True, adds month of year feature.
            drift_date (str/date): If provided, features before this date are zeroed out.
            drop_first (bool): Whether to drop the first category in one-hot encoding.

        Returns:
            data (pd.DataFrame): DataFrame indexed by date with seasonal features (one-hot encoded).
            columns_by_type (dict): Dictionary with lists of column names by feature type.
    """

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

# Example usage and print output for testing
if __name__ == '__main__':
    holiday_df, cols_by_event_name, cols_by_event_type = calendar_events_features(
        start_date='2020-12-01', 
        end_date='2025-02-01', 
        weekdays=True, 
        workingdays=True, 
        lags=2, 
        leads=1, 
        slope_coef=0, # Set to 0 for simple binary, or non-zero for pulse
        drift_date='2022-01-01',
        event_names=['natal', 'anonovo', 'carnaval'] # Optional: test with specific events
    )
    print("\n\n\n---- Holiday Features ----\n")
    print(holiday_df.tail(20))
    print("\n\n---- Holiday Features Sum ----\n")
    print(holiday_df.sum())
    print("\n\nColumns by Event Name:")
    for group_name, col_list in cols_by_event_name.items():
        print(f"\n  {group_name} ({len(col_list)}): \n{col_list[:5]}...") # Print first 5
    print("\nColumns by Event Type:")
    for group_name, col_list in cols_by_event_type.items():
        print(f"\n  {group_name} ({len(col_list)}): \n{col_list[:5]}...") # Print first 5


    seas_df, seas_cols_by_type = calendar_seasonal_features(
        start_date='2020-12-01', 
        end_date='2025-01-15', 
        dayofweek=True, 
        dayofweekpermonth=True, 
        dayofmonth=True, 
        monthofyear=True, 
        drift_date='2022-01-01',
        drop_first=False
    )
    print("\n\n\n---- Seasonal Features ----\n")
    print(seas_df.tail(20))
    print("\n\n---- Seasonal Features Sum ----\n")
    print(holiday_df.sum())
    print("\n\nSeasonal Columns by Type:")
    for type_name, col_list in seas_cols_by_type.items():
        print(f"\n  {type_name} ({len(col_list)}): \n{col_list}")


