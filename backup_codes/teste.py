

import pandas as pd
import numpy as np

target_df = pd.read_csv('data/groupby_transactions.csv', parse_dates=['date'])
# print(target_df.head())

target_df['date'] = pd.to_datetime(target_df['date'])
target_df['dayofweek'] = target_df['date'].dt.dayofweek

# Add weekend dummy (1 for weekend, 0 for weekday)
target_df['is_weekend'] = (target_df['dayofweek'] >= 5).astype(int)

# If you want separate dummies for Saturday and Sunday
target_df['is_dayofweek_7'] = (target_df['dayofweek'] == 5).astype(int)
target_df['is_dayofweek_6'] = (target_df['dayofweek'] == 4).astype(int)
target_df['is_dayofweek_5'] = (target_df['dayofweek'] == 3).astype(int)
target_df['is_dayofweek_4'] = (target_df['dayofweek'] == 2).astype(int)
target_df['is_dayofweek_3'] = (target_df['dayofweek'] == 1).astype(int)
target_df['is_dayofweek_2'] = (target_df['dayofweek'] == 0).astype(int)
target_df['is_dayofweek_1'] = (target_df['dayofweek'] == 6).astype(int)

# Apply exponential transformation to all dummy variables
dummy_cols = ['is_weekend', 'is_dayofweek_1', 'is_dayofweek_2', 'is_dayofweek_3', 
             'is_dayofweek_4', 'is_dayofweek_5', 'is_dayofweek_6', 'is_dayofweek_7']

for col in dummy_cols:
    target_df[col] = np.exp(target_df[col])

print(target_df[['date', 'transactions']+dummy_cols])

target_df.to_csv('data/groupby_transactions_2.csv', index=False)