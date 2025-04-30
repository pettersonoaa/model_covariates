import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet  

def load_series(csv_path, time_col='date', value_col='sales', name='y', freq='D'):
    df = pd.read_csv(csv_path, usecols=[time_col, value_col], index_col=0, parse_dates=[time_col])
    df = df.rename_axis('ds')
    df = df.groupby('ds')[value_col].sum().asfreq(freq)
    series = pd.Series(df, index=df.index, name=name)
    series = series.interpolate(method='time')
    return series


df_y = load_series(csv_path='data/groupby_train.csv', value_col='sales', name='y')
df_X = load_series(csv_path='data/groupby_transactions.csv', value_col='transactions', name='transactions')
df = pd.concat([df_y, df_X], axis=1)
df = df.reset_index()
df['transactions_lag30'] = df['transactions'].shift(30)
df['transactions_lag60'] = df['transactions'].shift(60)
df['transactions_lag90'] = df['transactions'].shift(90)
df['transactions_lag120'] = df['transactions'].shift(120)
# df = df.drop('transactions', axis='columns')
df = df.dropna()
# df = df.set_index('ds').map(np.log).reset_index()
df['transactions_log'] = df['transactions'].ffill().map(np.log)
# print(df)
# df['transactions_log'].plot()
# plt.show()




df_train, df_test = df[:-30], df[-30:]
df_test = df_test.drop('y', axis='columns')

model = Prophet(
    seasonality_mode='multiplicative',
    seasonality_prior_scale=20, 
)
model.add_regressor('transactions_log')
model.add_regressor('transactions_lag30')
model.add_regressor('transactions_lag60')
model.add_regressor('transactions_lag90')
model.add_regressor('transactions_lag120')
model.fit(df)

# 5. Prepare the future dataframe
# You must include the same covariate for future dates too!
future = model.make_future_dataframe(periods=90, freq='D') 

# Merge or manually add future values for covariate
# (Here we'll just re-use the last known value, but ideally predict it separately)
# future = future.merge(df[['ds', 'transactions']], how='left', on='ds')

fitted = model.predict(df_train)
forecast = model.predict(df_test)
# fig1 = model.plot(fitted)
fig1 = model.plot_components(fitted)
# fig2 = model.plot_components(forecast)








# 5. Merge actuals for plotting
df_plot = pd.merge(
    df[['ds', 'y']],
    fitted[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
    on='ds',
    how='outer'
)
df_plot = df_plot.tail(180)

# Set seaborn style manually for elegance
sns.set_style("whitegrid")
plt.figure(figsize=(16, 8))

# --- Plotting ---

# # 1. Plot uncertainty interval (with lower alpha, no thick fill)
# plt.fill_between(
#     df_plot['ds'],
#     df_plot['yhat_lower'],
#     df_plot['yhat_upper'],
#     color='cornflowerblue',
#     alpha=0.25,
#     label='Forecast Uncertainty'
# )

# 2. Plot Actual (past observations)
sns.lineplot(
    x='ds', y='y', data=df_plot,
    label='Actual', color='gray', linewidth=1.0
)

# 3. Plot Fitted (train values)
sns.lineplot(
    x='ds', y='yhat', data=df_plot,
    label='Fitted', color='red', linewidth=3.0, #linestyle='--'
)

# # Vertical line to separate past and future
# plt.axvline(x=df_test['ds'].min(), color='gray', linestyle='--', linewidth=1)

# Titles and labels
plt.title('Fitted Plot', fontsize=22, weight='bold', pad=20)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Value', fontsize=16)

# Improve x-axis
plt.xticks(rotation=45)
plt.gca().set_xticklabels([d.strftime('%b %Y') for d in df_plot['ds']])

# Improve grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Remove top and right spines
sns.despine()

# Legend
plt.legend(fontsize=13, loc='upper left')

# Tight layout
plt.tight_layout()











# 5. Merge actuals for plotting
df_plot = pd.merge(
    df[['ds', 'y']],
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
    on='ds',
    how='outer'
)
df_plot = pd.merge(
    df_plot[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']],
    fitted[['ds', 'yhat']].rename(columns={'yhat': 'fitted_yhat'}),
    on='ds',
    how='outer'
)
df_plot = df_plot.tail(365)

# Set seaborn style manually for elegance
sns.set_style("whitegrid")
plt.figure(figsize=(16, 8))

# --- Plotting ---

# 1. Plot uncertainty interval (with lower alpha, no thick fill)
plt.fill_between(
    df_plot['ds'],
    df_plot['yhat_lower'],
    df_plot['yhat_upper'],
    color='cornflowerblue',
    alpha=0.25,
    label='Forecast Uncertainty'
)

# 2. Plot Actual (past observations)
sns.lineplot(
    x='ds', y='y', data=df_plot,
    label='Actual', color='gray', linewidth=1.0
)

# 3. Plot Forecast (future values)
sns.lineplot(
    x='ds', y='yhat', data=df_plot,
    label='Forecast', color='royalblue', linewidth=3.0, #linestyle='--'
)

# 3. Plot Fitted (train values)
sns.lineplot(
    x='ds', y='fitted_yhat', data=df_plot,
    label='Fitted', color='red', linewidth=3.0, #linestyle='--'
)

# Vertical line to separate past and future
plt.axvline(x=df_test['ds'].min(), color='gray', linestyle='--', linewidth=1)

# Titles and labels
plt.title('Forecast Plot', fontsize=22, weight='bold', pad=20)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Value', fontsize=16)

# Improve x-axis
plt.xticks(rotation=45)
plt.gca().set_xticklabels([d.strftime('%b %Y') for d in df_plot['ds']])

# Improve grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Remove top and right spines
sns.despine()

# Legend
plt.legend(fontsize=13, loc='upper left')

# Tight layout
plt.tight_layout()

# Show
plt.show()


# # 8. (Optional) Export forecast
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('forecast_output.csv', index=False)
