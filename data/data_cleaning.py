import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import ta
import pickle


# Read the CSV file
df = pd.read_csv(r'C:\Users\osama\Documents\IS498-ML\data\Equites_Historical_Adjusted_Prices_Report.csv', index_col=False)

# Fill missing values with NaN
df.fillna(np.nan, inplace=True)

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Convert 'Company Name', 'Industry Group', and 'Symbol' to string type
df[['Company Name', 'Industry Group', 'Symbol']] = df[['Company Name', 'Industry Group', 'Symbol']].astype(str)

# Sort the DataFrame by the 'Date' column in ascending order and reset the index
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Select the best company for each industry based on volume and historical data
industries = df['Industry Group'].unique()
selected_company_records = []

for industry in industries:
    industry_df = df[df['Industry Group'] == industry]
    
    company_stats = industry_df.groupby('Company Name').agg({
        'Volume Traded': 'mean',
        'Date': 'count'
    })
    company_stats.columns = ['Avg Volume', 'Historical Data Count']
    
    company_stats['Volume Percentile'] = company_stats['Avg Volume'].rank(pct=True)
    company_stats['Historical Data Percentile'] = company_stats['Historical Data Count'].rank(pct=True)
    
    company_stats['Combined Score'] = (company_stats['Volume Percentile'] + company_stats['Historical Data Percentile']) / 2
    
    best_company = company_stats.nlargest(1, 'Combined Score').index[0]
    
    company_records = industry_df[industry_df['Company Name'] == best_company].sort_values('Date')
    
    selected_company_records.append(company_records)

selected_df = pd.concat(selected_company_records, ignore_index=True)

# Extract time-based features from the 'Date' column
selected_df['Day'] = selected_df['Date'].dt.day
selected_df['Month'] = selected_df['Date'].dt.month
selected_df['Year'] = selected_df['Date'].dt.year
selected_df['DayOfWeek'] = selected_df['Date'].dt.dayofweek
selected_df['Quarter'] = selected_df['Date'].dt.quarter

# Create rolling window features
window_size = 10
selected_df['RollingMean'] = selected_df['Close'].rolling(window=window_size).mean()
selected_df['RollingStd'] = selected_df['Close'].rolling(window=window_size).std()
selected_df['RollingMin'] = selected_df['Close'].rolling(window=window_size).min()
selected_df['RollingMax'] = selected_df['Close'].rolling(window=window_size).max()

# Calculate MACD
macd = ta.trend.MACD(selected_df['Close'])
selected_df['MACD'] = macd.macd()
selected_df['MACDSignal'] = macd.macd_signal()

# Calculate RSI
selected_df['RSI'] = ta.momentum.RSIIndicator(selected_df['Close']).rsi()

# Calculate Bollinger Bands
bollinger = ta.volatility.BollingerBands(selected_df['Close'])
selected_df['BB_Upper'] = bollinger.bollinger_hband()
selected_df['BB_Middle'] = bollinger.bollinger_mavg()
selected_df['BB_Lower'] = bollinger.bollinger_lband()

# Fill missing values with the mean of each column
columns_to_fill = ['RollingStd', 'RollingMean', 'RollingMin', 'RollingMax', 'MACD', 'MACDSignal', 'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower']
selected_df[columns_to_fill] = selected_df[columns_to_fill].fillna(selected_df[columns_to_fill].mean())

# Create lagged features for 'Close' price
lag_periods = [1, 2, 3, 5, 7]
for lag in lag_periods:
    selected_df[f'Close_lag_{lag}'] = selected_df['Close'].shift(lag)

# Select the top k features based on f-regression score
feature_columns = ['Open', 'High', 'Low', 'Volume Traded', 'RollingMean', 'RollingStd', 'MACD', 'RSI']
target_column = 'Close'
k = 10
selector = SelectKBest(score_func=f_regression, k=k)
X_selected = selector.fit_transform(selected_df[feature_columns], selected_df[target_column])

# Get the selected feature names
selected_features = selector.get_support(indices=True)
selected_feature_names = [feature_columns[i] for i in selected_features]
print("Selected Features:", selected_feature_names)

# Define the time steps for each strategy and risk level
strategies = {
    'Long_High_Risk': 90,
    'Long_Middle_Risk': 180,
    'Long_Low_Risk': 365,
    'Short_High_Risk': 5,
    'Short_Middle_Risk': 10,
    'Short_Low_Risk': 30
}

# Create columns for each strategy and risk level
for strategy in strategies:
    selected_df[strategy] = 0

# Assign labels for each strategy and risk level
for company in selected_df['Symbol'].unique():
    company_df = selected_df[selected_df['Symbol'] == company]
    
    for strategy, time_step in strategies.items():
        selected_df.loc[company_df.index, strategy] = (company_df['Close'].shift(-time_step) > company_df['Close']).astype(int)

# Encode categorical features
label_encoder = LabelEncoder()
selected_df[['Industry Group', 'Company Name']] = selected_df[['Industry Group', 'Company Name']].apply(label_encoder.fit_transform)

# Drop rows with missing values
selected_df.dropna(inplace=True)


    

# Save the final DataFrame to a CSV file
selected_df.to_csv('final_version.csv', index=False)

# Print the shape of the DataFrame
print("DataFrame Shape:", selected_df.shape)