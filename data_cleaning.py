import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ta
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression

# Read the CSV file
df = pd.read_csv(r'C:\Users\osama\Documents\IS498-ML\Equites_Historical_Adjusted_Prices_Report.csv', index_col=False)

# Fill missing values with NaN
df.fillna(np.nan, inplace=True)

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Convert 'Company Name', 'Industry Group', and 'Symbol' to string type
df['Company Name'] = df['Company Name'].astype(str)
df['Industry Group'] = df['Industry Group'].astype(str)
df['Symbol'] = df['Symbol'].astype(str)

# Sort the DataFrame by the 'Date' column in ascending order and reset the index
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Get the list of unique industries
industries = df['Industry Group'].unique()

# Initialize an empty list to store the selected companies' records
selected_company_records = []

# Iterate over each industry
for industry in industries:
    # Filter the DataFrame for the current industry
    industry_df = df[df['Industry Group'] == industry]
    
    # Calculate the average volume traded and historical data count for each company in the industry
    avg_volume = industry_df.groupby('Company Name')['Volume Traded'].mean()
    historical_data_count = industry_df.groupby('Company Name').size()
    
    # Combine the average volume and historical data count into a single DataFrame
    company_stats = pd.concat([avg_volume, historical_data_count], axis=1)
    company_stats.columns = ['Avg Volume', 'Historical Data Count']
    
    # Calculate the percentile ranks for average volume and historical data count
    company_stats['Volume Percentile'] = company_stats['Avg Volume'].rank(pct=True)
    company_stats['Historical Data Percentile'] = company_stats['Historical Data Count'].rank(pct=True)
    
    # Calculate the combined score based on volume and historical data percentiles
    company_stats['Combined Score'] = (company_stats['Volume Percentile'] + company_stats['Historical Data Percentile']) / 2
    
    # Select the company with the highest combined score from the industry
    best_company = company_stats.nlargest(1, 'Combined Score').index[0]
    
    # Filter and sort the records for the selected company
    company_records = industry_df[industry_df['Company Name'] == best_company].sort_values('Date')
    
    # Append the sorted records for the selected company to the list
    selected_company_records.append(company_records)

# Concatenate the selected company records into a single DataFrame
selected_df = pd.concat(selected_company_records, ignore_index=True)

# Specify the numerical columns to normalize
numerical_columns = ['Open', 'High', 'Low', 'Close', 'Volume Traded', 'Value Traded (SAR)', 'No. of Trades']

# Create a MinMaxScaler object and normalize the numerical columns
scaler = MinMaxScaler()
selected_df[numerical_columns] = scaler.fit_transform(selected_df[numerical_columns])

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

# Specify the columns for which to fill missing values with the mean
columns_to_fill = ['RollingStd', 'RollingMean', 'RollingMin', 'RollingMax', 'MACD', 'MACDSignal', 'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower']

# Fill missing values with the mean of each column
for column in columns_to_fill:
    selected_df[column] = selected_df[column].fillna(selected_df[column].mean())

# Create lagged features for 'Close' price
lag_periods = [1, 2, 3, 5, 7]
for lag in lag_periods:
    selected_df[f'Close_lag_{lag}'] = selected_df['Close'].shift(lag)

# Normalize the RSI values
selected_df['RSI'] = scaler.fit_transform(selected_df[['RSI']])

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
long_term_high_risk = 90
long_term_middle_risk = 180
long_term_low_risk = 365
short_term_high_risk = 5
short_term_middle_risk = 10
short_term_low_risk = 30

# Create columns for each strategy and risk level
selected_df['Long_High_Risk'] = np.zeros(len(selected_df))
selected_df['Long_Middle_Risk'] = np.zeros(len(selected_df))
selected_df['Long_Low_Risk'] = np.zeros(len(selected_df))
selected_df['Short_High_Risk'] = np.zeros(len(selected_df))
selected_df['Short_Middle_Risk'] = np.zeros(len(selected_df))
selected_df['Short_Low_Risk'] = np.zeros(len(selected_df))

# Get the list of unique company symbols
companies = selected_df['Symbol'].unique()

# Iterate through each company and assign labels for each strategy and risk level
for company in companies:
    company_df = selected_df[selected_df['Symbol'] == company]
    
    for i in range(len(company_df)):
        if i + long_term_high_risk < len(company_df) and company_df.iloc[i + long_term_high_risk]['Close'] > company_df.iloc[i]['Close']:
            selected_df.loc[company_df.index[i], 'Long_High_Risk'] = 1
        if i + long_term_middle_risk < len(company_df) and company_df.iloc[i + long_term_middle_risk]['Close'] > company_df.iloc[i]['Close']:
            selected_df.loc[company_df.index[i], 'Long_Middle_Risk'] = 1
        if i + long_term_low_risk < len(company_df) and company_df.iloc[i + long_term_low_risk]['Close'] > company_df.iloc[i]['Close']:
            selected_df.loc[company_df.index[i], 'Long_Low_Risk'] = 1
        if i + short_term_high_risk < len(company_df) and company_df.iloc[i + short_term_high_risk]['Close'] > company_df.iloc[i]['Close']:
            selected_df.loc[company_df.index[i], 'Short_High_Risk'] = 1
        if i + short_term_middle_risk < len(company_df) and company_df.iloc[i + short_term_middle_risk]['Close'] > company_df.iloc[i]['Close']:
            selected_df.loc[company_df.index[i], 'Short_Middle_Risk'] = 1
        if i + short_term_low_risk < len(company_df) and company_df.iloc[i + short_term_low_risk]['Close'] > company_df.iloc[i]['Close']:
            selected_df.loc[company_df.index[i], 'Short_Low_Risk'] = 1

# Convert label columns to integer type
label_columns = ['Long_High_Risk', 'Long_Middle_Risk', 'Long_Low_Risk',
                 'Short_High_Risk', 'Short_Middle_Risk', 'Short_Low_Risk']
selected_df[label_columns] = selected_df[label_columns].astype(int)


# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Encode the categorical features
selected_df['Industry Group'] = label_encoder.fit_transform(selected_df['Industry Group'])
selected_df['Company Name'] = label_encoder.fit_transform(selected_df['Company Name'])

# Print the updated DataFrame with the new label columns
print(selected_df.head())

# Save the final DataFrame to a CSV file
selected_df.to_csv('final_version.csv', index=False)

# Print the data types of the columns
print(selected_df.dtypes)

# Print the shape of the DataFrame
print(selected_df.shape)

# Print the column names of the DataFrame
print(selected_df.columns)