import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Read the CSV file
df = pd.read_csv(r'C:\Users\osama\Documents\IS498-ML\final_version.csv', index_col=False)

# Select a specific stock
selected_stock = 5110 # Replace with the desired stock symbol
selected_df = df[df['Symbol'] == selected_stock]

company_name = selected_df['Company Name'].iloc[0]


# Define the feature columns
feature_columns = ['RollingMean', 'MACD', 'RSI']

# Create a new DataFrame with only the feature columns and the 'Close' column
data = selected_df[feature_columns + ['Close']]

# Create a DatetimeIndex using the 'Year', 'Month', and 'Day' columns
data.index = pd.to_datetime(selected_df[['Year', 'Month', 'Day']])

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split the data into training and testing sets
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Prepare the training data
x_train = train_data[:-1, :-1]
y_train = train_data[1:, -1]

# Prepare the testing data
x_test = test_data[:-1, :-1]
y_test = test_data[1:, -1]

# Build the Random Forest model
model = RandomForestRegressor(n_estimators=200, random_state=40)

# Train the model
model.fit(x_train, y_train)

# Make predictions
predictions = model.predict(x_test)

# Inverse transform the predictions and actual values
inv_predictions = scaler.inverse_transform(np.concatenate((x_test, predictions.reshape(-1, 1)), axis=1))[:, -1]
inv_y_test = scaler.inverse_transform(np.concatenate((x_test, y_test.reshape(-1, 1)), axis=1))[:, -1]

# Calculate accuracy
threshold = 0.03  # Define the threshold for accuracy (e.g., 5%)
accurate_predictions = np.sum(np.abs(inv_predictions - inv_y_test) / inv_y_test <= threshold)
accuracy = accurate_predictions / len(inv_y_test) * 100

# Create a DataFrame for signals
signals_df = pd.DataFrame(index=data.index[-len(inv_y_test):])
signals_df['Close'] = inv_y_test
signals_df['Predicted'] = inv_predictions
signals_df['Signal'] = 0

# Initialize the portfolio with a starting value of 3000
portfolio_value = 3000

# Generate buy and sell signals based on predictions
for i in range(1, len(signals_df)):
    if signals_df['Predicted'][i] > signals_df['Predicted'][i - 1] * 1.02:  # Buy signal: 5% increase in predicted price
        signals_df['Signal'][i] = 1
    elif signals_df['Predicted'][i] < signals_df['Predicted'][i - 1] * 0.98:  # Sell signal: 5% decrease in predicted price
        signals_df['Signal'][i] = -1

# Implement risk management and update portfolio value
position = 0
risk_tolerance = 0.02  # Maximum allowable loss percentage

buy_price = 0
sell_price = 0
num_trades = 0

for i in range(len(signals_df)):
    if signals_df['Signal'][i] == 1 and position == 0:
        buy_price = signals_df['Close'][i]
        shares = portfolio_value // buy_price
        position = 1  # Buy
        num_trades += 1
        print(f"Trade {num_trades}: Bought {shares} shares at {buy_price:.2f}. Portfolio value: {portfolio_value:.2f}")
    elif signals_df['Signal'][i] == -1 and position == 1:
        sell_price = signals_df['Close'][i]
        portfolio_value += shares * sell_price
        position = 0  # Sell
        num_trades += 1
        print(f"Trade {num_trades}: Sold {shares} shares at {sell_price:.2f}. Portfolio value: {portfolio_value:.2f}")

    if position == 1 and signals_df['Close'][i] < buy_price * (1 - risk_tolerance):
        sell_price = signals_df['Close'][i]
        portfolio_value += shares * sell_price
        position = 0  # Stop loss triggered, sell immediately
        num_trades += 1
        print(f"Trade {num_trades}: Stop loss triggered. Sold {shares} shares at {sell_price:.2f}. Portfolio value: {portfolio_value:.2f}")


# Calculate the percentage of profit
initial_portfolio_value = 3000
percentage_profit = (portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100

# Plot the price curve and signals
plt.figure(figsize=(12, 6))
plt.plot(signals_df['Close'], label='Close Price')
buy_signals = signals_df[signals_df['Signal'] == 1]
sell_signals = signals_df[signals_df['Signal'] == -1]
plt.scatter(buy_signals.index, buy_signals['Close'], color='green', label='Buy Signal', marker='^', alpha=1)
plt.scatter(sell_signals.index, sell_signals['Close'], color='red', label='Sell Signal', marker='v', alpha=1)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Create a DataFrame for plotting
plotting_data = pd.DataFrame(
    {
        'Original Close Price': inv_y_test,
        'Predicted Close Price': inv_predictions
    },
    index=data.index[-len(inv_y_test):]
)


# Plot the original close price and predicted close price
plt.figure(figsize=(12, 6))
plt.plot(plotting_data['Original Close Price'], label='Original Close Price')
plt.plot(plotting_data['Predicted Close Price'], label='Predicted Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title(f'Original vs. Predicted Close Price for {company_name}')  # Add company name as the title
plt.legend()
plt.show()

# Calculate the accuracy and other performance measurements
mse = mean_squared_error(inv_y_test, inv_predictions)
r2 = r2_score(inv_y_test, inv_predictions)

print(f"Number of trades: {num_trades}")
print(f"Final portfolio value: {portfolio_value:.2f}")
print(f"Percentage profit: {percentage_profit:.2f}%")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Rest of the code remains the same
print(f"Accuracy: {accuracy:.2f}%")