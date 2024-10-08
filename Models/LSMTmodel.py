import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error, explained_variance_score

# Read the CSV file
df = pd.read_csv(r'C:\Users\osama\Documents\IS498-ML\final_version.csv', index_col=False)

# Select a specific stock
selected_stock = 5110 # Replace with the desired stock symbol
selected_df = df[df['Symbol'] == selected_stock]

# Get the company name for the selected stock
company_name = selected_df['Company Name'].iloc[0]

# Define the feature columns
feature_columns = ["% Change" ,'RollingMean', 'MACD', 'RSI']

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
x_train = []
y_train = []
for i in range(100, len(train_data)):
    x_train.append(train_data[i - 100:i, :-1])
    y_train.append(train_data[i, -1])
x_train, y_train = np.array(x_train), np.array(y_train)

# Prepare the testing data
x_test = []
y_test = []
for i in range(100, len(test_data)):
    x_test.append(test_data[i - 100:i, :-1])
    y_test.append(test_data[i, -1])
x_test, y_test = np.array(x_test), np.array(y_test)

# Reshape the input data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=4, epochs=10)

# Make predictions
predictions = model.predict(x_test)

# Inverse transform the predictions and actual values
inv_predictions = scaler.inverse_transform(np.concatenate((x_test[:, -1, :], predictions), axis=1))[:, -1]
inv_y_test = scaler.inverse_transform(np.concatenate((x_test[:, -1, :], y_test.reshape(-1, 1)), axis=1))[:, -1]


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
mae = mean_absolute_error(inv_y_test, inv_predictions)
print(mse)
print(mae)
