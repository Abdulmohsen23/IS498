from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


selected_df= pd.read_csv(r'C:\Users\osama\Documents\IS498-ML\final_version.csv', index_col=False)

print(selected_df.head(10))
# Assuming your data is stored in a DataFrame called 'selected_df'
# with all the features you mentioned

# Prepare the data for linear regression
X_price = selected_df[['Open', 'High', 'Low', 'Close', 'Change', '% Change', 'Volume Traded',
                       'Value Traded (SAR)', 'No. of Trades', 'Day', 'Month', 'Year',
                       'DayOfWeek', 'Quarter', 'RollingMean', 'RollingStd', 'RollingMin',
                       'RollingMax', 'MACD', 'MACDSignal', 'RSI', 'BB_Upper', 'BB_Middle',
                       'BB_Lower', 'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_5',
                       'Close_lag_7']]  # Features for price prediction
y_price = selected_df['Close']  # Target variable for price prediction

# Split the data into training and testing sets for linear regression
# Split the data into training and testing sets for linear regression
X_price_train, X_price_test, y_price_train, y_price_test = train_test_split(X_price, y_price, test_size=0.2, random_state=42)

# Create and train the linear regression model
price_model = LinearRegression()
price_model.fit(X_price_train, y_price_train)

# Create a MinMaxScaler instance for the target variable
target_scaler = MinMaxScaler()
selected_df['Close'] = target_scaler.fit_transform(selected_df[['Close']])

numerical_columns = ['Open', 'High', 'Low', 'Close', 'Volume Traded', 'Value Traded (SAR)', 'No. of Trades']

# Create a MinMaxScaler object and normalize the numerical columns
scaler = MinMaxScaler()
selected_df[numerical_columns] = scaler.fit_transform(selected_df[numerical_columns])

# Predict prices for the entire dataset
price_predictions = price_model.predict(X_price)  

# Inverse transform the price predictions if needed
price_predictions = target_scaler.inverse_transform(price_predictions.reshape(-1, 1)) 
price_predictions = price_predictions.flatten()

# Add the predictions as a column
selected_df['Predicted_Price'] = price_predictions 




# Prepare the data for logistic regression
X_decision = selected_df[['Open', 'High', 'Low', 'Close', 'Change', '% Change', 'Volume Traded',
                          'Value Traded (SAR)', 'No. of Trades', 'Day', 'Month', 'Year',
                          'DayOfWeek', 'Quarter', 'RollingMean', 'RollingStd', 'RollingMin',
                          'RollingMax', 'MACD', 'MACDSignal', 'RSI', 'BB_Upper', 'BB_Middle',
                          'BB_Lower', 'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_5',
                          'Close_lag_7']]  # Features for buy/sell decision
X_decision['Predicted_Price'] = price_predictions  # Add predicted price as a feature

# Prepare the target variables for logistic regression
y_long_high_risk = selected_df['Long_High_Risk']
y_long_middle_risk = selected_df['Long_Middle_Risk']
y_long_low_risk = selected_df['Long_Low_Risk']
y_short_high_risk = selected_df['Short_High_Risk']
y_short_middle_risk = selected_df['Short_Middle_Risk']
y_short_low_risk = selected_df['Short_Low_Risk']

# Split the data into training and testing sets for logistic regression
X_decision_train, X_decision_test, y_long_high_risk_train, y_long_high_risk_test, \
y_long_middle_risk_train, y_long_middle_risk_test, y_long_low_risk_train, y_long_low_risk_test, \
y_short_high_risk_train, y_short_high_risk_test, y_short_middle_risk_train, y_short_middle_risk_test, \
y_short_low_risk_train, y_short_low_risk_test = train_test_split(X_decision, y_long_high_risk, y_long_middle_risk,
                                                                 y_long_low_risk, y_short_high_risk, y_short_middle_risk,
                                                                 y_short_low_risk, test_size=0.2, random_state=42)

# Create and train the logistic regression models for each risk level
long_high_risk_model = LogisticRegression(max_iter=500000)
long_high_risk_model.fit(X_decision_train, y_long_high_risk_train)

long_middle_risk_model = LogisticRegression(max_iter=500000)
long_middle_risk_model.fit(X_decision_train, y_long_middle_risk_train)

long_low_risk_model = LogisticRegression(max_iter=500000)
long_low_risk_model.fit(X_decision_train, y_long_low_risk_train)

short_high_risk_model = LogisticRegression(max_iter=500000)
short_high_risk_model.fit(X_decision_train, y_short_high_risk_train)

short_middle_risk_model = LogisticRegression(max_iter=500000)
short_middle_risk_model.fit(X_decision_train, y_short_middle_risk_train)

short_low_risk_model = LogisticRegression(max_iter=500000)
short_low_risk_model.fit(X_decision_train, y_short_low_risk_train)

# Make buy/sell predictions using the trained logistic regression models
long_high_risk_predictions = long_high_risk_model.predict(X_decision_test)
long_middle_risk_predictions = long_middle_risk_model.predict(X_decision_test)
long_low_risk_predictions = long_low_risk_model.predict(X_decision_test)
short_high_risk_predictions = short_high_risk_model.predict(X_decision_test)
short_middle_risk_predictions = short_middle_risk_model.predict(X_decision_test)
short_low_risk_predictions = short_low_risk_model.predict(X_decision_test)

# Evaluate the performance of the logistic regression models
long_high_risk_accuracy = accuracy_score(y_long_high_risk_test, long_high_risk_predictions)
long_middle_risk_accuracy = accuracy_score(y_long_middle_risk_test, long_middle_risk_predictions)
long_low_risk_accuracy = accuracy_score(y_long_low_risk_test, long_low_risk_predictions)
short_high_risk_accuracy = accuracy_score(y_short_high_risk_test, short_high_risk_predictions)
short_middle_risk_accuracy = accuracy_score(y_short_middle_risk_test, short_middle_risk_predictions)
short_low_risk_accuracy = accuracy_score(y_short_low_risk_test, short_low_risk_predictions)

print("Long High Risk Accuracy:", long_high_risk_accuracy)
print("Long Middle Risk Accuracy:", long_middle_risk_accuracy)
print("Long Low Risk Accuracy:", long_low_risk_accuracy)
print("Short High Risk Accuracy:", short_high_risk_accuracy)
print("Short Middle Risk Accuracy:", short_middle_risk_accuracy)
print("Short Low Risk Accuracy:", short_low_risk_accuracy)

long_high_risk_precision = precision_score(y_long_high_risk_test, long_high_risk_predictions)
long_high_risk_recall = recall_score(y_long_high_risk_test, long_high_risk_predictions)

long_middle_risk_precision = precision_score(y_long_middle_risk_test, long_middle_risk_predictions)
long_middle_risk_recall = recall_score(y_long_middle_risk_test, long_middle_risk_predictions)

long_low_risk_precision = precision_score(y_long_low_risk_test, long_low_risk_predictions)
long_low_risk_recall = recall_score(y_long_low_risk_test, long_low_risk_predictions)

short_high_risk_precision = precision_score(y_short_high_risk_test, short_high_risk_predictions)
short_high_risk_recall = recall_score(y_short_high_risk_test, short_high_risk_predictions)

short_middle_risk_precision = precision_score(y_short_middle_risk_test, short_middle_risk_predictions)
short_middle_risk_recall = recall_score(y_short_middle_risk_test, short_middle_risk_predictions)

short_low_risk_precision = precision_score(y_short_low_risk_test, short_low_risk_predictions)
short_low_risk_recall = recall_score(y_short_low_risk_test, short_low_risk_predictions)

print("Long High Risk Precision:", long_high_risk_precision)
print("Long High Risk Recall:", long_high_risk_recall)
print("Long Middle Risk Precision:", long_middle_risk_precision)
print("Long Middle Risk Recall:", long_middle_risk_recall)
print("Long Low Risk Precision:", long_low_risk_precision)
print("Long Low Risk Recall:", long_low_risk_recall)
print("Short High Risk Precision:", short_high_risk_precision)
print("Short High Risk Recall:", short_high_risk_recall)
print("Short Middle Risk Precision:", short_middle_risk_precision)
print("Short Middle Risk Recall:", short_middle_risk_recall)
print("Short Low Risk Precision:", short_low_risk_precision)
print("Short Low Risk Recall:", short_low_risk_recall)

# Make predictions for new data
new_data = pd.DataFrame({'Open': [100], 'High': [110], 'Low': [95], 'Close': [105],
                         'Change': [5], '% Change': [0.05], 'Volume Traded': [1000],
                         'Value Traded (SAR)': [100000], 'No. of Trades': [100],
                         'Day': [15], 'Month': [6], 'Year': [2023], 'DayOfWeek': [1],
                         'Quarter': [2], 'RollingMean': [102], 'RollingStd': [2],
                         'RollingMin': [98], 'RollingMax': [108], 'MACD': [0.5],
                         'MACDSignal': [0.3], 'RSI': [60], 'BB_Upper': [110],
                         'BB_Middle': [105], 'BB_Lower': [100], 'Close_lag_1': [104],
                         'Close_lag_2': [103], 'Close_lag_3': [102], 'Close_lag_5': [100],
                         'Close_lag_7': [98]})  # Example new data

new_price_prediction = price_model.predict(new_data)
new_data['Predicted_Price'] = new_price_prediction

new_long_high_risk_prediction = long_high_risk_model.predict(new_data)
new_long_middle_risk_prediction = long_middle_risk_model.predict(new_data)
new_long_low_risk_prediction = long_low_risk_model.predict(new_data)
new_short_high_risk_prediction = short_high_risk_model.predict(new_data)
new_short_middle_risk_prediction = short_middle_risk_model.predict(new_data)
new_short_low_risk_prediction = short_low_risk_model.predict(new_data)

print("New Price Prediction:", new_price_prediction)
print("New Long High Risk Prediction:", new_long_high_risk_prediction)
print("New Long Middle Risk Prediction:", new_long_middle_risk_prediction)
print("New Long Low Risk Prediction:", new_long_low_risk_prediction)
print("New Short High Risk Prediction:", new_short_high_risk_prediction)
print("New Short Middle Risk Prediction:", new_short_middle_risk_prediction)
print("New Short Low Risk Prediction:", new_short_low_risk_prediction)

