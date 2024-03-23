import pandas as pd
import numpy as np
from scipy import stats
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.dummy import DummyClassifier


# Loading data
df = pd.read_csv('D:\ML\Data\ALrajhiData 1.1.csv')

# Inspect your data
# Example: Forward filling missing values 
df.fillna(method='ffill', inplace=True)

# Example: Imputation with the mean
df['Volume Traded'].fillna(df['Volume Traded'].mean(), inplace=True) 


# Example: Capping outliers above the 95th percentile
for feature in ['Open', 'High', 'Low', 'Close']:
    q75, q25 = np.percentile(df[feature], [75,25])
    iqr = q75 - q25
    upper_bound = q75 + (1.5 * iqr)
    df[feature] = np.clip(df[feature], None, upper_bound)

# Calculate Bollinger Bands
bb_result = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2) 

# Accessing the bands
df['BB_upper'] = bb_result.bollinger_hband()
df['BB_middle'] = bb_result.bollinger_mavg()
df['BB_lower'] = bb_result.bollinger_lband()




# Calculate short-term EMA (12-period) and long-term EMA (26-period)
short_ema = df['Close'].ewm(span=12, min_periods=12, adjust=False).mean()
long_ema = df['Close'].ewm(span=26, min_periods=26, adjust=False).mean()

# Calculate MACD Line
macd_line = short_ema - long_ema

# Calculate Signal Line (9-period EMA of MACD Line)
signal_line = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()

# Calculate MACD Histogram
macd_histogram = macd_line - signal_line

# Assign calculated values to DataFrame columns
df['MACD_Line'] = macd_line
df['Signal_Line'] = signal_line
df['MACD_Histogram'] = macd_histogram




df['Date'] = pd.to_datetime(df['Date']) 

# Simple Moving Average
df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10) 

# RSI 
df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

scaler = StandardScaler()
features_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume Traded']
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

df['Price_Direction'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)  # 1 represents 'up', 0 represents 'down'


# Target Variable (Assuming you've calculated it as discussed earlier)
y = df['Price_Direction']  


# Data Splitting 
X = df[['SMA_10', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'MACD_Line', 'Signal_Line', 'MACD_Histogram']] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Create an imputer object with strategy='mean'
imputer = SimpleImputer(strategy='mean')


# Fit the imputer on the training data
imputer.fit(X_train)


# Model Training (and onwards) ... 
model = LogisticRegression()

# Impute missing values in training and testing data
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Now, train your model using the imputed data
model.fit(X_train_imputed, y_train)
y_pred = model.predict(X_test_imputed)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1-Score:', f1_score(y_test, y_pred))

# Naive Baseline Comparison

dummy_clf = DummyClassifier(strategy="most_frequent") # Or try 'stratified', etc.
dummy_clf.fit(X_train_imputed, y_train)
dummy_y_pred = dummy_clf.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Naive Baseline Accuracy", accuracy_score(y_test, dummy_y_pred))


fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

