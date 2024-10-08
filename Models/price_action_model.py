import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class PriceActionModel:
    def __init__(self, window_size=100, n_estimators=100, max_depth=5, consecutive_signals=3):
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.consecutive_signals = consecutive_signals
        self.model = None

    def train(self, data):
        features = self.preprocess_data(data)
        target = self.generate_labels(data)
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth)
        self.model.fit(features, target)

    def preprocess_data(self, data):
        # Calculate higher highs and lower lows
        data.loc[:, 'Higher_High'] = (data['High'] > data['High'].shift(1)).astype(int)
        data.loc[:, 'Lower_Low'] = (data['Low'] < data['Low'].shift(1)).astype(int)

        # Add any other relevant features
        features = data[['Higher_High', 'Lower_Low', 'Volume Traded', 'MACD', 'RSI', 'BB_Upper', 'BB_Lower']]
        return features

    def generate_labels(self, data):
        # Generate buy/sell labels based on price action
        labels = pd.Series(0, index=data.index)
        labels.loc[(data['Higher_High'] == 1) & (data['Higher_High'].rolling(self.window_size).sum() >= self.consecutive_signals)] = 1
        labels.loc[(data['Lower_Low'] == 1) & (data['Lower_Low'].rolling(self.window_size).sum() >= self.consecutive_signals)] = -1
        return labels

    def predict(self, data):
        features = self.preprocess_data(data)
        predictions = self.model.predict(features)
        return predictions
