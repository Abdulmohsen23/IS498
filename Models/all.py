import numpy as np
import pandas as pd
import ta  # Technical analysis library

# Load and preprocess the data
df = pd.read_csv('final_version.csv', index_col=False)
# Implement your data preprocessing function

# Price Action Model
def price_action_model(data):
    """
    Implement the higher-high, lower-low price action patterns,
    pyramiding entry-exit strategy, and averaging strategy.
    Returns buy, sell, and hold signals.
    """
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0

    # Higher-high and lower-low patterns
    for i in range(1, len(data)):
        if data['Close'][i] > data['Close'][i - 1]:
            signals.loc[signals.index[i], 'signal'] = 1  # Buy signal
        elif data['Close'][i] < data['Close'][i - 1]:
            signals.loc[signals.index[i], 'signal'] = -1  # Sell signal

    # Pyramiding and averaging strategies
    # Implement your pyramiding and averaging strategies here
    # and update the 'signal' column accordingly

    signals['positions'] = signals['signal'].cumsum().apply(np.sign)
    return signals

# Technical Analysis Model
def technical_analysis_model(data):
    """
    Calculate technical indicators like ATR, CCI, and EMAs.
    Generate buy, sell, and hold signals based on thresholds.
    """
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0

    # Calculate technical indicators
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
    data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close'], window=20).cci()
    data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
    data['EMA_200'] = ta.trend.EMAIndicator(data['Close'], window=200).ema_indicator()

    # Generate signals based on thresholds
    for i in range(1, len(data)):
        if data['CCI'][i] > 100:  # Overbought
            signals.loc[signals.index[i], 'signal'] = -1  # Sell signal
        elif data['CCI'][i] < -100:  # Oversold
            signals.loc[signals.index[i], 'signal'] = 1  # Buy signal
        elif data['Close'][i] > data['EMA_50'][i] and data['Close'][i] > data['EMA_200'][i]:
            signals.loc[signals.index[i], 'signal'] = 1  # Buy signal
        elif data['Close'][i] < data['EMA_50'][i] and data['Close'][i] < data['EMA_200'][i]:
            signals.loc[signals.index[i], 'signal'] = -1  # Sell signal

    signals['positions'] = signals['signal'].cumsum().apply(np.sign)
    return signals

# Simple Exponential Smoothing
def exponential_smoothing(data, alpha=0.3):
    """
    Implement the Simple Exponential Smoothing technique.
    Generate trend signals based on smoothed values.
    """
    smoothed = data['Close'].ewm(alpha=alpha).mean()
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0

    for i in range(1, len(data)):
        if data['Close'][i] > smoothed[i]:
            signals.loc[signals.index[i], 'signal'] = 1  # Uptrend signal
        elif data['Close'][i] < smoothed[i]:
            signals.loc[signals.index[i], 'signal'] = -1  # Downtrend signal

    return signals

# Ensembling
def ensemble_signals(price_action, technical, smoothing, weights=[0.3, 0.4, 0.3]):
    """
    Combine signals from different models using an ensembling strategy.
    Generate final buy, sell, and hold signals.
    """
    signals = price_action * weights[0] + technical * weights[1] + smoothing * weights[2]
    signals['signal'] = np.sign(signals['signal'])
    signals['positions'] = signals['signal'].cumsum().apply(np.sign)
    return signals

def backtest(data, signals):
    """
    Implement a trading strategy based on the final ensemble signals.
    Backtest the strategy and calculate performance metrics.
    Plot the equity curve and buy/sell signals.
    """
    # Initialize portfolio
    portfolio = data['Close'].copy()
    portfolio[:] = 1000  # Initial capital
    position = 0

    # Trading logic
    for i in range(1, len(data)):
        try:
            if signals['signal'][i] == 1:  # Buy signal
                position = portfolio[i - 1] / data['Close'][i]
            elif signals['signal'][i] == -1:  # Sell signal
                portfolio[i] = portfolio[i - 1] + position * data['Close'][i]
                position = 0

            if i < len(data) - 1:
                portfolio[i + 1] = portfolio[i] * (1 + (data['Close'][i + 1] / data['Close'][i] - 1) * position)
        except Exception as e:
            print(f"Error at index {i}: {e}")
            continue

    # Drop the initial row with NaN values
    portfolio = portfolio.dropna()

    # Calculate performance metrics
    returns = portfolio.pct_change().dropna()
    annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    max_drawdown = (portfolio / portfolio.cummax()).min() - 1
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)

    print(f"Annual Return: {annual_return:.4f}")
    print(f"Maximum Drawdown: {max_drawdown:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

    # Plot equity curve and signals
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio)
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')

    buy_signals = signals.loc[signals['signal'] == 1]
    sell_signals = signals.loc[signals['signal'] == -1]
    plt.plot(buy_signals.index, data['Close'][buy_signals.index], '^', color='g', markersize=10)
    plt.plot(sell_signals.index, data['Close'][sell_signals.index], 'v', color='r', markersize=10)

    plt.show()

# Main execution
if __name__ == '__main__':
    # Generate signals from different models
    price_action_signals = price_action_model(df)
    technical_signals = technical_analysis_model(df)
    smoothing_signals = exponential_smoothing(df)

    # Ensemble signals
    ensemble_signals = ensemble_signals(price_action_signals, technical_signals, smoothing_signals)

    # Backtest and evaluate performance
    backtest(df, ensemble_signals)