import pandas as pd
import json

# Read the CSV file
df = pd.read_csv('final_version.csv')

# Convert the DataFrame to a list of dictionaries
data = df.to_dict('records')

# Create a list to store the chat conversations
chat_conversations = []

for row in data:
    conversation = [
        {"role": "system", "content": "You are a financial assistant that predicts stock trends and generates buy/sell signals based on the provided data."},
        {"role": "user", "content": f"Industry Group: {row['Industry Group']}, Symbol: {row['Symbol']}, Company Name: {row['Company Name']}, Date: {row['Date']}, Open: {row['Open']}, High: {row['High']}, Low: {row['Low']}, Close: {row['Close']}, Change: {row['Change']}, % Change: {row['% Change']}, Volume Traded: {row['Volume Traded']}, Value Traded (SAR): {row['Value Traded (SAR)']}, No. of Trades: {row['No. of Trades']}, Day: {row['Day']}, Month: {row['Month']}, Year: {row['Year']}, DayOfWeek: {row['DayOfWeek']}, Quarter: {row['Quarter']}, RollingMean: {row['RollingMean']}, RollingStd: {row['RollingStd']}, RollingMin: {row['RollingMin']}, RollingMax: {row['RollingMax']}, MACD: {row['MACD']}, MACDSignal: {row['MACDSignal']}, RSI: {row['RSI']}, BB_Upper: {row['BB_Upper']}, BB_Middle: {row['BB_Middle']}, BB_Lower: {row['BB_Lower']}, Close_lag_1: {row['Close_lag_1']}, Close_lag_2: {row['Close_lag_2']}, Close_lag_3: {row['Close_lag_3']}, Close_lag_5: {row['Close_lag_5']}, Close_lag_7: {row['Close_lag_7']}, Long_High_Risk: {row['Long_High_Risk']}, Long_Middle_Risk: {row['Long_Middle_Risk']}, Long_Low_Risk: {row['Long_Low_Risk']}, Short_High_Risk: {row['Short_High_Risk']}, Short_Middle_Risk: {row['Short_Middle_Risk']}, Short_Low_Risk: {row['Short_Low_Risk']}"
        }
    ]
    
    # Here, you need to determine the desired assistant response (completion) based on your data
    # For example, you could use a simple classification like:
    if row['Close'] > row['Open']:
        completion = "Based on the provided data, the stock trend is UP. I would recommend a BUY signal."
    else:
        completion = "Based on the provided data, the stock trend is DOWN. I would recommend a SELL signal."
    
    conversation.append({"role": "assistant", "content": completion})
    
    chat_conversations.append({"messages": conversation})

# Split the data into training and validation sets
# Assuming you want to use 80% for training and 20% for validation
split_index = int(0.8 * len(chat_conversations))
train_data = chat_conversations[:split_index]
val_data = chat_conversations[split_index:]

# Save the training and validation data as JSONL files
with open('train.jsonl', 'w', encoding='utf-8') as f:
    for conversation in train_data:
        json.dump(conversation, f)
        f.write('\n')

with open('val.jsonl', 'w', encoding='utf-8') as f:
    for conversation in val_data:
        json.dump(conversation, f)
        f.write('\n')