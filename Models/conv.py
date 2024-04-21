import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
# Assuming your data is stored in a DataFrame called 'selected_df'
# Extract the relevant features and target variables

selected_df= pd.read_csv(r'C:\Users\osama\Documents\IS498-ML\final_version.csv', index_col=False)

feature_columns = ['Open', 'High', 'Low', 'Volume Traded', 'RollingMean', 'RollingStd', 'MACD', 'RSI']
label_columns = ['Long_High_Risk', 'Long_Middle_Risk', 'Long_Low_Risk', 'Short_High_Risk', 'Short_Middle_Risk', 'Short_Low_Risk']

X = selected_df[feature_columns].values
y = selected_df[label_columns].values

# Define the number of timesteps and features
timesteps = 30  # Number of historical time steps to consider
features = X.shape[1]  # Number of features per time step

# Reshape the input data into a 3D format
X_reshaped = []
y_reshaped = []

for i in range(timesteps, len(X)):
    X_reshaped.append(X[i - timesteps:i])
    y_reshaped.append(y[i])

X_reshaped = np.array(X_reshaped)
y_reshaped = np.array(y_reshaped)

# Print the shapes of the reshaped data
print("X_reshaped shape:", X_reshaped.shape)
print("y_reshaped shape:", y_reshaped.shape)




# Define the custom dataset class
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNNModel(nn.Module):
    def __init__(self, num_timesteps, num_features, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * (num_timesteps // 8), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x.permute(0, 2, 1))
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Set the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)

train_dataset = StockDataset(X_train, y_train)
val_dataset = StockDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create an instance of the CNN model
num_timesteps = X_train.shape[1]
num_features = X_train.shape[2]
num_classes = y_train.shape[1]
model = CNNModel(num_timesteps, num_features, num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Train the model
num_epochs = 13
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            val_loss += criterion(outputs, batch_y).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

# Evaluate the model on the validation set
model.eval()
with torch.no_grad():
    val_predictions = []
    val_targets = []
    for batch_X, batch_y in val_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        val_predictions.extend(outputs.cpu().numpy())
        val_targets.extend(batch_y.cpu().numpy())

    val_predictions = np.array(val_predictions)
    val_targets = np.array(val_targets)

    # Compute evaluation metrics
    mse = np.mean((val_predictions - val_targets) ** 2)
    mae = np.mean(np.abs(val_predictions - val_targets))

    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation MAE: {mae:.4f}")

# Interpret the predictions and make decisions
threshold = 0.5
for i in range(len(val_predictions)):
    if val_predictions[i][0] > threshold:
        print(f"Sample {i+1}: Buy signal (Long_High_Risk)")
    elif val_predictions[i][1] > threshold:
        print(f"Sample {i+1}: Buy signal (Long_Middle_Risk)")
    elif val_predictions[i][2] > threshold:
        print(f"Sample {i+1}: Buy signal (Long_Low_Risk)")
    elif val_predictions[i][3] > threshold:
        print(f"Sample {i+1}: Sell signal (Short_High_Risk)")
    elif val_predictions[i][4] > threshold:
        print(f"Sample {i+1}: Sell signal (Short_Middle_Risk)")
    elif val_predictions[i][5] > threshold:
        print(f"Sample {i+1}: Sell signal (Short_Low_Risk)")
    else:
        print(f"Sample {i+1}: No significant signal")