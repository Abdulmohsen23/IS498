from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
# Assuming you have your preprocessed data stored in 'selected_df'
selected_df= pd.read_csv(r'C:\Users\osama\Documents\IS498-ML\final_version.csv', index_col=False)

# Specify the feature columns and target columns
feature_columns = ['Open', 'High', 'Low', 'Volume Traded', 'RollingMean', 'RollingStd', 'MACD', 'RSI', 'Industry Group', 'Company Name']
target_columns = ['Long_High_Risk', 'Long_Middle_Risk', 'Long_Low_Risk', 'Short_High_Risk', 'Short_Middle_Risk', 'Short_Low_Risk']

# Split the data into features (X) and targets (y)
X = selected_df[feature_columns]
y = selected_df[target_columns]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an XGBoost Classifier
xgb_classifier = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Train the classifier on the training data
xgb_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = xgb_classifier.predict(X_test)

# Evaluate the model's performance
for i, label in enumerate(target_columns):
    print(f"Label: {label}")
    print("Accuracy:", accuracy_score(y_test[label], y_pred[:, i]))
    print("Precision:", precision_score(y_test[label], y_pred[:, i]))
    print("Recall:", recall_score(y_test[label], y_pred[:, i]))
    print("F1-score:", f1_score(y_test[label], y_pred[:, i]))
    print()