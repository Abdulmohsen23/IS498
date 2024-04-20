from sklearn.ensemble import RandomForestClassifier
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

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)