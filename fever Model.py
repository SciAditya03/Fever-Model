import pandas as pd

# Load the dataset
data = pd.read_csv("enhanced_fever_medicine_recommendation.csv")

# Check the structure of the dataset
print(data.info())
print(data.head())

import pandas as pd

# Load the dataset
data = pd.read_csv("enhanced_fever_medicine_recommendation.csv")

# Display the structure and preview the dataset
print(data.info())
print(data.head())

print(data.columns)

# Define thresholds for fever, flu, or cold
def define_labels(row):
    # Convert relevant columns to numeric, handling errors
    row['Heart_Rate'] = pd.to_numeric(row['Heart_Rate'], errors='coerce')
    row['Blood_Pressure'] = pd.to_numeric(row['Blood_Pressure'], errors='coerce')
    row['Humidity'] = pd.to_numeric(row['Humidity'], errors='coerce')
    
    if row['Temperature'] > 37.5:  # High temperature threshold
        if row['Heart_Rate'] > 100 or row['Blood_Pressure'] > 120 or row['Humidity'] < 30:
            return 'flu'  # Flu-like symptoms
        return 'fever'  # Fever only
    elif row['Heart_Rate'] > 100 or row['Blood_Pressure'] > 120 or row['Humidity'] < 30:
        return 'cold'  # Cold-like symptoms
    return 'healthy'  # Normal readings

# Apply the function to create the condition column
data['condition'] = data.apply(define_labels, axis=1)

# Verify the column was created successfully
print(data['condition'].value_counts())

# Define features (X) and target (y)
X = data.drop('condition', axis=1)  # Drop the target column
y = data['condition']              # Target column

# Optional: Check the first few rows
print(X.head())
print(y.head())

from sklearn.preprocessing import LabelEncoder

# Encode categorical columns (if necessary)
categorical_columns = ['Headache', 'Fever_Severity']  # Replace with your column names
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

from sklearn.model_selection import train_test_split

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.info())

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical columns
X_train['Gender'] = label_encoder.fit_transform(X_train['Gender'])
X_test['Gender'] = label_encoder.transform(X_test['Gender'])  # Use the same encoder

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Ensure both train and test sets have the same columns
X_train, X_test = X_train.align(X_test, join='left', axis=1)
X_test = X_test.fillna(0)  # Fill missing columns with 0

from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Scale training and testing sets
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
rf_accuracy = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

import pandas as pd

# One-hot encode categorical variables in X_train and X_test
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Ensure both datasets have the same columns after encoding
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Train the Random Forest model
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred = rf_model.predict(X_test)

# Print evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

from sklearn.metrics import classification_report, confusion_matrix

# Predict on test data
y_pred = rf_model.predict(X_test)

# Print evaluation metrics
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

import joblib

# Save the Random Forest model to a .pkl file
joblib.dump(rf_model, "random_forest_model.pkl")

print("Random Forest model saved successfully as random_forest_model.pkl")

from google.colab import files

# Download the .pkl file
files.download("random_forest_model.pkl")

