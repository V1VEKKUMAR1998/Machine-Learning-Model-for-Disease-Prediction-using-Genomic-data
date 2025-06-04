import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('genotype_data_for_ml.csv')

# Drop SampleID as it's not a predictive feature
data = data.drop(columns=['SampleID'])

# Separate features and target
X = data.drop(columns=['disease_risk'])
y = data['disease_risk']

# Replace -1 with NaN for imputation
X = X.replace(-1, np.nan)

# Drop columns with all NaN values
X = X.dropna(axis=1, how='all')

# Check if there are any columns left after dropping
if X.empty:
    raise ValueError("No valid features remain after dropping columns with all missing values.")

# Impute missing values using the mode
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Verify data consistency
for col in X_imputed.columns:
    assert X_imputed[col].isin([0, 1, 2]).all(), f"Invalid values in {col}"

# Train-test split (requires more than one row)
if len(X_imputed) < 2:
    raise ValueError("Dataset has only one row. More samples are needed for train-test split.")
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Save preprocessed data
X_train.to_csv('X_train_preprocessed.csv', index=False)
X_test.to_csv('X_test_preprocessed.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Preprocessing complete.")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print("Preprocessed data saved as CSV files.")