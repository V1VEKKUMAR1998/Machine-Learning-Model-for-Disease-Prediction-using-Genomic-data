import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('genotype_data_for_ml.csv')

# Step 1: Drop SampleID as it's not a predictive feature
data = data.drop(columns=['SampleID'])

# Step 2: Separate features and target
X = data.drop(columns=['disease_risk'])
y = data['disease_risk']

# Step 3: Handle missing values (-1 represents missing data)
# Replace -1 with NaN for imputation
X = X.replace(-1, np.nan)

# Impute missing values using the mode (most frequent value) for each SNP
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Step 4: Verify data consistency
# Ensure all SNP values are in [0, 1, 2]
for col in X_imputed.columns:
    assert X_imputed[col].isin([0, 1, 2]).all(), f"Invalid values in {col}"

# Step 5: Train-test split (assuming more data is available)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Step 6: Save preprocessed data
X_train.to_csv('X_train_preprocessed.csv', index=False)
X_test.to_csv('X_test_preprocessed.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Print summary
print("Preprocessing complete.")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print("Preprocessed data saved as CSV files.")