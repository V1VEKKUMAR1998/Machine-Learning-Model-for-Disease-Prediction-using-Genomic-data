import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_excel('New Microsoft Excel Worksheet.xlsx')

# Step 1: Encode genotypes
def encode_genotype(row):
    ref = row['Reference']
    alt = row['Alter'].split(',')[0]  # Take first alternate allele if multiple
    genotype = row['Genotype']
    if genotype == f"{ref}{ref}":
        return 0  # Homozygous Reference
    elif genotype in [f"{ref}{alt}", f"{alt}{ref}"]:
        return 1  # Heterozygous
    elif genotype == f"{alt}{alt}":
        return 2  # Homozygous Variant
    else:
        return -1  # Invalid or missing

# Step 2: Encode disease risk
data['disease_risk'] = data['Results'].map({'Low risk': 0, 'High risk': 1})

# Step 3: Assign synthetic SampleIDs
data['SampleID'] = range(len(data))

# Step 4: Encode genotypes numerically
data['Genotype_Encoded'] = data.apply(encode_genotype, axis=1)

# Step 5: Create wide format (pivot table for genotypes)
pivot_data = data.pivot_table(index='SampleID', columns='Risk Allele', values='Genotype_Encoded', aggfunc='first').reset_index()

# Step 6: Merge disease_risk back into pivot_data
pivot_data = pivot_data.merge(data[['SampleID', 'disease_risk']].drop_duplicates(), on='SampleID', how='left')

# Step 7: Separate features and target
X = pivot_data.drop(columns=['SampleID', 'disease_risk'])
y = pivot_data['disease_risk']

# Step 8: Replace -1 with NaN for imputation
X = X.replace(-1, np.nan)

# Step 9: Drop columns with all NaN values
X = X.dropna(axis=1, how='all')

# Check if there are any columns left
if X.empty:
    raise ValueError("No valid features remain after dropping columns with all missing values.")

# Step 10: Impute missing values using the mode
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Step 11: Verify data consistency
for col in X_imputed.columns:
    assert X_imputed[col].isin([0, 1, 2]).all(), f"Invalid values in {col}"

# Step 12: Train-test split
if len(X_imputed) < 2:
    raise ValueError("Dataset has only one row. More samples are needed for train-test split.")
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Step 13: Save preprocessed data
X_train.to_csv('X_train_preprocessed.csv', index=False)
X_test.to_csv('X_test_preprocessed.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Preprocessing complete.")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print("Preprocessed data saved as CSV files.")