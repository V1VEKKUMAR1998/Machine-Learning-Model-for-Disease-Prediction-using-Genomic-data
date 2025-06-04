
import pandas as pd

pivot_df = df.pivot_table(
    index='SampleID',
    columns='Risk Allele',
    values='Genotype',
    aggfunc='first'  # or 'mode' or 'max' depending on your need
)
duplicates = df[df.duplicated(subset=['SampleID', 'Risk Allele'], keep=False)]
print(duplicates)

# Step 1: Load your Excel file
file_path = 'New Microsoft Excel Worksheet.xlsx'
df = pd.read_excel(file_path)

# Step 2: Check your columns and assign Sample IDs if missing
# If your data is per SNP per sample, you need a Sample ID column.
# If you don't have one, you can create one for demo like this:
if 'SampleID' not in df.columns:
    # This is a placeholder: adjust according to your data structure!
    # For example, if every 53 rows belong to one sample,
    # you can create SampleID like this:
    df['SampleID'] = (df.index // 53).astype(str)  # Adjust 53 if needed

# Step 3: Pivot the table
# We assume columns: 'SampleID', 'Risk Allele' (or SNP id), 'Genotype'
# Adjust the SNP identifier column name accordingly
pivot_df = df.pivot(index='SampleID', columns='Risk Allele', values='Genotype')

# Step 4: Encode genotypes AA=0, AG=1, GG=2, etc.
def encode_genotype(gt):
    if pd.isna(gt):
        return -1  # or you can use np.nan or 9 for missing
    gt = gt.upper()
    if gt == 'AA':
        return 0
    elif gt in ('AG', 'GA'):
        return 1
    elif gt == 'GG':
        return 2
    else:
        return -1  # unknown genotype

pivot_encoded = pivot_df.applymap(encode_genotype)

# Step 5: Add the target column from original df
# We need to get target (Result) per sample
# Assuming 'Results' column exists and is same for each SNP row per sample
results = df.groupby('SampleID')['Results'].first()

# Convert target to binary: High risk=1, Low risk=0
def encode_target(res):
    if isinstance(res, str) and 'high risk' in res.lower():
        return 1
    else:
        return 0

target = results.apply(encode_target)

# Step 6: Merge target with genotype data
final_df = pivot_encoded.copy()
final_df['disease_risk'] = target

# Step 7: Save to CSV for ML modeling
final_df.to_csv('genotype_data_for_ml.csv')

print("Data prepared and saved to genotype_data_for_ml.csv")
