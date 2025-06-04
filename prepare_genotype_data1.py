import pandas as pd

# ---- Step 1: Load Excel File ----
file_path = 'New Microsoft Excel Worksheet.xlsx'  # Replace with your file name
df = pd.read_excel(file_path)

# ---- Step 2: Add SampleID if missing ----
if 'SampleID' not in df.columns:
    # Assumes every N rows belong to one sample (adjust N as needed)
    N = 53  # Set this to number of SNPs per sample
    df['SampleID'] = (df.index // N).astype(str)

# ---- Step 3: Clean column names ----
df.columns = [col.strip() for col in df.columns]  # Strip whitespace
required_cols = ['SampleID', 'Risk Allele', 'Genotype', 'Results']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ---- Step 4: Pivot data to wide format using pivot_table ----
pivot_df = df.pivot_table(
    index='SampleID',
    columns='Risk Allele',
    values='Genotype',
    aggfunc='first'  # Handles duplicates safely
)

# ---- Step 5: Encode genotypes (AA=0, AG=1, GG=2) ----
def encode_genotype(gt):
    if pd.isna(gt):
        return -1  # Missing
    gt = gt.upper()
    if gt == 'AA':
        return 0
    elif gt in ['AG', 'GA']:
        return 1
    elif gt == 'GG':
        return 2
    else:
        return -1

pivot_encoded = pivot_df.applymap(encode_genotype)

# ---- Step 6: Get and encode Results column as target ----
# Assumes each sample has consistent result for all its rows
results_series = df.groupby('SampleID')['Results'].first()

def encode_result(res):
    if isinstance(res, str) and 'high risk' in res.lower():
        return 1
    else:
        return 0

target_series = results_series.apply(encode_result)

# ---- Step 7: Merge encoded genotypes and target ----
final_df = pivot_encoded.copy()
final_df['disease_risk'] = target_series

# ---- Step 8: Save to CSV ----
output_file = 'genotype_data_for_ml.csv'
final_df.to_csv(output_file)
print(f"✅ Data prepared and saved to: {output_file}")
