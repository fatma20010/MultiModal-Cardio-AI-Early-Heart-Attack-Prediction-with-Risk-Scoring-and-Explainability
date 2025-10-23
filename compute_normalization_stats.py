"""
Compute normalization statistics from training data
Run this once to get mean/std for each clinical feature
"""

import pandas as pd
import json

# Load your training data
df = pd.read_csv('multimodal_metadata_clean.csv')

# Get clinical columns (exclude non-clinical columns)
exclude_cols = ['patient_id', 'pcg_path', 'label_pcg', 'ecg_path', 'label_ecg', 'output']
clinical_cols = [c for c in df.columns if c not in exclude_cols]

print("Clinical columns found:", clinical_cols)
print("\nNormalization Statistics:\n")

# Compute mean and std for each feature
stats = {}
for col in clinical_cols:
    mean_val = df[col].mean()
    std_val = df[col].std()
    min_val = df[col].min()
    max_val = df[col].max()
    
    stats[col] = {
        'mean': float(mean_val),
        'std': float(std_val),
        'min': float(min_val),
        'max': float(max_val)
    }
    
    print(f"{col:15s}: mean={mean_val:10.4f}, std={std_val:10.4f}, min={min_val:10.4f}, max={max_val:10.4f}")

# Save to JSON
with open('normalization_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("\n✅ Statistics saved to 'normalization_stats.json'")
print("\n⚠️  NOTE: Your data is ALREADY NORMALIZED (z-scores)!")
print("   This means during training, you normalized with some original raw data")
print("   that we don't have access to anymore.")
print("\n   Solutions:")
print("   1. Use the normalized values directly from the CSV for testing")
print("   2. Find the original raw data and its statistics")
print("   3. Collect new raw patient data and normalize it with known stats")



