import pandas as pd

df = pd.read_csv('multimodal_metadata_clean.csv')

print("="*60)
print("📊 DATASET LABEL DISTRIBUTION")
print("="*60)
print("\nLabel counts:")
print(df['output'].value_counts())

print(f"\nTotal samples: {len(df)}")
print(f"Percentage of label 1.0: {(df['output'] == 1.0).mean() * 100:.2f}%")
print(f"Percentage of label 0.0: {(df['output'] == 0.0).mean() * 100:.2f}%")

if (df['output'] == 1.0).mean() > 0.99:
    print("\n⚠️  CRITICAL: Dataset is 99%+ positive class!")
    print("   This is why model outputs 0.9939 for everything.")
    print("   The model learned to ALWAYS predict 'abnormal'")
    print("\n💡 Solution: You need to balance your dataset or use class weights!")
elif (df['output'] == 1.0).mean() > 0.90:
    print("\n⚠️  WARNING: Dataset is highly imbalanced (90%+ positive)")
    print("   Model is biased toward predicting 'abnormal'")
else:
    print("\n✅ Dataset seems reasonably balanced")

print("="*60)



