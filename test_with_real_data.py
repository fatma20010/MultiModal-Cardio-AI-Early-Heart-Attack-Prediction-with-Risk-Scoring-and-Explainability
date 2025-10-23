"""
Test model with REAL clinical values to see if it varies
"""

import torch
from preprocessing import DataPreprocessor
from clinical_normalizer import ClinicalNormalizer
from model import MultiModalCardioAI

# Initialize
preprocessor = DataPreprocessor(max_pcg_len=500)
normalizer = ClinicalNormalizer()
model = MultiModalCardioAI(clinical_input_size=14)
model.load_state_dict(torch.load("best_multimodal_model.pth", map_location='cpu'))
model.eval()

# Load ECG/PCG
ecg_path = "C:/Users/msi/Desktop/Ecgdata/P001.png"
pcg_path = "C:/Users/msi/Desktop/Finalaudio/P001.wav"
ecg_tensor = preprocessor.preprocess_ecg(ecg_path)
pcg_tensor = preprocessor.preprocess_pcg(pcg_path)

print("="*70)
print("🧪 TESTING WITH DIFFERENT CLINICAL DATA")
print("="*70)

# Test 1: Healthy young patient
print("\n👤 Test 1: Healthy 30-year-old (should be LOW risk)")
clinical_healthy = {
    'age': 30, 'sex': 1, 'cp': 0, 'trtbps': 110, 'chol': 180,
    'fbs': 0, 'restecg': 0, 'thalachh': 170, 'exng': 0,
    'oldpeak': 0.0, 'slp': 0, 'caa': 0, 'thall': 0, 'temp': 98.6
}
normalized = normalizer.normalize_clinical_data(clinical_healthy)
normalized['98.6'] = normalized['temp']
clinical_tensor = preprocessor.preprocess_clinical(normalized)

with torch.no_grad():
    pred = model(ecg_tensor, pcg_tensor, clinical_tensor).item()
    print(f"   Prediction: {pred*100:.1f}%")

# Test 2: High-risk elderly patient
print("\n👤 Test 2: High-risk 70-year-old (should be HIGH risk)")
clinical_high_risk = {
    'age': 70, 'sex': 1, 'cp': 3, 'trtbps': 170, 'chol': 320,
    'fbs': 1, 'restecg': 2, 'thalachh': 100, 'exng': 1,
    'oldpeak': 4.0, 'slp': 2, 'caa': 3, 'thall': 2, 'temp': 98.6
}
normalized = normalizer.normalize_clinical_data(clinical_high_risk)
normalized['98.6'] = normalized['temp']
clinical_tensor = preprocessor.preprocess_clinical(normalized)

with torch.no_grad():
    pred = model(ecg_tensor, pcg_tensor, clinical_tensor).item()
    print(f"   Prediction: {pred*100:.1f}%")

# Test 3: Change ECG/PCG only (keep clinical same)
print("\n🔬 Test 3: Does ECG/PCG matter? (Same clinical, different images)")

# Same high-risk clinical, try different ECG files
print("\n   Same clinical data, but different ECG:")
for patient_id in ["P001", "P010", "P050"]:
    try:
        ecg_path_test = f"C:/Users/msi/Desktop/Ecgdata/{patient_id}.png"
        ecg_tensor_test = preprocessor.preprocess_ecg(ecg_path_test)
        
        with torch.no_grad():
            pred = model(ecg_tensor_test, pcg_tensor, clinical_tensor).item()
            print(f"   {patient_id} ECG: {pred*100:.1f}%")
    except:
        pass

# Test 4: Gray baseline ECG vs Real ECG
print("\n🖼️  Test 4: Real ECG vs Baseline ECG (same clinical/PCG)")
ecg_baseline = torch.ones_like(ecg_tensor) * 0.5

# Load fresh P001 ECG
ecg_real = preprocessor.preprocess_ecg("C:/Users/msi/Desktop/Ecgdata/P001.png")

# Use high-risk clinical
normalized = normalizer.normalize_clinical_data(clinical_high_risk)
normalized['98.6'] = normalized['temp']
clinical_tensor = preprocessor.preprocess_clinical(normalized)

with torch.no_grad():
    pred_real_ecg = model(ecg_real, pcg_tensor, clinical_tensor).item()
    pred_baseline_ecg = model(ecg_baseline, pcg_tensor, clinical_tensor).item()
    
    print(f"   With REAL ECG:     {pred_real_ecg*100:.1f}%")
    print(f"   With BASELINE ECG: {pred_baseline_ecg*100:.1f}%")
    print(f"   Difference:        {abs(pred_real_ecg - pred_baseline_ecg)*100:.2f}%")
    
    if abs(pred_real_ecg - pred_baseline_ecg) < 0.001:
        print("   ⚠️  ECG makes NO difference!")
    else:
        print("   ✅ ECG does make a difference!")

print("\n" + "="*70)
print("✅ Analysis complete!")
print("="*70)



