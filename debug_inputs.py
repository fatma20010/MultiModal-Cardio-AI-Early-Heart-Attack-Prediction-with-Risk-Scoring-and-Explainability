"""
Debug script to check if ECG/PCG inputs are being processed correctly
"""

import torch
from preprocessing import DataPreprocessor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

preprocessor = DataPreprocessor(max_pcg_len=500)

# Test with a sample patient
ecg_path = "C:/Users/msi/Desktop/Ecgdata/P001.png"
pcg_path = "C:/Users/msi/Desktop/Finalaudio/P001.wav"

print("="*70)
print("🔍 DEBUGGING ECG/PCG INPUTS")
print("="*70)

# Check ECG
print("\n📊 ECG Image:")
print(f"   Path: {ecg_path}")
try:
    ecg_img = Image.open(ecg_path)
    print(f"   ✅ Loaded: {ecg_img.size} pixels, mode: {ecg_img.mode}")
    
    ecg_tensor = preprocessor.preprocess_ecg(ecg_path)
    print(f"   ✅ Tensor shape: {ecg_tensor.shape}")
    print(f"   ✅ Min value: {ecg_tensor.min().item():.4f}")
    print(f"   ✅ Max value: {ecg_tensor.max().item():.4f}")
    print(f"   ✅ Mean value: {ecg_tensor.mean().item():.4f}")
    print(f"   ✅ Std value: {ecg_tensor.std().item():.4f}")
    
    # Check if image is diverse (not all the same color)
    if ecg_tensor.std().item() < 0.01:
        print("   ⚠️  WARNING: ECG image has very low variance - might be blank!")
    else:
        print("   ✅ ECG image has good variance (diverse pixels)")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Check PCG
print("\n🎵 PCG Audio:")
print(f"   Path: {pcg_path}")
try:
    pcg_tensor = preprocessor.preprocess_pcg(pcg_path)
    print(f"   ✅ Tensor shape: {pcg_tensor.shape}")
    print(f"   ✅ Min value: {pcg_tensor.min().item():.4f}")
    print(f"   ✅ Max value: {pcg_tensor.max().item():.4f}")
    print(f"   ✅ Mean value: {pcg_tensor.mean().item():.4f}")
    print(f"   ✅ Std value: {pcg_tensor.std().item():.4f}")
    
    # Check if audio has actual signal
    if pcg_tensor.std().item() < 0.001:
        print("   ⚠️  WARNING: PCG audio has very low variance - might be silence!")
    else:
        print("   ✅ PCG audio has good variance (actual signal)")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test with model
print("\n🤖 Model Forward Pass:")
try:
    from model import MultiModalCardioAI
    
    model = MultiModalCardioAI(clinical_input_size=14)
    model.load_state_dict(torch.load("best_multimodal_model.pth", map_location='cpu'))
    model.eval()
    
    # Test clinical data (all zeros = average patient)
    clinical_tensor = torch.zeros(1, 14)
    
    with torch.no_grad():
        # Full prediction
        output_full = model(ecg_tensor, pcg_tensor, clinical_tensor)
        print(f"   ✅ Full prediction: {output_full.item():.4f}")
        
        # Only ECG
        pcg_zero = torch.zeros_like(pcg_tensor)
        clinical_zero = torch.zeros_like(clinical_tensor)
        output_ecg_only = model(ecg_tensor, pcg_zero, clinical_zero)
        print(f"   📊 ECG only: {output_ecg_only.item():.4f}")
        
        # Only PCG
        ecg_gray = torch.ones_like(ecg_tensor) * 0.5
        output_pcg_only = model(ecg_gray, pcg_tensor, clinical_zero)
        print(f"   🎵 PCG only: {output_pcg_only.item():.4f}")
        
        # Only Clinical
        output_clinical_only = model(ecg_gray, pcg_zero, clinical_tensor)
        print(f"   📋 Clinical only: {output_clinical_only.item():.4f}")
        
        # All baselines
        output_baseline = model(ecg_gray, pcg_zero, clinical_zero)
        print(f"   ⚪ All baseline: {output_baseline.item():.4f}")
        
    print("\n📈 Analysis:")
    if abs(output_ecg_only.item() - output_baseline.item()) < 0.01:
        print("   ⚠️  ECG seems to have NO effect - might not be learning from images!")
    else:
        print(f"   ✅ ECG changes prediction by {abs(output_ecg_only.item() - output_baseline.item()):.4f}")
    
    if abs(output_pcg_only.item() - output_baseline.item()) < 0.01:
        print("   ⚠️  PCG seems to have NO effect - might not be learning from audio!")
    else:
        print(f"   ✅ PCG changes prediction by {abs(output_pcg_only.item() - output_baseline.item()):.4f}")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✅ Debugging complete!")
print("="*70)



