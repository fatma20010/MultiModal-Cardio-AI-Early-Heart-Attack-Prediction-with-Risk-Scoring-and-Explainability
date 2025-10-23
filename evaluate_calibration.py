"""
Calibration Evaluation Script

This script evaluates the calibration of your trained model on the dataset.
It generates calibration curves, ROC curves, and other performance metrics.

Usage:
    python evaluate_calibration.py
"""

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import numpy as np
import torchaudio
import torch.nn.functional as F
from model import MultiModalCardioAI
from calibration import ModelCalibrator

class MultiModalCardioDataset(Dataset):
    """Same dataset class used during training"""
    def __init__(self, csv_path, max_pcg_len=500):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=['pcg_path', 'output'])
        
        clinical_cols = [c for c in self.df.columns if c not in
                         ['patient_id', 'ecg_path', 'pcg_path', 'label_ecg', 'label_pcg', 'output']]
        self.df[clinical_cols] = self.df[clinical_cols].apply(pd.to_numeric, errors='coerce')
        self.df[clinical_cols] = self.df[clinical_cols].fillna(self.df[clinical_cols].mean())
        
        self.clinical_cols = clinical_cols
        self.max_pcg_len = max_pcg_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # ECG Image
        ecg_img = Image.open(row['ecg_path']).convert('RGB')
        ecg_img = torch.tensor(np.array(ecg_img)).permute(2, 0, 1).float() / 255.0
        
        # PCG Audio
        waveform, sr = torchaudio.load(row['pcg_path'])
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if waveform.shape[1] < self.max_pcg_len:
            waveform = F.pad(waveform, (0, self.max_pcg_len - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.max_pcg_len]
        
        # Clinical data
        clinical_data = torch.tensor(row[self.clinical_cols].values, dtype=torch.float)
        
        # Target
        target = torch.tensor(float(row['output']), dtype=torch.float).unsqueeze(0)
        
        return ecg_img, waveform, clinical_data, target

def main():
    print("\n" + "="*70)
    print("🔬 CALIBRATION EVALUATION")
    print("="*70 + "\n")
    
    # Configuration
    MODEL_PATH = "best_multimodal_model.pth"
    DATA_PATH = "multimodal_metadata_clean.csv"
    CLINICAL_INPUT_SIZE = 14
    MAX_PCG_LEN = 500
    BATCH_SIZE = 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"📍 Device: {DEVICE}")
    print(f"📂 Model: {MODEL_PATH}")
    print(f"📂 Data: {DATA_PATH}\n")
    
    # Load model
    print("Loading model...")
    model = MultiModalCardioAI(clinical_input_size=CLINICAL_INPUT_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded\n")
    
    # Load dataset
    print("Loading dataset...")
    try:
        dataset = MultiModalCardioDataset(DATA_PATH, max_pcg_len=MAX_PCG_LEN)
        
        # Use a subset for faster evaluation (or full dataset for complete analysis)
        # For full dataset, remove the subset line
        subset_size = min(len(dataset), 500)  # Evaluate on up to 500 samples
        indices = np.random.choice(len(dataset), subset_size, replace=False)
        subset = torch.utils.data.Subset(dataset, indices)
        
        dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"✅ Dataset loaded ({len(subset)} samples)\n")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\nPlease ensure:")
        print("  1. multimodal_metadata_clean.csv exists")
        print("  2. ECG and PCG file paths in the CSV are correct")
        print("  3. All data files are accessible")
        return
    
    # Initialize calibrator
    calibrator = ModelCalibrator(model, device=DEVICE)
    
    # Evaluate
    print("Evaluating model calibration...")
    print("This may take a few minutes...\n")
    
    try:
        results = calibrator.evaluate_on_dataset(dataloader)
        
        print("="*70)
        print("📊 EVALUATION RESULTS")
        print("="*70)
        print(f"\n✨ Expected Calibration Error (ECE): {results['metrics']['ece']:.4f}")
        
        interpretation = ""
        if results['metrics']['ece'] < 0.05:
            interpretation = "Excellent - Model is very well calibrated"
        elif results['metrics']['ece'] < 0.10:
            interpretation = "Good - Model calibration is acceptable"
        elif results['metrics']['ece'] < 0.15:
            interpretation = "Fair - Consider recalibration"
        else:
            interpretation = "Poor - Recalibration strongly recommended"
        
        print(f"   Interpretation: {interpretation}")
        
        print(f"\n🎯 ROC AUC: {results['metrics']['auc_roc']:.4f}")
        
        print("\n📈 Accuracy at different thresholds:")
        for key, value in results['metrics'].items():
            if key.startswith('accuracy_'):
                threshold = key.split('@')[1]
                print(f"   Threshold {threshold}: {value:.4f}")
        
        # Generate comprehensive calibration report
        print("\n" + "="*70)
        print("Generating calibration report with visualizations...")
        print("="*70 + "\n")
        
        summary = calibrator.generate_calibration_report(
            results['y_true'], 
            results['y_pred'],
            save_dir='calibration_report'
        )
        
        print("\n✅ Calibration report saved to 'calibration_report/' directory")
        print("\nGenerated files:")
        print("  📊 calibration_curve.png - Shows how well predicted probabilities match true outcomes")
        print("  📊 roc_curve.png - ROC curve and AUC score")
        print("  📊 pr_curve.png - Precision-Recall curve")
        print("  📄 summary.json - Complete evaluation metrics")
        
        # Risk distribution
        print("\n" + "="*70)
        print("🎯 RISK DISTRIBUTION")
        print("="*70)
        for level, count in summary['risk_distribution'].items():
            percentage = (count / summary['total_samples']) * 100
            print(f"   {level:12s}: {count:4d} patients ({percentage:5.1f}%)")
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
