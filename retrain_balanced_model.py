"""
Retrain Multimodal Model with Proper Balance
- Handles class imbalance
- Forces each modality to learn
- Monitors modality contributions during training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from PIL import Image
import torchaudio
import torch.nn.functional as F
from model import MultiModalCardioAI
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy

class BalancedMultiModalDataset(Dataset):
    def __init__(self, df, max_pcg_len=500):
        self.df = df.reset_index(drop=True)
        self.max_pcg_len = max_pcg_len
        
        # Get clinical columns
        self.clinical_cols = [c for c in df.columns if c not in
                             ['patient_id', 'ecg_path', 'pcg_path', 'label_ecg', 'label_pcg', 'output']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        try:
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
            
            # Clinical data - handle mixed types robustly
            clinical_values = []
            for col in self.clinical_cols:
                val = row[col]
                # Convert to float, handle any type
                try:
                    clinical_values.append(float(val))
                except (ValueError, TypeError):
                    # If conversion fails, use 0.0
                    clinical_values.append(0.0)
            
            clinical_data = torch.tensor(clinical_values, dtype=torch.float32)
            
            # Target (binarize it)
            target_val = float(row['output']) if not pd.isna(row['output']) else 0.0
            target = torch.tensor(1.0 if target_val >= 0.5 else 0.0, dtype=torch.float32).unsqueeze(0)
            
            return ecg_img, waveform, clinical_data, target
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a dummy sample
            return (torch.zeros(3, 224, 224), torch.zeros(1, self.max_pcg_len), 
                   torch.zeros(len(self.clinical_cols), dtype=torch.float32), 
                   torch.tensor([0.0], dtype=torch.float32))

def compute_class_weights(df):
    """Compute weights for balanced sampling"""
    # Binarize labels
    labels = (df['output'] >= 0.5).astype(int)
    
    class_counts = labels.value_counts()
    print(f"\nClass distribution:")
    print(f"  Class 0 (healthy): {class_counts.get(0, 0)} samples")
    print(f"  Class 1 (abnormal): {class_counts.get(1, 0)} samples")
    
    # Compute weights
    total = len(labels)
    weight_0 = total / (2 * class_counts.get(0, 1))
    weight_1 = total / (2 * class_counts.get(1, 1))
    
    sample_weights = labels.map({0: weight_0, 1: weight_1})
    
    print(f"\nSample weights:")
    print(f"  Class 0 weight: {weight_0:.4f}")
    print(f"  Class 1 weight: {weight_1:.4f}")
    
    return torch.DoubleTensor(sample_weights.values)

def validate_modality_contributions(model, val_loader, device):
    """Check if each modality is actually contributing"""
    model.eval()
    
    ecg_impacts = []
    pcg_impacts = []
    clinical_impacts = []
    
    with torch.no_grad():
        for i, (ecg, pcg, clinical, target) in enumerate(val_loader):
            if i >= 10:  # Test on 10 batches
                break
                
            ecg, pcg, clinical = ecg.to(device), pcg.to(device), clinical.to(device)
            
            # Full prediction
            pred_full = model(ecg, pcg, clinical).mean().item()
            
            # Ablations
            ecg_baseline = torch.ones_like(ecg) * 0.5
            pcg_baseline = torch.zeros_like(pcg)
            clinical_baseline = torch.zeros_like(clinical)
            
            pred_no_ecg = model(ecg_baseline, pcg, clinical).mean().item()
            pred_no_pcg = model(ecg, pcg_baseline, clinical).mean().item()
            pred_no_clinical = model(ecg, pcg, clinical_baseline).mean().item()
            
            ecg_impacts.append(abs(pred_full - pred_no_ecg))
            pcg_impacts.append(abs(pred_full - pred_no_pcg))
            clinical_impacts.append(abs(pred_full - pred_no_clinical))
    
    avg_ecg = np.mean(ecg_impacts) * 100
    avg_pcg = np.mean(pcg_impacts) * 100
    avg_clinical = np.mean(clinical_impacts) * 100
    
    total = avg_ecg + avg_pcg + avg_clinical
    if total > 0:
        ecg_pct = avg_ecg / total * 100
        pcg_pct = avg_pcg / total * 100
        clinical_pct = avg_clinical / total * 100
    else:
        ecg_pct = pcg_pct = clinical_pct = 33.33
    
    return ecg_pct, pcg_pct, clinical_pct

def train_with_modality_forcing():
    """
    Train model with:
    1. Class balancing
    2. Modality-specific losses
    3. Progress monitoring
    """
    
    print("="*70)
    print("🚀 RETRAINING MULTIMODAL MODEL")
    print("="*70)
    
    # Configuration
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    MAX_PCG_LEN = 500
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n📍 Device: {DEVICE}")
    print(f"📊 Batch size: {BATCH_SIZE}")
    print(f"🔄 Epochs: {NUM_EPOCHS}")
    
    # Load data
    print("\n📂 Loading dataset...")
    df = pd.read_csv('multimodal_metadata_clean.csv')
    df = df.dropna(subset=['pcg_path', 'output'])
    
    # Get clinical columns
    clinical_cols = [c for c in df.columns if c not in
                     ['patient_id', 'ecg_path', 'pcg_path', 'label_ecg', 'label_pcg', 'output']]
    
    # Convert all clinical columns to numeric, forcing errors to NaN
    for col in clinical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill NaN with column mean
    df[clinical_cols] = df[clinical_cols].fillna(df[clinical_cols].mean())
    
    # Drop any rows that still have NaN after filling
    df = df.dropna(subset=clinical_cols)
    
    print(f"✅ Loaded {len(df)} samples")
    print(f"✅ Clinical features: {len(clinical_cols)}")
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, 
                                         stratify=(df['output'] >= 0.5).astype(int))
    
    print(f"\n📊 Split:")
    print(f"   Training: {len(train_df)} samples")
    print(f"   Validation: {len(val_df)} samples")
    
    # Create datasets
    train_dataset = BalancedMultiModalDataset(train_df, max_pcg_len=MAX_PCG_LEN)
    val_dataset = BalancedMultiModalDataset(val_df, max_pcg_len=MAX_PCG_LEN)
    
    # Compute class weights for balanced sampling
    sample_weights = compute_class_weights(train_df)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = MultiModalCardioAI(clinical_input_size=len(clinical_cols))
    model.to(DEVICE)
    
    # Optimizer with different learning rates for different branches
    optimizer = torch.optim.Adam([
        {'params': model.ecg_encoder.parameters(), 'lr': LEARNING_RATE},
        {'params': model.ecg_fc.parameters(), 'lr': LEARNING_RATE},
        {'params': model.pcg_encoder.parameters(), 'lr': LEARNING_RATE * 2},  # Higher LR for PCG
        {'params': model.clinical_fc.parameters(), 'lr': LEARNING_RATE * 0.5},  # Lower LR for clinical
        {'params': model.fc_fusion.parameters(), 'lr': LEARNING_RATE}
    ])
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'ecg_contrib': [],
        'pcg_contrib': [],
        'clinical_contrib': []
    }
    
    best_val_loss = float('inf')
    best_model_wts = None
    patience = 5
    patience_counter = 0
    
    print("\n" + "="*70)
    print("🎯 Starting Training")
    print("="*70)
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (ecg, pcg, clinical, target) in enumerate(train_loader):
            ecg, pcg, clinical, target = ecg.to(DEVICE), pcg.to(DEVICE), clinical.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Main prediction
            outputs = model(ecg, pcg, clinical)
            main_loss = criterion(outputs, target)
            
            # Auxiliary losses to force each modality to learn
            # Create baseline inputs
            ecg_baseline = torch.ones_like(ecg) * 0.5
            pcg_baseline = torch.zeros_like(pcg)
            clinical_baseline = torch.zeros_like(clinical)
            
            # Each modality should be able to predict alone
            output_ecg_only = model(ecg, pcg_baseline, clinical_baseline)
            output_pcg_only = model(ecg_baseline, pcg, clinical_baseline)
            output_clinical_only = model(ecg_baseline, pcg_baseline, clinical)
            
            aux_loss_ecg = criterion(output_ecg_only, target)
            aux_loss_pcg = criterion(output_pcg_only, target)
            aux_loss_clinical = criterion(output_clinical_only, target)
            
            # Combined loss with modality forcing
            total_loss = main_loss + 0.2 * (aux_loss_ecg + aux_loss_pcg + aux_loss_clinical)
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
            # Accuracy
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == target).sum().item()
            train_total += target.size(0)
            
            if (batch_idx + 1) % 20 == 0:
                print(f"   Batch [{batch_idx+1}/{len(train_loader)}] Loss: {total_loss.item():.4f}")
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for ecg, pcg, clinical, target in val_loader:
                ecg, pcg, clinical, target = ecg.to(DEVICE), pcg.to(DEVICE), clinical.to(DEVICE), target.to(DEVICE)
                
                outputs = model(ecg, pcg, clinical)
                loss = criterion(outputs, target)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == target).sum().item()
                val_total += target.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Check modality contributions
        ecg_pct, pcg_pct, clinical_pct = validate_modality_contributions(model, val_loader, DEVICE)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['ecg_contrib'].append(ecg_pct)
        history['pcg_contrib'].append(pcg_pct)
        history['clinical_contrib'].append(clinical_pct)
        
        # Print epoch results
        print(f"\n📊 Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"   Modality Contributions: ECG={ecg_pct:.1f}%, PCG={pcg_pct:.1f}%, Clinical={clinical_pct:.1f}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"   ✅ New best model! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    # Save model
    torch.save(model.state_dict(), 'best_multimodal_model_retrained.pth')
    print(f"\n✅ Model saved as 'best_multimodal_model_retrained.pth'")
    
    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training & Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Modality contributions
    axes[1, 0].plot(history['ecg_contrib'], label='ECG')
    axes[1, 0].plot(history['pcg_contrib'], label='PCG')
    axes[1, 0].plot(history['clinical_contrib'], label='Clinical')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Contribution %')
    axes[1, 0].set_title('Modality Contributions Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].axhline(y=33.33, color='gray', linestyle='--', alpha=0.5)
    
    # Final contributions
    final_contribs = [history['ecg_contrib'][-1], history['pcg_contrib'][-1], history['clinical_contrib'][-1]]
    axes[1, 1].bar(['ECG', 'PCG', 'Clinical'], final_contribs, color=['#3b82f6', '#10b981', '#f59e0b'])
    axes[1, 1].set_ylabel('Contribution %')
    axes[1, 1].set_title('Final Modality Contributions')
    axes[1, 1].axhline(y=33.33, color='gray', linestyle='--', alpha=0.5, label='Balanced (33.33%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print(f"✅ Training plots saved as 'training_history.png'")
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Final Results:")
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"   Best Val Acc:  {history['val_acc'][np.argmin(history['val_loss'])]:.4f}")
    print(f"\n   Final Modality Contributions:")
    print(f"   - ECG:      {history['ecg_contrib'][-1]:.1f}%")
    print(f"   - PCG:      {history['pcg_contrib'][-1]:.1f}%")
    print(f"   - Clinical: {history['clinical_contrib'][-1]:.1f}%")
    
    return model, history

if __name__ == "__main__":
    model, history = train_with_modality_forcing()
