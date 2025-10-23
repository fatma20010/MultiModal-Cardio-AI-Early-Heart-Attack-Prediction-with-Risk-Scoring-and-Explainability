# 🚀 Model Retraining Guide

## What This Will Do

The retraining script will:
- ✅ **Balance your dataset** using weighted sampling
- ✅ **Force ECG/PCG branches to learn** using auxiliary losses
- ✅ **Monitor modality contributions** during training
- ✅ **Save the best model** with early stopping
- ✅ **Generate training plots** showing progress

---

## 🎯 Expected Results

### Before (Current Model):
```
ECG contribution:      0.0%  ❌
PCG contribution:      0.2%  ❌
Clinical contribution: 99.8% ❌
```

### After Retraining (Goal):
```
ECG contribution:      25-40%  ✅
PCG contribution:      25-40%  ✅
Clinical contribution: 20-35%  ✅
```

---

## 📋 Prerequisites

Make sure you have:
1. ✅ `multimodal_metadata_clean.csv` in current directory
2. ✅ All ECG images accessible at paths in CSV
3. ✅ All PCG audio files accessible at paths in CSV
4. ✅ Python packages installed (torch, pandas, matplotlib, sklearn)

---

## 🚀 How to Run

### Step 1: Start Training

```bash
python retrain_balanced_model.py
```

### Step 2: Watch Progress

You'll see output like:
```
🚀 RETRAINING MULTIMODAL MODEL
======================================
📍 Device: cuda (or cpu)
📊 Batch size: 16
🔄 Epochs: 30

Class distribution:
  Class 0 (healthy): 337 samples
  Class 1 (abnormal): 3447 samples

🎯 Starting Training
======================================

📊 Epoch 1/30
   Train Loss: 0.6234 | Train Acc: 0.6543
   Val Loss:   0.6012 | Val Acc:   0.6789
   Modality Contributions: ECG=15.2%, PCG=18.3%, Clinical=66.5%
   ✅ New best model! (Val Loss: 0.6012)

📊 Epoch 2/30
   ...
```

### Step 3: Wait for Completion

Training will take approximately:
- **CPU**: 2-4 hours ⏰
- **GPU**: 30-60 minutes ⚡

The script will:
- Save progress every epoch
- Stop early if not improving (patience=5)
- Show real-time modality contributions

---

## 📊 What Gets Generated

After training completes:

1. **`best_multimodal_model_retrained.pth`**
   - Your new trained model
   - Replace the old `best_multimodal_model.pth` with this

2. **`training_history.png`**
   - 4 plots showing:
     - Training/Validation Loss
     - Training/Validation Accuracy
     - Modality Contributions over time
     - Final contribution breakdown

---

## 🔧 Key Features of the Training

### 1. **Class Balancing**
Uses weighted sampling so model sees equal amounts of healthy/abnormal patients:
```python
sample_weights = compute_class_weights(train_df)
sampler = WeightedRandomSampler(sample_weights, ...)
```

### 2. **Modality Forcing**
Adds auxiliary losses that force each branch to learn:
```python
# Main loss (all modalities)
main_loss = criterion(outputs, target)

# Auxiliary losses (each modality alone)
aux_loss_ecg = criterion(output_ecg_only, target)
aux_loss_pcg = criterion(output_pcg_only, target)
aux_loss_clinical = criterion(output_clinical_only, target)

# Combined
total_loss = main_loss + 0.2 * (aux_loss_ecg + aux_loss_pcg + aux_loss_clinical)
```

### 3. **Different Learning Rates**
- ECG: 1x learning rate (standard)
- PCG: 2x learning rate (needs to catch up)
- Clinical: 0.5x learning rate (already learning too much)

### 4. **Real-Time Monitoring**
Every epoch, tests if ECG/PCG actually change predictions:
```python
validate_modality_contributions(model, val_loader, device)
```

---

## ⚙️ Configuration Options

You can edit these in `retrain_balanced_model.py`:

```python
BATCH_SIZE = 16        # Increase if you have lots of RAM/VRAM
NUM_EPOCHS = 30        # Increase for more training
LEARNING_RATE = 1e-4   # Decrease if training is unstable
```

---

## 🎯 Success Criteria

Training is successful if:

1. **Validation loss decreases** (going down over epochs)
2. **Accuracy improves** (>60% is good, >70% is great)
3. **Modalities are balanced**:
   - Each between 20-40%
   - None above 60%
   - None below 10%

4. **Different inputs give different outputs**:
   - Test manually after training

---

## 🐛 Troubleshooting

### Problem: "Out of memory" error
**Solution**: Reduce `BATCH_SIZE` to 8 or 4

### Problem: Training is very slow
**Solution**: 
- Reduce `NUM_EPOCHS` to 15-20
- Or let it run overnight

### Problem: Modalities still imbalanced after training
**Solution**: Increase auxiliary loss weight from 0.2 to 0.5:
```python
total_loss = main_loss + 0.5 * (aux_loss_ecg + aux_loss_pcg + aux_loss_clinical)
```

### Problem: Accuracy stays around 50%
**Solution**: This might mean your data quality needs improvement, but the modalities should still balance out

---

## 📈 What to Expect

### Early Epochs (1-5):
- Loss: ~0.6-0.7
- Accuracy: ~50-65%
- ECG: 10-20%, PCG: 10-20%, Clinical: 60-80%

### Middle Epochs (6-15):
- Loss: ~0.4-0.6
- Accuracy: ~60-75%
- ECG: 20-30%, PCG: 20-30%, Clinical: 40-60%

### Late Epochs (16-30):
- Loss: ~0.3-0.5
- Accuracy: ~65-80%
- ECG: 25-35%, PCG: 25-35%, Clinical: 30-45%

---

## 🔄 After Training

### Step 1: Replace Old Model
```bash
# Backup old model
mv best_multimodal_model.pth best_multimodal_model_OLD.pth

# Use new model
mv best_multimodal_model_retrained.pth best_multimodal_model.pth
```

### Step 2: Restart Backend
```bash
python app.py
```

### Step 3: Test API
Upload real patient data and check:
- ✅ Predictions vary (not always 99%)
- ✅ Different ECG images give different results
- ✅ Modality contributions are balanced

---

## 💡 Tips for Best Results

1. **Let it finish**: Don't interrupt training mid-epoch
2. **Monitor the plots**: If one modality stays at 0%, increase its learning rate
3. **Use GPU if available**: Much faster training
4. **Start overnight**: Training takes time, let it run while you sleep
5. **Save the plots**: Show them to your professor as evidence of proper training

---

## 🎓 For Your Professor

Include in your report:
- ✅ Training history plots
- ✅ Final modality contributions
- ✅ Explanation of balancing strategy
- ✅ Before/after comparison

This shows:
- Understanding of class imbalance
- Knowledge of multi-task learning
- Systematic problem-solving
- Professional ML engineering

---

## 🚀 Ready to Start?

```bash
python retrain_balanced_model.py
```

**Expected time:**
- CPU: 2-4 hours
- GPU: 30-60 minutes

Good luck! 🍀
