# 🫀 Multimodal Cardiac Abnormality Detection API v2.0

**Enhanced with Calibration & Explainability Features**

A Flask-based REST API for predicting cardiac abnormalities using a deep learning model that combines three types of medical data: ECG images, PCG (heart sound) audio, and clinical features.

## 🆕 What's New in v2.0

### ✨ Key Enhancements:
1. **5-Level Risk Stratification** - More granular risk assessment (was 3 levels)
2. **Explainability Analysis** - Understand which modality drove the prediction
3. **Clinical Feature Importance** - See which patient factors matter most
4. **Calibration Tools** - Evaluate model reliability with ECE and calibration curves
5. **Confidence Assessment** - Know when the model is uncertain

---

## 📋 Model Overview

**Output**: A probability between 0 and 1 representing the likelihood of cardiac abnormality

| Patient | Model Output | Risk Category | Interpretation |
|---------|--------------|---------------|----------------|
| #1 | 0.05 | Very Low Risk | Routine follow-up |
| #2 | 0.32 | Low Risk | Annual cardiovascular check |
| #3 | 0.55 | Moderate Risk | Further diagnostic testing |
| #4 | 0.72 | High Risk | Urgent medical evaluation |
| #5 | 0.95 | Very High Risk | Emergency referral |

## 🔬 Input Modalities

| Modality | Input Data | What It Represents |
|----------|------------|-------------------|
| 🫀 ECG (image) | 2D ECG graph images | Heart electrical activity pattern |
| 🔉 PCG (audio) | Heart sound waveform (.wav) | Heartbeat rhythm and murmurs |
| 📊 Clinical data | 14 numerical features | Patient-level medical indicators |

---

## 📦 Installation

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Verify model file exists:
Make sure `best_multimodal_model.pth` is in the same directory

---

## 🚀 Running the API

Start the Flask server:

```bash
python app.py
```

The API will be available at `http://localhost:5000`

---

## 🔌 API Endpoints

### 1. Home
```bash
GET /
```

Shows API information and available endpoints.

### 2. Health Check
```bash
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### 3. Risk Levels
```bash
GET /risk-levels
```

Get all 5 risk level definitions with thresholds and recommendations.

### 4. Basic Prediction
```bash
POST /predict
```

**Required Files**:
- `ecg_image`: ECG image file (PNG, JPG)
- `pcg_audio`: PCG audio file (WAV)

**Required Form Data** (14 clinical features):
- `age`, `sex`, `cp`, `trtbps`, `chol`, `fbs`, `restecg`, `thalachh`, 
- `exng`, `oldpeak`, `slp`, `caa`, `thall`, `temp`

**Response**:
```json
{
  "success": true,
  "probability": 0.8542,
  "percentage": "85.42%",
  "risk_assessment": {
    "level": 4,
    "category": "High Risk",
    "interpretation": "High probability of cardiac abnormality detected",
    "recommendation": "Urgent medical evaluation required",
    "action": "Schedule immediate cardiology consultation..."
  }
}
```

### 5. Detailed Prediction (NEW! ⭐)
```bash
POST /predict/detailed
```

Same input as basic prediction, but returns **full explainability analysis**:

**Response**:
```json
{
  "success": true,
  "probability": 0.8542,
  "percentage": "85.42%",
  "risk_assessment": {
    "level": 4,
    "category": "High Risk",
    "color": "#FF6D00",
    "interpretation": "High probability of cardiac abnormality detected",
    "recommendation": "Urgent medical evaluation required",
    "action": "Schedule immediate cardiology consultation..."
  },
  "explainability": {
    "modality_contributions": {
      "ecg": {
        "percentage": 45.2,
        "impact": 0.234,
        "interpretation": "ECG data moderately influenced this prediction"
      },
      "pcg": {
        "percentage": 35.8,
        "impact": 0.185,
        "interpretation": "PCG data moderately influenced this prediction"
      },
      "clinical": {
        "percentage": 19.0,
        "impact": 0.098,
        "interpretation": "Clinical data had minor influence on this prediction"
      }
    },
    "primary_driver": "ECG",
    "explanation": "The prediction was primarily driven by ECG data (45.2% contribution)...",
    "top_clinical_features": [
      {
        "feature": "oldpeak",
        "importance": 0.045,
        "interpretation": "ST depression had significant impact on prediction"
      },
      {
        "feature": "cp",
        "importance": 0.032,
        "interpretation": "Chest pain type had significant impact on prediction"
      }
      // ... top 5 features
    ],
    "confidence_assessment": {
      "level": "high",
      "near_decision_boundary": false,
      "modalities_balanced": false,
      "message": "Prediction confidence is high based on strong signal across modalities."
    },
    "summary": "This patient shows high risk of cardiac abnormality (probability: 85.4%). The prediction is primarily based on ECG data. Among clinical features, oldpeak is the most influential."
  }
}
```

---

## 🧪 Testing the API

### Basic Test:
```bash
python test_api.py
```

This will test:
- ✅ Home endpoint
- ✅ Health check
- ✅ Risk levels
- ✅ Basic prediction
- ✅ Detailed prediction with explainability

### Calibration Evaluation:
```bash
python evaluate_calibration.py
```

This generates:
- 📊 Calibration curve (how well predicted probabilities match reality)
- 📊 ROC curve (discrimination ability)
- 📊 Precision-Recall curve
- 📄 Summary report with Expected Calibration Error (ECE)

**Output**: All files saved to `calibration_report/` directory

---

## 📊 Understanding the Results

### 1. Risk Levels (5-tier system)

| Level | Range | Category | Action |
|-------|-------|----------|--------|
| 1 | 0.0-0.2 | Very Low | Routine follow-up |
| 2 | 0.2-0.4 | Low | Annual cardiovascular check |
| 3 | 0.4-0.6 | Moderate | Further diagnostic testing (stress test, echo) |
| 4 | 0.6-0.8 | High | Urgent cardiology consultation |
| 5 | 0.8-1.0 | Very High | Emergency referral if symptomatic |

### 2. Modality Contributions

Shows which type of data influenced the prediction most:
- **ECG**: Heart electrical patterns
- **PCG**: Heart sound analysis
- **Clinical**: Patient demographics and lab results

**Example**:
- ECG: 45% → Primary driver
- PCG: 35% → Secondary contributor
- Clinical: 20% → Supporting evidence

### 3. Clinical Feature Importance

Identifies which specific clinical factors matter most for this patient:
- **oldpeak** (ST depression) → Indicates ischemia
- **cp** (chest pain type) → Symptom severity
- **thalachh** (max heart rate) → Exercise capacity
- etc.

### 4. Confidence Assessment

Indicates prediction reliability:
- **High**: Strong signals, far from decision boundary
- **Good**: Clear signals, reasonable confidence
- **Moderate**: Near boundary or conflicting signals → Consider additional tests

### 5. Expected Calibration Error (ECE)

Measures how well predicted probabilities match real outcomes:
- **< 0.05**: Excellent calibration
- **0.05-0.10**: Good calibration
- **0.10-0.15**: Fair calibration
- **> 0.15**: Poor calibration (consider recalibration)

---

## 💻 Example Usage

### Python:
```python
import requests

# Prepare data
files = {
    'ecg_image': open('patient_ecg.png', 'rb'),
    'pcg_audio': open('patient_pcg.wav', 'rb')
}

clinical_data = {
    'age': 3.048, 'sex': 0.166, 'cp': 6.683,
    'trtbps': 2.908, 'chol': -0.497, 'fbs': 8.869,
    'restecg': -4.769, 'thalachh': -0.405,
    'exng': -0.169, 'oldpeak': 4.308,
    'slp': -4.910, 'caa': -0.171,
    'thall': -5.186, 'temp': 0.497
}

# Get detailed prediction with explainability
response = requests.post(
    'http://localhost:5000/predict/detailed', 
    files=files, 
    data=clinical_data
)

result = response.json()
print(f"Probability: {result['probability']}")
print(f"Risk Level: {result['risk_assessment']['category']}")
print(f"Primary Driver: {result['explainability']['primary_driver']}")
print(f"Summary: {result['explainability']['summary']}")
```

### cURL:
```bash
curl -X POST http://localhost:5000/predict/detailed \
  -F "ecg_image=@patient_ecg.png" \
  -F "pcg_audio=@patient_pcg.wav" \
  -F "age=3.048" \
  -F "sex=0.166" \
  -F "cp=6.683" \
  # ... other clinical features
```

---

## 📁 Project Structure

```
Data_Wrangling/
├── app.py                          # Flask API (enhanced v2.0)
├── model.py                        # Model architecture
├── preprocessing.py                # Data preprocessing utilities
├── calibration.py                  # NEW: Calibration tools
├── explainability.py               # NEW: Explainability module
├── best_multimodal_model.pth      # Trained model weights
├── multimodal_metadata_clean.csv  # Training dataset
├── requirements.txt               # Python dependencies
├── test_api.py                    # API testing script (updated)
├── evaluate_calibration.py        # NEW: Calibration evaluation
├── README.md                      # This file
└── calibration_report/            # Generated calibration analysis
    ├── calibration_curve.png
    ├── roc_curve.png
    ├── pr_curve.png
    └── summary.json
```

---

## 🎯 Clinical Interpretation Guide

### When to Trust the Prediction:
✅ **High confidence** assessment
✅ **Multiple modalities** contributing (balanced)
✅ **Clear risk category** (far from boundaries)
✅ **Top clinical features** align with symptoms

### When to Exercise Caution:
⚠️ **Moderate confidence** assessment
⚠️ **Prediction near 0.4-0.6** range (decision boundary)
⚠️ **Conflicting modality signals**
⚠️ **ECE > 0.10** in calibration report

### Always Remember:
⚠️ This is a **decision support tool**, not a replacement for clinical judgment
⚠️ Use in conjunction with standard diagnostic protocols
⚠️ Consult with qualified cardiologists for final diagnosis
⚠️ Consider patient history and additional clinical context

---

## 🛠️ Advanced Features

### 1. Batch Prediction
Process multiple patients by calling the API in a loop or use the dataset evaluation script.

### 2. Custom Thresholds
Modify risk thresholds in `calibration.py` → `get_risk_stratification()` based on your clinical needs.

### 3. Model Recalibration
If ECE is high, consider implementing:
- Platt scaling
- Temperature scaling
- Isotonic regression

### 4. Feature Analysis
Analyze clinical feature importance patterns across your patient population using the explainability module.

---

## 📈 Performance Metrics

Evaluate your model with `evaluate_calibration.py`:

```bash
python evaluate_calibration.py
```

**Metrics Provided**:
- Expected Calibration Error (ECE)
- ROC AUC
- Precision-Recall AUC
- Accuracy at multiple thresholds (0.3, 0.5, 0.7)
- Risk distribution across patient population

---

## 🔒 Security & Production Notes

This is a **development/testing API**. For production:
- ✅ Add authentication (API keys, OAuth)
- ✅ Implement rate limiting
- ✅ Add input validation and sanitization
- ✅ Use HTTPS/TLS encryption
- ✅ Add comprehensive logging
- ✅ Set up monitoring and alerts
- ✅ Deploy with production WSGI server (Gunicorn, uWSGI)
- ✅ Implement HIPAA compliance measures for healthcare data

---

## 📚 Citations & Acknowledgments

If you use this system in research, please cite:
- ResNet architecture for ECG analysis
- Your dataset sources
- Calibration metrics (Guo et al., 2017)

---

## 🐛 Troubleshooting

**Issue**: Model not loading
- **Solution**: Verify `best_multimodal_model.pth` exists

**Issue**: CUDA out of memory
- **Solution**: API automatically uses CPU if CUDA unavailable

**Issue**: Audio loading errors
- **Solution**: Ensure `soundfile` is installed and WAV files are valid

**Issue**: High ECE score
- **Solution**: Consider model recalibration or collecting more training data

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the calibration report
3. Test with sample data first
4. Check that all dependencies are installed

---

## 🎉 Version History

**v2.0** (Current)
- Added 5-level risk stratification
- Added explainability features (modality contributions)
- Added clinical feature importance
- Added calibration evaluation tools
- Enhanced API with detailed prediction endpoint

**v1.0**
- Basic prediction API
- 3-level risk assessment
- ECG, PCG, and clinical data fusion

---

**Made with ❤️ for better cardiac care**
