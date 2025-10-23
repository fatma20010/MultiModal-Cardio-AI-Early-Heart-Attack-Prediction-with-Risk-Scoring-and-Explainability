# 🫀 Multimodal Cardiac Abnormality Detection API

A Flask-based REST API for predicting cardiac abnormalities using a deep learning model that combines three types of medical data: ECG images, PCG (heart sound) audio, and clinical features.

## 📋 Model Overview

**Output**: A probability between 0 and 1 representing the likelihood of cardiac abnormality

| Patient | Model Output | Interpretation |
|---------|--------------|----------------|
| #1 | 0.05 | 5% chance → likely healthy |
| #2 | 0.72 | 72% chance → likely abnormal |
| #3 | 0.95 | 95% chance → high risk |

## 🔬 Input Modalities

| Modality | Input Data | What It Represents |
|----------|------------|-------------------|
| 🫀 ECG (image) | 2D ECG graph images | Heart electrical activity pattern |
| 🔉 PCG (audio) | Heart sound waveform (.wav) | Heartbeat rhythm and murmurs |
| 📊 Clinical data | 14 numerical features | Patient-level medical indicators |

## 📦 Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Verify model file exists**:
   - Make sure `best_multimodal_model.pth` is in the same directory

## 🚀 Running the API

Start the Flask server:

```bash
python app.py
```

The API will be available at `http://localhost:5000`

## 🔌 API Endpoints

### 1. Health Check
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

### 2. Prediction
```bash
POST /predict
```

**Required Files**:
- `ecg_image`: ECG image file (PNG, JPG)
- `pcg_audio`: PCG audio file (WAV)

**Required Form Data** (14 clinical features):
- `age`: Patient age (normalized)
- `sex`: Sex (normalized)
- `cp`: Chest pain type
- `trtbps`: Resting blood pressure
- `chol`: Cholesterol level
- `fbs`: Fasting blood sugar
- `restecg`: Resting ECG results
- `thalachh`: Maximum heart rate achieved
- `exng`: Exercise induced angina
- `oldpeak`: ST depression
- `slp`: Slope of peak exercise ST segment
- `caa`: Number of major vessels
- `thall`: Thalassemia
- `temp`: Temperature/Oxygen saturation

**Example using Python**:
```python
import requests

files = {
    'ecg_image': open('patient_ecg.png', 'rb'),
    'pcg_audio': open('patient_pcg.wav', 'rb')
}

clinical_data = {
    'age': 3.048,
    'sex': 0.166,
    'cp': 6.683,
    'trtbps': 2.908,
    'chol': -0.497,
    'fbs': 8.869,
    'restecg': -4.769,
    'thalachh': -0.405,
    'exng': -0.169,
    'oldpeak': 4.308,
    'slp': -4.910,
    'caa': -0.171,
    'thall': -5.186,
    'temp': 0.497
}

response = requests.post('http://localhost:5000/predict', 
                        files=files, 
                        data=clinical_data)
print(response.json())
```

**Example using cURL**:
```bash
curl -X POST http://localhost:5000/predict \
  -F "ecg_image=@patient_ecg.png" \
  -F "pcg_audio=@patient_pcg.wav" \
  -F "age=3.048" \
  -F "sex=0.166" \
  -F "cp=6.683" \
  -F "trtbps=2.908" \
  -F "chol=-0.497" \
  -F "fbs=8.869" \
  -F "restecg=-4.769" \
  -F "thalachh=-0.405" \
  -F "exng=-0.169" \
  -F "oldpeak=4.308" \
  -F "slp=-4.910" \
  -F "caa=-0.171" \
  -F "thall=-5.186" \
  -F "temp=0.497"
```

**Response**:
```json
{
  "success": true,
  "probability": 0.8542,
  "percentage": "85.42%",
  "interpretation": "High risk - Abnormality likely detected",
  "risk_level": "high"
}
```

## 🧪 Testing the API

Run the test script:

```bash
python test_api.py
```

**Note**: Update the file paths in `test_api.py` to point to actual patient data from your dataset.

## 📁 Project Structure

```
Data_Wrangling/
├── app.py                          # Flask API application
├── model.py                        # Model architecture definition
├── preprocessing.py                # Data preprocessing utilities
├── best_multimodal_model.pth      # Trained model weights
├── multimodal_metadata_clean.csv  # Training dataset metadata
├── requirements.txt               # Python dependencies
├── test_api.py                    # API testing script
└── README.md                      # This file
```

## 🎯 Risk Level Interpretation

| Probability Range | Risk Level | Interpretation |
|-------------------|------------|----------------|
| 0.0 - 0.3 | Low | Likely healthy |
| 0.3 - 0.7 | Moderate | Further examination recommended |
| 0.7 - 1.0 | High | Abnormality likely detected |

## ⚙️ Model Architecture

- **ECG Branch**: ResNet18 CNN → 128-dim features
- **PCG Branch**: 1D CNN on waveform → 128-dim features
- **Clinical Branch**: Fully connected layers → 32-dim features
- **Fusion**: Concatenation → FC layers → Sigmoid output

## 🛠️ Troubleshooting

**Issue**: Model not loading
- **Solution**: Verify `best_multimodal_model.pth` exists in the directory

**Issue**: CUDA out of memory
- **Solution**: The API will automatically use CPU if CUDA is unavailable

**Issue**: File format errors
- **Solution**: Ensure ECG images are RGB and PCG audio is WAV format

## 📝 Notes

- Clinical features in the CSV are already normalized (z-scores)
- ECG images are automatically converted to RGB and normalized
- PCG audio is automatically converted to mono and padded/trimmed to 500 samples
- Temporary files are automatically cleaned up after prediction

## 🔒 Security Note

This is a development/testing API. For production use:
- Add authentication
- Add rate limiting
- Add input validation
- Use HTTPS
- Add logging and monitoring
