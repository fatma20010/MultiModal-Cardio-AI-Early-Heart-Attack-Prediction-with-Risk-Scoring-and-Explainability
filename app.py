from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import os
import json
from model import MultiModalCardioAI
from preprocessing import DataPreprocessor
from calibration import ModelCalibrator
from explainability import ModalityExplainer
from report_generator import CardiacReportGenerator
from clinical_normalizer import ClinicalNormalizer

app = Flask(__name__)

# Enable CORS for frontend connection
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173", 
            "http://localhost:3000", 
            "http://localhost:8080",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8080",
            "http://127.0.0.1:3000",
            # Allow all origins for Lovable deployment
            "*"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Configuration
MODEL_PATH = "best_multimodal_model.pth"
CLINICAL_INPUT_SIZE = 14  # Number of clinical features
MAX_PCG_LEN = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
print("Loading model...")
model = MultiModalCardioAI(clinical_input_size=CLINICAL_INPUT_SIZE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f"Model loaded successfully on {DEVICE}")

# Initialize modules
preprocessor = DataPreprocessor(max_pcg_len=MAX_PCG_LEN)
calibrator = ModelCalibrator(model, device=DEVICE)
explainer = ModalityExplainer(model, device=DEVICE)
report_generator = CardiacReportGenerator()
clinical_normalizer = ClinicalNormalizer()

@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        "message": "🫀 Multimodal Cardiac Abnormality Detection API",
        "version": "2.0 - Enhanced with Calibration & Explainability",
        "status": "online",
        "endpoints": {
            "/predict": "POST - Make a prediction (basic)",
            "/predict/detailed": "POST - Make a prediction with explainability",
            "/health": "GET - Check API health",
            "/risk-levels": "GET - Get risk level definitions"
        },
        "features": [
            "✅ 5-level risk stratification",
            "✅ Modality contribution analysis",
            "✅ Clinical feature importance",
            "✅ Confidence assessment"
        ]
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "device": str(DEVICE)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict cardiac abnormality from multimodal data
    
    Expected input:
    - Form data with files: 'ecg_image', 'pcg_audio'
    - Form data with clinical features (14 values):
        age, sex, cp, trtbps, chol, fbs, restecg, thalachh, 
        exng, oldpeak, slp, caa, thall, temp
    
    Returns:
    - JSON with probability score and interpretation
    """
    try:
        # Check if files are present
        if 'ecg_image' not in request.files:
            return jsonify({"error": "Missing 'ecg_image' file"}), 400
        if 'pcg_audio' not in request.files:
            return jsonify({"error": "Missing 'pcg_audio' file"}), 400
        
        ecg_file = request.files['ecg_image']
        pcg_file = request.files['pcg_audio']
        
        # Save uploaded files temporarily
        ecg_path = f"temp_ecg_{os.urandom(8).hex()}.png"
        pcg_path = f"temp_pcg_{os.urandom(8).hex()}.wav"
        
        ecg_file.save(ecg_path)
        pcg_file.save(pcg_path)
        
        try:
            # Get RAW clinical data from form
            raw_clinical_data = {
                'age': float(request.form.get('age', 54.0)),
                'sex': float(request.form.get('sex', 1.0)),
                'cp': float(request.form.get('cp', 0.0)),
                'trtbps': float(request.form.get('trtbps', 120.0)),
                'chol': float(request.form.get('chol', 240.0)),
                'fbs': float(request.form.get('fbs', 0.0)),
                'restecg': float(request.form.get('restecg', 0.0)),
                'thalachh': float(request.form.get('thalachh', 150.0)),
                'exng': float(request.form.get('exng', 0.0)),
                'oldpeak': float(request.form.get('oldpeak', 0.0)),
                'slp': float(request.form.get('slp', 1.0)),
                'caa': float(request.form.get('caa', 0.0)),
                'thall': float(request.form.get('thall', 2.0)),
                'temp': float(request.form.get('temp', 98.6))
            }
            
            # 🔧 NORMALIZE raw clinical data to z-scores
            normalized_clinical_data = clinical_normalizer.normalize_clinical_data(raw_clinical_data)
            normalized_clinical_data['98.6'] = normalized_clinical_data['temp']
            
            # Preprocess inputs
            ecg_tensor = preprocessor.preprocess_ecg(ecg_path).to(DEVICE)
            pcg_tensor = preprocessor.preprocess_pcg(pcg_path).to(DEVICE)
            clinical_tensor = preprocessor.preprocess_clinical(normalized_clinical_data).to(DEVICE)
            
            # Make prediction
            with torch.no_grad():
                output = model(ecg_tensor, pcg_tensor, clinical_tensor)
                probability = output.item()
            
            # Get enhanced risk stratification
            risk_info = calibrator.get_risk_stratification(probability)
            
            # Prepare response
            response = {
                "success": True,
                "probability": round(probability, 4),
                "percentage": f"{round(probability * 100, 2)}%",
                "risk_assessment": {
                    "level": risk_info['level'],
                    "category": risk_info['category'],
                    "interpretation": risk_info['interpretation'],
                    "recommendation": risk_info['recommendation'],
                    "action": risk_info['action']
                }
            }
            
            return jsonify(response)
            
        finally:
            # Clean up temporary files
            if os.path.exists(ecg_path):
                os.remove(ecg_path)
            if os.path.exists(pcg_path):
                os.remove(pcg_path)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/predict/detailed', methods=['POST'])
def predict_detailed():
    """
    Predict with full explainability analysis
    
    Same input as /predict endpoint, but returns:
    - Modality contribution analysis
    - Clinical feature importance
    - Confidence assessment
    - Detailed explanations
    """
    try:
        # Check if files are present
        if 'ecg_image' not in request.files:
            return jsonify({"error": "Missing 'ecg_image' file"}), 400
        if 'pcg_audio' not in request.files:
            return jsonify({"error": "Missing 'pcg_audio' file"}), 400
        
        ecg_file = request.files['ecg_image']
        pcg_file = request.files['pcg_audio']
        
        # Save uploaded files temporarily
        ecg_path = f"temp_ecg_{os.urandom(8).hex()}.png"
        pcg_path = f"temp_pcg_{os.urandom(8).hex()}.wav"
        
        ecg_file.save(ecg_path)
        pcg_file.save(pcg_path)
        
        try:
            # Get RAW clinical data from form
            raw_clinical_data = {
                'age': float(request.form.get('age', 54.0)),
                'sex': float(request.form.get('sex', 1.0)),
                'cp': float(request.form.get('cp', 0.0)),
                'trtbps': float(request.form.get('trtbps', 120.0)),
                'chol': float(request.form.get('chol', 240.0)),
                'fbs': float(request.form.get('fbs', 0.0)),
                'restecg': float(request.form.get('restecg', 0.0)),
                'thalachh': float(request.form.get('thalachh', 150.0)),
                'exng': float(request.form.get('exng', 0.0)),
                'oldpeak': float(request.form.get('oldpeak', 0.0)),
                'slp': float(request.form.get('slp', 1.0)),
                'caa': float(request.form.get('caa', 0.0)),
                'thall': float(request.form.get('thall', 2.0)),
                'temp': float(request.form.get('temp', 98.6))
            }
            
            # 🔧 NORMALIZE raw clinical data to z-scores
            normalized_clinical_data = clinical_normalizer.normalize_clinical_data(raw_clinical_data)
            normalized_clinical_data['98.6'] = normalized_clinical_data['temp']
            
            # Preprocess inputs
            ecg_tensor = preprocessor.preprocess_ecg(ecg_path).to(DEVICE)
            pcg_tensor = preprocessor.preprocess_pcg(pcg_path).to(DEVICE)
            clinical_tensor = preprocessor.preprocess_clinical(normalized_clinical_data).to(DEVICE)
            
            # Make prediction
            with torch.no_grad():
                output = model(ecg_tensor, pcg_tensor, clinical_tensor)
                probability = output.item()
            
            # Get enhanced risk stratification
            risk_info = calibrator.get_risk_stratification(probability)
            
            # Get explainability analysis
            explanation_report = explainer.generate_explanation_report(
                ecg_tensor, pcg_tensor, clinical_tensor, probability
            )
            
            # Prepare comprehensive response
            response = {
                "success": True,
                "probability": round(probability, 4),
                "percentage": f"{round(probability * 100, 2)}%",
                "risk_assessment": {
                    "level": risk_info['level'],
                    "category": risk_info['category'],
                    "color": risk_info['color'],
                    "interpretation": risk_info['interpretation'],
                    "recommendation": risk_info['recommendation'],
                    "action": risk_info['action']
                },
                "explainability": {
                    "modality_contributions": {
                        "ecg": explanation_report['modality_analysis']['contributions']['ecg'],
                        "pcg": explanation_report['modality_analysis']['contributions']['pcg'],
                        "clinical": explanation_report['modality_analysis']['contributions']['clinical']
                    },
                    "primary_driver": explanation_report['modality_analysis']['primary_driver'],
                    "explanation": explanation_report['modality_analysis']['explanation'],
                    "top_clinical_features": explanation_report['clinical_feature_importance']['top_5_features'],
                    "confidence_assessment": explanation_report['confidence_assessment'],
                    "summary": explanation_report['summary']
                }
            }
            
            return jsonify(response)
            
        finally:
            # Clean up temporary files
            if os.path.exists(ecg_path):
                os.remove(ecg_path)
            if os.path.exists(pcg_path):
                os.remove(pcg_path)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/risk-levels', methods=['GET'])
def risk_levels():
    """Get all risk level definitions"""
    levels = []
    for prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
        levels.append(calibrator.get_risk_stratification(prob))
    
    return jsonify({
        "risk_levels": levels,
        "description": "5-level risk stratification system for cardiac abnormality detection"
    })

@app.route('/generate-report', methods=['POST'])
def generate_report():
    """
    Generate PDF report from prediction results
    
    Expects JSON data with prediction results
    Returns PDF file for download
    """
    try:
        # Get prediction data from request
        prediction_data = request.json
        
        if not prediction_data:
            return jsonify({"error": "No prediction data provided"}), 400
        
        # Generate unique filename
        timestamp = str(int(os.urandom(4).hex(), 16))
        report_filename = f"cardiac_report_{timestamp}.pdf"
        report_path = os.path.join("reports", report_filename)
        
        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        
        # Generate PDF report
        report_generator.generate_report(prediction_data, report_path)
        
        # Send file
        return send_file(
            report_path,
            as_attachment=True,
            download_name=f"Cardiac_Assessment_Report_{timestamp}.pdf",
            mimetype='application/pdf'
        )
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Report generation failed: {str(e)}"
        }), 500
    finally:
        # Clean up report file after sending (optional)
        # Uncomment if you want to delete after download
        # if os.path.exists(report_path):
        #     os.remove(report_path)
        pass

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🫀 MULTIMODAL CARDIAC ABNORMALITY DETECTION API")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Clinical Features: {CLINICAL_INPUT_SIZE}")
    print("="*60 + "\n")
    
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
