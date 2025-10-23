import requests
import json

# API endpoint
API_URL = "http://localhost:5000"

def test_home():
    """Test the home endpoint"""
    print("\n" + "="*60)
    print("Testing Home Endpoint")
    print("="*60)
    
    response = requests.get(f"{API_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_health():
    """Test the health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_risk_levels():
    """Test the risk levels endpoint"""
    print("\n" + "="*60)
    print("Testing Risk Levels Endpoint")
    print("="*60)
    
    response = requests.get(f"{API_URL}/risk-levels")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_prediction(ecg_path, pcg_path, clinical_data):
    """
    Test the basic prediction endpoint
    
    Args:
        ecg_path: Path to ECG image file
        pcg_path: Path to PCG audio file
        clinical_data: Dictionary with clinical features
    """
    print("\n" + "="*60)
    print("Testing Basic Prediction Endpoint")
    print("="*60)
    
    # Prepare files
    files = {
        'ecg_image': open(ecg_path, 'rb'),
        'pcg_audio': open(pcg_path, 'rb')
    }
    
    # Prepare form data
    data = clinical_data
    
    try:
        response = requests.post(f"{API_URL}/predict", files=files, data=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    finally:
        # Close file handles
        files['ecg_image'].close()
        files['pcg_audio'].close()

def test_prediction_detailed(ecg_path, pcg_path, clinical_data):
    """
    Test the detailed prediction endpoint with explainability
    
    Args:
        ecg_path: Path to ECG image file
        pcg_path: Path to PCG audio file
        clinical_data: Dictionary with clinical features
    """
    print("\n" + "="*60)
    print("Testing Detailed Prediction Endpoint (with Explainability)")
    print("="*60)
    
    # Prepare files
    files = {
        'ecg_image': open(ecg_path, 'rb'),
        'pcg_audio': open(pcg_path, 'rb')
    }
    
    # Prepare form data
    data = clinical_data
    
    try:
        response = requests.post(f"{API_URL}/predict/detailed", files=files, data=data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n🎯 Prediction: {result['probability']:.4f} ({result['percentage']})")
            print(f"\n📊 Risk Assessment:")
            print(f"   Level: {result['risk_assessment']['level']} - {result['risk_assessment']['category']}")
            print(f"   {result['risk_assessment']['interpretation']}")
            print(f"   Action: {result['risk_assessment']['action']}")
            
            print(f"\n🔍 Explainability Analysis:")
            print(f"   Primary Driver: {result['explainability']['primary_driver']}")
            print(f"\n   Modality Contributions:")
            for modality, contrib in result['explainability']['modality_contributions'].items():
                print(f"      {modality.upper():10s}: {contrib['percentage']:.1f}% - {contrib['interpretation']}")
            
            print(f"\n   Top Clinical Features:")
            for i, feature in enumerate(result['explainability']['top_clinical_features'], 1):
                print(f"      {i}. {feature['feature']:12s}: {feature['interpretation']}")
            
            print(f"\n   Confidence: {result['explainability']['confidence_assessment']['level']}")
            print(f"   {result['explainability']['confidence_assessment']['message']}")
            
            print(f"\n💡 Summary:")
            print(f"   {result['explainability']['summary']}")
        else:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return response.status_code == 200
    finally:
        # Close file handles
        files['ecg_image'].close()
        files['pcg_audio'].close()

def main():
    """Main test function"""
    print("\n" + "🫀"*30)
    print("MULTIMODAL CARDIAC API TEST SCRIPT - v2.0")
    print("Enhanced with Calibration & Explainability")
    print("🫀"*30)
    
    # Test 1: Home page
    print("\n[TEST 1] Home Page")
    test_home()
    
    # Test 2: Health check
    print("\n[TEST 2] Health Check")
    health_ok = test_health()
    
    if not health_ok:
        print("\n❌ Health check failed! Make sure the API is running.")
        print("   Run: python app.py")
        return
    
    print("\n✅ Health check passed!")
    
    # Test 3: Risk Levels
    print("\n[TEST 3] Risk Levels")
    test_risk_levels()
    
    # Sample clinical data (these are normalized values from your dataset)
    sample_clinical_data = {
        'age': 3.0484125000485958,
        'sex': 0.165876602604127,
        'cp': 6.682988904687991,
        'trtbps': 2.9083992336096136,
        'chol': -0.4971892863463477,
        'fbs': 8.86942313043338,
        'restecg': -4.769133982402904,
        'thalachh': -0.4046691447127727,
        'exng': -0.1685209527976911,
        'oldpeak': 4.307615072719734,
        'slp': -4.909948864025057,
        'caa': -0.1714670948070741,
        'thall': -5.185516525604089,
        'temp': 0.4967060552526283
    }
    
    # You need to update these paths to actual files from your dataset
    ecg_sample = "C:/Users/msi/Desktop/Ecgdata/P001.png"
    pcg_sample = "C:/Users/msi/Desktop/Finalaudio/P001.wav"
    
    print("\n" + "="*60)
    print("Prediction Tests")
    print("="*60)
    print(f"\nUsing sample data:")
    print(f"  ECG: {ecg_sample}")
    print(f"  PCG: {pcg_sample}")
    
    try:
        # Test 4: Basic Prediction
        print("\n[TEST 4] Basic Prediction")
        prediction_ok = test_prediction(ecg_sample, pcg_sample, sample_clinical_data)
        if prediction_ok:
            print("\n✅ Basic prediction test passed!")
        else:
            print("\n❌ Basic prediction test failed!")
        
        # Test 5: Detailed Prediction with Explainability
        print("\n[TEST 5] Detailed Prediction with Explainability")
        detailed_ok = test_prediction_detailed(ecg_sample, pcg_sample, sample_clinical_data)
        if detailed_ok:
            print("\n✅ Detailed prediction test passed!")
        else:
            print("\n❌ Detailed prediction test failed!")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  File not found: {e}")
        print("\nTo run prediction tests:")
        print("1. Update ecg_sample and pcg_sample paths in test_api.py")
        print("2. Use actual patient data from your dataset")
        print("3. Run this script again")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("All Tests Complete!")
    print("="*60)
    print("\n💡 Next Steps:")
    print("   1. Try the detailed prediction endpoint for full explainability")
    print("   2. Run evaluate_calibration.py to assess model calibration")
    print("   3. Check the 5-level risk stratification system")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
