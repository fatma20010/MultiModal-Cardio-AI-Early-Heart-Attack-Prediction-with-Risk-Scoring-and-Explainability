"""
Clinical Data Normalizer
Converts raw clinical values to z-scores using standard medical reference ranges
"""

import numpy as np

class ClinicalNormalizer:
    """
    Normalizes raw clinical data to z-scores
    Handles both numeric values and human-readable string labels
    Based on standard medical reference ranges and typical heart disease dataset statistics
    """
    
    def __init__(self):
        # Standard reference ranges and statistics for heart disease datasets
        # These are typical means and standard deviations from UCI Heart Disease datasets
        self.normalization_params = {
            'age': {'mean': 54.0, 'std': 9.0},           # Years (typical: 29-77)
            'sex': {'mean': 0.68, 'std': 0.47},          # 0=Female, 1=Male
            'cp': {'mean': 0.97, 'std': 1.03},           # Chest pain type (0-3)
            'trtbps': {'mean': 131.6, 'std': 17.5},      # Resting BP in mm Hg (typical: 94-200)
            'chol': {'mean': 246.3, 'std': 51.8},        # Cholesterol in mg/dL (typical: 126-564)
            'fbs': {'mean': 0.15, 'std': 0.36},          # Fasting blood sugar > 120 mg/dL (0 or 1)
            'restecg': {'mean': 0.53, 'std': 0.53},      # Resting ECG results (0-2)
            'thalachh': {'mean': 149.6, 'std': 22.9},    # Max heart rate achieved (typical: 71-202)
            'exng': {'mean': 0.33, 'std': 0.47},         # Exercise induced angina (0 or 1)
            'oldpeak': {'mean': 1.04, 'std': 1.16},      # ST depression (typical: 0-6.2)
            'slp': {'mean': 1.40, 'std': 0.62},          # Slope of peak exercise ST segment (0-2)
            'caa': {'mean': 0.73, 'std': 1.02},          # Number of major vessels (0-4)
            'thall': {'mean': 2.31, 'std': 0.61},        # Thalassemia (0-3)
            'temp': {'mean': 98.6, 'std': 0.5}           # Body temperature in °F or SpO2
        }
        
        # Mapping dictionaries for string values to numeric codes
        self.value_mappings = {
            'sex': {
                'female': 0, 'f': 0, 'woman': 0, '0': 0, 0: 0,
                'male': 1, 'm': 1, 'man': 1, '1': 1, 1: 1
            },
            'cp': {
                # Frontend uses 1-4, but model needs 0-3
                'typical angina': 0, 'typical': 0, '1': 0, 1: 0,
                'atypical angina': 1, 'atypical': 1, '2': 1, 2: 1,
                'non-anginal pain': 2, 'non-anginal': 2, 'non anginal': 2, '3': 2, 3: 2,
                'asymptomatic': 3, '4': 3, 4: 3,
                # Also accept 0-3 directly
                '0': 0, 0: 0
            },
            'fbs': {
                'true': 1, 'yes': 1, '1': 1, 1: 1, '>120': 1,
                'false': 0, 'no': 0, '0': 0, 0: 0, '≤120': 0, '<=120': 0
            },
            'restecg': {
                'normal': 0, '0': 0, 0: 0,
                'st-t wave abnormality': 1, 'st-t abnormality': 1, 'abnormality': 1, '1': 1, 1: 1,
                'left ventricular hypertrophy': 2, 'lvh': 2, 'hypertrophy': 2, '2': 2, 2: 2
            },
            'exng': {
                'yes': 1, 'true': 1, '1': 1, 1: 1,
                'no': 0, 'false': 0, '0': 0, 0: 0
            },
            'slp': {
                # Frontend uses 1-3, but model needs 0-2
                'upsloping': 0, 'up': 0, '1': 0, 1: 0,
                'flat': 1, '2': 1, 2: 1,
                'downsloping': 2, 'down': 2, '3': 2, 3: 2,
                # Also accept 0-2 directly
                '0': 0, 0: 0
            },
            'thall': {
                # Frontend uses 3,6,7 but model needs 0,1,2
                'normal': 0, '3': 0, 3: 0,
                'fixed defect': 1, 'fixed': 1, '6': 1, 6: 1,
                'reversible defect': 2, 'reversible': 2, '7': 2, 7: 2,
                # Also accept 0-2 directly
                '0': 0, 0: 0, '1': 1, 1: 1, '2': 2, 2: 2
            }
        }
    
    def convert_to_numeric(self, feature_name, value):
        """
        Convert string/categorical values to numeric codes
        
        Args:
            feature_name: Name of the clinical feature
            value: Raw value (can be string or number)
            
        Returns:
            float: Numeric value
        """
        # If already numeric, return as float
        if isinstance(value, (int, float)):
            # Handle special encoding conversions
            if feature_name == 'cp' and value in [1, 2, 3, 4]:
                return float(value - 1)  # Convert 1-4 to 0-3
            elif feature_name == 'slp' and value in [1, 2, 3]:
                return float(value - 1)  # Convert 1-3 to 0-2
            elif feature_name == 'thall' and value in [3, 6, 7]:
                mapping = {3: 0, 6: 1, 7: 2}
                return float(mapping[value])
            return float(value)
        
        # Convert string to lowercase for matching
        value_str = str(value).lower().strip()
        
        # Check if feature has a mapping
        if feature_name in self.value_mappings:
            if value_str in self.value_mappings[feature_name]:
                return float(self.value_mappings[feature_name][value_str])
            # Try to convert to int first (in case it's a string number)
            try:
                num_val = int(value_str)
                if num_val in self.value_mappings[feature_name]:
                    return float(self.value_mappings[feature_name][num_val])
            except (ValueError, KeyError):
                pass
        
        # If no mapping found, try to convert directly to float
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Cannot convert value '{value}' for feature '{feature_name}'")
    
    def normalize_value(self, feature_name, raw_value):
        """
        Normalize a single feature value to z-score
        
        Args:
            feature_name: Name of the clinical feature
            raw_value: Raw clinical value (numeric or string)
            
        Returns:
            float: Z-score normalized value
        """
        if feature_name not in self.normalization_params:
            raise ValueError(f"Unknown feature: {feature_name}")
        
        # Convert to numeric first
        numeric_value = self.convert_to_numeric(feature_name, raw_value)
        
        params = self.normalization_params[feature_name]
        z_score = (numeric_value - params['mean']) / params['std']
        
        return z_score
    
    def normalize_clinical_data(self, raw_clinical_data):
        """
        Normalize all clinical features from raw values to z-scores
        
        Args:
            raw_clinical_data: Dictionary with raw clinical values
                {
                    'age': 45,
                    'sex': 1,
                    'cp': 2,
                    'trtbps': 120,
                    'chol': 240,
                    ... etc
                }
        
        Returns:
            dict: Normalized clinical data (z-scores)
        """
        normalized = {}
        
        feature_order = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 
                        'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 
                        'caa', 'thall', 'temp']
        
        for feature in feature_order:
            raw_value = float(raw_clinical_data.get(feature, 0))
            
            # Handle special case for 98.6 column name
            if feature == 'temp':
                actual_feature = '98.6' if '98.6' in raw_clinical_data else 'temp'
                raw_value = float(raw_clinical_data.get(actual_feature, raw_clinical_data.get('temp', 98.6)))
            
            normalized[feature] = self.normalize_value(feature, raw_value)
        
        return normalized
    
    def get_reference_ranges(self):
        """Get typical reference ranges for clinical features"""
        ranges = {
            'age': {'min': 20, 'max': 90, 'unit': 'years', 'typical': '29-77'},
            'sex': {'values': '0=Female, 1=Male'},
            'cp': {'values': '0=Typical Angina, 1=Atypical Angina, 2=Non-anginal Pain, 3=Asymptomatic'},
            'trtbps': {'min': 90, 'max': 200, 'unit': 'mm Hg', 'normal': '90-140'},
            'chol': {'min': 100, 'max': 600, 'unit': 'mg/dL', 'normal': '<200 desirable'},
            'fbs': {'values': '0=≤120 mg/dL, 1=>120 mg/dL'},
            'restecg': {'values': '0=Normal, 1=ST-T Wave Abnormality, 2=Left Ventricular Hypertrophy'},
            'thalachh': {'min': 60, 'max': 220, 'unit': 'bpm', 'normal': '60-100 resting'},
            'exng': {'values': '0=No, 1=Yes'},
            'oldpeak': {'min': 0, 'max': 7, 'unit': 'ST depression'},
            'slp': {'values': '0=Upsloping, 1=Flat, 2=Downsloping'},
            'caa': {'values': '0-4 major vessels colored by fluoroscopy'},
            'thall': {'values': '0=Normal, 1=Fixed Defect, 2=Reversible Defect, 3=Unknown'},
            'temp': {'min': 97, 'max': 100, 'unit': '°F or SpO2'}
        }
        return ranges
    
    def denormalize_value(self, feature_name, z_score):
        """
        Convert z-score back to raw value (for display purposes)
        
        Args:
            feature_name: Name of the clinical feature
            z_score: Z-score normalized value
            
        Returns:
            float: Original scale value (approximate)
        """
        if feature_name not in self.normalization_params:
            raise ValueError(f"Unknown feature: {feature_name}")
        
        params = self.normalization_params[feature_name]
        raw_value = (z_score * params['std']) + params['mean']
        
        return raw_value


# Example usage
if __name__ == "__main__":
    normalizer = ClinicalNormalizer()
    
    # Example: Patient with normal clinical values
    patient_data = {
        'age': 45,          # 45 years old
        'sex': 1,           # Male
        'cp': 2,            # Non-anginal pain
        'trtbps': 120,      # 120 mm Hg
        'chol': 240,        # 240 mg/dL
        'fbs': 0,           # < 120 mg/dL
        'restecg': 0,       # Normal
        'thalachh': 150,    # 150 bpm
        'exng': 0,          # No
        'oldpeak': 1.0,     # 1.0 ST depression
        'slp': 1,           # Flat
        'caa': 0,           # 0 vessels
        'thall': 2,         # Reversible defect
        'temp': 98.6        # Normal temperature
    }
    
    print("Raw Clinical Data:")
    for key, value in patient_data.items():
        print(f"  {key:12s}: {value}")
    
    print("\nNormalized (Z-scores):")
    normalized = normalizer.normalize_clinical_data(patient_data)
    for key, value in normalized.items():
        print(f"  {key:12s}: {value:8.4f}")
    
    print("\n✅ Use these z-scores for model input!")
