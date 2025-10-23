import torch
import torch.nn as nn
import numpy as np
import copy

class ModalityExplainer:
    """
    Explainability module to understand modality contributions
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize the explainer
        
        Args:
            model: The trained MultiModalCardioAI model
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def get_modality_contributions(self, ecg_tensor, pcg_tensor, clinical_tensor):
        """
        Calculate contribution of each modality using baseline ablation study
        Uses mean/baseline values instead of zeros for more accurate attribution
        
        Args:
            ecg_tensor: ECG image tensor [1, 3, H, W]
            pcg_tensor: PCG audio tensor [1, 1, max_pcg_len]
            clinical_tensor: Clinical features tensor [1, num_features]
            
        Returns:
            dict: Contribution scores and explanations for each modality
        """
        self.model.eval()
        
        with torch.no_grad():
            # Full prediction
            full_output = self.model(ecg_tensor, pcg_tensor, clinical_tensor)
            full_prob = full_output.item()
            
            # Create baseline/neutral inputs (mean values, not zeros!)
            # For images: use mean pixel value (gray image)
            ecg_baseline = torch.ones_like(ecg_tensor) * 0.5  # Mid-gray
            
            # For audio: use mean value (silence/baseline)
            pcg_baseline = torch.zeros_like(pcg_tensor)  # Silence for audio is appropriate
            
            # For clinical: all zeros (already z-scores, zero is the mean)
            clinical_baseline = torch.zeros_like(clinical_tensor)
            
            # Ablation: Replace ECG with baseline
            output_no_ecg = self.model(ecg_baseline, pcg_tensor, clinical_tensor)
            prob_no_ecg = output_no_ecg.item()
            
            # Ablation: Replace PCG with baseline
            output_no_pcg = self.model(ecg_tensor, pcg_baseline, clinical_tensor)
            prob_no_pcg = output_no_pcg.item()
            
            # Ablation: Replace Clinical with baseline
            output_no_clinical = self.model(ecg_tensor, pcg_tensor, clinical_baseline)
            prob_no_clinical = output_no_clinical.item()
            
            # Also test with only each modality (to get positive contribution)
            output_only_ecg = self.model(ecg_tensor, pcg_baseline, clinical_baseline)
            prob_only_ecg = output_only_ecg.item()
            
            output_only_pcg = self.model(ecg_baseline, pcg_tensor, clinical_baseline)
            prob_only_pcg = output_only_pcg.item()
            
            output_only_clinical = self.model(ecg_baseline, pcg_baseline, clinical_tensor)
            prob_only_clinical = output_only_clinical.item()
            
            # Calculate contributions using both approaches and average them
            # Approach 1: Removal impact (how much it drops when removed)
            ecg_removal_impact = abs(full_prob - prob_no_ecg)
            pcg_removal_impact = abs(full_prob - prob_no_pcg)
            clinical_removal_impact = abs(full_prob - prob_no_clinical)
            
            # Approach 2: Individual contribution (baseline prediction)
            baseline_output = self.model(ecg_baseline, pcg_baseline, clinical_baseline)
            baseline_prob = baseline_output.item()
            
            ecg_individual = abs(prob_only_ecg - baseline_prob)
            pcg_individual = abs(prob_only_pcg - baseline_prob)
            clinical_individual = abs(prob_only_clinical - baseline_prob)
            
            # Combined contribution (average of both methods)
            ecg_contribution = (ecg_removal_impact + ecg_individual) / 2
            pcg_contribution = (pcg_removal_impact + pcg_individual) / 2
            clinical_contribution = (clinical_removal_impact + clinical_individual) / 2
            
            # Normalize to percentages
            total = ecg_contribution + pcg_contribution + clinical_contribution
            
            if total > 0:
                ecg_pct = (ecg_contribution / total) * 100
                pcg_pct = (pcg_contribution / total) * 100
                clinical_pct = (clinical_contribution / total) * 100
            else:
                # If all contributions are zero, assume equal contribution
                ecg_pct = pcg_pct = clinical_pct = 33.33
        
        # Determine primary driver
        contributions = {
            'ECG': ecg_pct,
            'PCG': pcg_pct,
            'Clinical': clinical_pct
        }
        primary_modality = max(contributions, key=contributions.get)
        
        return {
            'full_prediction': float(full_prob),
            'contributions': {
                'ecg': {
                    'percentage': float(ecg_pct),
                    'impact': float(ecg_contribution),
                    'interpretation': self._interpret_contribution(ecg_pct, 'ECG')
                },
                'pcg': {
                    'percentage': float(pcg_pct),
                    'impact': float(pcg_contribution),
                    'interpretation': self._interpret_contribution(pcg_pct, 'PCG')
                },
                'clinical': {
                    'percentage': float(clinical_pct),
                    'impact': float(clinical_contribution),
                    'interpretation': self._interpret_contribution(clinical_pct, 'Clinical')
                }
            },
            'primary_driver': primary_modality,
            'explanation': self._generate_explanation(primary_modality, contributions)
        }
    
    def _interpret_contribution(self, percentage, modality_name):
        """Generate interpretation text for contribution percentage"""
        if percentage > 50:
            return f"{modality_name} data strongly influenced this prediction"
        elif percentage > 35:
            return f"{modality_name} data moderately influenced this prediction"
        else:
            return f"{modality_name} data had minor influence on this prediction"
    
    def _generate_explanation(self, primary_modality, contributions):
        """Generate human-readable explanation of the prediction"""
        sorted_modalities = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        
        explanation = f"The prediction was primarily driven by {primary_modality} data "
        explanation += f"({contributions[primary_modality]:.1f}% contribution). "
        
        second_modality, second_pct = sorted_modalities[1]
        if second_pct > 25:
            explanation += f"{second_modality} data also played a significant role "
            explanation += f"({second_pct:.1f}% contribution)."
        
        return explanation
    
    def get_feature_importance(self, ecg_tensor, pcg_tensor, clinical_tensor, 
                              feature_names=None):
        """
        Calculate importance of individual clinical features
        
        Args:
            ecg_tensor: ECG image tensor
            pcg_tensor: PCG audio tensor
            clinical_tensor: Clinical features tensor [1, num_features]
            feature_names: List of feature names (optional)
            
        Returns:
            dict: Feature importance scores
        """
        if feature_names is None:
            feature_names = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 
                           'restecg', 'thalachh', 'exng', 'oldpeak', 
                           'slp', 'caa', 'thall', 'temp']
        
        self.model.eval()
        
        with torch.no_grad():
            # Baseline prediction
            baseline_output = self.model(ecg_tensor, pcg_tensor, clinical_tensor)
            baseline_prob = baseline_output.item()
            
            feature_importance = {}
            
            # Test each feature by zeroing it out
            for idx, feature_name in enumerate(feature_names):
                clinical_ablated = clinical_tensor.clone()
                clinical_ablated[0, idx] = 0
                
                output = self.model(ecg_tensor, pcg_tensor, clinical_ablated)
                prob = output.item()
                
                # Importance is the change in prediction
                importance = abs(baseline_prob - prob)
                feature_importance[feature_name] = float(importance)
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # Get top 5 features
        top_5 = sorted_features[:5]
        
        return {
            'all_features': feature_importance,
            'top_5_features': [
                {
                    'feature': name,
                    'importance': importance,
                    'interpretation': self._interpret_feature(name, importance)
                }
                for name, importance in top_5
            ]
        }
    
    def _interpret_feature(self, feature_name, importance):
        """Interpret clinical feature importance"""
        feature_descriptions = {
            'age': 'Patient age',
            'sex': 'Patient sex',
            'cp': 'Chest pain type',
            'trtbps': 'Resting blood pressure',
            'chol': 'Cholesterol level',
            'fbs': 'Fasting blood sugar',
            'restecg': 'Resting ECG results',
            'thalachh': 'Maximum heart rate',
            'exng': 'Exercise induced angina',
            'oldpeak': 'ST depression',
            'slp': 'Slope of ST segment',
            'caa': 'Number of major vessels',
            'thall': 'Thalassemia',
            'temp': 'Body temperature/SpO2'
        }
        
        description = feature_descriptions.get(feature_name, feature_name)
        
        if importance > 0.05:
            level = "critical"
        elif importance > 0.02:
            level = "significant"
        elif importance > 0.01:
            level = "moderate"
        else:
            level = "minor"
        
        return f"{description} had {level} impact on prediction"
    
    def generate_explanation_report(self, ecg_tensor, pcg_tensor, clinical_tensor,
                                   prediction_probability):
        """
        Generate comprehensive explanation report
        
        Args:
            ecg_tensor: ECG image tensor
            pcg_tensor: PCG audio tensor
            clinical_tensor: Clinical features tensor
            prediction_probability: The model's prediction
            
        Returns:
            dict: Complete explanation report
        """
        # Get modality contributions
        modality_contrib = self.get_modality_contributions(
            ecg_tensor, pcg_tensor, clinical_tensor
        )
        
        # Get feature importance
        feature_imp = self.get_feature_importance(
            ecg_tensor, pcg_tensor, clinical_tensor
        )
        
        # Generate confidence assessment
        confidence = self._assess_confidence(
            modality_contrib['contributions'], 
            prediction_probability
        )
        
        return {
            'prediction': float(prediction_probability),
            'modality_analysis': modality_contrib,
            'clinical_feature_importance': feature_imp,
            'confidence_assessment': confidence,
            'summary': self._generate_summary(
                modality_contrib, feature_imp, prediction_probability
            )
        }
    
    def _assess_confidence(self, contributions, probability):
        """Assess prediction confidence based on various factors"""
        
        # Check if prediction is near decision boundary
        near_boundary = 0.4 < probability < 0.6
        
        # Check if contributions are balanced (might indicate uncertainty)
        contrib_values = [
            contributions['ecg']['percentage'],
            contributions['pcg']['percentage'],
            contributions['clinical']['percentage']
        ]
        max_contrib = max(contrib_values)
        is_balanced = max_contrib < 50
        
        # Determine confidence level
        if near_boundary or is_balanced:
            confidence_level = "moderate"
            message = "Prediction confidence is moderate. Consider additional diagnostic tests."
        elif probability < 0.2 or probability > 0.8:
            confidence_level = "high"
            message = "Prediction confidence is high based on strong signal across modalities."
        else:
            confidence_level = "good"
            message = "Prediction confidence is good."
        
        return {
            'level': confidence_level,
            'near_decision_boundary': near_boundary,
            'modalities_balanced': is_balanced,
            'message': message
        }
    
    def _generate_summary(self, modality_contrib, feature_imp, probability):
        """Generate human-readable summary"""
        risk_term = "high risk" if probability > 0.5 else "low risk"
        primary = modality_contrib['primary_driver']
        top_feature = feature_imp['top_5_features'][0]['feature']
        
        summary = f"This patient shows {risk_term} of cardiac abnormality (probability: {probability:.1%}). "
        summary += f"The prediction is primarily based on {primary} data. "
        summary += f"Among clinical features, {top_feature} is the most influential."
        
        return summary
