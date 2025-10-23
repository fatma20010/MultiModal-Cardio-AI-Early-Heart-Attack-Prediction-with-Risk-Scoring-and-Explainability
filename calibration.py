import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pandas as pd

class ModelCalibrator:
    """
    Calibration and evaluation utilities for the cardiac abnormality detection model
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize the calibrator
        
        Args:
            model: The trained MultiModalCardioAI model
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
    def expected_calibration_error(self, y_true, y_pred, n_bins=10):
        """
        Calculate Expected Calibration Error (ECE)
        
        Args:
            y_true: True binary labels (0 or 1)
            y_pred: Predicted probabilities (0-1)
            n_bins: Number of bins for calibration
            
        Returns:
            float: ECE score (lower is better, 0 is perfect)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred, bins[:-1]) - 1
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_acc = np.mean(y_true[mask])
                bin_conf = np.mean(y_pred[mask])
                bin_weight = np.sum(mask) / len(y_pred)
                ece += bin_weight * np.abs(bin_acc - bin_conf)
        
        return ece
    
    def plot_calibration_curve(self, y_true, y_pred, n_bins=10, save_path=None):
        """
        Plot calibration curve
        
        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities
            n_bins: Number of bins
            save_path: Path to save the plot (optional)
            
        Returns:
            fig: Matplotlib figure object
        """
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot calibration curve
        ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
        
        # Plot perfect calibration
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        
        # Calculate ECE
        ece = self.expected_calibration_error(y_true, y_pred, n_bins)
        
        ax.set_xlabel('Predicted Probability', fontsize=14)
        ax.set_ylabel('True Probability', fontsize=14)
        ax.set_title(f'Calibration Curve\nECE = {ece:.4f}', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, y_true, y_pred, save_path=None):
        """
        Plot ROC curve
        
        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities
            save_path: Path to save the plot
            
        Returns:
            fig: Matplotlib figure object
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('ROC Curve', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, y_true, y_pred, save_path=None):
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities
            save_path: Path to save the plot
            
        Returns:
            fig: Matplotlib figure object
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title('Precision-Recall Curve', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_risk_stratification(self, probability):
        """
        Enhanced 5-level risk stratification
        
        Args:
            probability: Predicted probability (0-1)
            
        Returns:
            dict: Risk level, category, color, and recommendations
        """
        if probability < 0.2:
            return {
                'level': 1,
                'category': 'Very Low Risk',
                'color': '#00C853',  # Green
                'interpretation': 'Very low probability of cardiac abnormality',
                'recommendation': 'Routine follow-up recommended',
                'action': 'Continue regular health monitoring'
            }
        elif probability < 0.4:
            return {
                'level': 2,
                'category': 'Low Risk',
                'color': '#64DD17',  # Light green
                'interpretation': 'Low probability of cardiac abnormality',
                'recommendation': 'Standard care pathway',
                'action': 'Annual cardiovascular check recommended'
            }
        elif probability < 0.6:
            return {
                'level': 3,
                'category': 'Moderate Risk',
                'color': '#FFC107',  # Amber
                'interpretation': 'Moderate probability of cardiac abnormality',
                'recommendation': 'Further diagnostic testing advised',
                'action': 'Consult with cardiologist for additional tests (stress test, echo)'
            }
        elif probability < 0.8:
            return {
                'level': 4,
                'category': 'High Risk',
                'color': '#FF6D00',  # Orange
                'interpretation': 'High probability of cardiac abnormality detected',
                'recommendation': 'Urgent medical evaluation required',
                'action': 'Schedule immediate cardiology consultation and comprehensive cardiac workup'
            }
        else:
            return {
                'level': 5,
                'category': 'Very High Risk',
                'color': '#D50000',  # Red
                'interpretation': 'Very high probability of cardiac abnormality',
                'recommendation': 'Immediate medical attention required',
                'action': 'Emergency cardiology referral - consider ER evaluation if symptomatic'
            }
    
    def evaluate_on_dataset(self, dataloader):
        """
        Evaluate model on a dataset and return predictions
        
        Args:
            dataloader: PyTorch DataLoader with test data
            
        Returns:
            dict: Contains y_true, y_pred, and metrics
        """
        y_true = []
        y_pred = []
        
        self.model.eval()
        with torch.no_grad():
            for ecg, pcg, clinical, target in dataloader:
                ecg = ecg.to(self.device)
                pcg = pcg.to(self.device)
                clinical = clinical.to(self.device)
                
                outputs = self.model(ecg, pcg, clinical)
                
                y_true.extend(target.cpu().numpy().flatten())
                y_pred.extend(outputs.cpu().numpy().flatten())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        ece = self.expected_calibration_error(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Calculate accuracy at different thresholds
        thresholds = [0.3, 0.5, 0.7]
        accuracies = {}
        for thresh in thresholds:
            y_pred_binary = (y_pred >= thresh).astype(int)
            acc = np.mean(y_pred_binary == y_true)
            accuracies[f'accuracy_@{thresh}'] = acc
        
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'metrics': {
                'ece': ece,
                'auc_roc': roc_auc,
                **accuracies
            }
        }
    
    def generate_calibration_report(self, y_true, y_pred, save_dir='calibration_report'):
        """
        Generate a comprehensive calibration report with plots
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            save_dir: Directory to save the report
            
        Returns:
            dict: Summary statistics
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate plots
        self.plot_calibration_curve(y_true, y_pred, 
                                   save_path=f'{save_dir}/calibration_curve.png')
        self.plot_roc_curve(y_true, y_pred,
                           save_path=f'{save_dir}/roc_curve.png')
        self.plot_precision_recall_curve(y_true, y_pred,
                                        save_path=f'{save_dir}/pr_curve.png')
        
        # Calculate metrics
        ece = self.expected_calibration_error(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)
        
        # Risk distribution
        risk_distribution = {
            'Very Low': np.sum(y_pred < 0.2),
            'Low': np.sum((y_pred >= 0.2) & (y_pred < 0.4)),
            'Moderate': np.sum((y_pred >= 0.4) & (y_pred < 0.6)),
            'High': np.sum((y_pred >= 0.6) & (y_pred < 0.8)),
            'Very High': np.sum(y_pred >= 0.8)
        }
        
        summary = {
            'calibration': {
                'ece': float(ece),
                'interpretation': 'Excellent' if ece < 0.05 else 'Good' if ece < 0.10 else 'Fair' if ece < 0.15 else 'Poor'
            },
            'discrimination': {
                'auc_roc': float(roc_auc),
                'auc_pr': float(pr_auc)
            },
            'risk_distribution': risk_distribution,
            'total_samples': len(y_true)
        }
        
        # Save summary as JSON
        import json
        with open(f'{save_dir}/summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Calibration Report Generated in '{save_dir}/'")
        print(f"   - Expected Calibration Error: {ece:.4f} ({summary['calibration']['interpretation']})")
        print(f"   - ROC AUC: {roc_auc:.4f}")
        print(f"   - PR AUC: {pr_auc:.4f}")
        
        return summary
