import torch
import torchaudio
import torch.nn.functional as F
from PIL import Image
import numpy as np

class DataPreprocessor:
    def __init__(self, max_pcg_len=500):
        """
        Preprocessing utilities for multimodal cardiac data
        
        Args:
            max_pcg_len: Maximum length for PCG audio (default: 500)
        """
        self.max_pcg_len = max_pcg_len
        
    def preprocess_ecg(self, ecg_path):
        """
        Preprocess ECG image
        
        Args:
            ecg_path: Path to ECG image file
            
        Returns:
            torch.Tensor: Preprocessed ECG image tensor [1, 3, H, W]
        """
        try:
            # Load image
            ecg_img = Image.open(ecg_path).convert('RGB')
            
            # Convert to tensor and normalize to [0, 1]
            ecg_tensor = torch.tensor(np.array(ecg_img)).permute(2, 0, 1).float() / 255.0
            
            # Add batch dimension
            ecg_tensor = ecg_tensor.unsqueeze(0)
            
            return ecg_tensor
        except Exception as e:
            raise ValueError(f"Error preprocessing ECG image: {str(e)}")
    
    def preprocess_pcg(self, pcg_path):
        """
        Preprocess PCG audio
        
        Args:
            pcg_path: Path to PCG audio file (.wav)
            
        Returns:
            torch.Tensor: Preprocessed PCG audio tensor [1, 1, max_pcg_len]
        """
        try:
            # Load audio
            waveform, sr = torchaudio.load(pcg_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Pad or trim to max_pcg_len
            if waveform.shape[1] < self.max_pcg_len:
                waveform = F.pad(waveform, (0, self.max_pcg_len - waveform.shape[1]))
            else:
                waveform = waveform[:, :self.max_pcg_len]
            
            # Add batch dimension
            waveform = waveform.unsqueeze(0)
            
            return waveform
        except Exception as e:
            raise ValueError(f"Error preprocessing PCG audio: {str(e)}")
    
    def preprocess_clinical(self, clinical_data):
        """
        Preprocess clinical data
        
        Args:
            clinical_data: Dictionary or list of clinical features
                          Expected features (in order): age, sex, cp, trtbps, chol, fbs, 
                          restecg, thalachh, exng, oldpeak, slp, caa, thall, temp
            
        Returns:
            torch.Tensor: Preprocessed clinical data tensor [1, num_features]
        """
        try:
            # Convert to list if dictionary
            if isinstance(clinical_data, dict):
                # Expected order of features
                feature_order = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 
                               'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 
                               'caa', 'thall', '98.6']
                clinical_list = [float(clinical_data.get(key, 0.0)) for key in feature_order]
            else:
                clinical_list = [float(x) for x in clinical_data]
            
            # Convert to tensor
            clinical_tensor = torch.tensor(clinical_list, dtype=torch.float).unsqueeze(0)
            
            return clinical_tensor
        except Exception as e:
            raise ValueError(f"Error preprocessing clinical data: {str(e)}")
