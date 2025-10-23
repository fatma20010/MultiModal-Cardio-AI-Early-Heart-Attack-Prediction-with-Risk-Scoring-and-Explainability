import axios from 'axios';

// API URL configuration for different environments
const getApiUrl = () => {
  // Check if we're in production
  if (import.meta.env.PROD) {
    // In production, use environment variable or fallback to same origin
    return import.meta.env.VITE_API_URL || window.location.origin;
  }
  // In development, use localhost
  return import.meta.env.VITE_API_URL || 'http://localhost:5000';
};

// LOVABLE DEPLOYMENT FIX: Use same origin for production deployment
const API_URL = getApiUrl();

// Debug logging
console.log('API_URL:', API_URL);
console.log('Environment:', import.meta.env.MODE);
console.log('Production:', import.meta.env.PROD);

export interface ClinicalData {
  age: number;
  sex: number;
  cp: number;
  trtbps: number;
  chol: number;
  fbs: number;
  restecg: number;
  thalachh: number;
  exng: number;
  oldpeak: number;
  slp: number;
  caa: number;
  thall: number;
  temp: number;
}

export interface PredictionResponse {
  success: boolean;
  probability: number;
  percentage: string;
  risk_assessment: {
    level: number;
    category: string;
    interpretation: string;
    recommendation: string;
    action: string;
    color?: string;
  };
}

export interface DetailedPredictionResponse extends PredictionResponse {
  explainability?: {
    modality_contributions: {
      ecg: {
        percentage: number;
        impact: number;
        interpretation: string;
      };
      pcg: {
        percentage: number;
        impact: number;
        interpretation: string;
      };
      clinical: {
        percentage: number;
        impact: number;
        interpretation: string;
      };
    };
    primary_driver: string;
    explanation: string;
    top_clinical_features: Array<{
      feature: string;
      importance: number;
      interpretation: string;
    }>;
    confidence_assessment: {
      level: string;
      near_decision_boundary: boolean;
      modalities_balanced: boolean;
      message: string;
    };
    summary: string;
  };
}

export const predictRisk = async (
  ecgFile: File,
  pcgFile: File,
  clinicalData: ClinicalData,
  detailed: boolean = false
): Promise<PredictionResponse | DetailedPredictionResponse> => {
  const formData = new FormData();
  
  // Append files
  formData.append('ecg_image', ecgFile);
  formData.append('pcg_audio', pcgFile);
  
  // Append clinical data as strings
  Object.entries(clinicalData).forEach(([key, value]) => {
    formData.append(key, value.toString());
  });

  const endpoint = detailed ? '/predict/detailed' : '/predict';
  
  try {
    const response = await axios.post(`${API_URL}${endpoint}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const message = error.response?.data?.error || error.message;
      throw new Error(message);
    }
    throw error;
  }
};

export const downloadReport = async (predictionData: DetailedPredictionResponse): Promise<Blob> => {
  try {
    const response = await axios.post(`${API_URL}/generate-report`, predictionData, {
      responseType: 'blob',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error('Failed to generate report');
    }
    throw error;
  }
};
