/**
 * Type definitions for PlantHealth AI frontend
 */

// OOD Detection info
export interface OODInfo {
  is_ood: boolean;
  max_probability: number;
  entropy: number;
  recommendation: string;
  in_distribution_votes: number;
  total_votes: number;
}

// Prediction response from the API
export interface PredictionResponse {
  success: boolean;
  top_prediction: {
    class_name: string;
    confidence: number;
    plant: string;
    condition: string;
    class_index: number;
  } | null;
  predictions: TopPrediction[];
  inference_time_ms: number;
  ood_detection?: OODInfo | null;
  warning?: string | null;
}

// Individual prediction in top predictions list
export interface TopPrediction {
  class_name: string;
  confidence: number;
  plant: string;
  condition: string;
  class_index: number;
}

// Error response from the API
export interface ErrorResponse {
  detail: string;
}

// Health check response
export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_type: string;
  device: string;
  num_classes: number;
}

// Upload state
export type UploadStatus = 'idle' | 'uploading' | 'success' | 'error';

// Component props
export interface ImageUploadProps {
  onPrediction: (result: PredictionResponse, imageUrl: string) => void;
  onError: (error: string) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
}

export interface PredictionResultProps {
  result: PredictionResponse | null;
  imageUrl: string | null;
}

export interface ConfidenceBarProps {
  label: string;
  confidence: number;
  isTop?: boolean;
}
