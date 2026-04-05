/**
 * API service for PlantHealth AI backend communication
 */
import axios, { AxiosError } from 'axios';
import type { PredictionResponse, HealthResponse, ErrorResponse } from '../types';

// API base URL - uses Vite proxy in development
const API_BASE_URL = '/api';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 second timeout for predictions
});

/**
 * Upload an image and get plant disease prediction
 */
export async function predictDisease(imageFile: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('file', imageFile);

  try {
    const response = await apiClient.post<PredictionResponse>('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<ErrorResponse>;
      if (axiosError.response?.data?.detail) {
        throw new Error(axiosError.response.data.detail);
      }
      if (axiosError.code === 'ECONNABORTED') {
        throw new Error('Request timed out. Please try again.');
      }
      if (!axiosError.response) {
        throw new Error('Cannot connect to server. Please ensure the backend is running.');
      }
    }
    throw new Error('An unexpected error occurred. Please try again.');
  }
}

/**
 * Check API health status
 */
export async function checkHealth(): Promise<HealthResponse> {
  try {
    const response = await apiClient.get<HealthResponse>('/health');
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error) && !error.response) {
      throw new Error('Cannot connect to server');
    }
    throw error;
  }
}

/**
 * Format confidence as percentage string
 * Note: Backend returns confidence as 0-100
 */
export function formatConfidence(confidence: number): string {
  return `${confidence.toFixed(1)}%`;
}

/**
 * Get severity level based on confidence
 * Note: Backend returns confidence as 0-100
 */
export function getConfidenceLevel(confidence: number): 'high' | 'medium' | 'low' {
  if (confidence >= 80) return 'high';
  if (confidence >= 50) return 'medium';
  return 'low';
}

/**
 * Format class name for display (e.g., "Apple___Black_rot" -> "Apple - Black Rot")
 */
export function formatClassName(className: string): string {
  return className
    .replace(/___/g, ' - ')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
