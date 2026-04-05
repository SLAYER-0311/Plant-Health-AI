import { useState, useEffect } from 'react';
import { ImageUpload, PredictionResult } from './components';
import { checkHealth } from './services/api';
import type { PredictionResponse, HealthResponse } from './types';

function App() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState<HealthResponse | null>(null);
  const [serverError, setServerError] = useState<string | null>(null);

  // Check server health on mount
  useEffect(() => {
    const checkServerHealth = async () => {
      try {
        const health = await checkHealth();
        setServerStatus(health);
        setServerError(null);
      } catch {
        setServerError('Cannot connect to server. Please ensure the backend is running.');
      }
    };

    checkServerHealth();
    // Recheck every 30 seconds
    const interval = setInterval(checkServerHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handlePrediction = (prediction: PredictionResponse, imgUrl: string) => {
    setResult(prediction);
    setImageUrl(imgUrl);
    setError(null);
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
    setResult(null);
  };

  return (
    <div className="min-h-screen py-8 px-4">
      {/* Header */}
      <header className="text-center mb-10">
        <div className="flex items-center justify-center gap-3 mb-4">
          <svg
            className="w-12 h-12 text-green-600 dark:text-green-400"
            fill="currentColor"
            viewBox="0 0 24 24"
          >
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
          </svg>
          <h1 className="text-4xl md:text-5xl font-bold text-gray-800 dark:text-white">
            PlantHealth AI
          </h1>
        </div>
        <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
          Upload an image of your plant to detect diseases using advanced AI. 
          Get instant diagnosis with confidence scores and treatment recommendations.
        </p>
      </header>

      {/* Server Status Indicator */}
      <div className="max-w-xl mx-auto mb-6">
        {serverError ? (
          <div className="bg-red-100 dark:bg-red-900/30 border border-red-300 dark:border-red-800 rounded-lg p-3 text-center">
            <p className="text-red-700 dark:text-red-300 text-sm flex items-center justify-center gap-2">
              <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></span>
              {serverError}
            </p>
          </div>
        ) : serverStatus ? (
          <div className="bg-green-100 dark:bg-green-900/30 border border-green-300 dark:border-green-800 rounded-lg p-3 text-center">
            <p className="text-green-700 dark:text-green-300 text-sm flex items-center justify-center gap-2">
              <span className="w-2 h-2 bg-green-500 rounded-full"></span>
              Server connected | Model: {serverStatus.model_type} | Device: {serverStatus.device}
            </p>
          </div>
        ) : (
          <div className="bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg p-3 text-center">
            <p className="text-gray-600 dark:text-gray-400 text-sm flex items-center justify-center gap-2">
              <span className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></span>
              Checking server connection...
            </p>
          </div>
        )}
      </div>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto">
        {/* Upload Section */}
        <ImageUpload
          onPrediction={handlePrediction}
          onError={handleError}
          isLoading={isLoading}
          setIsLoading={setIsLoading}
        />

        {/* Error Display */}
        {error && (
          <div className="mt-6 max-w-xl mx-auto bg-red-100 dark:bg-red-900/30 border border-red-300 dark:border-red-800 rounded-lg p-4">
            <div className="flex items-center">
              <svg className="w-5 h-5 text-red-500 mr-3" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <p className="text-red-700 dark:text-red-300">{error}</p>
            </div>
          </div>
        )}

        {/* Results Display */}
        <PredictionResult result={result} imageUrl={imageUrl} />
      </main>

      {/* Footer */}
      <footer className="mt-16 text-center text-gray-500 dark:text-gray-400 text-sm">
        <p className="mb-2">
          Powered by Deep Learning | Supports 38 plant disease classes
        </p>
        <p>
          Built with PyTorch, FastAPI, React & Tailwind CSS
        </p>
      </footer>
    </div>
  );
}

export default App;
