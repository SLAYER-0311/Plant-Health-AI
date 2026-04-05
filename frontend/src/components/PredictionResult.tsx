import type { PredictionResultProps, ConfidenceBarProps } from '../types';
import { formatConfidence, getConfidenceLevel } from '../services/api';

function ConfidenceBar({ label, confidence, isTop = false }: ConfidenceBarProps) {
  const level = getConfidenceLevel(confidence);
  const percentage = confidence; // Backend already returns 0-100

  const colorClasses = {
    high: 'bg-green-500',
    medium: 'bg-yellow-500',
    low: 'bg-red-500',
  };

  return (
    <div className={`${isTop ? 'mb-4' : 'mb-2'}`}>
      <div className="flex justify-between items-center mb-1">
        <span className={`${isTop ? 'font-semibold text-gray-800 dark:text-gray-200' : 'text-sm text-gray-600 dark:text-gray-400'}`}>
          {label}
        </span>
        <span className={`${isTop ? 'font-bold' : 'text-sm'} ${
          level === 'high' ? 'text-green-600 dark:text-green-400' :
          level === 'medium' ? 'text-yellow-600 dark:text-yellow-400' :
          'text-red-600 dark:text-red-400'
        }`}>
          {formatConfidence(confidence)}
        </span>
      </div>
      <div className={`w-full bg-gray-200 dark:bg-gray-700 rounded-full ${isTop ? 'h-3' : 'h-2'}`}>
        <div
          className={`${colorClasses[level]} rounded-full transition-all duration-500 ${isTop ? 'h-3' : 'h-2'}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

function StatusBadge({ condition }: { condition: string }) {
  const isHealthy = condition.toLowerCase() === 'healthy';

  return (
    <span className={`
      inline-flex items-center px-3 py-1 rounded-full text-sm font-medium
      ${isHealthy 
        ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' 
        : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'}
    `}>
      {isHealthy ? (
        <svg className="w-4 h-4 mr-1.5" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
        </svg>
      ) : (
        <svg className="w-4 h-4 mr-1.5" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
        </svg>
      )}
      {condition}
    </span>
  );
}

export function PredictionResult({ result, imageUrl }: PredictionResultProps) {
  if (!result) return null;

  const { top_prediction: prediction, predictions: top_predictions, inference_time_ms: processing_time_ms, ood_detection, warning } = result;

  // Check if OOD detected
  const isOOD = ood_detection?.is_ood || false;

  return (
    <div className="w-full max-w-2xl mx-auto mt-8 animate-fade-in">
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl overflow-hidden">
        {/* Header */}
        <div className={`px-6 py-4 ${isOOD ? 'bg-gradient-to-r from-red-600 to-orange-600' : 'bg-gradient-to-r from-green-600 to-emerald-600'}`}>
          <h2 className="text-xl font-bold text-white">
            {isOOD ? 'Image Analysis - Warning' : 'Analysis Results'}
          </h2>
          <p className="text-white/90 text-sm mt-1">
            Processed in {processing_time_ms.toFixed(0)}ms
          </p>
        </div>

        <div className="p-6">
          {/* OOD Warning */}
          {isOOD && (
            <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border-2 border-red-300 dark:border-red-800">
              <div className="flex items-start">
                <svg className="w-6 h-6 text-red-600 dark:text-red-400 mt-0.5 mr-3 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
                <div className="flex-1">
                  <h5 className="font-bold text-red-800 dark:text-red-200 text-lg mb-2">
                    ⚠️ Not a Plant Leaf
                  </h5>
                  <p className="text-red-700 dark:text-red-300 mb-3">
                    {warning || ood_detection?.recommendation || 'This image does not appear to be a plant leaf.'}
                  </p>
                  
                  {/* OOD Detection Details */}
                  {ood_detection && (
                    <div className="mt-3 pt-3 border-t border-red-200 dark:border-red-800">
                      <p className="text-sm font-semibold text-red-800 dark:text-red-200 mb-2">
                        Detection Details:
                      </p>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div className="bg-white dark:bg-gray-900/50 p-2 rounded">
                          <span className="text-gray-600 dark:text-gray-400">Confidence:</span>
                          <span className="ml-2 font-semibold text-red-600 dark:text-red-400">
                            {(ood_detection.max_probability * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="bg-white dark:bg-gray-900/50 p-2 rounded">
                          <span className="text-gray-600 dark:text-gray-400">Entropy:</span>
                          <span className="ml-2 font-semibold text-red-600 dark:text-red-400">
                            {ood_detection.entropy.toFixed(2)}
                          </span>
                        </div>
                        <div className="bg-white dark:bg-gray-900/50 p-2 rounded col-span-2">
                          <span className="text-gray-600 dark:text-gray-400">Detection Votes:</span>
                          <span className="ml-2 font-semibold text-red-600 dark:text-red-400">
                            {ood_detection.in_distribution_votes}/{ood_detection.total_votes} methods agree it's a valid leaf
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  <p className="text-sm text-red-600 dark:text-red-400 mt-3 font-medium">
                    Please upload a clear image of a plant leaf for accurate disease detection.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Image thumbnail (always show) */}
          {imageUrl && (
            <div className="flex justify-center mb-6">
              <img
                src={imageUrl}
                alt="Analyzed image"
                className="max-w-xs rounded-lg shadow-md"
              />
            </div>
          )}

          {/* Only show predictions if NOT OOD */}
          {!isOOD && prediction && (
            <>
              {/* Main prediction card */}
              <div className="mb-6">
                {/* Prediction details */}
                <div>
                  <div className="flex items-center gap-3 mb-3">
                    <h3 className="text-2xl font-bold text-gray-800 dark:text-white">
                      {prediction.plant}
                    </h3>
                    <StatusBadge condition={prediction.condition} />
                  </div>

                  {prediction.condition.toLowerCase() !== 'healthy' && (
                    <p className="text-lg text-gray-600 dark:text-gray-300 mb-4">
                      Detected: <span className="font-semibold text-red-600 dark:text-red-400">{prediction.condition}</span>
                    </p>
                  )}

                  <ConfidenceBar
                    label="Confidence"
                    confidence={prediction.confidence}
                    isTop
                  />
                </div>
              </div>

              {/* Top predictions */}
              {top_predictions.length > 1 && (
                <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
                  <h4 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
                    Other Possibilities
                  </h4>
                  <div className="space-y-2">
                    {top_predictions.slice(1, 5).map((pred, idx) => (
                      <ConfidenceBar
                        key={idx}
                        label={`${pred.plant} - ${pred.condition}`}
                        confidence={pred.confidence}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* OOD Info for valid leaves (optional debug info) */}
              {ood_detection && !isOOD && (
                <div className="mt-4 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                  <p className="text-xs text-green-700 dark:text-green-300">
                    ✓ Valid leaf confirmed ({ood_detection.in_distribution_votes}/{ood_detection.total_votes} detection methods agree)
                  </p>
                </div>
              )}

              {/* Recommendations */}
              {prediction.condition.toLowerCase() !== 'healthy' && (
                <div className="mt-6 p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
                  <div className="flex items-start">
                    <svg className="w-5 h-5 text-amber-500 mt-0.5 mr-3 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                    </svg>
                    <div>
                      <h5 className="font-semibold text-amber-800 dark:text-amber-200">
                        Recommendation
                      </h5>
                      <p className="text-sm text-amber-700 dark:text-amber-300 mt-1">
                        This plant appears to have <strong>{prediction.condition}</strong>. 
                        Consider consulting with a plant pathologist or agricultural expert 
                        for proper treatment options.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {prediction.condition.toLowerCase() === 'healthy' && (
                <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                  <div className="flex items-start">
                    <svg className="w-5 h-5 text-green-500 mt-0.5 mr-3 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    <div>
                      <h5 className="font-semibold text-green-800 dark:text-green-200">
                        Great News!
                      </h5>
                      <p className="text-sm text-green-700 dark:text-green-300 mt-1">
                        Your <strong>{prediction.plant}</strong> appears to be healthy! 
                        Continue with regular care and monitoring.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
