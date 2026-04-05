import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => setFile(e.target.files[0]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult(null);
    setError(null);

    if (!file) {
      setError('Please upload an image.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:7860/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Error with API request');
      }

      const data = await response.json();
      setResult(data);
      setFile(null);
    } catch (err) {
      setError('Failed to get response.');
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>PlantHealth AI</h1>
        <p>Upload a photo of a plant leaf to diagnose diseases.</p>

        <form onSubmit={handleSubmit}>
          <input type="file" accept="image/*" onChange={handleChange} />
          <button type="submit">Submit</button>
        </form>

        {result && (
          <div className="result">
            <h2>Prediction</h2>
            <p><strong>Class:</strong> {result.class}</p>
            <p><strong>Confidence:</strong> {result.confidence}</p>
          </div>
        )}

        {error && <p className="error">{error}</p>}
      </header>
    </div>
  );
}

export default App;