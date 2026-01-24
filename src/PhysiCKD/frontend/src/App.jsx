import React, { useState } from 'react';
import Header from './components/Header';
import PredictionForm from './components/PredictionForm';
import ResultsDashboard from './components/ResultsDashboard';
import { generateMockPrediction } from './utils/predictionLogic';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [formData, setFormData] = useState({});

  const handlePrediction = (result, data) => {
    setFormData(data);
    setPrediction(result);
  };

  const handleClear = () => {
    setPrediction(null);
    setFormData({});
  };

  return (
    <div className="container">
      <Header />
      <div className="main-content">
        <PredictionForm onSubmit={handlePrediction} onClear={handleClear} />
        <section className="results-section">
          <div className="results-container">
            <h2>Prediction Results</h2>
            <ResultsDashboard prediction={prediction} formData={formData} />
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;
