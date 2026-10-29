import React from 'react';
import { 
  CheckCircle2, 
  RotateCcw, 
  Eye, 
  BarChart3,
  Stethoscope,
  Activity,
  Layers
} from 'lucide-react';
import { API_BASE_URL } from '../services/api';

export default function AnalysisResults({ 
  prediction, 
  recording, 
  explanation, 
  explanationError,
  diseasePrediction,
  onReset 
}) {
  if (!prediction && !recording && !diseasePrediction) return null;

  const predictedClass = prediction?.predicted_class || 'Unknown';
  const probScores = prediction?.probability_scores || null;

  const predictedDisease = diseasePrediction?.predicted_disease || 'Screening In Progress';
  const diseaseProbs = diseasePrediction?.probability_scores || null;

  // Grad-CAM artifact image URL
  const gradCamArtifactUrl = explanation?.artifact_path
    ? `${API_BASE_URL.replace(/\/+$/, '')}/${explanation.artifact_path.replace(/^\/+/, '')}`
    : null;

  return (
    <section className="results-container glass-panel" id="analysis-results" style={{ padding: '1.5rem', background: '#ffffff', borderRadius: '12px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05)' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', borderBottom: '1px solid #f1f5f9', paddingBottom: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '1.35rem', fontWeight: 700, margin: 0, color: '#0f172a' }}>
            AI Prediction Results
          </h2>
          <span style={{ fontSize: '0.85rem', color: '#64748b' }}>
            Audio: {recording?.original_filename || 'Recording'}
          </span>
        </div>

        <button 
          type="button" 
          onClick={onReset} 
          className="btn btn-secondary"
          style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
        >
          <RotateCcw size={15} /> Analyze Another
        </button>
      </div>

      {/* Grid: Model A & Model B Predictions */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.25rem', marginBottom: '1.5rem' }}>
        
        {/* Model A: Sound Classification */}
        <div style={{ border: '1px solid #e2e8f0', borderRadius: '10px', padding: '1.25rem', background: '#f8fafc' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '0.75rem', color: '#0284c7' }}>
            <Activity size={18} />
            <span style={{ fontSize: '0.85rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Model A: Sound Category
            </span>
          </div>

          <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#0f172a', marginBottom: '1rem' }}>
            {predictedClass}
          </div>

          {/* Probabilities */}
          {probScores && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {Object.entries(probScores).map(([clsName, score]) => {
                const numericScore = typeof score === 'number' ? score : parseFloat(score) || 0;
                const percent = Math.round(numericScore * 100);
                const isTop = clsName === predictedClass;

                return (
                  <div key={clsName}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: isTop ? 700 : 500, color: isTop ? '#0f172a' : '#64748b', marginBottom: '3px' }}>
                      <span>{clsName}</span>
                      <span className="font-mono">{percent}%</span>
                    </div>
                    <div style={{ height: '7px', background: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
                      <div 
                        style={{ 
                          width: `${percent}%`, 
                          height: '100%', 
                          background: isTop ? '#0284c7' : '#cbd5e1', 
                          borderRadius: '4px',
                          transition: 'width 0.4s ease'
                        }} 
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Model B: Respiratory Disease Prediction */}
        <div style={{ border: '1px solid #e2e8f0', borderRadius: '10px', padding: '1.25rem', background: '#faf5ff' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '0.75rem', color: '#7c3aed' }}>
            <Stethoscope size={18} />
            <span style={{ fontSize: '0.85rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Model B: Disease Prediction
            </span>
          </div>

          <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#0f172a', marginBottom: '1rem' }}>
            {predictedDisease}
          </div>

          {/* Probabilities */}
          {diseaseProbs ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {Object.entries(diseaseProbs).map(([dName, score]) => {
                const numericScore = typeof score === 'number' ? score : parseFloat(score) || 0;
                const percent = Math.round(numericScore * 100);
                const isTop = dName === predictedDisease;

                let barColor = '#7c3aed';
                if (dName.toLowerCase().includes('healthy')) barColor = '#10b981';
                else if (dName.toLowerCase().includes('pneumonia')) barColor = '#ef4444';
                else if (dName.toLowerCase().includes('wheez')) barColor = '#f59e0b';

                return (
                  <div key={dName}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: isTop ? 700 : 500, color: isTop ? '#0f172a' : '#64748b', marginBottom: '3px' }}>
                      <span>{dName}</span>
                      <span className="font-mono">{percent}%</span>
                    </div>
                    <div style={{ height: '7px', background: '#e9d5ff', borderRadius: '4px', overflow: 'hidden' }}>
                      <div 
                        style={{ 
                          width: `${percent}%`, 
                          height: '100%', 
                          background: isTop ? barColor : '#cbd5e1', 
                          borderRadius: '4px',
                          transition: 'width 0.4s ease'
                        }} 
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div style={{ color: '#64748b', fontSize: '0.9rem' }}>Evaluating disease probability scores...</div>
          )}
        </div>

      </div>

      {/* Visual Interpretability (Heatmap) */}
      {gradCamArtifactUrl && (
        <div style={{ border: '1px solid #e2e8f0', borderRadius: '10px', padding: '1.25rem', marginBottom: '1.5rem', background: '#f8fafc' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '0.75rem', color: '#0f172a', fontWeight: 600, fontSize: '0.95rem' }}>
            <Eye size={18} color="#d97706" />
            <span>Grad-CAM Time-Frequency Saliency Heatmap</span>
          </div>
          <div style={{ textAlign: 'center', background: '#ffffff', padding: '8px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
            <img 
              src={gradCamArtifactUrl} 
              alt={`Grad-CAM heatmap for ${predictedClass}`} 
              style={{ maxWidth: '100%', maxHeight: '280px', objectFit: 'contain', borderRadius: '6px' }} 
            />
          </div>
        </div>
      )}

      {/* Bottom Minimal Notice */}
      <div style={{ fontSize: '0.8rem', color: '#94a3b8', textAlign: 'center' }}>
        Research-assistance predictions only. Not a medical diagnosis.
      </div>
    </section>
  );
}
