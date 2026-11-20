import React, { useState } from 'react';
import { 
  RotateCcw, 
  Activity, 
  ShieldAlert, 
  Eye, 
  Sparkles, 
  FileText, 
  Info, 
  CheckCircle2, 
  AlertTriangle,
  Stethoscope,
  Maximize2
} from 'lucide-react';
import WaveformMotif from './common/WaveformMotif';
import { API_BASE_URL } from '../services/api';

const CLASS_EXPLANATIONS = {
  'Normal': {
    title: 'Normal Vesicular Breath Sounds',
    description: 'Clear, unhindered respiratory airflow throughout inhalation and exhalation without continuous wheezing or discontinuous crackling.'
  },
  'CAS': {
    title: 'Continuous Adventitious Sounds (Wheezes)',
    description: 'Musical, continuous airway vibrations produced by airflow through narrowed, inflamed, or obstructed tracheobronchial passages.'
  },
  'DAS': {
    title: 'Discontinuous Adventitious Sounds (Crackles)',
    description: 'Explosive, brief non-musical sounds typically caused by the sudden reopening of collapsed small airways or fluid accumulation in peripheral alveoli.'
  },
  'CAS & DAS': {
    title: 'Mixed Adventitious Sounds (Wheezes & Crackles)',
    description: 'Co-occurrence of both continuous musical wheezing and discontinuous crackling popping sounds within the same respiratory recording.'
  }
};

const DISEASE_EXPLANATIONS = {
  'Healthy Control': 'Acoustic respiratory profile consistent with healthy pediatric control patterns.',
  'Pneumonia': 'Acoustic signature displaying focal consolidation indicators and discontinuous high-energy bursts.',
  'Wheezing Disorders': 'Acoustic signature exhibiting continuous tonal airway impedance patterns.'
};

export default function AnalysisResults({ 
  prediction, 
  recording, 
  explanation, 
  explanationError,
  diseasePrediction,
  onReset 
}) {
  const [activeTab, setActiveTab] = useState('acoustic'); // 'acoustic' | 'disease' | 'interpretability'
  const [showGradCamModal, setShowGradCamModal] = useState(false);

  if (!prediction && !recording && !diseasePrediction) return null;

  const predictedClass = prediction?.predicted_class || 'Unknown';
  const probScores = prediction?.probability_scores || null;
  const soundMeta = CLASS_EXPLANATIONS[predictedClass] || {
    title: predictedClass,
    description: 'Acoustic auscultation sound category identified by the CNN model.'
  };

  const predictedDisease = diseasePrediction?.predicted_disease || null;
  const diseaseProbs = diseasePrediction?.probability_scores || null;
  const diseaseMeta = predictedDisease ? DISEASE_EXPLANATIONS[predictedDisease] : null;

  // Grad-CAM artifact URL
  const gradCamArtifactUrl = explanation?.artifact_path
    ? `${API_BASE_URL.replace(/\/+$/, '')}/${explanation.artifact_path.replace(/^\/+/, '')}`
    : null;

  return (
    <section className="results-signature-experience" id="analysis-results">
      
      {/* Editorial Header */}
      <div className="results-top-banner">
        <div>
          <div className="results-eyebrow">
            <span className="live-dot" /> Verified Acoustic Analysis Output
          </div>
          <h2 className="results-main-title">
            Intelligent Auscultation Findings
          </h2>
          <div className="results-meta-strip">
            <span className="meta-item">
              <strong>Source:</strong> {recording?.original_filename || 'Auscultation Audio'}
            </span>
            {recording?.anatomical_location && (
              <span className="meta-item">
                <strong>Site:</strong> {recording.anatomical_location.replace(/_/g, ' ')}
              </span>
            )}
            {recording?.patient_age_group && (
              <span className="meta-item">
                <strong>Age Group:</strong> {recording.patient_age_group}
              </span>
            )}
            {prediction?.inference_duration_ms && (
              <span className="meta-item font-mono">
                {prediction.inference_duration_ms}ms latency
              </span>
            )}
          </div>
        </div>

        <button 
          type="button" 
          onClick={onReset} 
          className="btn btn-secondary analyze-another-btn"
        >
          <RotateCcw size={15} /> Analyze Another
        </button>
      </div>

      {/* Mandatory Scientific Research Disclaimer (Dignified Warm Tone) */}
      <div className="research-disclaimer-box" role="note">
        <ShieldAlert size={20} className="disclaimer-icon" />
        <div className="disclaimer-text">
          <strong>Clinical Research Prototype & Disclaimer:</strong> This system is an investigational decision-support aid. Results represent probabilistic algorithmic estimations and do <strong>not</strong> constitute a medical diagnosis. All findings require independent evaluation by a licensed healthcare professional.
        </div>
      </div>

      {/* Primary Results Tabs */}
      <div className="results-tab-nav" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'acoustic'}
          onClick={() => setActiveTab('acoustic')}
          className={`tab-btn ${activeTab === 'acoustic' ? 'active' : ''}`}
        >
          <Activity size={16} />
          <span>Acoustic Classification (Model A)</span>
        </button>

        {diseasePrediction && (
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'disease'}
            onClick={() => setActiveTab('disease')}
            className={`tab-btn ${activeTab === 'disease' ? 'active' : ''}`}
          >
            <Stethoscope size={16} />
            <span>Disease Screening Indication (Model B)</span>
          </button>
        )}

        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'interpretability'}
          onClick={() => setActiveTab('interpretability')}
          className={`tab-btn ${activeTab === 'interpretability' ? 'active' : ''}`}
        >
          <Eye size={16} />
          <span>Grad-CAM Saliency & Spectrogram</span>
        </button>
      </div>

      {/* TAB 1: ACOUSTIC SOUND CLASSIFICATION (MODEL A) */}
      {activeTab === 'acoustic' && (
        <div className="tab-pane-content">
          
          <div className="acoustic-hero-card">
            <div className="acoustic-result-badge-col">
              <span className="model-label-pill">Model A — 2D-CNN Auscultation</span>
              
              <div className="predicted-sound-class">
                {predictedClass}
              </div>

              <div className="sound-class-friendly-title">
                {soundMeta.title}
              </div>

              <p className="sound-class-description">
                {soundMeta.description}
              </p>
            </div>

            {/* Probability Distribution */}
            <div className="probabilities-panel">
              <h4 className="prob-panel-heading">
                Softmax Class Probabilities
              </h4>
              <p className="prob-panel-sub">
                Acoustic feature confidence across all four sound categories:
              </p>

              {probScores ? (
                <div className="prob-list">
                  {Object.entries(probScores).map(([clsName, score]) => {
                    const numericScore = typeof score === 'number' ? score : parseFloat(score) || 0;
                    const percent = Math.round(numericScore * 100);
                    const isTop = clsName === predictedClass;

                    return (
                      <div key={clsName} className={`prob-row-item ${isTop ? 'highlighted' : ''}`}>
                        <div className="prob-item-header">
                          <span className="prob-item-name">
                            {clsName}
                            {clsName === 'CAS' && <small> (Wheezes)</small>}
                            {clsName === 'DAS' && <small> (Crackles)</small>}
                            {clsName === 'Normal' && <small> (Clear)</small>}
                            {clsName === 'CAS & DAS' && <small> (Mixed)</small>}
                          </span>
                          <span className="prob-item-value font-mono">
                            {percent}%
                          </span>
                        </div>
                        <div className="prob-track-bg">
                          <div 
                            className={`prob-bar-fill ${isTop ? 'top-fill' : ''}`}
                            style={{ width: `${percent}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="text-muted">Probability distribution unavailable.</p>
              )}
            </div>
          </div>

          {/* Model Provenance & Parameters (Progressive Disclosure) */}
          <div className="provenance-strip">
            <div className="provenance-item">
              <span className="prov-label">Model Registry ID</span>
              <span className="prov-val font-mono">{prediction?.model_version_str || 'sprsound-cnn-v1-fold0'}</span>
            </div>
            <div className="provenance-item">
              <span className="prov-label">Sampling Standard</span>
              <span className="prov-val">8,000 Hz, 16-bit Mono</span>
            </div>
            <div className="provenance-item">
              <span className="prov-label">Bandpass Window</span>
              <span className="prov-val">50 Hz – 2,000 Hz (Order 5)</span>
            </div>
            <div className="provenance-item">
              <span className="prov-label">Representation</span>
              <span className="prov-val">128 Mel-Bands Spectrogram</span>
            </div>
          </div>

        </div>
      )}

      {/* TAB 2: DISEASE SCREENING (MODEL B) — CLEARLY SEGREGATED */}
      {activeTab === 'disease' && (
        <div className="tab-pane-content">
          
          <div className="disease-screening-card">
            
            {/* Caution banner specific to disease screening */}
            <div className="disease-caution-callout">
              <Info size={18} />
              <div>
                <strong>Research Screening Indication Only:</strong> The pediatric disease model provides research-grade classification for epidemiological studies. It is strictly segregated from clinical diagnosis.
              </div>
            </div>

            <div className="disease-finding-layout">
              <div>
                <span className="model-label-pill">Model B — Pediatric Disease Screening</span>
                <div className="predicted-disease-title">
                  {predictedDisease || 'Screening In Progress'}
                </div>
                {diseaseMeta && (
                  <p className="disease-meta-desc">{diseaseMeta}</p>
                )}
              </div>

              {/* Disease Probabilities */}
              {diseaseProbs && (
                <div className="disease-probs-panel">
                  <h4 className="prob-panel-heading">Screening Likelihood Distribution</h4>
                  <div className="prob-list">
                    {Object.entries(diseaseProbs).map(([catName, score]) => {
                      const numericScore = typeof score === 'number' ? score : parseFloat(score) || 0;
                      const percent = Math.round(numericScore * 100);
                      const isTop = catName === predictedDisease;

                      return (
                        <div key={catName} className={`prob-row-item ${isTop ? 'highlighted' : ''}`}>
                          <div className="prob-item-header">
                            <span className="prob-item-name">{catName}</span>
                            <span className="prob-item-value font-mono">{percent}%</span>
                          </div>
                          <div className="prob-track-bg">
                            <div 
                              className={`prob-bar-fill ${isTop ? 'top-fill-disease' : ''}`}
                              style={{ width: `${percent}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

          </div>

        </div>
      )}

      {/* TAB 3: GRAD-CAM INTERPRETABILITY */}
      {activeTab === 'interpretability' && (
        <div className="tab-pane-content">
          
          <div className="interpretability-card">
            <div className="interpretability-header">
              <div>
                <h3 className="interpretability-title">
                  Grad-CAM Class Activation Saliency
                </h3>
                <p className="interpretability-sub">
                  Visualizes time-frequency coordinates in the Log-Mel Spectrogram that exerted the highest gradient influence on the CNN's decision.
                </p>
              </div>

              {gradCamArtifactUrl && (
                <button
                  type="button"
                  onClick={() => setShowGradCamModal(true)}
                  className="btn btn-ghost"
                  title="Expand Heatmap"
                >
                  <Maximize2 size={15} /> Expand View
                </button>
              )}
            </div>

            {gradCamArtifactUrl ? (
              <div className="gradcam-image-wrapper">
                <img 
                  src={gradCamArtifactUrl} 
                  alt="Grad-CAM Saliency Spectrogram" 
                  className="gradcam-img"
                />
                
                <div className="gradcam-legend-bar">
                  <span className="legend-label">Low Saliency</span>
                  <div className="legend-gradient" />
                  <span className="legend-label">High Saliency (Decisive)</span>
                </div>

                <div className="gradcam-caption">
                  <strong>Reading this visualization:</strong> Warmer bands (yellow to deep red) indicate specific frequency ranges and time windows where anomalous auscultation patterns (e.g. musical wheeze harmonics or crackle bursts) activated the convolutional filters.
                </div>
              </div>
            ) : (
              <div className="gradcam-pending-state">
                <Info size={28} className="text-muted" />
                <h4>Grad-CAM Heatmap Not Available</h4>
                <p>
                  {explanationError || 'Grad-CAM interpretation has not been generated for this recording attempt, or the explanation service is pending.'}
                </p>
              </div>
            )}
          </div>

        </div>
      )}

      {/* Grad-CAM Fullscreen Modal */}
      {showGradCamModal && gradCamArtifactUrl && (
        <div className="modal-backdrop" onClick={() => setShowGradCamModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h4>Grad-CAM Spectrogram Saliency (High Resolution)</h4>
              <button 
                type="button" 
                onClick={() => setShowGradCamModal(false)}
                className="btn btn-ghost"
              >
                Close
              </button>
            </div>
            <div style={{ textAlign: 'center', padding: '1.5rem' }}>
              <img 
                src={gradCamArtifactUrl} 
                alt="High-resolution Grad-CAM Spectrogram" 
                style={{ maxWidth: '100%', maxHeight: '75vh', borderRadius: 'var(--radius-md)' }}
              />
            </div>
          </div>
        </div>
      )}

    </section>
  );
}
