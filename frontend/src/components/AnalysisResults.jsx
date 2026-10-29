import React from 'react';
import { 
  CheckCircle2, 
  RotateCcw, 
  FileAudio, 
  Clock, 
  Layers, 
  Eye, 
  Info, 
  AlertTriangle, 
  BarChart3,
  Sliders
} from 'lucide-react';
import { getCategoryDetails } from '../utils/clinicalCategories';
import { API_BASE_URL } from '../services/api';

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

export default function AnalysisResults({ 
  prediction, 
  recording, 
  explanation, 
  explanationError,
  onReset 
}) {
  if (!prediction && !recording) return null;

  const predictedClass = prediction?.predicted_class || 'Pending Classification';
  const categoryInfo = getCategoryDetails(predictedClass);
  const probScores = prediction?.probability_scores || null;
  const hasProbabilities = probScores && Object.keys(probScores).length > 0;

  // Resolve Grad-CAM artifact URL safely
  const gradCamArtifactUrl = explanation?.artifact_path
    ? `${API_BASE_URL.replace(/\/+$/, '')}/${explanation.artifact_path.replace(/^\/+/, '')}`
    : null;

  return (
    <section className="results-container glass-panel" id="analysis-results">
      {/* Results Header */}
      <div className="results-header">
        <div className="results-header-left">
          <span className="badge badge-emerald">
            <CheckCircle2 size={14} /> Diagnostic Acoustic Evaluation
          </span>
          <h2 className="results-title">Auscultation Analysis Results</h2>
          <p className="results-subtitle">
            Feature extraction, deep convolutional pattern classification, and Grad-CAM interpretability.
          </p>
        </div>

        <button 
          type="button" 
          onClick={onReset} 
          className="btn-secondary results-reset-btn"
          id="btn-analyze-another-top"
        >
          <RotateCcw size={15} /> Analyze Another Recording
        </button>
      </div>

      {/* Main Prediction Banner Card */}
      <div className="prediction-banner-card">
        <div className="prediction-banner-top">
          <div className="prediction-category-group">
            <span className="prediction-label-prefix">Predicted Respiratory Category</span>
            <div className="prediction-category-title-row">
              <h3 className="prediction-category-name">{categoryInfo?.label || predictedClass}</h3>
              <span className={`badge ${categoryInfo?.badgeClass || 'badge-cyan'} text-sm`}>
                {predictedClass}
              </span>
            </div>
          </div>

          {prediction?.inference_duration_ms !== undefined && prediction?.inference_duration_ms !== null && (
            <div className="prediction-latency-badge font-mono">
              <Clock size={13} /> {prediction.inference_duration_ms} ms forward pass
            </div>
          )}
        </div>

        {/* Plain-English Clinical Explanation */}
        <div className="prediction-explanation-box">
          <div className="explanation-icon-col">
            <Info size={18} className="text-cyan" />
          </div>
          <div className="explanation-text-content">
            <div className="explanation-lead font-semibold">Clinical Acoustic Interpretation</div>
            <p className="explanation-body">
              {categoryInfo?.clinicalSummary || 'Acoustic pattern evaluated against pediatric auscultation dataset.'}
            </p>
            <div className="acoustic-props-row">
              <div className="acoustic-prop-item">
                <span className="prop-name">Sound Profile:</span>
                <span className="prop-val">{categoryInfo?.soundType || 'Acoustic signal'}</span>
              </div>
              <div className="acoustic-prop-item">
                <span className="prop-name">Typical Duration:</span>
                <span className="prop-val">{categoryInfo?.duration || 'Variable'}</span>
              </div>
              <div className="acoustic-prop-item">
                <span className="prop-name">Bandpass Spectrum:</span>
                <span className="prop-val">{categoryInfo?.frequencyRange || '50 Hz – 2000 Hz'}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Class Probabilities Distribution (Rendered ONLY if returned by backend) */}
      {hasProbabilities && (
        <div className="results-subcard">
          <div className="subcard-header">
            <div className="subcard-title-group">
              <BarChart3 size={18} className="text-cyan" />
              <h4 className="subcard-title">Class Softmax Probabilities</h4>
            </div>
            <span className="subcard-hint">
              Output scores returned directly from active CNN softmax head
            </span>
          </div>

          <div className="probability-bars-grid">
            {Object.entries(probScores).map(([clsName, score]) => {
              const numericScore = typeof score === 'number' ? score : parseFloat(score) || 0;
              const percent = Math.round(numericScore * 100);
              const isTop = clsName === predictedClass;
              const itemCategory = getCategoryDetails(clsName);

              return (
                <div 
                  key={clsName} 
                  className={`probability-item ${isTop ? 'probability-item-top' : ''}`}
                >
                  <div className="prob-label-row">
                    <span className="prob-class-name">
                      {clsName}
                      {isTop && <span className="top-indicator">Predicted</span>}
                    </span>
                    <span className="prob-percentage font-mono">{percent}%</span>
                  </div>

                  <div className="prob-progress-track">
                    <div 
                      className={`prob-progress-fill ${isTop ? 'fill-accent' : 'fill-muted'}`}
                      style={{ 
                        width: `${Math.max(percent, 2)}%`,
                        backgroundColor: isTop ? itemCategory?.themeColor : undefined 
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Technical Recording Details Grid */}
      {recording && (
        <div className="results-subcard">
          <div className="subcard-header">
            <div className="subcard-title-group">
              <FileAudio size={18} className="text-cyan" />
              <h4 className="subcard-title">Audio Stream Properties</h4>
            </div>
            <span className="subcard-hint">
              Validated acoustic parameters stored in database
            </span>
          </div>

          <div className="metadata-metrics-grid">
            <div className="meta-metric-card">
              <span className="meta-metric-label">Original Filename</span>
              <span className="meta-metric-value font-mono text-ellipsis" title={recording.original_filename}>
                {recording.original_filename || 'Audio Recording'}
              </span>
            </div>

            <div className="meta-metric-card">
              <span className="meta-metric-label">Duration</span>
              <span className="meta-metric-value font-mono">
                {recording.duration_seconds ? `${recording.duration_seconds} s` : 'N/A'}
              </span>
            </div>

            <div className="meta-metric-card">
              <span className="meta-metric-label">Sampling Rate</span>
              <span className="meta-metric-value font-mono">
                {recording.sample_rate ? `${recording.sample_rate} Hz` : 'N/A'}
              </span>
            </div>

            <div className="meta-metric-card">
              <span className="meta-metric-label">Channels</span>
              <span className="meta-metric-value font-mono">
                {recording.channels === 1 ? 'Mono (1 Ch)' : recording.channels === 2 ? 'Stereo (2 Ch)' : `${recording.channels} Ch`}
              </span>
            </div>

            <div className="meta-metric-card">
              <span className="meta-metric-label">File Size</span>
              <span className="meta-metric-value font-mono">
                {formatBytes(recording.file_size_bytes)}
              </span>
            </div>

            <div className="meta-metric-card">
              <span className="meta-metric-label">Format Signature</span>
              <span className="meta-metric-value font-mono uppercase">
                {recording.file_format || 'WAV'}
              </span>
            </div>

            {recording.anatomical_location && (
              <div className="meta-metric-card">
                <span className="meta-metric-label">Anatomical Location</span>
                <span className="meta-metric-value">
                  {recording.anatomical_location_display || recording.anatomical_location}
                </span>
              </div>
            )}

            {recording.patient_age_group && (
              <div className="meta-metric-card">
                <span className="meta-metric-label">Patient Age Group</span>
                <span className="meta-metric-value">
                  {recording.patient_age_group_display || recording.patient_age_group}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Visualizations: Mel-Spectrogram & Grad-CAM Heatmap */}
      <div className="results-visualizations-row">
        {/* Mel-Spectrogram Panel */}
        <div className="visualization-card">
          <div className="viz-card-header">
            <div className="viz-card-title-group">
              <Sliders size={18} className="text-cyan" />
              <h4 className="viz-card-title">Time-Frequency Mel-Spectrogram</h4>
            </div>
            <span className="badge badge-cyan text-xs">Acoustic Feature</span>
          </div>

          <div className="viz-card-body">
            {recording?.spectrogram_url ? (
              <div className="viz-image-wrapper">
                <img 
                  src={recording.spectrogram_url} 
                  alt="Mel-Spectrogram visualization" 
                  className="viz-image"
                />
              </div>
            ) : (
              <div className="viz-status-banner">
                <div className="viz-status-icon">
                  <Layers size={24} className="text-cyan" />
                </div>
                <div className="viz-status-text">
                  <div className="viz-status-heading">In-Memory Preprocessing Active</div>
                  <p className="viz-status-desc">
                    Raw auscultation waveforms are filtered with a 5th-order Butterworth bandpass (50–2000 Hz) 
                    and transformed into 128 Mel-frequency bins internally for the 2D-CNN input tensor. 
                    Image export is not configured on this endpoint.
                  </p>
                </div>
              </div>
            )}
            
            <div className="viz-spec-specs font-mono">
              <span>n_mels: 128</span>
              <span>n_fft: 1024</span>
              <span>hop_length: 512</span>
              <span>f_min: 50 Hz</span>
              <span>f_max: 2000 Hz</span>
            </div>
          </div>
        </div>

        {/* Grad-CAM Saliency Heatmap Panel */}
        <div className="visualization-card">
          <div className="viz-card-header">
            <div className="viz-card-title-group">
              <Eye size={18} className="text-amber" />
              <h4 className="viz-card-title">Grad-CAM Saliency Explanation</h4>
            </div>
            <span className="badge badge-amber text-xs">Explainable AI</span>
          </div>

          <div className="viz-card-body">
            {gradCamArtifactUrl ? (
              <div className="viz-image-wrapper">
                <img 
                  src={gradCamArtifactUrl} 
                  alt={`Grad-CAM heatmap for ${predictedClass}`} 
                  className="viz-image"
                />
                <div className="viz-image-caption">
                  <span>Target Class: <strong className="font-mono">{predictedClass}</strong></span>
                  {explanation?.structured_data?.conv_layer && (
                    <span>Layer: <code className="inline-code">{explanation.structured_data.conv_layer}</code></span>
                  )}
                </div>
              </div>
            ) : (
              <div className="viz-status-banner viz-status-banner-warning">
                <div className="viz-status-icon">
                  <AlertTriangle size={24} className="text-amber" />
                </div>
                <div className="viz-status-text">
                  <div className="viz-status-heading">Grad-CAM Artifact Status</div>
                  <p className="viz-status-desc">
                    {explanationError || explanation?.error_details || (
                      "Grad-CAM requires access to the trained convolutional layers and weights of the CNN model. Authentic saliency heatmaps are generated only from backpropagated convolutional gradients and are never synthesized."
                    )}
                  </p>
                </div>
              </div>
            )}

            <div className="interpretability-disclaimer">
              <Info size={14} className="shrink-0 text-cyan mt-0.5" />
              <span>
                Grad-CAM highlights time-frequency regions in the Mel-Spectrogram that contributed 
                most heavily to the model's classification. It is an algorithmic interpretability tool 
                and does not constitute medical advice or clinical diagnosis.
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Action Section */}
      <div className="results-footer-actions">
        <button 
          type="button" 
          onClick={onReset} 
          className="btn-primary"
          id="btn-analyze-another-bottom"
        >
          <RotateCcw size={16} /> Analyze Another Recording
        </button>
      </div>
    </section>
  );
}
