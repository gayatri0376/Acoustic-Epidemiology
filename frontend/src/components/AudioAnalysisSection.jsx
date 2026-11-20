import React, { useState, useRef, useCallback } from 'react';
import { 
  UploadCloud, 
  FileAudio, 
  AlertCircle, 
  AlertTriangle, 
  Sparkles, 
  X, 
  Loader2,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  CheckCircle2,
  ShieldCheck,
  Stethoscope,
  Activity
} from 'lucide-react';
import AudioPlayer from './AudioPlayer';
import WaveformMotif from './common/WaveformMotif';
import { 
  uploadAudioRecording, 
  requestPrediction, 
  requestExplanation, 
  requestDiseasePrediction 
} from '../services/api';

const MAX_FILE_SIZE = 25 * 1024 * 1024; // 25 MB
const ALLOWED_EXTENSIONS = ['.wav', '.flac', '.ogg', '.mp3'];

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

const ANATOMICAL_SITES = [
  { value: 'TRACHEA', label: 'Trachea' },
  { value: 'ANTERIOR_UPPER_RIGHT', label: 'Anterior Upper Right' },
  { value: 'ANTERIOR_UPPER_LEFT', label: 'Anterior Upper Left' },
  { value: 'ANTERIOR_LOWER_RIGHT', label: 'Anterior Lower Right' },
  { value: 'ANTERIOR_LOWER_LEFT', label: 'Anterior Lower Left' },
  { value: 'POSTERIOR_UPPER_RIGHT', label: 'Posterior Upper Right' },
  { value: 'POSTERIOR_UPPER_LEFT', label: 'Posterior Upper Left' },
  { value: 'POSTERIOR_LOWER_RIGHT', label: 'Posterior Lower Right' },
  { value: 'POSTERIOR_LOWER_LEFT', label: 'Posterior Lower Left' },
  { value: 'LATERAL_RIGHT', label: 'Lateral Right' },
  { value: 'LATERAL_LEFT', label: 'Lateral Left' },
];

const AGE_GROUPS = [
  { value: 'NEONATE', label: 'Neonate (0–28 days)' },
  { value: 'INFANT', label: 'Infant (1–12 months)' },
  { value: 'TODDLER', label: 'Toddler (1–3 years)' },
  { value: 'PRESCHOOL', label: 'Preschool (3–5 years)' },
  { value: 'SCHOOL_AGE', label: 'School-age (6–12 years)' },
  { value: 'ADOLESCENT', label: 'Adolescent (13–18 years)' },
];

export default function AudioAnalysisSection({ 
  token, 
  onRequireAuth, 
  onAnalysisSuccess,
  isAnalyzing,
  setIsAnalyzing
}) {
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [fileDuration, setFileDuration] = useState(null);
  const [validationError, setValidationError] = useState(null);
  
  // Real processing pipeline stages: 'uploading' | 'feature_extraction' | 'cnn_inference' | 'disease_screening' | 'gradcam_saliency'
  const [currentStage, setCurrentStage] = useState(null);
  const [stageProgress, setStageProgress] = useState({});
  const [serverError, setServerError] = useState(null);
  const [modelNotice, setModelNotice] = useState(null);
  
  // Optional clinical auscultation metadata
  const [showMeta, setShowMeta] = useState(false);
  const [anatomicalLocation, setAnatomicalLocation] = useState('');
  const [patientAgeGroup, setPatientAgeGroup] = useState('');
  const [patientPseudoId, setPatientPseudoId] = useState('');

  const validateAudioFile = useCallback((file) => {
    if (!file) return 'No audio file selected.';

    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      return `Unsupported file format (${ext}). Supported formats: ${ALLOWED_EXTENSIONS.join(', ')}.`;
    }

    if (file.size > MAX_FILE_SIZE) {
      return `File size (${formatBytes(file.size)}) exceeds the maximum allowed limit of 25 MB.`;
    }

    return null;
  }, []);

  const handleFileSelection = (file) => {
    setValidationError(null);
    setServerError(null);
    setModelNotice(null);

    const err = validateAudioFile(file);
    if (err) {
      setValidationError(err);
      setSelectedFile(null);
      setFileDuration(null);
      return;
    }

    setSelectedFile(file);
    setFileDuration(null);
  };

  const handleInputChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelection(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleFileSelection(file);
    }
  };

  const handleResetFile = () => {
    setSelectedFile(null);
    setFileDuration(null);
    setValidationError(null);
    setServerError(null);
    setModelNotice(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Run the multi-stage real acoustic analysis pipeline
  const handleStartAnalysis = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setServerError(null);
    setModelNotice(null);
    setStageProgress({});

    try {
      // Stage 1: Upload and Ingestion
      setCurrentStage('uploading');
      setStageProgress(prev => ({ ...prev, uploading: 'active' }));

      const formData = new FormData();
      formData.append('audio_file', selectedFile);
      if (anatomicalLocation) formData.append('anatomical_location', anatomicalLocation);
      if (patientAgeGroup) formData.append('patient_age_group', patientAgeGroup);
      if (patientPseudoId.trim()) formData.append('patient_pseudo_id', patientPseudoId.trim());

      const uploadRes = await uploadAudioRecording(formData, token);
      if (!uploadRes.success) {
        if (uploadRes.statusCode === 401 && onRequireAuth) {
          onRequireAuth();
          return;
        }
        throw new Error(uploadRes.error || uploadRes.data?.error || 'Auscultation file upload and validation failed.');
      }

      setStageProgress(prev => ({ ...prev, uploading: 'done' }));
      const recording = uploadRes.data;

      // Stage 2: Time-Frequency Preprocessing & Acoustic CNN Classification (Model A)
      setCurrentStage('cnn_inference');
      setStageProgress(prev => ({ ...prev, cnn_inference: 'active' }));

      const predRes = await requestPrediction(recording.id, token);
      if (!predRes.success) {
        if (predRes.isModelUnavailable) {
          setModelNotice('Acoustic CNN model is pending deployment or offline. Recording has been securely saved.');
          setIsAnalyzing(false);
          return;
        }
        throw new Error(predRes.errorMessage || 'Acoustic CNN classification encountered an error.');
      }

      setStageProgress(prev => ({ ...prev, cnn_inference: 'done' }));
      const prediction = predRes.data;

      // Stage 3: Pediatric Disease Screening Model (Model B)
      setCurrentStage('disease_screening');
      setStageProgress(prev => ({ ...prev, disease_screening: 'active' }));

      let diseasePrediction = null;
      try {
        const diseaseRes = await requestDiseasePrediction(recording.id, token);
        if (diseaseRes.success) {
          diseasePrediction = diseaseRes.data;
          setStageProgress(prev => ({ ...prev, disease_screening: 'done' }));
        } else {
          setStageProgress(prev => ({ ...prev, disease_screening: 'unavailable' }));
        }
      } catch (err) {
        setStageProgress(prev => ({ ...prev, disease_screening: 'unavailable' }));
      }

      // Stage 4: Grad-CAM Explainability Layer
      setCurrentStage('gradcam_saliency');
      setStageProgress(prev => ({ ...prev, gradcam_saliency: 'active' }));

      let explanation = null;
      let explanationError = null;

      try {
        const expRes = await requestExplanation(prediction.id, token, 'GRAD_CAM');
        if (expRes.success) {
          explanation = expRes.data;
          setStageProgress(prev => ({ ...prev, gradcam_saliency: 'done' }));
        } else {
          explanationError = expRes.errorMessage || 'Grad-CAM interpretation is pending or unsupported for this model version.';
          setStageProgress(prev => ({ ...prev, gradcam_saliency: 'unavailable' }));
        }
      } catch (err) {
        explanationError = err.message;
        setStageProgress(prev => ({ ...prev, gradcam_saliency: 'unavailable' }));
      }

      // Complete
      setCurrentStage('complete');
      if (onAnalysisSuccess) {
        onAnalysisSuccess({
          recording,
          prediction,
          diseasePrediction,
          explanation,
          explanationError
        });
      }
    } catch (err) {
      setServerError(err.message || 'An unexpected failure occurred during acoustic analysis.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="audio-upload-experience">
      
      {/* Hidden file input */}
      <input 
        ref={fileInputRef}
        type="file" 
        accept={ALLOWED_EXTENSIONS.join(',')}
        onChange={handleInputChange}
        className="hidden-file-input"
        aria-hidden="true"
        disabled={isAnalyzing}
      />

      {/* Validation Error Banner */}
      {validationError && (
        <div className="alert alert-error" role="alert" style={{ marginBottom: '1rem' }}>
          <AlertCircle size={18} />
          <div style={{ flex: 1 }}>
            <strong>Invalid Audio File:</strong> {validationError}
          </div>
          <button 
            type="button" 
            onClick={() => setValidationError(null)} 
            className="alert-close-btn"
            aria-label="Dismiss alert"
          >
            <X size={15} />
          </button>
        </div>
      )}

      {/* Server / Network Error Banner */}
      {serverError && (
        <div className="alert alert-error" role="alert" style={{ marginBottom: '1rem' }}>
          <AlertTriangle size={18} />
          <div style={{ flex: 1 }}>
            <strong>Analysis Failed:</strong> {serverError}
          </div>
          <button 
            type="button" 
            onClick={() => setServerError(null)} 
            className="alert-close-btn"
            aria-label="Dismiss alert"
          >
            <X size={15} />
          </button>
        </div>
      )}

      {/* Notice Banner */}
      {modelNotice && (
        <div className="alert alert-warn" role="alert" style={{ marginBottom: '1rem' }}>
          <AlertCircle size={18} />
          <div>{modelNotice}</div>
        </div>
      )}

      {/* STEP 1: DROPZONE (When no file is selected) */}
      {!selectedFile && (
        <div 
          className={`upload-signature-dropzone ${isDragOver ? 'drag-over' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click(); }}
          aria-label="Select or drop pediatric auscultation audio recording"
        >
          {/* Subtle respiratory wave backdrop */}
          <div className="dropzone-wave-backdrop">
            <WaveformMotif variant="hero" height={60} color="var(--accent-eucalyptus)" />
          </div>

          <div className="dropzone-icon-ring">
            <UploadCloud size={28} />
          </div>

          <div className="dropzone-content">
            <h3 className="dropzone-heading">
              Select or Drop Respiratory Recording
            </h3>
            <p className="dropzone-sub">
              Upload pediatric auscultation sound in <span className="format-highlight">.wav</span>, <span className="format-highlight">.flac</span>, <span className="format-highlight">.ogg</span>, or <span className="format-highlight">.mp3</span> (up to 25 MB).
            </p>
          </div>

          <div className="dropzone-badge-row">
            <span className="spec-badge">
              <ShieldCheck size={13} /> Strict 16-bit PCM Verification
            </span>
            <span className="spec-badge">
              <Activity size={13} /> 50–2000 Hz Butterworth Filter
            </span>
          </div>
        </div>
      )}

      {/* STEP 2: SELECTED FILE INSPECTION & WAVEFORM PREVIEW */}
      {selectedFile && !isAnalyzing && (
        <div className="selected-audio-workspace">
          
          {/* Header row with file info and actions */}
          <div className="selected-audio-header">
            <div className="file-info-col">
              <div className="file-icon-square">
                <FileAudio size={22} />
              </div>
              <div>
                <h4 className="file-name-display">{selectedFile.name}</h4>
                <div className="file-meta-pills">
                  <span className="file-pill font-mono">{formatBytes(selectedFile.size)}</span>
                  <span className="file-pill">Format: {selectedFile.name.split('.').pop().toUpperCase()}</span>
                  {fileDuration && (
                    <span className="file-pill font-mono">Duration: {fileDuration.toFixed(1)}s</span>
                  )}
                </div>
              </div>
            </div>

            <div className="file-action-buttons">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="btn btn-ghost"
                title="Select a different audio recording"
              >
                <RefreshCw size={14} /> Replace
              </button>
              <button
                type="button"
                onClick={handleResetFile}
                className="btn btn-ghost text-danger"
                title="Remove recording"
              >
                <X size={15} /> Remove
              </button>
            </div>
          </div>

          {/* Interactive Audio Waveform & Player */}
          <div className="audio-preview-container">
            <div className="preview-label">
              <span>Acoustic Waveform Inspection</span>
              <span className="preview-hint">Click along the waveform to scrub audio</span>
            </div>
            <AudioPlayer 
              file={selectedFile} 
              onDurationDetected={(dur) => setFileDuration(dur)} 
            />
          </div>

          {/* Collapsible Clinical Auscultation Metadata (Progressive Disclosure) */}
          <div className="clinical-meta-accordion">
            <button
              type="button"
              onClick={() => setShowMeta(!showMeta)}
              className="meta-accordion-toggle"
              aria-expanded={showMeta}
            >
              <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Stethoscope size={16} color="var(--accent-primary)" />
                <strong>Clinical Auscultation Context</strong> 
                <span className="badge-optional">Optional</span>
              </span>
              {showMeta ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>

            {showMeta && (
              <div className="meta-accordion-body">
                <div className="meta-form-grid">
                  
                  <div className="field-group">
                    <label htmlFor="patient-pseudo-id">Patient Research ID</label>
                    <input 
                      id="patient-pseudo-id"
                      type="text" 
                      placeholder="e.g. PED-2026-084"
                      value={patientPseudoId}
                      onChange={(e) => setPatientPseudoId(e.target.value)}
                      className="input-text"
                    />
                    <small className="field-hint">De-identified pseudonymized research tag.</small>
                  </div>

                  <div className="field-group">
                    <label htmlFor="anatomical-site">Auscultation Anatomical Site</label>
                    <select
                      id="anatomical-site"
                      value={anatomicalLocation}
                      onChange={(e) => setAnatomicalLocation(e.target.value)}
                      className="input-select"
                    >
                      <option value="">Unspecified Auscultation Site</option>
                      {ANATOMICAL_SITES.map(s => (
                        <option key={s.value} value={s.value}>{s.label}</option>
                      ))}
                    </select>
                    <small className="field-hint">Chest wall location where sound was captured.</small>
                  </div>

                  <div className="field-group">
                    <label htmlFor="pediatric-age">Pediatric Age Category</label>
                    <select
                      id="pediatric-age"
                      value={patientAgeGroup}
                      onChange={(e) => setPatientAgeGroup(e.target.value)}
                      className="input-select"
                    >
                      <option value="">Unspecified Pediatric Age</option>
                      {AGE_GROUPS.map(g => (
                        <option key={g.value} value={g.value}>{g.label}</option>
                      ))}
                    </select>
                    <small className="field-hint">Developmental stage of pediatric patient.</small>
                  </div>

                </div>
              </div>
            )}
          </div>

          {/* Primary Action Button: Analyze Recording */}
          <div className="analyze-action-bar">
            <button
              type="button"
              onClick={handleStartAnalysis}
              className="btn btn-primary btn-large signature-analyze-btn"
            >
              <Sparkles size={18} />
              <span>Analyze Auscultation Recording</span>
            </button>
            <p className="analyze-reassurance">
              Executes Butterworth 50–2000 Hz filtering, Log-Mel Spectrogram extraction, and dual CNN inference.
            </p>
          </div>

        </div>
      )}

      {/* STEP 3: TRANSPARENT GUIDED ANALYSIS FLOW (During processing) */}
      {isAnalyzing && (
        <div className="guided-analysis-progress-card">
          <div className="progress-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <Loader2 size={20} className="spinner-teal" />
              <h4 style={{ margin: 0 }}>Intelligent Acoustic Analysis in Progress</h4>
            </div>
            <WaveformMotif variant="pulse" height={20} color="var(--accent-primary)" />
          </div>

          <p className="progress-sub">
            Analyzing <strong>{selectedFile?.name}</strong> across genuine backend processing pipelines.
          </p>

          <div className="pipeline-steps-list">
            
            {/* Step 1: Ingestion */}
            <div className={`pipeline-step ${stageProgress.uploading === 'done' ? 'completed' : stageProgress.uploading === 'active' ? 'active' : ''}`}>
              <div className="step-indicator">
                {stageProgress.uploading === 'done' ? <CheckCircle2 size={16} /> : <span className="step-dot" />}
              </div>
              <div className="step-content">
                <div className="step-title">Audio Ingestion & Binary Header Inspection</div>
                <div className="step-desc">Verifying 16-bit PCM WAV, sample rate conformity, and SNR integrity.</div>
              </div>
            </div>

            {/* Step 2: CNN Inference */}
            <div className={`pipeline-step ${stageProgress.cnn_inference === 'done' ? 'completed' : stageProgress.cnn_inference === 'active' ? 'active' : ''}`}>
              <div className="step-indicator">
                {stageProgress.cnn_inference === 'done' ? <CheckCircle2 size={16} /> : <span className="step-dot" />}
              </div>
              <div className="step-content">
                <div className="step-title">Time-Frequency Preprocessing & Acoustic CNN</div>
                <div className="step-desc">Generating Log-Mel Spectrogram (128 bands) and classifying breath sounds (Normal, Wheezes, Crackles).</div>
              </div>
            </div>

            {/* Step 3: Disease Screening */}
            <div className={`pipeline-step ${stageProgress.disease_screening === 'done' ? 'completed' : stageProgress.disease_screening === 'active' ? 'active' : stageProgress.disease_screening === 'unavailable' ? 'skipped' : ''}`}>
              <div className="step-indicator">
                {stageProgress.disease_screening === 'done' ? <CheckCircle2 size={16} /> : stageProgress.disease_screening === 'unavailable' ? <span className="step-skipped">-</span> : <span className="step-dot" />}
              </div>
              <div className="step-content">
                <div className="step-title">Pediatric Disease Screening Model (Model B)</div>
                <div className="step-desc">Evaluating likelihood indicators for Healthy Control, Pneumonia, or Wheezing Disorders.</div>
              </div>
            </div>

            {/* Step 4: Grad-CAM */}
            <div className={`pipeline-step ${stageProgress.gradcam_saliency === 'done' ? 'completed' : stageProgress.gradcam_saliency === 'active' ? 'active' : stageProgress.gradcam_saliency === 'unavailable' ? 'skipped' : ''}`}>
              <div className="step-indicator">
                {stageProgress.gradcam_saliency === 'done' ? <CheckCircle2 size={16} /> : stageProgress.gradcam_saliency === 'unavailable' ? <span className="step-skipped">-</span> : <span className="step-dot" />}
              </div>
              <div className="step-content">
                <div className="step-title">Grad-CAM Interpretability & Saliency Map</div>
                <div className="step-desc">Calculating gradient activations over last conv layer to identify decisive time-frequency coordinates.</div>
              </div>
            </div>

          </div>
        </div>
      )}

    </div>
  );
}
