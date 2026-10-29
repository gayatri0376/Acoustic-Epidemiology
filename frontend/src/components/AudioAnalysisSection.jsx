import React, { useState, useRef, useCallback } from 'react';
import { 
  UploadCloud, 
  FileAudio, 
  AlertCircle, 
  AlertTriangle, 
  AudioWaveform, 
  Sparkles, 
  X, 
  Loader2,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import AudioPlayer from './AudioPlayer';
import { uploadAudioRecording, requestPrediction, requestExplanation } from '../services/api';

const MAX_FILE_SIZE = 25 * 1024 * 1024; // 25 MB
const ALLOWED_EXTENSIONS = ['.wav', '.flac', '.ogg', '.mp3'];
const MIN_DURATION_SECONDS = 0.2;

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

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
  const [analysisPhase, setAnalysisPhase] = useState(null); // 'uploading' | 'predicting' | 'explaining'
  const [serverError, setServerError] = useState(null);
  const [modelNotice, setModelNotice] = useState(null);
  
  // Optional clinical metadata
  const [showMeta, setShowMeta] = useState(false);
  const [anatomicalLocation, setAnatomicalLocation] = useState('');
  const [patientAgeGroup, setPatientAgeGroup] = useState('');

  const validateAudioFile = useCallback((file) => {
    if (!file) return 'No file selected.';

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

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setFileDuration(null);
    setValidationError(null);
    setServerError(null);
    setModelNotice(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDurationDetected = (duration) => {
    setFileDuration(duration);
    if (duration < MIN_DURATION_SECONDS) {
      setValidationError(
        `Audio duration (${duration.toFixed(2)}s) is too short (< 0.2s) for pediatric respiratory analysis.`
      );
    } else {
      setValidationError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setValidationError('Please select or drop an audio recording first.');
      return;
    }

    const checkErr = validateAudioFile(selectedFile);
    if (checkErr) {
      setValidationError(checkErr);
      return;
    }

    if (fileDuration !== null && fileDuration < MIN_DURATION_SECONDS) {
      setValidationError(
        `Audio duration (${fileDuration.toFixed(2)}s) is too short for clinical analysis.`
      );
      return;
    }

    if (!token) {
      setServerError({
        type: 'auth',
        title: 'Authentication Required',
        message: 'Uploading and analyzing recordings requires an authorized API session.',
        action: 'Sign in to continue',
      });
      onRequireAuth();
      return;
    }

    setIsAnalyzing(true);
    setServerError(null);
    setModelNotice(null);

    try {
      // Step 1: Upload Audio Recording
      setAnalysisPhase('uploading');
      const formData = new FormData();
      formData.append('audio_file', selectedFile);
      if (anatomicalLocation) {
        formData.append('anatomical_location', anatomicalLocation);
      }
      if (patientAgeGroup) {
        formData.append('patient_age_group', patientAgeGroup);
      }

      const uploadRes = await uploadAudioRecording(formData, token);

      if (!uploadRes.success) {
        setIsAnalyzing(false);
        setAnalysisPhase(null);

        if (uploadRes.statusCode === 401) {
          setServerError({
            type: 'auth',
            title: 'Session Expired or Unauthorized',
            message: 'Your authentication token is invalid or has expired. Please sign in again.',
            action: 'Re-authenticate',
          });
          onRequireAuth();
          return;
        }

        const backendMsg = 
          uploadRes.data?.audio_file?.[0] || 
          uploadRes.data?.error || 
          uploadRes.data?.detail || 
          uploadRes.data?.message || 
          'Audio recording could not be uploaded.';

        setServerError({
          type: 'validation',
          title: 'Upload Validation Error',
          message: typeof backendMsg === 'string' ? backendMsg : JSON.stringify(backendMsg),
        });
        return;
      }

      const recordingData = uploadRes.data?.recording || uploadRes.data;
      const recordingId = recordingData?.id;

      if (!recordingId) {
        throw new Error('Server did not return a valid recording identifier.');
      }

      // Step 2: Request CNN Inference
      setAnalysisPhase('predicting');
      const predictRes = await requestPrediction(recordingId, token);

      if (!predictRes.success) {
        setIsAnalyzing(false);
        setAnalysisPhase(null);

        if (predictRes.isModelUnavailable) {
          // Model is not active in registry
          const noticeMsg = 
            predictRes.data?.message || 
            predictRes.data?.error || 
            'No active trained CNN model weights are currently deployed in the model registry.';
          
          setModelNotice({
            title: 'ML Model Currently Undeployed',
            message: noticeMsg,
            recording: recordingData,
            predictionId: predictRes.data?.prediction_id,
          });
          return;
        }

        setServerError({
          type: 'inference',
          title: 'Model Inference Failed',
          message: predictRes.errorMessage || 'Deep learning inference could not be completed on this recording.',
        });
        return;
      }

      const predictionData = predictRes.data;

      // Step 3: Request Grad-CAM Explanation
      setAnalysisPhase('explaining');
      let explanationData = null;
      let explanationError = null;

      try {
        const explRes = await requestExplanation(predictionData.id, token, 'GRAD_CAM');
        if (explRes.success) {
          explanationData = explRes.data;
        } else {
          explanationError = explRes.errorMessage || 'Grad-CAM generation is currently unsupported or unavailable for this model configuration.';
        }
      } catch (explErr) {
        explanationError = `Explanation request error: ${explErr.message}`;
      }

      // Complete!
      setIsAnalyzing(false);
      setAnalysisPhase(null);

      onAnalysisSuccess({
        recording: recordingData,
        prediction: predictionData,
        explanation: explanationData,
        explanationError,
      });

    } catch (err) {
      setIsAnalyzing(false);
      setAnalysisPhase(null);
      setServerError({
        type: 'network',
        title: 'Connection Error',
        message: `Could not complete analysis: ${err.message}. Ensure the Django backend is running at http://127.0.0.1:8000.`,
      });
    }
  };

  return (
    <section id="analysis-section" className="section-container">
      <div className="section-header text-center">
        <span className="badge badge-cyan">
          <AudioWaveform size={14} /> Diagnostic Acoustic Workspace
        </span>
        <h1 className="section-title">Analyze Respiratory Sounds</h1>
        <p className="section-description">
          Upload pediatric respiratory audio recordings for automated preprocessing, 
          2D Mel-Spectrogram feature extraction, CNN acoustic classification, and Grad-CAM interpretability.
        </p>
      </div>

      <div className="analysis-workspace-card glass-panel">
        {/* Hidden native file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".wav,.flac,.ogg,.mp3,audio/wav,audio/x-wav,audio/flac,audio/ogg,audio/mpeg"
          onChange={handleInputChange}
          className="hidden-file-input"
          aria-label="Upload audio file"
        />

        {/* Drag & Drop Upload Zone */}
        {!selectedFile ? (
          <div
            className={`drop-zone ${isDragOver ? 'drop-zone-active' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                fileInputRef.current?.click();
              }
            }}
          >
            <div className="drop-zone-icon-glow">
              <UploadCloud size={38} className="text-cyan drop-icon" />
            </div>

            <h3 className="drop-zone-title">
              Drag &amp; drop respiratory audio recording here
            </h3>
            <p className="drop-zone-subtitle">
              or click anywhere in this area to browse local files
            </p>

            <button
              type="button"
              className="btn-primary browse-btn"
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
            >
              Browse Audio File
            </button>

            {/* Validation specs info banner */}
            <div className="drop-zone-rules">
              <span className="rule-badge">Formats: .WAV, .FLAC, .OGG, .MP3</span>
              <span className="rule-badge">Max Size: 25 MB</span>
              <span className="rule-badge">Min Duration: 0.2 s</span>
            </div>
          </div>
        ) : (
          /* Selected File View & Player */
          <div className="selected-file-card">
            <div className="file-header-row">
              <div className="file-meta-group">
                <div className="file-icon-box">
                  <FileAudio size={24} className="text-cyan" />
                </div>
                <div className="file-details">
                  <span className="file-name" title={selectedFile.name}>
                    {selectedFile.name}
                  </span>
                  <div className="file-submetrics font-mono">
                    <span>{formatBytes(selectedFile.size)}</span>
                    <span className="dot-sep">•</span>
                    <span className="uppercase">{selectedFile.name.split('.').pop()}</span>
                    {fileDuration !== null && (
                      <>
                        <span className="dot-sep">•</span>
                        <span>{fileDuration.toFixed(2)} seconds</span>
                      </>
                    )}
                  </div>
                </div>
              </div>

              <div className="file-actions">
                <button
                  type="button"
                  onClick={handleRemoveFile}
                  className="btn-icon-danger"
                  title="Remove selected recording"
                  disabled={isAnalyzing}
                  aria-label="Remove recording"
                >
                  <X size={16} />
                </button>
              </div>
            </div>

            {/* In-browser Audio Player for Pre-listening */}
            <div className="player-wrapper">
              <div className="player-label-row">
                <span className="text-xs font-semibold text-secondary uppercase tracking-wider">
                  Pre-Submission Auscultation Player
                </span>
                <span className="text-xs text-muted">Verify recording quality before analysis</span>
              </div>
              <AudioPlayer 
                file={selectedFile} 
                onDurationDetected={handleDurationDetected} 
              />
            </div>

            {/* Optional Clinical Metadata Toggle */}
            <div className="meta-accordion-toggle">
              <button
                type="button"
                className="btn-accordion"
                onClick={() => setShowMeta(!showMeta)}
              >
                <span>Optional Clinical Auscultation Metadata</span>
                {showMeta ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>
            </div>

            {showMeta && (
              <div className="meta-form-grid">
                <div className="form-group">
                  <label className="form-label" htmlFor="sel-anatomical-location">
                    Anatomical Location
                  </label>
                  <select
                    id="sel-anatomical-location"
                    value={anatomicalLocation}
                    onChange={(e) => setAnatomicalLocation(e.target.value)}
                    className="form-select"
                    disabled={isAnalyzing}
                  >
                    <option value="">Unspecified Location</option>
                    <option value="ANTERIOR_CHEST">Anterior Chest</option>
                    <option value="POSTERIOR_CHEST">Posterior Chest</option>
                    <option value="TRACHEA">Trachea</option>
                    <option value="LATERAL_CHEST">Lateral Chest</option>
                    <option value="UNKNOWN">Unknown</option>
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="sel-patient-age">
                    Patient Age Group
                  </label>
                  <select
                    id="sel-patient-age"
                    value={patientAgeGroup}
                    onChange={(e) => setPatientAgeGroup(e.target.value)}
                    className="form-select"
                    disabled={isAnalyzing}
                  >
                    <option value="">Unspecified Age</option>
                    <option value="INFANT">Infant (0 – 1 year)</option>
                    <option value="CHILD">Child (1 – 12 years)</option>
                    <option value="ADOLESCENT">Adolescent (12 – 18 years)</option>
                    <option value="UNKNOWN">Unknown</option>
                  </select>
                </div>
              </div>
            )}

            {/* Analyze Audio Action Row */}
            <div className="analysis-cta-row">
              <button
                type="button"
                onClick={handleAnalyze}
                className="btn-primary btn-large-analyze"
                disabled={isAnalyzing || !!validationError}
                id="btn-analyze-audio"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 size={18} className="animate-spin" />
                    <span>Analyzing Audio Stream...</span>
                  </>
                ) : (
                  <>
                    <Sparkles size={18} />
                    <span>Analyze Audio</span>
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Real-time Validation Error Banner */}
        {validationError && (
          <div className="alert-banner alert-banner-error" role="alert">
            <AlertCircle size={18} className="shrink-0" />
            <div className="alert-content">
              <strong>Invalid Audio Recording: </strong>
              <span>{validationError}</span>
            </div>
          </div>
        )}

        {/* Server & API Error Banner */}
        {serverError && (
          <div className="alert-banner alert-banner-error" role="alert">
            <AlertCircle size={18} className="shrink-0" />
            <div className="alert-content">
              <strong>{serverError.title}: </strong>
              <span>{serverError.message}</span>
              {serverError.type === 'auth' && (
                <button
                  type="button"
                  onClick={onRequireAuth}
                  className="alert-action-link"
                >
                  {serverError.action || 'Sign In Now'}
                </button>
              )}
            </div>
          </div>
        )}

        {/* Honest Model Unavailable Notice (HTTP 503) */}
        {modelNotice && (
          <div className="alert-banner alert-banner-warning" role="alert">
            <AlertTriangle size={18} className="shrink-0 text-amber" />
            <div className="alert-content">
              <strong>{modelNotice.title}: </strong>
              <span>{modelNotice.message}</span>
              <p className="model-notice-footnote">
                Note: Audio was successfully recorded and validated with ID{' '}
                <code className="inline-code">{modelNotice.recording?.id}</code>. 
                In compliance with strict scientific standards, this platform never returns fabricated 
                or simulated predictions when model weights are not loaded.
              </p>
            </div>
          </div>
        )}

        {/* Multi-stage In-Flight Loading Indicator */}
        {isAnalyzing && (
          <div className="analysis-progress-panel">
            <div className="progress-waveform-row">
              <span className="audio-bar"></span>
              <span className="audio-bar"></span>
              <span className="audio-bar"></span>
              <span className="audio-bar"></span>
              <span className="audio-bar"></span>
              <span className="audio-bar"></span>
              <span className="audio-bar"></span>
              <span className="audio-bar"></span>
            </div>

            <div className="progress-steps-list font-mono text-sm">
              <div className={`progress-step-item ${analysisPhase === 'uploading' ? 'step-active' : 'step-done'}`}>
                <span className="step-dot"></span>
                <span>1. Uploading &amp; validating audio stream header and sample rate</span>
              </div>
              <div className={`progress-step-item ${analysisPhase === 'predicting' ? 'step-active' : analysisPhase === 'explaining' ? 'step-done' : 'step-pending'}`}>
                <span className="step-dot"></span>
                <span>2. Computing 2D Mel-Spectrogram &amp; executing 2D-CNN forward pass</span>
              </div>
              <div className={`progress-step-item ${analysisPhase === 'explaining' ? 'step-active' : 'step-pending'}`}>
                <span className="step-dot"></span>
                <span>3. Backpropagating convolutional gradients for Grad-CAM saliency</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
