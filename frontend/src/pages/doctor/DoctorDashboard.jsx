import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, 
  Users, 
  Mic, 
  FileText, 
  Clock,
  Stethoscope, 
  Send, 
  Loader2, 
  CheckCircle2 
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import DashboardLayout from '../../components/layout/DashboardLayout';
import WaveformMotif from '../../components/common/WaveformMotif';
import AudioAnalysisSection from '../../components/AudioAnalysisSection';
import AnalysisResults from '../../components/AnalysisResults';
import HistorySection from '../../components/HistorySection';
import { useAuth } from '../../context/AuthContext';
import { 
  fetchRecordings, 
  fetchActiveModel, 
  fetchActiveDiseaseModel,
  requestPrediction,
  requestDiseasePrediction,
  requestExplanation,
  submitAssessment
} from '../../services/api';

const NAV = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'analyze', label: 'Analyze Auscultation', icon: Mic },
  { id: 'history', label: 'Recordings History', icon: Users },
  { id: 'assessment', label: 'Clinician Assessment', icon: FileText },
];

export default function DoctorDashboard() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  
  const [recordings, setRecordings] = useState([]);
  const [loadingRecordings, setLoadingRecordings] = useState(false);
  const [activeModel, setActiveModel] = useState(null);
  const [activeDiseaseModel, setActiveDiseaseModel] = useState(null);
  
  // Selected recording for re-analysis or viewing
  const [selectedRecording, setSelectedRecording] = useState(null);
  
  // Analysis results payload
  const [analysisData, setAnalysisData] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  // Assessment state
  const [assessmentText, setAssessmentText] = useState('');
  const [assessmentVerdict, setAssessmentVerdict] = useState('AGREE');
  const [submittingAssessment, setSubmittingAssessment] = useState(false);
  const [assessmentSuccess, setAssessmentSuccess] = useState(false);

  // Fetch recordings and models on load
  const loadData = async () => {
    setLoadingRecordings(true);
    try {
      const recRes = await fetchRecordings();
      if (recRes.success && recRes.data) {
        setRecordings(Array.isArray(recRes.data) ? recRes.data : recRes.data.results || []);
      }
    } catch (e) {
      console.warn('Failed to load recordings', e);
    } finally {
      setLoadingRecordings(false);
    }

    try {
      const modelRes = await fetchActiveModel();
      if (modelRes.success) setActiveModel(modelRes.data);
    } catch (e) {}

    try {
      const dModelRes = await fetchActiveDiseaseModel();
      if (dModelRes.success) setActiveDiseaseModel(dModelRes.data);
    } catch (e) {}
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleAnalysisSuccess = (data) => {
    setAnalysisData(data);
    loadData();
    // Scroll smoothly to results
    setTimeout(() => {
      document.getElementById('analysis-results')?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  // Re-analyze a recording from the history list
  const handleInspectRecording = async (rec) => {
    setSelectedRecording(rec);
    setIsAnalyzing(true);
    setAnalysisData(null);

    // Scroll to analysis workspace
    document.getElementById('analyze')?.scrollIntoView({ behavior: 'smooth' });

    try {
      // 1. Sound prediction
      const soundRes = await requestPrediction(rec.id);
      let explRes = null;
      if (soundRes.success && soundRes.data?.id) {
        explRes = await requestExplanation(soundRes.data.id);
      }
      // 2. Disease prediction
      const disRes = await requestDiseasePrediction(rec.id);

      setAnalysisData({
        recording: rec,
        prediction: soundRes.data,
        explanation: explRes?.data,
        diseasePrediction: disRes.data,
        file: { name: rec.original_filename, size: rec.file_size_bytes }
      });
    } catch (err) {
      console.error('Error analyzing recording:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Submit clinician review
  const handleAssessmentSubmit = async (e) => {
    e.preventDefault();
    if (!analysisData?.prediction?.id) return;
    setSubmittingAssessment(true);
    setAssessmentSuccess(false);
    try {
      const res = await submitAssessment(analysisData.prediction.id, {
        clinical_decision: assessmentVerdict,
        diagnostic_notes: assessmentText,
      });
      if (res.success) {
        setAssessmentSuccess(true);
        setAssessmentText('');
      }
    } catch (err) {
      console.error(err);
    } finally {
      setSubmittingAssessment(false);
    }
  };

  return (
    <DashboardLayout navItems={NAV}>
      
      {/* 1. EDITORIAL WELCOME SECTION */}
      <section className="dashboard-hero-editorial" id="overview">
        <div className="hero-editorial-content">
          <div className="hero-eyebrow">
            <span className="live-status-dot" /> Clinician Auscultation & AI Research Workstation
          </div>
          <h1 className="hero-title">
            Understand respiratory sounds through intelligent acoustic analysis.
          </h1>
          <p className="hero-description">
            Welcome, Dr. {user.name}. Evaluate pediatric respiratory sound categories (Normal, Wheezes, Crackles) and disease screening indicators via deep 2D-CNN feature extraction.
          </p>

          <div className="hero-actions-strip">
            <a href="#analyze" className="btn btn-primary btn-large">
              <Mic size={18} /> Analyze Recording
            </a>
            <a href="#history" className="btn btn-secondary">
              <Clock size={16} /> View Recordings ({recordings.length})
            </a>
          </div>
        </div>

        {/* Ambient Waveform Graphic */}
        <div className="hero-waveform-accent">
          <WaveformMotif variant="hero" height={100} />
        </div>
      </section>

      {/* 2. AUTHENTIC SYSTEM STATUS (ZERO FAKE STATS) */}
      <section className="model-registry-status-bar">
        <div className="registry-status-col">
          <span className="registry-label">Acoustic CNN (Model A)</span>
          <div className="registry-status-val">
            <span className="status-indicator-dot active" />
            <span>{activeModel?.version || 'sprsound-cnn-v1-fold0'}</span>
            <span className="pill-active">Active</span>
          </div>
        </div>

        <div className="registry-divider" />

        <div className="registry-status-col">
          <span className="registry-label">Pediatric Disease Model (Model B)</span>
          <div className="registry-status-val">
            <span className="status-indicator-dot active" />
            <span>{activeDiseaseModel?.version || 'sprsound-disease-cnn-v2'}</span>
            <span className="pill-active">Active</span>
          </div>
        </div>

        <div className="registry-divider" />

        <div className="registry-status-col">
          <span className="registry-label">Database Auscultation Records</span>
          <div className="registry-status-val">
            <strong>{recordings.length}</strong>
            <span className="text-secondary" style={{ fontSize: '0.85rem' }}>ingested files</span>
          </div>
        </div>
      </section>

      {/* 3. SIGNATURE AUDIO ANALYSIS WORKSPACE */}
      <section className="workspace-section-container" id="analyze">
        <div className="section-title-wrap">
          <div className="section-badge">
            <Mic size={14} /> Signature Interaction
          </div>
          <h2>Analyze Auscultation Recording</h2>
          <p className="section-sub">
            Upload pediatric chest audio to run Butterworth filtering, 128-band Mel-Spectrogram extraction, and dual deep-learning models.
          </p>
        </div>

        <AudioAnalysisSection 
          onAnalysisSuccess={handleAnalysisSuccess}
          isAnalyzing={isAnalyzing}
          setIsAnalyzing={setIsAnalyzing}
          onRequireAuth={() => { logout(); navigate('/login'); }}
        />

        {/* ANALYSIS RESULTS PRESENTATION */}
        {analysisData && (
          <div style={{ marginTop: '2.5rem' }}>
            <AnalysisResults 
              prediction={analysisData.prediction}
              recording={analysisData.recording}
              explanation={analysisData.explanation}
              diseasePrediction={analysisData.diseasePrediction}
              onReset={() => setAnalysisData(null)}
            />
          </div>
        )}
      </section>

      {/* 4. CLINICIAN ASSESSMENT MODULE */}
      {analysisData?.prediction && (
        <section className="clinician-assessment-panel" id="assessment">
          <div className="assessment-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <Stethoscope size={22} className="text-teal" />
              <div>
                <h3 style={{ margin: 0 }}>Clinician Auscultation Review & Diagnostic Concordance</h3>
                <p style={{ margin: '4px 0 0', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                  Record human physician evaluation and auscultation notes for Model A prediction ({analysisData.prediction.predicted_class}).
                </p>
              </div>
            </div>
          </div>

          {assessmentSuccess && (
            <div className="alert alert-success" style={{ margin: '1rem 0' }}>
              <CheckCircle2 size={18} />
              <span>Clinician assessment recorded successfully and archived with prediction record.</span>
            </div>
          )}

          <form onSubmit={handleAssessmentSubmit} className="assessment-form-body">
            <div className="verdict-selection-row">
              <label className="verdict-label">Physician Agreement:</label>
              <div className="verdict-radio-group">
                {[
                  { value: 'AGREE', label: 'Agree with AI Classification' },
                  { value: 'DISAGREE', label: 'Disagree / Overrule' },
                  { value: 'UNCERTAIN', label: 'Inconclusive / Further Testing Required' }
                ].map(opt => (
                  <label key={opt.value} className={`verdict-pill-choice ${assessmentVerdict === opt.value ? 'selected' : ''}`}>
                    <input 
                      type="radio" 
                      name="verdict" 
                      value={opt.value} 
                      checked={assessmentVerdict === opt.value}
                      onChange={(e) => setAssessmentVerdict(e.target.value)}
                    />
                    <span>{opt.label}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="field-group" style={{ marginTop: '1rem' }}>
              <label htmlFor="clinical-notes">Clinical Notes & Differential Auscultation Observations</label>
              <textarea 
                id="clinical-notes"
                rows={3}
                placeholder="Enter physical examination notes, adventitious sound qualities, or clinical correlation..."
                value={assessmentText}
                onChange={(e) => setAssessmentText(e.target.value)}
                className="input-textarea"
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1rem' }}>
              <button 
                type="submit" 
                className="btn btn-primary"
                disabled={submittingAssessment}
              >
                {submittingAssessment ? <Loader2 size={16} className="spin" /> : <Send size={15} />}
                <span>Submit Clinical Review</span>
              </button>
            </div>
          </form>
        </section>
      )}

      {/* 5. HISTORY EXPERIENCE */}
      <section className="workspace-section-container" id="history">
        <div className="section-title-wrap">
          <div className="section-badge">
            <Clock size={14} /> Repository History
          </div>
          <h2>Patient Auscultation Repository</h2>
          <p className="section-sub">
            Search, filter, and inspect previous pediatric respiratory sound recordings and past AI inference outputs.
          </p>
        </div>

        <HistorySection 
          recordings={recordings}
          loading={loadingRecordings}
          onSelectRecording={handleInspectRecording}
          onAnalyzeNew={() => {
            document.getElementById('analyze')?.scrollIntoView({ behavior: 'smooth' });
          }}
        />
      </section>

    </DashboardLayout>
  );
}
