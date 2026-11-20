import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, 
  UserRound, 
  ClipboardList, 
  Mic, 
  FileText, 
  RotateCcw,
  Clock,
  Sparkles,
  ShieldCheck,
  CheckCircle2,
  Info
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import DashboardLayout from '../../components/layout/DashboardLayout';
import WaveformMotif from '../../components/common/WaveformMotif';
import AudioAnalysisSection from '../../components/AudioAnalysisSection';
import AnalysisResults from '../../components/AnalysisResults';
import HistorySection from '../../components/HistorySection';
import { useAuth } from '../../context/AuthContext';
import { fetchRecordings, requestPrediction, requestExplanation, requestDiseasePrediction } from '../../services/api';

const NAV = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'upload', label: 'Submit Recording', icon: Mic },
  { id: 'history', label: 'My Recordings History', icon: ClipboardList },
  { id: 'profile', label: 'My Account', icon: UserRound },
];

export default function PatientDashboard() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [recordings, setRecordings] = useState([]);
  const [loadingRecordings, setLoadingRecordings] = useState(false);
  const [analysisData, setAnalysisData] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const loadData = async () => {
    setLoadingRecordings(true);
    try {
      const res = await fetchRecordings();
      if (res.success && res.data) {
        setRecordings(Array.isArray(res.data) ? res.data : res.data.results || []);
      }
    } catch (e) {
      console.warn('Failed to load user recordings', e);
    } finally {
      setLoadingRecordings(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleAnalysisSuccess = (data) => {
    setAnalysisData(data);
    loadData();
    setTimeout(() => {
      document.getElementById('analysis-results')?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  const handleInspectRecording = async (rec) => {
    setIsAnalyzing(true);
    setAnalysisData(null);
    document.getElementById('upload')?.scrollIntoView({ behavior: 'smooth' });

    try {
      const soundRes = await requestPrediction(rec.id);
      let explRes = null;
      if (soundRes.success && soundRes.data?.id) {
        explRes = await requestExplanation(soundRes.data.id);
      }
      const disRes = await requestDiseasePrediction(rec.id);

      setAnalysisData({
        recording: rec,
        prediction: soundRes.data,
        explanation: explRes?.data,
        diseasePrediction: disRes.data,
        file: { name: rec.original_filename, size: rec.file_size_bytes }
      });
    } catch (err) {
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <DashboardLayout navItems={NAV}>
      
      {/* 1. HERO SECTION */}
      <section className="dashboard-hero-editorial" id="overview">
        <div className="hero-editorial-content">
          <div className="hero-eyebrow">
            <span className="live-status-dot" /> Acoustic Contributor Portal
          </div>
          <h1 className="hero-title">
            Understand respiratory sounds through intelligent acoustic analysis.
          </h1>
          <p className="hero-description">
            Welcome, {user.name}. Upload pediatric auscultation sounds for automated classification and clinician-guided review.
          </p>

          <div className="hero-actions-strip">
            <a href="#upload" className="btn btn-primary btn-large">
              <Mic size={18} /> Submit Audio Recording
            </a>
            <a href="#history" className="btn btn-secondary">
              <ClipboardList size={16} /> My Previous Uploads ({recordings.length})
            </a>
          </div>
        </div>

        <div className="hero-waveform-accent">
          <WaveformMotif variant="hero" height={100} />
        </div>
      </section>

      {/* 2. AUTHENTIC OVERVIEW STRIP (ZERO FAKE STATS) */}
      <section className="model-registry-status-bar">
        <div className="registry-status-col">
          <span className="registry-label">My Uploaded Audio</span>
          <div className="registry-status-val">
            <strong>{recordings.length}</strong>
            <span className="text-secondary" style={{ fontSize: '0.85rem' }}>recordings in repository</span>
          </div>
        </div>

        <div className="registry-divider" />

        <div className="registry-status-col">
          <span className="registry-label">Account Verification</span>
          <div className="registry-status-val">
            <CheckCircle2 size={16} color="var(--status-success)" />
            <span style={{ color: 'var(--status-success)', fontWeight: 600 }}>Active / Permitted</span>
          </div>
        </div>

        <div className="registry-divider" />

        <div className="registry-status-col">
          <span className="registry-label">Data Privacy</span>
          <div className="registry-status-val">
            <ShieldCheck size={16} color="var(--accent-primary)" />
            <span style={{ fontSize: '0.875rem' }}>De-identified research storage</span>
          </div>
        </div>
      </section>

      {/* 3. SIGNATURE AUDIO UPLOAD WORKSPACE */}
      <section className="workspace-section-container" id="upload">
        <div className="section-title-wrap">
          <div className="section-badge">
            <Mic size={14} /> Submit Auscultation
          </div>
          <h2>Upload Pediatric Respiratory Recording</h2>
          <p className="section-sub">
            Submit a respiratory audio file (.wav, .flac, .ogg, or .mp3) for automated acoustic feature extraction and classification.
          </p>
        </div>

        <AudioAnalysisSection 
          onAnalysisSuccess={handleAnalysisSuccess}
          isAnalyzing={isAnalyzing}
          setIsAnalyzing={setIsAnalyzing}
          onRequireAuth={() => { logout(); navigate('/login'); }}
        />

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

      {/* 4. HISTORY SECTION */}
      <section className="workspace-section-container" id="history">
        <div className="section-title-wrap">
          <div className="section-badge">
            <Clock size={14} /> My History
          </div>
          <h2>My Respiratory Sound History</h2>
          <p className="section-sub">
            Review past audio submissions, waveforms, and automated classification findings.
          </p>
        </div>

        <HistorySection 
          recordings={recordings}
          loading={loadingRecordings}
          onSelectRecording={handleInspectRecording}
          onAnalyzeNew={() => {
            document.getElementById('upload')?.scrollIntoView({ behavior: 'smooth' });
          }}
        />
      </section>

      {/* 5. USER PROFILE INFORMATION */}
      <section className="workspace-section-container" id="profile" style={{ marginBottom: '3rem' }}>
        <div className="profile-details-card">
          <div className="profile-card-header">
            <UserRound size={20} className="text-teal" />
            <div>
              <h3 style={{ margin: 0 }}>Contributor Account Details</h3>
              <p style={{ margin: '2px 0 0', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                System identity and role configuration.
              </p>
            </div>
          </div>

          <div className="profile-grid">
            <div className="profile-item">
              <span className="profile-label">Username:</span>
              <span className="profile-value font-mono">{user.name}</span>
            </div>
            <div className="profile-item">
              <span className="profile-label">Registered Email:</span>
              <span className="profile-value">{user.email}</span>
            </div>
            <div className="profile-item">
              <span className="profile-label">Role Category:</span>
              <span className="profile-value">Acoustic Contributor / Uploader</span>
            </div>
          </div>
        </div>
      </section>

    </DashboardLayout>
  );
}
