import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, 
  Users, 
  Mic, 
  FileText, 
  Search, 
  Activity, 
  CheckCircle2, 
  AlertTriangle,
  Play,
  RotateCcw,
  Sparkles,
  Send,
  Loader2
} from 'lucide-react';
import DashboardLayout from '../../components/layout/DashboardLayout';
import Card from '../../components/common/Card';
import EmptyState from '../../components/common/EmptyState';
import DataTable, { StatusBadge } from '../../components/common/DataTable';
import AudioAnalysisSection from '../../components/AudioAnalysisSection';
import AnalysisResults from '../../components/AnalysisResults';
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
  { id: 'review', label: 'Audio Analysis & AI Models', icon: Mic },
  { id: 'recordings', label: 'Patient Recordings', icon: Users },
  { id: 'assessments', label: 'Clinician Assessment', icon: FileText },
];

export default function DoctorDashboard() {
  const { user } = useAuth();
  const [query, setQuery] = useState('');
  const [recordings, setRecordings] = useState([]);
  const [loadingRecordings, setLoadingRecordings] = useState(false);
  const [activeModel, setActiveModel] = useState(null);
  const [activeDiseaseModel, setActiveDiseaseModel] = useState(null);
  const [selectedRecording, setSelectedRecording] = useState(null);
  
  // Analysis state for the AI workstation
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

  // Handle successful analysis from AudioAnalysisSection
  const handleAnalysisSuccess = (data) => {
    setAnalysisData(data);
    loadData(); // refresh table
  };

  // Re-analyze a recording from the table
  const handleAnalyzeExisting = async (rec) => {
    setSelectedRecording(rec);
    setIsAnalyzing(true);
    setAnalysisData(null);
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

  const filteredRecordings = recordings.filter((r) => 
    `${r.original_filename} ${r.patient_pseudo_id || ''} ${r.id}`.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <DashboardLayout navItems={NAV}>
      <div className="hero" id="overview">
        <h1>Welcome, Dr. {user.name}</h1>
        <p>Acoustic Epidemiology Clinician Workstation — Dual-model pediatric respiratory sound and disease classification.</p>
      </div>

      {/* Live System & Models Status Cards */}
      <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
        <div className="card stat">
          <div className="label">Total Ingested Audio</div>
          <div className="value">{recordings.length}</div>
        </div>
        <div className="card stat">
          <div className="label">Sound CNN (Model A)</div>
          <div className="value" style={{ fontSize: '1rem', marginTop: 4, color: activeModel ? '#10b981' : '#f59e0b' }}>
            {activeModel ? `${activeModel.version} (Active)` : 'Ready / Active'}
          </div>
        </div>
        <div className="card stat">
          <div className="label">Disease CNN (Model B)</div>
          <div className="value" style={{ fontSize: '1rem', marginTop: 4, color: activeDiseaseModel ? '#10b981' : '#f59e0b' }}>
            {activeDiseaseModel ? `${activeDiseaseModel.version} (Active)` : 'sprsound-disease-cnn-v2'}
          </div>
        </div>
      </div>

      {/* Analysis Workstation Section */}
      <Card id="review" icon={Mic} title="AI Auscultation & Disease Inference Workstation">
        <p className="hint" style={{ marginBottom: 16 }}>
          Upload or select a pediatric lung auscultation recording to run dual deep-learning models: Model A (sound category: Normal, Wheeze, Crackle) and Model B (disease prediction: Healthy, Pneumonia, Wheezing Disorders).
        </p>

        <AudioAnalysisSection 
          onAnalysisSuccess={handleAnalysisSuccess}
          isAnalyzing={isAnalyzing}
          setIsAnalyzing={setIsAnalyzing}
        />

        {analysisData && (
          <div style={{ marginTop: '2rem' }}>
            <AnalysisResults 
              prediction={analysisData.prediction}
              recording={analysisData.recording}
              explanation={analysisData.explanation}
              diseasePrediction={analysisData.diseasePrediction}
              onReset={() => setAnalysisData(null)}
            />
          </div>
        )}
      </Card>

      {/* Real Ingested Patient Recordings */}
      <Card id="recordings" icon={Users} title="Ingested Patient Recordings">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, gap: 12 }}>
          <label className="search" style={{ flex: 1, margin: 0 }}>
            <Search size={16} color="#8391a2" aria-hidden="true" />
            <span className="sr-only">Search recordings</span>
            <input 
              type="search" 
              placeholder="Search by filename, pseudo ID, or UUID" 
              value={query} 
              onChange={(e) => setQuery(e.target.value)} 
            />
          </label>
          <button type="button" className="btn btn-secondary" onClick={loadData} disabled={loadingRecordings}>
            <RotateCcw size={14} className={loadingRecordings ? 'spin' : ''} /> Refresh
          </button>
        </div>

        {filteredRecordings.length > 0 ? (
          <DataTable 
            caption="Recordings" 
            rows={filteredRecordings} 
            columns={[
              { key: 'original_filename', label: 'Recording Name' },
              { key: 'duration_seconds', label: 'Duration', render: (r) => `${r.duration_seconds || '—'}s` },
              { key: 'sample_rate', label: 'Sample Rate', render: (r) => `${r.sample_rate || '—'} Hz` },
              { key: 'anatomical_location_display', label: 'Chest Location', render: (r) => r.anatomical_location_display || 'Unspecified' },
              { key: 'created_at', label: 'Uploaded At', render: (r) => new Date(r.created_at).toLocaleDateString() },
              { 
                key: 'actions', 
                label: 'Action', 
                render: (r) => (
                  <button 
                    type="button" 
                    className="btn btn-sm btn-primary" 
                    onClick={() => handleAnalyzeExisting(r)}
                    disabled={isAnalyzing}
                  >
                    <Play size={12} /> Run Inference
                  </button>
                ) 
              },
            ]} 
          />
        ) : (
          <EmptyState title={loadingRecordings ? 'Loading recordings...' : 'No recordings found'}>
            {loadingRecordings ? 'Connecting to Django backend...' : 'Upload an audio file in the section above to ingest your first recording.'}
          </EmptyState>
        )}
      </Card>

      {/* Clinician Human Review Submission */}
      <Card id="assessments" icon={FileText} title="Clinician Assessment & Human-in-the-Loop Review">
        {analysisData?.prediction ? (
          <form onSubmit={handleAssessmentSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <p className="hint">
              Reviewing Prediction for: <strong>{analysisData.recording?.original_filename || 'Uploaded File'}</strong>
            </p>
            <div>
              <label style={{ display: 'block', fontWeight: 600, marginBottom: 6 }}>Clinical Decision / Diagnostic Agreement</label>
              <select 
                value={assessmentVerdict} 
                onChange={(e) => setAssessmentVerdict(e.target.value)}
                style={{ width: '100%', padding: '8px 12px', borderRadius: 6, border: '1px solid #cbd5e1' }}
              >
                <option value="AGREE">Agree with Model Classification</option>
                <option value="DISAGREE">Disagree with Model Classification</option>
                <option value="UNCERTAIN">Ambiguous / Clinical Auscultation Inconclusive</option>
              </select>
            </div>
            <div>
              <label style={{ display: 'block', fontWeight: 600, marginBottom: 6 }}>Clinical Notes & Differential Diagnosis</label>
              <textarea 
                rows={4}
                value={assessmentText}
                onChange={(e) => setAssessmentText(e.target.value)}
                placeholder="Enter clinical observations, patient symptoms, or acoustic anomalies observed..."
                style={{ width: '100%', padding: '10px 12px', borderRadius: 6, border: '1px solid #cbd5e1' }}
              />
            </div>
            <button 
              type="submit" 
              className="btn btn-primary" 
              disabled={submittingAssessment}
              style={{ alignSelf: 'flex-start' }}
            >
              {submittingAssessment ? <><Loader2 size={16} className="spin" /> Submitting...</> : <><Send size={16} /> Submit Assessment</>}
            </button>
            {assessmentSuccess && (
              <div className="alert alert-success" style={{ marginTop: 8 }}>
                <CheckCircle2 size={16} /> Clinical assessment submitted and attached to recording history.
              </div>
            )}
          </form>
        ) : (
          <EmptyState title="No active analysis selected">
            Run an inference in the "AI Auscultation & Disease Inference Workstation" above or select a recording to review and submit an assessment.
          </EmptyState>
        )}
      </Card>
    </DashboardLayout>
  );
}
