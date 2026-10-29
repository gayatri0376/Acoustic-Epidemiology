import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, 
  UserRound, 
  ClipboardList, 
  Mic, 
  FileText, 
  RotateCcw,
  Clock
} from 'lucide-react';
import DashboardLayout from '../../components/layout/DashboardLayout';
import Card from '../../components/common/Card';
import EmptyState from '../../components/common/EmptyState';
import DataTable from '../../components/common/DataTable';
import AudioAnalysisSection from '../../components/AudioAnalysisSection';
import AnalysisResults from '../../components/AnalysisResults';
import { useAuth } from '../../context/AuthContext';
import { fetchRecordings } from '../../services/api';

const NAV = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'profile', label: 'My Profile', icon: UserRound },
  { id: 'upload', label: 'Upload Respiratory Recording', icon: Mic },
  { id: 'history', label: 'My Recordings History', icon: ClipboardList },
];

export default function PatientDashboard() {
  const { user } = useAuth();
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
  };

  return (
    <DashboardLayout navItems={NAV}>
      <div className="hero" id="overview">
        <h1>Welcome, {user.name}</h1>
        <p>Acoustic Epidemiology Patient Portal — Ingest respiratory audio for research and clinician review.</p>
      </div>

      <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
        <div className="card stat">
          <div className="label">My Uploaded Audio</div>
          <div className="value">{recordings.length}</div>
        </div>
        <div className="card stat">
          <div className="label">Account Status</div>
          <div className="value" style={{ fontSize: '1rem', marginTop: 4, color: '#10b981' }}>Active / Verified</div>
        </div>
      </div>

      {/* Profile Card */}
      <Card id="profile" icon={UserRound} title="My Profile & Clinical Identifiers">
        <dl className="kv" style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '8px 16px' }}>
          <dt style={{ fontWeight: 600 }}>Username:</dt>
          <dd>{user.name}</dd>
          <dt style={{ fontWeight: 600 }}>Email Address:</dt>
          <dd>{user.email}</dd>
          <dt style={{ fontWeight: 600 }}>System Role:</dt>
          <dd>Acoustic Data Contributor ({user.role})</dd>
        </dl>
      </Card>

      {/* Upload Section */}
      <Card id="upload" icon={Mic} title="Submit Respiratory Sound for Analysis">
        <p className="hint" style={{ marginBottom: 16 }}>
          Upload your pediatric audio recording (.wav, .flac, .ogg, or .mp3) for feature extraction and AI classification.
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

      {/* History Card */}
      <Card id="history" icon={ClipboardList} title="My Uploaded Audio Recordings">
        {recordings.length > 0 ? (
          <DataTable 
            caption="My Recordings" 
            rows={recordings} 
            columns={[
              { key: 'original_filename', label: 'File Name' },
              { key: 'duration_seconds', label: 'Duration', render: (r) => `${r.duration_seconds || '—'}s` },
              { key: 'sample_rate', label: 'Sample Rate', render: (r) => `${r.sample_rate || '—'} Hz` },
              { key: 'created_at', label: 'Upload Date', render: (r) => new Date(r.created_at).toLocaleDateString() },
            ]} 
          />
        ) : (
          <EmptyState title={loadingRecordings ? 'Loading recordings...' : 'No recordings submitted yet'}>
            {loadingRecordings ? 'Checking backend database...' : 'Upload your first audio file above to view history.'}
          </EmptyState>
        )}
      </Card>
    </DashboardLayout>
  );
}
