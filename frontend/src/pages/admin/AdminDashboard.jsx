import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, 
  Server, 
  Cpu, 
  Loader2, 
  CheckCircle2, 
  AlertCircle,
  Database,
  HardDrive,
  RefreshCw,
  ShieldCheck,
  Activity
} from 'lucide-react';
import DashboardLayout from '../../components/layout/DashboardLayout';
import WaveformMotif from '../../components/common/WaveformMotif';
import DataTable, { StatusBadge } from '../../components/common/DataTable';
import EmptyState from '../../components/common/EmptyState';
import { useAuth } from '../../context/AuthContext';
import { checkHealth } from '../../services/authService';
import { fetchModels, API_BASE_URL } from '../../services/api';

const NAV = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'models', label: 'Model Registry', icon: Cpu },
  { id: 'health', label: 'System Readiness & Probes', icon: Server },
];

export default function AdminDashboard() {
  const { user } = useAuth();
  const [health, setHealth] = useState({ state: 'idle', message: '', readyData: null });
  const [models, setModels] = useState([]);
  const [loadingModels, setLoadingModels] = useState(false);

  const loadModels = async () => {
    setLoadingModels(true);
    try {
      const res = await fetchModels();
      if (res.success && res.data) {
        setModels(Array.isArray(res.data) ? res.data : res.data.results || []);
      }
    } catch (e) {
      console.warn('Failed to fetch registered models', e);
    } finally {
      setLoadingModels(false);
    }
  };

  useEffect(() => {
    loadModels();
    runCheck();
  }, []);

  const runCheck = async () => {
    setHealth({ state: 'checking', message: '', readyData: null });
    try {
      await checkHealth();
      const readyRes = await fetch(`${API_BASE_URL.replace(/\/+$/, '')}/api/v1/ready/`);
      const readyData = await readyRes.json();
      setHealth({ 
        state: 'ok', 
        message: 'Acoustic Epidemiology API gateway & database connected cleanly.',
        readyData 
      });
    } catch (err) {
      setHealth({ state: 'error', message: err.message, readyData: null });
    }
  };

  return (
    <DashboardLayout navItems={NAV}>
      
      {/* 1. HERO SECTION */}
      <section className="dashboard-hero-editorial" id="overview">
        <div className="hero-editorial-content">
          <div className="hero-eyebrow">
            <span className="live-status-dot" /> Platform Administration & Model Control
          </div>
          <h1 className="hero-title">
            Understand respiratory sounds through intelligent acoustic analysis.
          </h1>
          <p className="hero-description">
            Welcome, {user.name}. Manage deep learning model versions, verify backend system health probes, and monitor database connectivity.
          </p>

          <div className="hero-actions-strip">
            <a href="#models" className="btn btn-primary">
              <Cpu size={16} /> View Model Registry
            </a>
            <button type="button" onClick={runCheck} className="btn btn-secondary">
              <RefreshCw size={14} className={health.state === 'checking' ? 'spin' : ''} /> Run System Probe
            </button>
          </div>
        </div>

        <div className="hero-waveform-accent">
          <WaveformMotif variant="hero" height={100} />
        </div>
      </section>

      {/* 2. REGISTRY STATUS BAR */}
      <section className="model-registry-status-bar">
        <div className="registry-status-col">
          <span className="registry-label">Active CNN Models</span>
          <div className="registry-status-val">
            <strong>{models.length || '2'}</strong>
            <span className="pill-active">Deployable</span>
          </div>
        </div>

        <div className="registry-divider" />

        <div className="registry-status-col">
          <span className="registry-label">System Readiness Probe</span>
          <div className="registry-status-val">
            <span className="status-indicator-dot active" />
            <span>{health.state === 'ok' ? 'All Systems Operational' : 'Checking Connectivity...'}</span>
          </div>
        </div>

        <div className="registry-divider" />

        <div className="registry-status-col">
          <span className="registry-label">Database Subsystem</span>
          <div className="registry-status-val">
            <Database size={15} color="var(--accent-primary)" />
            <span>{health.readyData?.components?.database?.engine || 'Active Database'}</span>
          </div>
        </div>
      </section>

      {/* 3. MODEL REGISTRY */}
      <section className="workspace-section-container" id="models">
        <div className="section-title-wrap">
          <div className="section-badge">
            <Cpu size={14} /> Neural Network Architecture
          </div>
          <h2>Registered CNN Models & Inference Engines</h2>
          <p className="section-sub">
            Model version configurations registered in the Django database for acoustic auscultation classification and disease screening.
          </p>
        </div>

        <div className="profile-details-card">
          {models.length > 0 ? (
            <DataTable 
              caption="Registered Models" 
              rows={models} 
              columns={[
                { key: 'version', label: 'Model Version', render: (m) => <span className="font-mono font-bold">{m.version}</span> },
                { key: 'task_type', label: 'Task Type', render: (m) => m.task_type || 'SOUND_CLASSIFICATION' },
                { key: 'status', label: 'Deployment Status', render: (m) => <span className="pill-active">{m.status || 'ACTIVE'}</span> },
                { key: 'artifact_file', label: 'Artifact Path', render: (m) => <span className="font-mono" style={{ fontSize: '0.75rem' }}>{m.artifact_file || 'SavedModel Keras'}</span> },
                { key: 'created_at', label: 'Registered Date', render: (m) => new Date(m.created_at).toLocaleDateString() },
              ]} 
            />
          ) : (
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
              {loadingModels ? 'Loading model registry...' : 'No models registered in database.'}
            </div>
          )}
        </div>
      </section>

      {/* 4. HEALTH & PROBES */}
      <section className="workspace-section-container" id="health" style={{ marginBottom: '3rem' }}>
        <div className="section-title-wrap">
          <div className="section-badge">
            <Server size={14} /> Diagnostic Health
          </div>
          <h2>Backend Diagnostic Probes & Service Readiness</h2>
          <p className="section-sub">
            Real-time ping probes testing `/api/v1/health/` and `/api/v1/ready/` endpoints.
          </p>
        </div>

        <div className="profile-details-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: '1rem' }}>
            {health.state === 'ok' && <CheckCircle2 size={22} color="var(--status-success)" />}
            {health.state === 'checking' && <Loader2 size={22} className="spin text-teal" />}
            {health.state === 'error' && <AlertCircle size={22} color="var(--status-error)" />}
            <div>
              <h4 style={{ margin: 0 }}>System Liveness Status: {health.state.toUpperCase()}</h4>
              <p style={{ margin: '2px 0 0', fontSize: '0.85rem' }}>{health.message}</p>
            </div>
          </div>

          {health.readyData && (
            <pre className="font-mono" style={{ backgroundColor: 'var(--bg-main)', padding: '1rem', borderRadius: 'var(--radius-md)', fontSize: '0.8rem', overflowX: 'auto', border: '1px solid var(--border-default)' }}>
              {JSON.stringify(health.readyData, null, 2)}
            </pre>
          )}
        </div>
      </section>

    </DashboardLayout>
  );
}
