import React from 'react';
import { Activity } from 'lucide-react';


export default function Navbar({ backendStatus }) {
  return (
    <header className="navbar-container">
      <div className="navbar-content">
        <a href="#hero" className="navbar-brand" id="brand-link">
          <div className="brand-logo-glow">
            <Activity className="brand-icon" size={24} />
          </div>
          <div className="brand-text">
            <span className="brand-title">Acoustic Epidemiology</span>
            <span className="brand-subtitle">Pediatric Respiratory AI</span>
          </div>
        </a>

        <nav className="navbar-nav">
          <a href="#health-section" className="nav-link" id="nav-health">System Health</a>
          <a href="#apps-section" className="nav-link" id="nav-apps">Backend Apps</a>
          <a href="#pipeline-section" className="nav-link" id="nav-pipeline">CNN Pipeline</a>
          <a href="#team-section" className="nav-link" id="nav-team">Research Team</a>
        </nav>

        <div className="navbar-actions">
          <div className={`status-pill ${backendStatus.isHealthy ? 'status-pill-success' : backendStatus.loading ? 'status-pill-loading' : 'status-pill-error'}`}>
            <span className={`pulse-indicator ${backendStatus.isHealthy ? 'pulse-success' : backendStatus.loading ? 'pulse-loading' : 'pulse-error'}`}></span>
            <span className="status-label">
              {backendStatus.loading
                ? 'Probing API...'
                : backendStatus.isHealthy
                ? 'Backend Online (200 OK)'
                : 'Backend Disconnected'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
