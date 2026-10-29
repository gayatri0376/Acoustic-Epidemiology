import React from 'react';
import { AudioWaveform, Key, LogOut } from 'lucide-react';

export default function Navbar({ token, onOpenAuth, onLogout }) {
  return (
    <header className="navbar-container">
      <div className="navbar-content">
        <a href="#analysis-section" className="navbar-brand" id="brand-link">
          <div className="brand-logo-glow">
            <AudioWaveform className="brand-icon" size={22} />
          </div>
          <div className="brand-text">
            <span className="brand-title">Acoustic Epidemiology</span>
            <span className="brand-subtitle">Pediatric Respiratory Sound Analysis</span>
          </div>
        </a>

        <nav className="navbar-nav">
          <a href="#analysis-section" className="nav-link" id="nav-analysis">
            Analyze Audio
          </a>
          <a href="#categories-section" className="nav-link" id="nav-categories">
            Target Categories
          </a>
          <a href="#methodology-section" className="nav-link" id="nav-methodology">
            CNN Methodology
          </a>
        </nav>

        <div className="navbar-actions">
          {token ? (
            <div className="auth-pill-active">
              <span className="auth-status-dot"></span>
              <span className="auth-pill-label">Authorized Session</span>
              <button
                type="button"
                onClick={onLogout}
                className="auth-logout-btn"
                title="Disconnect API session"
                aria-label="Disconnect session"
              >
                <LogOut size={13} />
              </button>
            </div>
          ) : (
            <button
              type="button"
              onClick={onOpenAuth}
              className="btn-auth-trigger"
              id="header-auth-btn"
            >
              <Key size={14} />
              <span>Sign In / API Key</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
