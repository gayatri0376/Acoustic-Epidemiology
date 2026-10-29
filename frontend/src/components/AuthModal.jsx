import React, { useState } from 'react';
import { Key, Lock, User, Mail, AlertCircle, CheckCircle2, X } from 'lucide-react';
import { loginUser, registerUser } from '../services/api';

export default function AuthModal({ isOpen, onClose, token, onAuthSuccess }) {
  const [activeTab, setActiveTab] = useState('login'); // 'login' | 'register' | 'token'
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [rawToken, setRawToken] = useState(token || '');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  if (!isOpen) return null;

  const handleLogin = async (e) => {
    e.preventDefault();
    if (!username.trim() || !password) {
      setError('Please provide both username and password.');
      return;
    }
    setLoading(true);
    setError(null);
    setSuccessMsg(null);

    const res = await loginUser(username.trim(), password);
    setLoading(false);

    if (res.success && res.data?.token) {
      setSuccessMsg('Authentication successful.');
      onAuthSuccess(res.data.token);
      setTimeout(() => {
        onClose();
      }, 700);
    } else {
      const err = res.error || (res.data && (res.data.error || res.data.detail || JSON.stringify(res.data))) || 'Authentication failed.';
      setError(typeof err === 'string' ? err : 'Invalid credentials.');
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    if (!username.trim() || !password) {
      setError('Username and password are required.');
      return;
    }
    setLoading(true);
    setError(null);
    setSuccessMsg(null);

    const payload = {
      username: username.trim(),
      password,
      email: email.trim() || undefined,
    };

    const res = await registerUser(payload);
    setLoading(false);

    if (res.success && res.data?.token) {
      setSuccessMsg('Registration successful. You are now authenticated.');
      onAuthSuccess(res.data.token);
      setTimeout(() => {
        onClose();
      }, 700);
    } else {
      const err = res.error || (res.data && (res.data.error || res.data.detail || JSON.stringify(res.data))) || 'Registration failed.';
      setError(typeof err === 'string' ? err : 'Registration could not be completed.');
    }
  };

  const handleManualToken = (e) => {
    e.preventDefault();
    const cleanToken = rawToken.trim();
    if (!cleanToken) {
      setError('Please enter a valid API token.');
      return;
    }
    onAuthSuccess(cleanToken);
    setSuccessMsg('API token saved.');
    setTimeout(() => {
      onClose();
    }, 500);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card glass-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-group">
            <div className="modal-icon-badge">
              <Key size={18} className="text-cyan" />
            </div>
            <div>
              <h3 className="modal-title">API Authentication</h3>
              <p className="modal-subtitle">Authenticate to access Django REST API endpoints</p>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close modal">
            <X size={18} />
          </button>
        </div>

        <div className="modal-tabs">
          <button
            type="button"
            className={`modal-tab ${activeTab === 'login' ? 'active' : ''}`}
            onClick={() => { setActiveTab('login'); setError(null); }}
          >
            Sign In
          </button>
          <button
            type="button"
            className={`modal-tab ${activeTab === 'register' ? 'active' : ''}`}
            onClick={() => { setActiveTab('register'); setError(null); }}
          >
            Register
          </button>
          <button
            type="button"
            className={`modal-tab ${activeTab === 'token' ? 'active' : ''}`}
            onClick={() => { setActiveTab('token'); setError(null); }}
          >
            Direct Token
          </button>
        </div>

        {error && (
          <div className="alert-banner alert-banner-error">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        {successMsg && (
          <div className="alert-banner alert-banner-success">
            <CheckCircle2 size={16} />
            <span>{successMsg}</span>
          </div>
        )}

        {activeTab === 'login' && (
          <form onSubmit={handleLogin} className="modal-form">
            <div className="form-group">
              <label className="form-label" htmlFor="auth-username">Username</label>
              <div className="input-with-icon">
                <User size={16} className="input-icon" />
                <input
                  id="auth-username"
                  type="text"
                  className="form-input"
                  placeholder="e.g. researcher"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoComplete="username"
                  required
                />
              </div>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="auth-password">Password</label>
              <div className="input-with-icon">
                <Lock size={16} className="input-icon" />
                <input
                  id="auth-password"
                  type="password"
                  className="form-input"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  required
                />
              </div>
            </div>

            <button type="submit" className="btn-primary modal-action-btn" disabled={loading}>
              {loading ? 'Authenticating...' : 'Sign In'}
            </button>
          </form>
        )}

        {activeTab === 'register' && (
          <form onSubmit={handleRegister} className="modal-form">
            <div className="form-group">
              <label className="form-label" htmlFor="reg-username">Username</label>
              <div className="input-with-icon">
                <User size={16} className="input-icon" />
                <input
                  id="reg-username"
                  type="text"
                  className="form-input"
                  placeholder="e.g. clinician_user"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoComplete="username"
                  required
                />
              </div>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="reg-email">Email (Optional)</label>
              <div className="input-with-icon">
                <Mail size={16} className="input-icon" />
                <input
                  id="reg-email"
                  type="email"
                  className="form-input"
                  placeholder="researcher@hospital.edu"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoComplete="email"
                />
              </div>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="reg-password">Password</label>
              <div className="input-with-icon">
                <Lock size={16} className="input-icon" />
                <input
                  id="reg-password"
                  type="password"
                  className="form-input"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="new-password"
                  required
                />
              </div>
            </div>

            <button type="submit" className="btn-primary modal-action-btn" disabled={loading}>
              {loading ? 'Creating Account...' : 'Register & Authorize'}
            </button>
          </form>
        )}

        {activeTab === 'token' && (
          <form onSubmit={handleManualToken} className="modal-form">
            <div className="form-group">
              <label className="form-label" htmlFor="manual-token">Django REST Auth Token</label>
              <div className="input-with-icon">
                <Key size={16} className="input-icon" />
                <input
                  id="manual-token"
                  type="text"
                  className="form-input font-mono"
                  placeholder="e.g. 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b"
                  value={rawToken}
                  onChange={(e) => setRawToken(e.target.value)}
                  required
                />
              </div>
              <p className="form-hint">
                Paste an active token issued by <code className="inline-code">/api/v1/auth/login/</code> or created via Django admin.
              </p>
            </div>

            <button type="submit" className="btn-primary modal-action-btn">
              Save Token
            </button>
          </form>
        )}
      </div>
    </div>
  );
}

