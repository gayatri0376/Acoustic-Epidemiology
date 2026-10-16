import React from 'react';
import { 
  Server, 
  CheckCircle2, 
  AlertTriangle, 
  RefreshCw, 
  Clock, 
  Activity, 
  Terminal, 
  Zap
} from 'lucide-react';


export default function HealthCheckCard({ 
  healthState, 
  onRefresh, 
  apiBaseUrl 
}) {
  const { loading, isHealthy, data, error, latency, lastChecked, statusCode } = healthState;

  return (
    <section id="health-section" className="section-container">
      <div className="section-header">
        <span className="badge badge-cyan">
          <Activity size={14} /> Step 2 Integration
        </span>
        <h2 className="section-title">Backend REST API Health Check</h2>
        <p className="section-description">
          Live connectivity verification between React (Vite) and Django REST Framework via{' '}
          <code className="inline-code">GET /api/v1/health/</code>.
        </p>
      </div>

      <div className="health-card glass-panel" id="health-control-card">
        <div className="health-card-topbar">
          <div className="endpoint-info">
            <Server size={18} className="text-cyan" />
            <span className="endpoint-method">GET</span>
            <span className="endpoint-url">
              {apiBaseUrl}/api/v1/health/
            </span>
          </div>

          <button 
            id="btn-health-check"
            className="btn-primary" 
            onClick={onRefresh} 
            disabled={loading}
            aria-label="Probe Backend Health"
          >
            <RefreshCw size={16} className={loading ? 'spin-icon' : ''} />
            {loading ? 'Pinging Backend...' : 'Ping Endpoint'}
          </button>
        </div>

        {/* State Display: Loading, Success, Error */}
        <div className="health-card-body">
          {loading && (
            <div className="state-container state-loading" id="health-state-loading">
              <div className="spinner-orbit">
                <div className="spinner-core"></div>
              </div>
              <div className="state-text-group">
                <h4 className="state-heading">Pinging Django REST Service...</h4>
                <p className="state-desc">
                  Dispatching HTTP GET request to <code className="inline-code">{apiBaseUrl}/api/v1/health/</code>
                </p>
              </div>
            </div>
          )}

          {!loading && isHealthy && (
            <div className="state-container state-success" id="health-state-success">
              <div className="success-banner">
                <div className="success-icon-badge">
                  <CheckCircle2 size={24} className="text-emerald" />
                </div>
                <div className="state-text-group">
                  <div className="state-header-row">
                    <h4 className="state-heading text-emerald">
                      Backend Healthy & Operational
                    </h4>
                    <span className="badge badge-emerald">HTTP {statusCode || 200} OK</span>
                  </div>
                  <p className="state-desc">
                    Django REST Framework successfully responded to the acoustic epidemiology health check probe.
                  </p>
                </div>
              </div>

              {/* Metrics row */}
              <div className="metrics-grid">
                <div className="metric-box">
                  <span className="metric-label"><Zap size={14} /> Round-trip Latency</span>
                  <span className="metric-value text-cyan">{latency !== null ? `${latency} ms` : '—'}</span>
                </div>
                <div className="metric-box">
                  <span className="metric-label"><Activity size={14} /> Service Name</span>
                  <span className="metric-value">{data?.service || 'Acoustic Epidemiology API'}</span>
                </div>
                <div className="metric-box">
                  <span className="metric-label"><Clock size={14} /> Server Timestamp</span>
                  <span className="metric-value font-mono text-sm">
                    {data?.timestamp ? new Date(data.timestamp).toLocaleTimeString() : 'Just now'}
                  </span>
                </div>
              </div>

              {/* Raw JSON inspection */}
              <div className="raw-response-wrapper">
                <div className="response-header">
                  <span className="response-label">HTTP Response Payload</span>
                  <span className="response-status text-emerald">200 OK</span>
                </div>
                <pre className="json-viewer" id="health-json-output">
                  {JSON.stringify(data, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {!loading && !isHealthy && (
            <div className="state-container state-error" id="health-state-error">
              <div className="error-banner">
                <div className="error-icon-badge">
                  <AlertTriangle size={24} className="text-rose" />
                </div>
                <div className="state-text-group">
                  <div className="state-header-row">
                    <h4 className="state-heading text-rose">
                      Cannot Connect to Backend
                    </h4>
                    <span className="badge badge-rose">Offline</span>
                  </div>
                  <p className="state-desc">
                    {error || 'Unable to establish a connection to the Django REST Framework server.'}
                  </p>
                </div>
              </div>

              <div className="troubleshooting-card">
                <div className="troubleshooting-title">
                  <Terminal size={16} /> How to Start the Django Server:
                </div>
                <p className="troubleshooting-desc">
                  Open a terminal tab in your project root and start the Django development server:
                </p>
                <div className="command-block">
                  <code>cd backend &amp;&amp; source .venv/bin/activate &amp;&amp; python manage.py runserver</code>
                </div>
                <div className="troubleshooting-actions">
                  <button 
                    id="btn-retry-health"
                    className="btn-secondary" 
                    onClick={onRefresh}
                  >
                    <RefreshCw size={14} /> Retry Connection
                  </button>
                  <span className="troubleshooting-note">
                    Ensuring CORS origin allows <code className="inline-code">http://localhost:5173</code>
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer info bar */}
        <div className="health-card-footer">
          <div className="footer-meta">
            <Clock size={14} className="text-muted" />
            <span>
              Last Checked: {lastChecked ? lastChecked.toLocaleTimeString() : 'Never'}
            </span>
          </div>
          <div className="footer-link">
            <span className="text-muted">Target host:</span>
            <code className="inline-code">{apiBaseUrl}</code>
          </div>
        </div>
      </div>
    </section>
  );
}
