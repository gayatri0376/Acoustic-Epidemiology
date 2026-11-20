import { useState } from 'react';
import { Navigate } from 'react-router-dom';
import { 
  Eye, 
  EyeOff, 
  Mail, 
  Lock, 
  Loader2, 
  UserRound, 
  Stethoscope, 
  ShieldCheck, 
  AudioWaveform, 
  Activity, 
  FileText, 
  AlertCircle, 
  FlaskConical,
  Sparkles,
  ShieldAlert
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { ROLE_HOME } from '../../config/config';
import Logo from '../../components/common/Logo';
import WaveformMotif from '../../components/common/WaveformMotif';

const ROLE_OPTIONS = [
  { value: 'patient', label: 'Patient / Uploader', icon: UserRound, desc: 'Submit respiratory sounds' },
  { value: 'doctor', label: 'Doctor / Clinician', icon: Stethoscope, desc: 'Review & diagnostic assessment' },
  { value: 'admin', label: 'Administrator', icon: ShieldCheck, desc: 'System & model management' },
];

function friendlyMessage(err) {
  if (err.kind === 'network' || err.kind === 'config' || err.kind === 'role') return err.message;
  if (err.status === 400 || err.status === 401) return 'Invalid email/username or password.';
  if (err.status === 403) return 'This account is currently inactive. Contact your system administrator.';
  if (err.status >= 500) return 'The server encountered an error. Please try again shortly.';
  return err.message || 'Something went wrong. Please check your credentials.';
}

export default function Login() {
  const { user, login } = useAuth();
  const [role, setRole] = useState('doctor');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [errors, setErrors] = useState({});
  const [formError, setFormError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  // If already signed in, navigate to respective dashboard
  if (user) return <Navigate to={ROLE_HOME[user.role]} replace />;

  const validate = () => {
    const e = {};
    if (!email.trim()) e.email = 'Email or username is required.';
    if (!password) e.password = 'Password is required.';
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleSubmit = async (ev) => {
    ev.preventDefault();
    setFormError('');
    if (!validate()) return;
    setSubmitting(true);
    try {
      await login(email.trim(), password, role);
    } catch (err) {
      setFormError(friendlyMessage(err));
      setSubmitting(false);
    }
  };

  const handleFillDemo = (demoEmail, demoRole) => {
    setEmail(demoEmail);
    setPassword('SecurePassword123!');
    setRole(demoRole);
    setFormError('');
  };

  return (
    <div className="auth-signature-experience">
      
      {/* LEFT: EDITORIAL BRAND PANEL */}
      <section className="auth-brand-showcase" aria-label="Acoustic Epidemiology Research Information">
        <div className="auth-brand-inner">
          <Logo light={false} />

          <div className="auth-hero-copy">
            <span className="auth-mission-tag">
              <Sparkles size={13} /> Clinical Research Platform
            </span>
            <h1 className="auth-headline">
              Understand respiratory sounds through intelligent acoustic analysis.
            </h1>
            <p className="auth-lead">
              Automated classification of pediatric respiratory auscultation sounds using convolutional neural networks, Butterworth bandpass filtering, and Grad-CAM interpretability.
            </p>
          </div>

          {/* Ambient Waveform Graphic */}
          <div className="auth-waveform-art">
            <WaveformMotif variant="hero" height={80} color="var(--accent-primary)" />
          </div>

          <div className="auth-features-list">
            <div className="auth-feature-item">
              <div className="feature-icon-box">
                <AudioWaveform size={18} />
              </div>
              <div>
                <strong>Acoustic Waveform Analysis</strong>
                <p>128-band Log-Mel Spectrogram representations of chest auscultations.</p>
              </div>
            </div>

            <div className="auth-feature-item">
              <div className="feature-icon-box">
                <Activity size={18} />
              </div>
              <div>
                <strong>Dual Deep-Learning Models</strong>
                <p>Separated sound category (Normal, Wheeze, Crackle) and disease screening.</p>
              </div>
            </div>

            <div className="auth-feature-item">
              <div className="feature-icon-box">
                <FileText size={18} />
              </div>
              <div>
                <strong>Clinician Diagnostic Review</strong>
                <p>Transparent interpretability heatmaps and physician concurrence notes.</p>
              </div>
            </div>
          </div>

          <div className="auth-disclaimer-foot">
            <ShieldAlert size={15} style={{ flexShrink: 0, marginTop: 2 }} />
            <span>
              Investigational research decision-support aid. AI predictions do not constitute a medical diagnosis and do not replace licensed clinical judgement.
            </span>
          </div>
        </div>
      </section>

      {/* RIGHT: CLEAN SIGN-IN FORM */}
      <main className="auth-form-canvas">
        <div className="auth-card-wrapper">
          
          <div className="auth-card-header">
            <h2>Sign In to Workspace</h2>
            <p className="auth-card-sub">
              Access your clinical auscultation and research portal.
            </p>
          </div>

          {/* Quick Demo Credentials Helper */}
          <div className="quick-access-box" role="note">
            <div className="quick-access-title">
              <FlaskConical size={14} /> Quick Demo Logins (Password: <code>SecurePassword123!</code>)
            </div>
            <div className="quick-access-pills">
              <button 
                type="button" 
                onClick={() => handleFillDemo('doctor@example.com', 'doctor')}
                className="quick-fill-pill"
              >
                Doctor (doctor@example.com)
              </button>
              <button 
                type="button" 
                onClick={() => handleFillDemo('patient@example.com', 'patient')}
                className="quick-fill-pill"
              >
                Patient (patient@example.com)
              </button>
              <button 
                type="button" 
                onClick={() => handleFillDemo('admin@example.com', 'admin')}
                className="quick-fill-pill"
              >
                Admin (admin@example.com)
              </button>
            </div>
          </div>

          {formError && (
            <div className="alert alert-error" role="alert">
              <AlertCircle size={16} /> <span>{formError}</span>
            </div>
          )}

          <form className="auth-form-body" onSubmit={handleSubmit} noValidate>
            
            {/* Role Category Selector */}
            <fieldset className="role-selection-fieldset">
              <legend className="role-legend">Select System Role</legend>
              <div className="role-toggle-grid">
                {ROLE_OPTIONS.map((opt) => {
                  const Icon = opt.icon;
                  const isSelected = role === opt.value;
                  return (
                    <label 
                      key={opt.value} 
                      className={`role-toggle-card ${isSelected ? 'selected' : ''}`}
                    >
                      <input 
                        type="radio" 
                        name="auth_role" 
                        value={opt.value} 
                        checked={isSelected}
                        onChange={(e) => setRole(e.target.value)}
                        className="sr-only"
                      />
                      <Icon size={18} className="role-card-icon" />
                      <span className="role-card-title">{opt.label}</span>
                    </label>
                  );
                })}
              </div>
            </fieldset>

            {/* Email / Username Input */}
            <div className="field-group">
              <label htmlFor="auth-email">Email Address or Username</label>
              <div className="input-with-icon">
                <Mail size={16} className="input-icon" />
                <input 
                  id="auth-email"
                  type="text" 
                  autoComplete="username"
                  placeholder="e.g. doctor@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className={`input-text ${errors.email ? 'has-error' : ''}`}
                />
              </div>
              {errors.email && <span className="field-error-msg">{errors.email}</span>}
            </div>

            {/* Password Input */}
            <div className="field-group">
              <label htmlFor="auth-password">Password</label>
              <div className="input-with-icon">
                <Lock size={16} className="input-icon" />
                <input 
                  id="auth-password"
                  type={showPassword ? 'text' : 'password'} 
                  autoComplete="current-password"
                  placeholder="Enter your account password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className={`input-text ${errors.password ? 'has-error' : ''}`}
                />
                <button 
                  type="button" 
                  onClick={() => setShowPassword(!showPassword)}
                  className="password-toggle-btn"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
              {errors.password && <span className="field-error-msg">{errors.password}</span>}
            </div>

            {/* Submit Button */}
            <button 
              type="submit" 
              className="btn btn-primary btn-large btn-full"
              disabled={submitting}
            >
              {submitting ? (
                <>
                  <Loader2 size={18} className="spin" />
                  <span>Verifying Credentials...</span>
                </>
              ) : (
                <span>Sign In to System</span>
              )}
            </button>

          </form>

        </div>
      </main>

    </div>
  );
}
