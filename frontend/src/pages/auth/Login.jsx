import { useState } from 'react';
import { Navigate } from 'react-router-dom';
import { Eye, EyeOff, Mail, Lock, Loader2, UserRound, Stethoscope, ShieldCheck, AudioWaveform, Activity, FileText, AlertCircle, FlaskConical } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { DEMO_MODE, ROLE_HOME } from '../../config/config';
import { DEMO_PASSWORD } from '../../services/demoData';
import Logo from '../../components/common/Logo';

const ROLE_OPTIONS = [
  { value: 'patient', label: 'Patient', icon: UserRound },
  { value: 'doctor', label: 'Doctor', icon: Stethoscope },
  { value: 'admin', label: 'Admin', icon: ShieldCheck },
];

// Turn an error into a message a person can act on.
function friendlyMessage(err) {
  if (err.kind === 'network' || err.kind === 'config' || err.kind === 'role') return err.message;
  if (err.status === 400 || err.status === 401) return 'Invalid email or password.';
  if (err.status === 403) return 'This account is not allowed to sign in. Contact your administrator.';
  if (err.status >= 500) return 'The server had a problem. Please try again in a moment.';
  return err.message || 'Something went wrong. Please try again.';
}

export default function Login() {
  const { user, login } = useAuth();
  const [role, setRole] = useState('patient'); // a request only; the server decides the real role
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [errors, setErrors] = useState({});
  const [formError, setFormError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  // Already signed in? Go to the dashboard for the VERIFIED role.
  if (user) return <Navigate to={ROLE_HOME[user.role]} replace />;

  const validate = () => {
    const e = {};
    if (!email.trim()) e.email = 'Email is required.';
    else if (!/^\S+@\S+\.\S+$/.test(email.trim())) e.email = 'Enter a valid email address.';
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
      await login(email.trim(), password, role); // success -> `user` is set and we redirect
    } catch (err) {
      setFormError(friendlyMessage(err));
      setSubmitting(false);
    }
  };

  return (
    <div className="auth">
      <section className="auth-brand" aria-label="About the project">
        <Logo light />
        <div>
          <h1>Acoustic Epidemiology Research System</h1>
          <p className="lead">Automated classification of pediatric respiratory diseases using respiratory sound analysis.</p>
          <ul className="auth-points">
            <li><span className="ico"><AudioWaveform size={18} aria-hidden="true" /></span>Respiratory sound recordings</li>
            <li><span className="ico"><Activity size={18} aria-hidden="true" /></span>Research-assistance analysis results</li>
            <li><span className="ico"><FileText size={18} aria-hidden="true" /></span>Assessment history and reports</li>
          </ul>
        </div>
        <p className="auth-foot">A research-assistance tool. Results are not a medical diagnosis and do not replace a clinician’s judgement.</p>
      </section>

      <main className="auth-main">
        <form className="auth-card" onSubmit={handleSubmit} noValidate>
          <div>
            <h2>Sign in</h2>
            <p className="sub">Use the account issued to you for this system.</p>
          </div>

          {DEMO_MODE && (
            <div className="alert alert-warn demo-box" role="note">
              <FlaskConical size={16} aria-hidden="true" />
              <div>
                <strong>Demo mode: simulated login, not real authentication.</strong>
                <div>Try <code>patient@demo.test</code>, <code>doctor@demo.test</code> or <code>admin@demo.test</code></div>
                <div>Password: <code>{DEMO_PASSWORD}</code></div>
              </div>
            </div>
          )}

          {formError && (
            <div className="alert alert-error" role="alert">
              <AlertCircle size={16} aria-hidden="true" /> <span>{formError}</span>
            </div>
          )}

          <fieldset className="roles">
            <legend>Sign in as</legend>
            {ROLE_OPTIONS.map(({ value, label, icon: Icon }) => (
              <label className="role" key={value}>
                <input type="radio" name="role" value={value} checked={role === value} onChange={() => setRole(value)} />
                <span><Icon size={20} aria-hidden="true" />{label}</span>
              </label>
            ))}
          </fieldset>

          <div className="field">
            <label htmlFor="email">Email</label>
            <div className="input-wrap">
              <Mail className="lead" size={18} aria-hidden="true" />
              <input id="email" type="email" autoComplete="username" placeholder="name@example.com"
                value={email} onChange={(e) => setEmail(e.target.value)}
                aria-invalid={!!errors.email} aria-describedby={errors.email ? 'email-err' : undefined} />
            </div>
            {errors.email && <span id="email-err" className="field-error">{errors.email}</span>}
          </div>

          <div className="field">
            <label htmlFor="password">Password</label>
            <div className="input-wrap">
              <Lock className="lead" size={18} aria-hidden="true" />
              <input id="password" type={showPassword ? 'text' : 'password'} autoComplete="current-password" placeholder="Enter your password"
                value={password} onChange={(e) => setPassword(e.target.value)}
                aria-invalid={!!errors.password} aria-describedby={errors.password ? 'pw-err' : undefined} />
              <button type="button" className="toggle" onClick={() => setShowPassword((s) => !s)}
                aria-label={showPassword ? 'Hide password' : 'Show password'} aria-pressed={showPassword}>
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
            {errors.password && <span id="pw-err" className="field-error">{errors.password}</span>}
          </div>

          <button className="btn btn-primary btn-block" type="submit" disabled={submitting}>
            {submitting ? (<><Loader2 className="spin" size={18} aria-hidden="true" /> Signing in…</>) : 'Sign in'}
          </button>

          <p className="hint">
            Accounts are issued by your administrator. Self-registration and password reset are not connected yet.
          </p>
        </form>
      </main>
    </div>
  );
}
