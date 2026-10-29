import { useState } from 'react';
import { LogOut, Menu, X, FlaskConical } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { DEMO_MODE } from '../../config/config';
import Logo from '../common/Logo';

// Shared shell for all three dashboards. navItems: [{ id, label, icon }]
// Sidebar links scroll to sections on the same page (no extra routes are invented).
export default function DashboardLayout({ navItems, children }) {
  const { user, logout } = useAuth();
  const [open, setOpen] = useState(false); // mobile sidebar
  const [busy, setBusy] = useState(false);

  const handleLogout = async () => { setBusy(true); await logout(); };

  return (
    <>
      {DEMO_MODE && (
        <div className="demo-banner" role="note">
          <FlaskConical size={14} aria-hidden="true" style={{ verticalAlign: '-2px' }} /> DEMO MODE: synthetic data and a simulated login. Not real authentication.
        </div>
      )}
      <div className="shell">
        <aside className={`sidebar ${open ? 'open' : ''}`} aria-label="Sidebar">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Logo light />
            <button className="toggle menu-btn" style={{ position: 'static', color: '#fff' }} onClick={() => setOpen(false)} aria-label="Close menu"><X size={20} /></button>
          </div>
          <nav className="nav" aria-label="Dashboard sections">
            {navItems.map(({ id, label, icon: Icon }) => (
              <a key={id} href={`#${id}`} onClick={() => setOpen(false)}>
                <Icon size={18} aria-hidden="true" /> {label}
              </a>
            ))}
          </nav>
          <p className="side-user">Research-assistance tool. Results are not a medical diagnosis.</p>
        </aside>
        <div className={`scrim ${open ? 'open' : ''}`} onClick={() => setOpen(false)} aria-hidden="true" />

        <div>
          <header className="topbar">
            <div className="who">
              <button className="toggle menu-btn" style={{ position: 'static' }} onClick={() => setOpen(true)} aria-label="Open menu"><Menu size={20} /></button>
              <span className="name">{user.name}</span>
              <span className="badge">{user.role}</span>
            </div>
            <button className="btn btn-ghost" onClick={handleLogout} disabled={busy}>
              <LogOut size={16} aria-hidden="true" /> Log out
            </button>
          </header>
          <main className="content">{children}</main>
        </div>
      </div>
    </>
  );
}
