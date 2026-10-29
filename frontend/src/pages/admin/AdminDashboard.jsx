import { useState } from 'react';
import { LayoutDashboard, UsersRound, Server, Loader2 } from 'lucide-react';
import DashboardLayout from '../../components/layout/DashboardLayout';
import Card from '../../components/common/Card';
import EmptyState from '../../components/common/EmptyState';
import DataTable from '../../components/common/DataTable';
import { useAuth } from '../../context/AuthContext';
import { DEMO_MODE } from '../../config/config';
import { DEMO_DATA } from '../../services/demoData';
import { checkHealth } from '../../services/authService';

const NAV = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'users', label: 'User management', icon: UsersRound },
  { id: 'health', label: 'System health', icon: Server },
];

export default function AdminDashboard() {
  const { user } = useAuth();
  const [health, setHealth] = useState({ state: 'idle', message: '' }); // idle | checking | ok | error
  const demo = DEMO_MODE ? DEMO_DATA.admin : null;

  // This is a REAL request to the backend health endpoint (even in demo mode).
  const runCheck = async () => {
    setHealth({ state: 'checking', message: '' });
    try {
      await checkHealth();
      setHealth({ state: 'ok', message: 'Backend responded successfully.' });
    } catch (err) {
      setHealth({ state: 'error', message: err.message });
    }
  };

  return (
    <DashboardLayout navItems={NAV}>
      <div className="hero" id="overview">
        <h1>Welcome, {user.name}</h1>
        <p>System overview and account administration.</p>
      </div>

      <Card id="users" icon={UsersRound} title="User management">
        {demo ? (
          <DataTable caption="Users (synthetic)" rows={demo.users} columns={[
            { key: 'name', label: 'Name' }, { key: 'email', label: 'Email' },
            { key: 'role', label: 'Role', render: (r) => <span className="badge">{r.role}</span> },
            { key: 'active', label: 'Active', render: (r) => (r.active ? 'Yes' : 'No') },
          ]} />
        ) : <EmptyState title="User management not connected">Listing, approving and deactivating accounts needs admin-only backend endpoints. These are not connected yet, and the backend must enforce admin permissions.</EmptyState>}
      </Card>

      <Card id="health" icon={Server} title="System health">
        <p className="hint" style={{ marginBottom: 12 }}>Sends a live request to the backend health endpoint.</p>
        <button className="btn btn-primary" onClick={runCheck} disabled={health.state === 'checking'}>
          {health.state === 'checking' ? <><Loader2 className="spin" size={16} aria-hidden="true" /> Checking…</> : 'Check API health'}
        </button>
        <div aria-live="polite" style={{ marginTop: 12 }}>
          {health.state === 'ok' && <span className="badge ok">{health.message}</span>}
          {health.state === 'error' && <div className="alert alert-error">{health.message}</div>}
        </div>
      </Card>
    </DashboardLayout>
  );
}
