import { LayoutDashboard, UserRound, ClipboardList, Mic, FileText, Upload } from 'lucide-react';
import DashboardLayout from '../../components/layout/DashboardLayout';
import Card from '../../components/common/Card';
import EmptyState from '../../components/common/EmptyState';
import DataTable, { StatusBadge } from '../../components/common/DataTable';
import { useAuth } from '../../context/AuthContext';
import { DEMO_MODE } from '../../config/config';
import { DEMO_DATA } from '../../services/demoData';

const NAV = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'profile', label: 'My profile', icon: UserRound },
  { id: 'assessments', label: 'Assessment history', icon: ClipboardList },
  { id: 'upload', label: 'Upload recording', icon: Mic },
  { id: 'results', label: 'Results & reports', icon: FileText },
];
const NOT_CONNECTED = 'This section will show your real records once the backend endpoint is connected.';

export default function PatientDashboard() {
  const { user } = useAuth();
  const demo = DEMO_MODE ? DEMO_DATA.patient : null; // real data would be fetched here instead

  return (
    <DashboardLayout navItems={NAV}>
      <div className="hero" id="overview">
        <h1>Welcome, {user.name}</h1>
        <p>View your assessments and recordings. You can only see your own records.</p>
      </div>

      <div className="grid">
        <div className="card stat"><div className="label">Assessments</div><div className="value">{demo ? demo.assessments.length : '—'}</div></div>
        <div className="card stat"><div className="label">Awaiting review</div><div className="value">{demo ? demo.assessments.filter((a) => a.status === 'Under review').length : '—'}</div></div>
      </div>

      <Card id="profile" icon={UserRound} title="My profile">
        {demo ? (
          <dl className="kv">{Object.entries(demo.profile).map(([k, v]) => (<div key={k} style={{ display: 'contents' }}><dt>{k}</dt><dd>{v}</dd></div>))}</dl>
        ) : <EmptyState title="Profile not connected">{NOT_CONNECTED}</EmptyState>}
      </Card>

      <Card id="assessments" icon={ClipboardList} title="Assessment history">
        {demo ? (
          <DataTable caption="Assessment history" rows={demo.assessments} columns={[
            { key: 'id', label: 'ID' }, { key: 'date', label: 'Date' }, { key: 'type', label: 'Type' },
            { key: 'status', label: 'Status', render: (r) => <StatusBadge status={r.status} /> },
          ]} />
        ) : <EmptyState title="No assessments to show">{NOT_CONNECTED}</EmptyState>}
      </Card>

      <Card id="upload" icon={Mic} title="Upload respiratory recording">
        <p className="hint" style={{ marginBottom: 12 }}>Recording upload will be enabled after the recordings endpoint and its accepted file format are confirmed in the backend.</p>
        <button className="btn btn-primary" disabled><Upload size={16} aria-hidden="true" /> Upload recording (not connected)</button>
      </Card>

      <Card id="results" icon={FileText} title="Results & reports">
        <EmptyState title="No results yet">{DEMO_MODE ? 'Demo mode never fabricates analysis results.' : NOT_CONNECTED}</EmptyState>
        <p className="hint" style={{ marginTop: 12 }}>Analysis output is research assistance only and is not a medical diagnosis.</p>
      </Card>
    </DashboardLayout>
  );
}
