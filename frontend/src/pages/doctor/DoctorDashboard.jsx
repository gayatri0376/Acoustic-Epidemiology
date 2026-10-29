import { useState } from 'react';
import { LayoutDashboard, Users, Mic, FileText, Search } from 'lucide-react';
import DashboardLayout from '../../components/layout/DashboardLayout';
import Card from '../../components/common/Card';
import EmptyState from '../../components/common/EmptyState';
import DataTable, { StatusBadge } from '../../components/common/DataTable';
import { useAuth } from '../../context/AuthContext';
import { DEMO_MODE } from '../../config/config';
import { DEMO_DATA } from '../../services/demoData';

const NAV = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'patients', label: 'My patients', icon: Users },
  { id: 'review', label: 'Recording review', icon: Mic },
  { id: 'reports', label: 'Reports', icon: FileText },
];
const NOT_CONNECTED = 'This section will list only patients your account is authorised to see, once the backend endpoint is connected.';

export default function DoctorDashboard() {
  const { user } = useAuth();
  const [query, setQuery] = useState('');
  const demo = DEMO_MODE ? DEMO_DATA.doctor : null;
  const rows = demo ? demo.patients.filter((p) => `${p.name} ${p.id}`.toLowerCase().includes(query.toLowerCase())) : [];

  return (
    <DashboardLayout navItems={NAV}>
      <div className="hero" id="overview">
        <h1>Welcome, {user.name}</h1>
        <p>Review recordings and analysis results for your authorised patients.</p>
      </div>
      <div className="grid">
        <div className="card stat"><div className="label">Patients</div><div className="value">{demo ? demo.patients.length : '—'}</div></div>
        <div className="card stat"><div className="label">Awaiting review</div><div className="value">{demo ? demo.patients.filter((p) => p.status === 'Under review').length : '—'}</div></div>
      </div>

      <Card id="patients" icon={Users} title="My patients">
        {demo ? (
          <>
            <label className="search">
              <Search size={16} color="#8391a2" aria-hidden="true" />
              <span className="sr-only">Search patients</span>
              <input type="search" placeholder="Search by name or ID" value={query} onChange={(e) => setQuery(e.target.value)} />
            </label>
            {rows.length ? (
              <DataTable caption="Patients" rows={rows} columns={[
                { key: 'id', label: 'Patient ID' }, { key: 'name', label: 'Name' }, { key: 'age', label: 'Age' },
                { key: 'lastAssessment', label: 'Last assessment' },
                { key: 'status', label: 'Status', render: (r) => <StatusBadge status={r.status} /> },
              ]} />
            ) : <EmptyState title="No matching patients">Try a different search.</EmptyState>}
          </>
        ) : <EmptyState title="Patient list not connected">{NOT_CONNECTED}</EmptyState>}
      </Card>

      <Card id="review" icon={Mic} title="Recording review">
        <EmptyState title="No recordings loaded">Recording review needs the recordings endpoint and the analysis output format to be confirmed.</EmptyState>
      </Card>

      <Card id="reports" icon={FileText} title="Reports">
        <EmptyState title="No reports yet">Reports will appear here when the backend provides them.</EmptyState>
        <p className="hint" style={{ marginTop: 12 }}>Model output is research assistance and not a diagnosis. Clinical decisions remain with the clinician.</p>
      </Card>
    </DashboardLayout>
  );
}
