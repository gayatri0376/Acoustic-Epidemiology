// SYNTHETIC data for DEMO MODE only. None of this is real, and no one is a real patient.
export const DEMO_PASSWORD = 'demo1234';

export const DEMO_USERS = {
  'patient@demo.test': { id: 'demo-p1', name: 'Demo Patient', email: 'patient@demo.test', role: 'patient' },
  'doctor@demo.test': { id: 'demo-d1', name: 'Dr. Demo Doctor', email: 'doctor@demo.test', role: 'doctor' },
  'admin@demo.test': { id: 'demo-a1', name: 'Demo Admin', email: 'admin@demo.test', role: 'admin' },
};

export const DEMO_DATA = {
  patient: {
    profile: { 'Patient ID': 'DEMO-0001', Age: '6 years', Gender: 'Female', Registered: '2026-01-12' },
    assessments: [
      { id: 'A-101', date: '2026-02-03', type: 'Lung sound recording', status: 'Completed' },
      { id: 'A-102', date: '2026-03-18', type: 'Lung sound recording', status: 'Under review' },
      { id: 'A-103', date: '2026-04-09', type: 'Follow-up recording', status: 'Pending upload' },
    ],
  },
  doctor: {
    patients: [
      { id: 'DEMO-0001', name: 'Demo Child A', age: 6, lastAssessment: '2026-03-18', status: 'Under review' },
      { id: 'DEMO-0002', name: 'Demo Child B', age: 4, lastAssessment: '2026-03-02', status: 'Completed' },
      { id: 'DEMO-0003', name: 'Demo Child C', age: 9, lastAssessment: '2026-02-21', status: 'Completed' },
    ],
  },
  admin: {
    users: [
      { id: 'demo-p1', name: 'Demo Patient', email: 'patient@demo.test', role: 'patient', active: true },
      { id: 'demo-d1', name: 'Dr. Demo Doctor', email: 'doctor@demo.test', role: 'doctor', active: true },
      { id: 'demo-a1', name: 'Demo Admin', email: 'admin@demo.test', role: 'admin', active: true },
    ],
  },
};
