import React from 'react';
import { Users, Award, Code2, Database, Brain, CheckSquare } from 'lucide-react';


const TEAM = [
  {
    name: 'Gayatri',
    role: 'Group Leader',
    badge: 'Project Lead',
    badgeClass: 'badge-amber',
    icon: Award,
    responsibility: 'Project architecture, backend coordination, system integration, and research planning.'
  },
  {
    name: 'Shreya',
    role: 'Frontend Developer',
    badge: 'UI/UX & Client',
    badgeClass: 'badge-cyan',
    icon: Code2,
    responsibility: 'React (Vite) interface, landing page experience, interactive acoustic visualizers, and API services.'
  },
  {
    name: 'Samruddhi',
    role: 'Dataset & Preprocessing',
    badge: 'Data Pipeline',
    badgeClass: 'badge-indigo',
    icon: Database,
    responsibility: 'Pediatric respiratory audio curation, bandpass filter implementations, and Mel-spectrogram generation.'
  },
  {
    name: 'Mohini',
    role: 'Machine Learning',
    badge: 'Deep Learning',
    badgeClass: 'badge-emerald',
    icon: Brain,
    responsibility: 'CNN architecture design, model training, evaluation metrics, and Grad-CAM explainability synthesis.'
  },
  {
    name: 'Vishal',
    role: 'Testing & Research Docs',
    badge: 'QA & Docs',
    badgeClass: 'badge-rose',
    icon: CheckSquare,
    responsibility: 'End-to-end integration tests, health verification suites, research documentation, and clinical logs.'
  }
];

export default function TeamSection() {
  return (
    <section id="team-section" className="section-container">
      <div className="section-header">
        <span className="badge badge-amber">
          <Users size={14} /> Research Group
        </span>
        <h2 className="section-title">Project Team &amp; Responsibilities</h2>
        <p className="section-description">
          Cross-disciplinary team driving the Acoustic Epidemiology research prototype from raw signal to clinical assessment.
        </p>
      </div>

      <div className="team-grid">
        {TEAM.map((member) => {
          const Icon = member.icon;
          return (
            <div key={member.name} className="team-card glass-panel" id={`team-card-${member.name.toLowerCase()}`}>
              <div className="team-card-header">
                <div className="team-avatar">
                  <Icon size={20} className="text-cyan" />
                </div>
                <div className="team-info">
                  <h3 className="team-name">{member.name}</h3>
                  <span className="team-role">{member.role}</span>
                </div>
              </div>
              <span className={`badge ${member.badgeClass} team-badge`}>{member.badge}</span>
              <p className="team-responsibility">{member.responsibility}</p>
            </div>
          );
        })}
      </div>
    </section>
  );
}
