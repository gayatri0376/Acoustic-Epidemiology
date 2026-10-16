import React from 'react';
import { 
  Users, 
  Mic, 
  Cpu, 
  Eye, 
  FileCheck, 
  Check, 
  Code,
  Layers
} from 'lucide-react';

const MODULES = [
  {
    id: 'accounts',
    name: 'accounts',
    title: 'Clinician & Researcher Accounts',
    icon: Users,
    badgeColor: 'badge-indigo',
    description: 'Manages user authentication, clinician roles (pediatrician, pulmonologist, researcher), and access control.',
    features: ['Custom User model', 'JWT/Session authentication', 'Role-based permissions', 'Audit log tracking'],
    path: 'backend/accounts/'
  },
  {
    id: 'recordings',
    name: 'recordings',
    title: 'Acoustic Audio Ingestion',
    icon: Mic,
    badgeColor: 'badge-cyan',
    description: 'Handles pediatric respiratory audio uploads, sampling rate verification (44.1kHz / 16kHz), and stethoscope acoustic metadata.',
    features: ['WAV/FLAC audio ingestion', 'Auscultation chest location tags', 'Audio duration/SNR validation', 'Storage management'],
    path: 'backend/recordings/'
  },
  {
    id: 'ml_inference',
    name: 'ml_inference',
    title: 'Spectrogram & CNN Inference',
    icon: Cpu,
    badgeColor: 'badge-emerald',
    description: 'Preprocesses raw audio into time-frequency Mel-Spectrograms and invokes the 2D-CNN classification model.',
    features: ['Librosa feature extraction', 'Log-Mel Spectrogram generator', 'Model inference runner', 'Classification probabilities'],
    path: 'backend/ml_inference/'
  },
  {
    id: 'explanations',
    name: 'explanations',
    title: 'Grad-CAM Explainability Engine',
    icon: Eye,
    badgeColor: 'badge-amber',
    description: 'Extracts gradient-weighted class activation maps (Grad-CAM) to visualize which spectral frequencies triggered diagnosis.',
    features: ['Grad-CAM heatmap generator', 'Spectrogram overlay synthesis', 'Clinical interpretability maps', 'Frequency band highlights'],
    path: 'backend/explanations/'
  },
  {
    id: 'assessments',
    name: 'assessments',
    title: 'Clinical Assessments & Reports',
    icon: FileCheck,
    badgeColor: 'badge-rose',
    description: 'Aggregates predictions, diagnostic confidence scores, and pediatric severity indexing into printable clinician reports.',
    features: ['Disease risk stratification', 'Longitudinal audio comparisons', 'Clinician notes & confirmation', 'Structured JSON reports'],
    path: 'backend/assessments/'
  }
];

export default function AppModulesGrid() {
  return (
    <section id="apps-section" className="section-container">
      <div className="section-header">
        <span className="badge badge-indigo">
          <Layers size={14} /> Modular Backend Architecture
        </span>
        <h2 className="section-title">Django Applications Scaffolded</h2>
        <p className="section-description">
          The backend is modularized into 5 dedicated Django applications, each handling a distinct domain of the acoustic epidemiology workflow.
        </p>
      </div>

      <div className="modules-grid">
        {MODULES.map((mod) => {
          const IconComponent = mod.icon;
          return (
            <div key={mod.id} className="module-card glass-panel" id={`card-module-${mod.id}`}>
              <div className="module-card-header">
                <div className={`module-icon-wrap ${mod.id}`}>
                  <IconComponent size={22} />
                </div>
                <div className="module-title-wrap">
                  <div className="module-name-row">
                    <span className="module-app-name font-mono">{mod.name}</span>
                    <span className={`badge ${mod.badgeColor}`}>Installed</span>
                  </div>
                  <h3 className="module-title">{mod.title}</h3>
                </div>
              </div>

              <p className="module-description">{mod.description}</p>

              <div className="module-features-list">
                {mod.features.map((feat, idx) => (
                  <div key={idx} className="feature-item">
                    <Check size={14} className="feature-check text-emerald" />
                    <span>{feat}</span>
                  </div>
                ))}
              </div>

              <div className="module-card-footer">
                <span className="module-path font-mono">
                  <Code size={13} /> {mod.path}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
