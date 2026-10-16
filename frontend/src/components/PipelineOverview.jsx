import React from 'react';
import { 
  Sliders, 
  BarChart2, 
  BrainCircuit, 
  Eye, 
  Sparkles
} from 'lucide-react';


const STEPS = [
  {
    step: '01',
    title: 'Acoustic Capture',
    tech: 'Pediatric Auscultation',
    icon: Sliders,
    badge: 'Raw Audio',
    color: 'border-cyan',
    desc: 'Pediatric chest sounds captured via digital stethoscope. Audio normalized, split into respiration cycles, and resampled to 16kHz.'
  },
  {
    step: '02',
    title: 'Denoising & Filters',
    tech: 'SciPy & NumPy',
    icon: BarChart2,
    badge: 'Signal Prep',
    color: 'border-indigo',
    desc: '5th-order Butterworth bandpass filter (50Hz - 2000Hz) attenuates ambient room noise, heart beats, and friction artifacts.'
  },
  {
    step: '03',
    title: 'Log Mel-Spectrogram',
    tech: 'Librosa / STFT',
    icon: BrainCircuit,
    badge: 'Time-Frequency',
    color: 'border-blue',
    desc: 'Short-Time Fourier Transform (STFT) transforms 1D temporal pressure waveforms into 128-band Log-Mel 2D energy spectrograms.'
  },
  {
    step: '04',
    title: '2D-CNN Classifier',
    tech: 'TensorFlow / Keras',
    icon: BrainCircuit,
    badge: 'Deep Learning',
    color: 'border-emerald',
    desc: 'Multi-layer 2D Convolutional Neural Network extracts spatial-temporal acoustic features to predict disease categories (Normal, Wheeze, Crackle).'
  },
  {
    step: '05',
    title: 'Grad-CAM Saliency',
    tech: 'Explainable AI',
    icon: Eye,
    badge: 'Clinician Trust',
    color: 'border-amber',
    desc: 'Gradient-weighted Class Activation Mapping overlays heatmap activations onto spectrograms, proving clinical basis of prediction.'
  }
];

export default function PipelineOverview() {
  return (
    <section id="pipeline-section" className="section-container">
      <div className="section-header">
        <span className="badge badge-cyan">
          <Sparkles size={14} /> Methodology
        </span>
        <h2 className="section-title">Acoustic Deep Learning Pipeline</h2>
        <p className="section-description">
          The planned multi-stage pipeline converting non-invasive pediatric chest sounds into explainable diagnostic predictions.
        </p>
      </div>

      <div className="pipeline-steps-wrapper">
        <div className="pipeline-grid">
          {STEPS.map((item) => {
            const Icon = item.icon;
            return (
              <div key={item.step} className="pipeline-card glass-panel" id={`pipeline-step-${item.step}`}>
                <div className="pipeline-card-top">
                  <span className="pipeline-step-badge font-mono">{item.step}</span>
                  <span className="badge badge-cyan text-xs">{item.badge}</span>
                </div>

                <div className="pipeline-icon-box">
                  <Icon size={24} className="text-cyan" />
                </div>

                <h3 className="pipeline-title">{item.title}</h3>
                <span className="pipeline-tech font-mono">{item.tech}</span>
                <p className="pipeline-desc">{item.desc}</p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
