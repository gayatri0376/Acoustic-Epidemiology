import React from 'react';
import { 
  Sliders, 
  Filter, 
  BrainCircuit, 
  Eye, 
  AudioWaveform,
  Sparkles
} from 'lucide-react';

const PIPELINE_STAGES = [
  {
    step: '01',
    title: 'Acoustic Capture',
    subtitle: 'Pediatric Auscultation',
    icon: AudioWaveform,
    badge: 'Raw Audio',
    desc: 'Pediatric respiratory sounds recorded via electronic stethoscope. Audio is validated for RIFF/WAV headers, minimum 0.2s duration, and sampled at 8kHz / 16kHz.'
  },
  {
    step: '02',
    title: 'Zero-Phase Bandpass',
    subtitle: 'Butterworth Denoising',
    icon: Filter,
    badge: '50 Hz – 2000 Hz',
    desc: 'A 5th-order digital Butterworth bandpass filter attenuates heart sounds (<50 Hz), patient motion artifacts, and ambient room noise above 2000 Hz.'
  },
  {
    step: '03',
    title: 'Log Mel-Spectrogram',
    subtitle: 'Time-Frequency Extraction',
    icon: Sliders,
    badge: '128 Mel Bands',
    desc: 'Short-Time Fourier Transform (STFT) with 1024 FFT length and 512 hop length transforms 1D waveforms into 2D time-frequency energy distributions normalized to [0, 1].'
  },
  {
    step: '04',
    title: '2D-CNN Classification',
    subtitle: 'Deep Feature Learning',
    icon: BrainCircuit,
    badge: 'Softmax Output',
    desc: 'Multi-layer 2D Convolutional Neural Network extracts spatial-temporal acoustic features to classify recordings into Normal, CAS, DAS, or CAS & DAS.'
  },
  {
    step: '05',
    title: 'Grad-CAM Saliency',
    subtitle: 'Visual Explainability',
    icon: Eye,
    badge: 'Gradient Attribution',
    desc: 'Gradient-weighted Class Activation Mapping backpropagates gradients from the final Conv2D layer to project acoustic focus regions onto the spectrogram.'
  }
];

export default function PipelineOverview() {
  return (
    <section id="methodology-section" className="section-container">
      <div className="section-header text-center">
        <span className="badge badge-cyan">
          <Sparkles size={14} /> Technical Architecture
        </span>
        <h2 className="section-title">Acoustic Deep Learning Pipeline</h2>
        <p className="section-description">
          End-to-end clinical AI workflow translating pediatric auscultation sounds into verified, interpretable classifications.
        </p>
      </div>

      <div className="pipeline-steps-wrapper">
        <div className="pipeline-grid">
          {PIPELINE_STAGES.map((item) => {
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
                <span className="pipeline-tech font-mono">{item.subtitle}</span>
                <p className="pipeline-desc">{item.desc}</p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
