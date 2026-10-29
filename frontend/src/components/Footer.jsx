import React from 'react';
import { AudioWaveform, ShieldCheck } from 'lucide-react';

const CURRENT_YEAR = new Date().getFullYear();

export default function Footer() {
  return (
    <footer className="footer-container">
      <div className="footer-content">
        <div className="footer-top">
          <div className="footer-brand">
            <div className="footer-brand-header">
              <AudioWaveform size={20} className="text-cyan" />
              <span className="footer-brand-title">Acoustic Epidemiology</span>
            </div>
            <span className="footer-brand-desc">
              Automated Classification of Pediatric Respiratory Sounds Using 2D-CNNs and Time-Frequency Mel-Spectrogram Feature Extraction.
            </span>
          </div>

          <div className="footer-tags">
            <span className="chip">SPRSound Protocol</span>
            <span className="chip">2D-CNN Spectrogram</span>
            <span className="chip">Grad-CAM XAI</span>
            <span className="chip">Librosa STFT</span>
            <span className="chip">Django REST 3.18</span>
          </div>
        </div>

        <div className="footer-disclaimer-box">
          <ShieldCheck size={16} className="text-cyan shrink-0" />
          <span>
            <strong>Research Investigational Notice:</strong> This software is an academic computer-aided acoustic analysis tool developed for pediatric pulmonary sound research. Classification outputs and Grad-CAM saliency visualizations are intended solely for investigative evaluation and do not constitute certified medical diagnosis or treatment recommendations.
          </span>
        </div>

        <div className="footer-bottom">
          <span>&copy; {CURRENT_YEAR} Acoustic Epidemiology Research Project</span>
          <span className="text-muted">Pediatric Respiratory Auscultation Analysis</span>
        </div>
      </div>
    </footer>
  );
}
