import React, { useState, useEffect, useCallback } from 'react';
import Navbar from './components/Navbar';
import HealthCheckCard from './components/HealthCheckCard';
import AppModulesGrid from './components/AppModulesGrid';
import PipelineOverview from './components/PipelineOverview';
import TeamSection from './components/TeamSection';
import { checkBackendHealth, API_BASE_URL } from './services/api';
import { 
  Activity, 
  ArrowRight, 
  Layers, 
  Sparkles, 
  Cpu
} from 'lucide-react';
import './App.css';

export default function App() {
  const [healthState, setHealthState] = useState({
    loading: true,
    isHealthy: false,
    data: null,
    error: null,
    latency: null,
    lastChecked: null,
    statusCode: null,
  });

  const performHealthCheck = useCallback(async () => {
    setHealthState((prev) => ({ ...prev, loading: true }));
    const result = await checkBackendHealth();
    setHealthState({
      loading: false,
      isHealthy: result.success,
      data: result.data || null,
      error: result.error || null,
      latency: result.latency || null,
      lastChecked: new Date(),
      statusCode: result.statusCode || null,
    });
  }, []);

  useEffect(() => {
    let isMounted = true;
    checkBackendHealth().then((result) => {
      if (isMounted) {
        setHealthState({
          loading: false,
          isHealthy: result.success,
          data: result.data || null,
          error: result.error || null,
          latency: result.latency || null,
          lastChecked: new Date(),
          statusCode: result.statusCode || null,
        });
      }
    });
    return () => {
      isMounted = false;
    };
  }, []);


  return (
    <div className="app-wrapper">
      {/* Top Navigation */}
      <Navbar backendStatus={{ isHealthy: healthState.isHealthy, loading: healthState.loading }} />

      <main>
        {/* Hero Section */}
        <section id="hero" className="hero-section">
          <div className="hero-glow-sphere"></div>

          <div className="hero-content">
            <div className="hero-tag-badge">
              <span className="badge badge-cyan">
                <Sparkles size={14} /> Pediatric Respiratory Sound AI • Step 2 Scaffold
              </span>
            </div>

            <h1 className="hero-title">
              Automated Classification of <br />
              <span className="gradient-text">Pediatric Respiratory Diseases</span>
            </h1>

            <p className="hero-description">
              A clinical research prototype powered by 2D Convolutional Neural Networks, 
              Time-Frequency Mel-Spectrogram feature extraction, and Grad-CAM explainability 
              to classify non-invasive pediatric auscultation sounds.
            </p>

            <div className="hero-cta-group">
              <a href="#health-section" className="btn-primary" id="cta-health-btn">
                <Activity size={16} /> Test Backend Health
              </a>
              <a href="#apps-section" className="btn-secondary" id="cta-apps-btn">
                <Layers size={16} /> View Scaffolded Apps
              </a>
              <a href="#pipeline-section" className="btn-secondary" id="cta-pipeline-btn">
                <Cpu size={16} /> CNN Pipeline <ArrowRight size={14} />
              </a>
            </div>

            {/* Soundwave Preview Banner */}
            <div className="waveform-preview-card glass-panel" id="waveform-hero-card">
              <div className="waveform-meta">
                <div className="audio-bars-container">
                  <span className="audio-bar"></span>
                  <span className="audio-bar"></span>
                  <span className="audio-bar"></span>
                  <span className="audio-bar"></span>
                  <span className="audio-bar"></span>
                  <span className="audio-bar"></span>
                </div>
                <div className="waveform-info">
                  <div className="waveform-title">Pediatric Auscultation Stream</div>
                  <div className="waveform-subtitle">Bandpass Filtered (50 Hz – 2000 Hz) • 16 kHz Sample Rate</div>
                </div>
              </div>

              <div className="tech-chips-row">
                <span className="chip">Django REST 3.18</span>
                <span className="chip">React 19 + Vite</span>
                <span className="chip">Librosa STFT</span>
                <span className="chip">2D-CNN</span>
                <span className="chip">Grad-CAM</span>
              </div>
            </div>
          </div>
        </section>

        {/* Backend REST API Health Check Component */}
        <HealthCheckCard 
          healthState={healthState} 
          onRefresh={performHealthCheck} 
          apiBaseUrl={API_BASE_URL}
        />

        {/* 5 Scaffolded Django Apps */}
        <AppModulesGrid />

        {/* Acoustic Pipeline Section */}
        <PipelineOverview />

        {/* Project Research Team */}
        <TeamSection />
      </main>

      {/* Footer */}
      <footer className="footer-container">
        <div className="footer-content">
          <div className="footer-top">
            <div className="footer-brand">
              <span className="footer-brand-title">Acoustic Epidemiology Research System</span>
              <span className="footer-brand-desc">
                Automated Pediatric Respiratory Sound Classification &amp; Explainability
              </span>
            </div>

            <div className="footer-stack-chips">
              <span className="chip">Django 6.1.2</span>
              <span className="chip">REST Framework</span>
              <span className="chip">React + Vite</span>
              <span className="chip">SQLite / MySQL</span>
              <span className="chip">TensorFlow</span>
            </div>
          </div>

          <div className="footer-bottom">
            <span>Step 2 Complete • Backend Apps &amp; Frontend Foundation Initialized</span>
            <span>Local API Base: <code className="inline-code">{API_BASE_URL}</code></span>
          </div>
        </div>
      </footer>
    </div>
  );
}
