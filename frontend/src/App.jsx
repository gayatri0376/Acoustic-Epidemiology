import React, { useState } from 'react';
import Navbar from './components/Navbar';
import AudioAnalysisSection from './components/AudioAnalysisSection';
import AnalysisResults from './components/AnalysisResults';
import ClinicalCategoriesReference from './components/ClinicalCategoriesReference';
import PipelineOverview from './components/PipelineOverview';
import Footer from './components/Footer';
import AuthModal from './components/AuthModal';
import './App.css';

const TOKEN_STORAGE_KEY = 'acoustic_auth_token';

export default function App() {
  const [token, setToken] = useState(() => {
    return localStorage.getItem(TOKEN_STORAGE_KEY) || '';
  });
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);

  // Sync token changes to localStorage
  const handleAuthSuccess = (newToken) => {
    setToken(newToken);
    if (newToken) {
      localStorage.setItem(TOKEN_STORAGE_KEY, newToken);
    } else {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
    }
  };

  const handleLogout = () => {
    setToken('');
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  };

  const handleAnalysisSuccess = (resultData) => {
    setAnalysisResult(resultData);
    // Smooth scroll to results
    setTimeout(() => {
      const resultsEl = document.getElementById('analysis-results');
      if (resultsEl) {
        resultsEl.scrollIntoView({ behavior: 'smooth' });
      }
    }, 100);
  };

  const handleResetAnalysis = () => {
    setAnalysisResult(null);
    // Smooth scroll to top analysis area
    setTimeout(() => {
      const analysisEl = document.getElementById('analysis-section');
      if (analysisEl) {
        analysisEl.scrollIntoView({ behavior: 'smooth' });
      }
    }, 100);
  };

  return (
    <div className="app-wrapper">
      {/* Clean Minimal Header */}
      <Navbar
        token={token}
        onOpenAuth={() => setIsAuthModalOpen(true)}
        onLogout={handleLogout}
      />

      <main className="main-content">
        {/* Main Analysis Section / Results Section */}
        {analysisResult ? (
          <AnalysisResults
            prediction={analysisResult.prediction}
            recording={analysisResult.recording}
            explanation={analysisResult.explanation}
            explanationError={analysisResult.explanationError}
            onReset={handleResetAnalysis}
          />
        ) : (
          <AudioAnalysisSection
            token={token}
            onRequireAuth={() => setIsAuthModalOpen(true)}
            onAnalysisSuccess={handleAnalysisSuccess}
            isAnalyzing={isAnalyzing}
            setIsAnalyzing={setIsAnalyzing}
          />
        )}

        {/* Clinical Taxonomy: 4 Target Categories */}
        <ClinicalCategoriesReference />

        {/* Acoustic Deep Learning Pipeline Methodology */}
        <PipelineOverview />
      </main>

      {/* Clean Academic Healthcare Footer */}
      <Footer />

      {/* Non-intrusive API Authentication Modal */}
      <AuthModal
        isOpen={isAuthModalOpen}
        onClose={() => setIsAuthModalOpen(false)}
        token={token}
        onAuthSuccess={handleAuthSuccess}
      />
    </div>
  );
}
