import React from 'react';
import { Stethoscope } from 'lucide-react';
import { TARGET_CATEGORIES } from '../utils/clinicalCategories';

export default function ClinicalCategoriesReference() {
  return (
    <section id="categories-section" className="section-container">
      <div className="section-header text-center">
        <span className="badge badge-cyan">
          <Stethoscope size={14} /> Diagnostic Acoustic Taxonomy
        </span>
        <h2 className="section-title">Target Respiratory Categories</h2>
        <p className="section-description">
          The four evaluated pediatric auscultation classes based on the SPRSound pediatric respiratory acoustics standard.
        </p>
      </div>

      <div className="categories-grid">
        {Object.values(TARGET_CATEGORIES).map((cat) => (
          <div key={cat.code} className="category-card glass-panel" id={`cat-card-${cat.code.toLowerCase().replace(/[^a-z0-9]/g, '-')}`}>
            <div className="category-card-header">
              <span className={`badge ${cat.badgeClass}`}>
                {cat.code}
              </span>
              <span className="category-sound-type text-xs font-mono text-muted">
                {cat.soundType}
              </span>
            </div>

            <h3 className="category-title">{cat.label}</h3>
            <p className="category-short-desc">{cat.shortDescription}</p>

            <div className="category-specs-list">
              <div className="category-spec-row">
                <span className="spec-label">Acoustic Duration</span>
                <span className="spec-val font-mono">{cat.duration}</span>
              </div>
              <div className="category-spec-row">
                <span className="spec-label">Frequency Spectrum</span>
                <span className="spec-val font-mono">{cat.frequencyRange}</span>
              </div>
            </div>

            <p className="category-clinical-desc">{cat.clinicalSummary}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
