import React from 'react';

/**
 * Acoustic Epidemiology Brand Identity.
 * "Understand respiratory sounds through intelligent acoustic analysis."
 */
export default function Logo({ light = false, subtitle = true }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      {/* Signature Respiratory Emblem: stethoscope curve + acoustic wave */}
      <span 
        style={{ 
          width: 40, 
          height: 40, 
          borderRadius: 'var(--radius-md)', 
          background: 'linear-gradient(135deg, var(--accent-primary) 0%, #1D6360 100%)', 
          display: 'grid', 
          placeItems: 'center', 
          color: '#ffffff',
          boxShadow: '0 2px 8px -2px rgba(36, 124, 120, 0.35)',
          flexShrink: 0,
          border: '1px solid rgba(255, 255, 255, 0.15)'
        }}
      >
        <svg 
          width="24" 
          height="24" 
          viewBox="0 0 24 24" 
          fill="none" 
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Stylized respiratory acoustic curve */}
          <path 
            d="M2 12h3l2-4 2 8 2-6 2 4 2-2h7" 
            stroke="currentColor" 
            strokeWidth="2" 
            strokeLinecap="round" 
            strokeLinejoin="round" 
          />
          {/* Subtle upper harmonic dot */}
          <circle cx="11" cy="6" r="1.2" fill="var(--accent-coral)" />
        </svg>
      </span>

      <div style={{ display: 'flex', flexDirection: 'column' }}>
        <span 
          style={{ 
            fontWeight: 'var(--font-weight-bold)', 
            fontSize: 'var(--font-size-md)', 
            lineHeight: 1.15, 
            color: light ? 'var(--text-inverse)' : 'var(--text-primary)',
            letterSpacing: 'var(--letter-spacing-tight)',
            fontFamily: 'var(--font-display)'
          }}
        >
          Acoustic Epidemiology
        </span>
        {subtitle && (
          <span 
            style={{ 
              fontSize: '0.6875rem', 
              color: light ? 'rgba(255,255,255,0.78)' : 'var(--accent-eucalyptus)',
              letterSpacing: '0.03em',
              textTransform: 'uppercase',
              fontWeight: 600,
              marginTop: 2
            }}
          >
            Respiratory Acoustic AI
          </span>
        )}
      </div>
    </div>
  );
}
