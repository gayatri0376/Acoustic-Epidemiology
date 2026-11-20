import React from 'react';

/**
 * Reusable Respiratory Waveform Visual Motif.
 * Communicates: sound, respiratory health, calm scientific cadence.
 *
 * Variants:
 * - 'hero': Ambient flowing respiratory wave for hero & welcome sections
 * - 'divider': Subtle horizontal acoustic wave divider
 * - 'pulse': Animated gentle breathing cadence
 * - 'bars': Stylized acoustic frequency bars
 */
export default function WaveformMotif({ 
  variant = 'hero', 
  height = 48, 
  color = 'currentColor', 
  className = '',
  style = {} 
}) {
  if (variant === 'divider') {
    return (
      <div 
        className={`waveform-divider ${className}`}
        style={{ 
          width: '100%', 
          overflow: 'hidden', 
          lineHeight: 0,
          color,
          opacity: 0.45,
          margin: '1.25rem 0',
          ...style 
        }} 
        aria-hidden="true"
      >
        <svg 
          viewBox="0 0 1200 32" 
          fill="none" 
          xmlns="http://www.w3.org/2000/svg"
          style={{ width: '100%', height: Math.min(height, 28) }}
          preserveAspectRatio="none"
        >
          <path 
            d="M0 16 Q 75 16 150 16 T 300 16 Q 340 16 360 8 T 380 24 T 400 6 T 420 26 T 440 16 Q 500 16 600 16 T 780 16 Q 810 16 825 11 T 840 21 T 855 8 T 870 23 T 885 16 Q 960 16 1200 16" 
            stroke="currentColor" 
            strokeWidth="1.5" 
            strokeLinecap="round"
          />
        </svg>
      </div>
    );
  }

  if (variant === 'pulse') {
    return (
      <div 
        className={`waveform-breathing-indicator ${className}`} 
        style={{ 
          display: 'inline-flex', 
          alignItems: 'center', 
          gap: '3px',
          height: `${height}px`,
          color,
          ...style 
        }}
        aria-label="Respiratory cadence active"
      >
        <span className="wave-bar bar-1" />
        <span className="wave-bar bar-2" />
        <span className="wave-bar bar-3" />
        <span className="wave-bar bar-4" />
        <span className="wave-bar bar-5" />
        <span className="wave-bar bar-4" />
        <span className="wave-bar bar-3" />
        <span className="wave-bar bar-2" />
        <span className="wave-bar bar-1" />
      </div>
    );
  }

  if (variant === 'bars') {
    return (
      <svg 
        viewBox="0 0 64 24" 
        fill="none" 
        xmlns="http://www.w3.org/2000/svg" 
        style={{ height: `${height}px`, width: 'auto', ...style }}
        className={className}
        aria-hidden="true"
      >
        <rect x="2" y="9" width="3" height="6" rx="1.5" fill={color} opacity="0.5" />
        <rect x="8" y="6" width="3" height="12" rx="1.5" fill={color} opacity="0.7" />
        <rect x="14" y="3" width="3" height="18" rx="1.5" fill={color} />
        <rect x="20" y="8" width="3" height="8" rx="1.5" fill={color} opacity="0.6" />
        <rect x="26" y="2" width="3" height="20" rx="1.5" fill={color} />
        <rect x="32" y="5" width="3" height="14" rx="1.5" fill={color} opacity="0.8" />
        <rect x="38" y="1" width="3" height="22" rx="1.5" fill={color} />
        <rect x="44" y="7" width="3" height="10" rx="1.5" fill={color} opacity="0.7" />
        <rect x="50" y="4" width="3" height="16" rx="1.5" fill={color} opacity="0.9" />
        <rect x="56" y="9" width="3" height="6" rx="1.5" fill={color} opacity="0.5" />
      </svg>
    );
  }

  // Default 'hero' ambient flow
  return (
    <div 
      className={`waveform-hero-ambient ${className}`}
      style={{ 
        width: '100%', 
        overflow: 'hidden', 
        lineHeight: 0,
        ...style 
      }}
      aria-hidden="true"
    >
      <svg 
        viewBox="0 0 1000 120" 
        fill="none" 
        xmlns="http://www.w3.org/2000/svg"
        style={{ width: '100%', height: `${height}px` }}
        preserveAspectRatio="none"
      >
        {/* Soft Eucalyptus primary breath wave */}
        <path 
          d="M0 60 C 150 15, 250 105, 400 60 C 550 15, 650 105, 800 60 C 900 30, 950 85, 1000 60" 
          stroke="var(--accent-eucalyptus)" 
          strokeWidth="1.8" 
          strokeLinecap="round"
          strokeOpacity="0.45"
        />
        {/* Medical teal acoustic auscultation pulse */}
        <path 
          d="M0 60 Q 200 60 280 60 T 320 25 T 340 95 T 360 40 T 380 75 T 400 60 Q 600 60 680 60 T 710 35 T 730 85 T 750 45 T 770 60 Q 900 60 1000 60" 
          stroke="var(--accent-primary)" 
          strokeWidth="1.75" 
          strokeLinecap="round"
          strokeOpacity="0.65"
        />
        {/* Pale sage ambient grounding wave */}
        <path 
          d="M0 60 C 100 85, 300 35, 500 60 C 700 85, 900 35, 1000 60" 
          stroke="var(--accent-eucalyptus-light)" 
          strokeWidth="1.2" 
          strokeLinecap="round"
          strokeOpacity="0.3"
        />
      </svg>
    </div>
  );
}
