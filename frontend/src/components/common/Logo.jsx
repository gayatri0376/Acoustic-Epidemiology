import { AudioWaveform } from 'lucide-react';

// Text-based brand identity (icon + wordmark).
export default function Logo({ light = false }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <span style={{ width: 38, height: 38, borderRadius: 11, background: '#117f86', display: 'grid', placeItems: 'center', color: '#fff' }}>
        <AudioWaveform size={22} aria-hidden="true" />
      </span>
      <span style={{ fontWeight: 700, fontSize: '1.05rem', lineHeight: 1.1, color: light ? '#fff' : '#263b50' }}>
        Acoustic<br />Epidemiology
      </span>
    </div>
  );
}
