import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { Play, Pause, Volume2, VolumeX, RotateCcw } from 'lucide-react';

function formatTime(seconds) {
  if (isNaN(seconds) || seconds === null) return '0:00';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
}

/**
 * Premium Respiratory Audio Player with Interactive Waveform Canvas.
 * Generates true acoustic peak visualization from audio data or smooth procedural waveform.
 */
export default function AudioPlayer({ 
  file, 
  onDurationDetected,
  compact = false,
  autoPlay = false 
}) {
  const audioRef = useRef(null);
  const canvasRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [peaks, setPeaks] = useState([]);
  const [isDecoding, setIsDecoding] = useState(false);

  // Object URL for audio element
  const audioUrl = useMemo(() => {
    if (!file) return null;
    if (typeof file === 'string') return file;
    if (file instanceof Blob || file instanceof File) {
      return URL.createObjectURL(file);
    }
    return null;
  }, [file]);

  useEffect(() => {
    return () => {
      if (audioUrl && typeof file !== 'string') {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl, file]);

  // Extract actual audio peaks using Web Audio API decodeAudioData
  useEffect(() => {
    if (!file || typeof window === 'undefined') return;

    let isMounted = true;
    const NUM_BARS = 64;

    const extractPeaks = async () => {
      try {
        setIsDecoding(true);
        let arrayBuffer;
        if (file instanceof Blob || file instanceof File) {
          arrayBuffer = await file.slice(0).arrayBuffer();
        } else if (typeof file === 'string') {
          const res = await fetch(file);
          arrayBuffer = await res.arrayBuffer();
        }

        if (!arrayBuffer || !isMounted) return;

        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) throw new Error('Web Audio not supported');
        const audioCtx = new AudioCtx();
        
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        if (!isMounted) return;

        const channelData = audioBuffer.getChannelData(0); // Mono channel
        const blockSize = Math.floor(channelData.length / NUM_BARS);
        const extractedPeaks = [];

        for (let i = 0; i < NUM_BARS; i++) {
          const start = i * blockSize;
          let sum = 0;
          for (let j = 0; j < blockSize; j++) {
            sum += Math.abs(channelData[start + j] || 0);
          }
          // Normalize peak amplitude between 0.15 and 0.95 for aesthetic clarity
          const avg = sum / blockSize;
          const scaled = Math.min(0.95, Math.max(0.12, avg * 3.5));
          extractedPeaks.push(scaled);
        }

        if (isMounted) {
          setPeaks(extractedPeaks);
          if (audioBuffer.duration && (!duration || duration === 0)) {
            setDuration(audioBuffer.duration);
            if (onDurationDetected) onDurationDetected(audioBuffer.duration);
          }
        }
        audioCtx.close();
      } catch (err) {
        // Fallback procedural peaks that simulate genuine respiratory wheeze/breath rhythm
        if (isMounted) {
          const fallback = [];
          for (let i = 0; i < NUM_BARS; i++) {
            const angle = (i / NUM_BARS) * Math.PI * 4;
            const envelope = Math.sin((i / NUM_BARS) * Math.PI);
            const val = 0.2 + 0.6 * envelope * (0.5 + 0.5 * Math.sin(angle * 2.3));
            fallback.push(Math.max(0.12, Math.min(0.95, val)));
          }
          setPeaks(fallback);
        }
      } finally {
        if (isMounted) setIsDecoding(false);
      }
    };

    extractPeaks();
    return () => { isMounted = false; };
  }, [file]);

  // Handle metadata loaded
  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      const dur = audioRef.current.duration;
      if (!isNaN(dur) && isFinite(dur) && dur > 0) {
        setDuration(dur);
        if (onDurationDetected) onDurationDetected(dur);
      }
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleEnded = () => {
    setIsPlaying(false);
    setCurrentTime(0);
  };

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().then(() => {
        setIsPlaying(true);
      }).catch((err) => {
        console.warn('Playback error:', err);
        setIsPlaying(false);
      });
    }
  };

  const handleReset = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
      setCurrentTime(0);
    }
  };

  const handleSeekFromCanvas = (e) => {
    if (!canvasRef.current || duration <= 0) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    const newTime = ratio * duration;
    setCurrentTime(newTime);
    if (audioRef.current) {
      audioRef.current.currentTime = newTime;
    }
  };

  const toggleMute = () => {
    if (!audioRef.current) return;
    const nextMuted = !isMuted;
    setIsMuted(nextMuted);
    audioRef.current.muted = nextMuted;
  };

  // Render Waveform on Canvas
  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const bars = peaks.length > 0 ? peaks : Array.from({ length: 64 }, () => 0.25);
    const numBars = bars.length;
    const barWidth = Math.max(2, (width / numBars) - 1.8);
    const progress = duration > 0 ? (currentTime / duration) : 0;

    bars.forEach((peak, i) => {
      const x = i * (width / numBars);
      const barHeight = Math.max(4, peak * (height - 8));
      const y = (height - barHeight) / 2;
      const isPlayed = (i / numBars) <= progress;

      // Color scheme: Played is Medical Teal (#247C78), Unplayed is Pale Sage (#CCDCD5)
      ctx.fillStyle = isPlayed ? '#247C78' : '#D0DDD7';
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(x, y, barWidth, barHeight, 2);
      } else {
        ctx.rect(x, y, barWidth, barHeight);
      }
      ctx.fill();
    });

    // Draw active playhead needle
    if (progress > 0 && progress <= 1) {
      const needleX = progress * width;
      ctx.fillStyle = '#1D6360';
      ctx.fillRect(needleX - 1, 0, 2, height);
    }
  }, [peaks, currentTime, duration]);

  useEffect(() => {
    drawWaveform();
  }, [drawWaveform]);

  // Resize canvas according to display width
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * (window.devicePixelRatio || 1);
      canvas.height = rect.height * (window.devicePixelRatio || 1);
      const ctx = canvas.getContext('2d');
      ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
      drawWaveform();
    }
  }, [peaks, drawWaveform]);

  return (
    <div className={`clinical-audio-player ${compact ? 'compact' : ''}`}>
      <audio
        ref={audioRef}
        src={audioUrl || ''}
        onLoadedMetadata={handleLoadedMetadata}
        onTimeUpdate={handleTimeUpdate}
        onEnded={handleEnded}
        preload="metadata"
      />

      {/* Main Playback Row */}
      <div className="player-inner-wrap">
        
        {/* Play/Pause Button */}
        <button
          type="button"
          onClick={togglePlay}
          className="player-play-btn"
          aria-label={isPlaying ? 'Pause auscultation audio' : 'Play auscultation audio'}
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" style={{ marginLeft: 2 }} />}
        </button>

        {/* Interactive Waveform Canvas Container */}
        <div 
          className="waveform-canvas-container"
          onClick={handleSeekFromCanvas}
          title="Click to seek anywhere along the acoustic recording"
          role="slider"
          aria-label="Audio scrubber waveform"
          aria-valuemin={0}
          aria-valuemax={duration || 100}
          aria-valuenow={currentTime}
        >
          <canvas
            ref={canvasRef}
            className="waveform-canvas"
            style={{ width: '100%', height: compact ? '36px' : '48px', cursor: 'pointer' }}
          />
        </div>

        {/* Time Elapsed & Duration */}
        <div className="player-time-badge" aria-label="Playback time">
          <span className="current font-mono">{formatTime(currentTime)}</span>
          <span className="divider">/</span>
          <span className="total font-mono">{formatTime(duration)}</span>
        </div>

        {/* Controls: Reset & Volume */}
        <div className="player-extra-controls">
          <button
            type="button"
            onClick={handleReset}
            className="player-icon-btn"
            title="Reset to beginning"
            aria-label="Reset to beginning"
          >
            <RotateCcw size={15} />
          </button>
          
          <button
            type="button"
            onClick={toggleMute}
            className={`player-icon-btn ${isMuted ? 'muted' : ''}`}
            title={isMuted ? 'Unmute' : 'Mute'}
            aria-label={isMuted ? 'Unmute' : 'Mute'}
          >
            {isMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
          </button>
        </div>
      </div>
    </div>
  );
}
