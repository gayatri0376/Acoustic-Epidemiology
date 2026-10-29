import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Play, Pause, Volume2, VolumeX, RotateCcw } from 'lucide-react';

function formatTime(seconds) {
  if (isNaN(seconds) || seconds === null) return '0:00';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
}

export default function AudioPlayer({ file, onDurationDetected }) {
  const audioRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);

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

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      const dur = audioRef.current.duration;
      if (!isNaN(dur) && isFinite(dur)) {
        setDuration(dur);
        if (onDurationDetected) {
          onDurationDetected(dur);
        }
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

  const handleSeek = (e) => {
    const newTime = parseFloat(e.target.value);
    setCurrentTime(newTime);
    if (audioRef.current) {
      audioRef.current.currentTime = newTime;
    }
  };

  const handleReset = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
      setCurrentTime(0);
    }
  };

  const toggleMute = () => {
    if (!audioRef.current) return;
    const nextMuted = !isMuted;
    setIsMuted(nextMuted);
    audioRef.current.muted = nextMuted;
  };

  const handleVolumeChange = (e) => {
    const val = parseFloat(e.target.value);
    setVolume(val);
    if (audioRef.current) {
      audioRef.current.volume = val;
      if (val === 0) {
        setIsMuted(true);
      } else if (isMuted) {
        setIsMuted(false);
      }
    }
  };

  if (!audioUrl) return null;

  return (
    <div className="audio-player-card">
      <audio
        ref={audioRef}
        src={audioUrl}
        onLoadedMetadata={handleLoadedMetadata}
        onTimeUpdate={handleTimeUpdate}
        onEnded={handleEnded}
        preload="metadata"
      />

      <div className="player-main-row">
        {/* Play/Pause Button */}
        <button
          type="button"
          onClick={togglePlay}
          className="player-play-btn"
          aria-label={isPlaying ? 'Pause audio' : 'Play audio'}
        >
          {isPlaying ? <Pause size={18} /> : <Play size={18} className="translate-x-0.5" />}
        </button>

        {/* Audio Wave Bars (animated when playing) */}
        <div className={`player-wave-bars ${isPlaying ? 'playing' : ''}`}>
          <span className="player-bar"></span>
          <span className="player-bar"></span>
          <span className="player-bar"></span>
          <span className="player-bar"></span>
          <span className="player-bar"></span>
          <span className="player-bar"></span>
          <span className="player-bar"></span>
          <span className="player-bar"></span>
        </div>

        {/* Scrubber & Timestamps */}
        <div className="player-scrubber-group">
          <input
            type="range"
            min={0}
            max={duration || 100}
            step="0.01"
            value={currentTime}
            onChange={handleSeek}
            className="player-seek-slider"
            aria-label="Audio timeline"
          />
          <div className="player-timestamps">
            <span className="font-mono text-xs">{formatTime(currentTime)}</span>
            <span className="font-mono text-xs text-muted">/ {formatTime(duration)}</span>
          </div>
        </div>

        {/* Volume & Reset Controls */}
        <div className="player-aux-controls">
          <button
            type="button"
            onClick={handleReset}
            className="player-icon-btn"
            title="Rewind to start"
            aria-label="Rewind"
          >
            <RotateCcw size={15} />
          </button>

          <button
            type="button"
            onClick={toggleMute}
            className="player-icon-btn"
            title={isMuted ? 'Unmute' : 'Mute'}
            aria-label={isMuted ? 'Unmute' : 'Mute'}
          >
            {isMuted || volume === 0 ? <VolumeX size={16} /> : <Volume2 size={16} />}
          </button>

          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={isMuted ? 0 : volume}
            onChange={handleVolumeChange}
            className="player-volume-slider"
            aria-label="Volume level"
          />
        </div>
      </div>
    </div>
  );
}
