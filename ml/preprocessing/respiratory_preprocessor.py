"""
Respiratory Audio Preprocessor & Feature Extraction Module
Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases

Standardized preprocessing and time-frequency feature extraction pipeline
aligned with the native SPRSound pediatric dataset (8,000 Hz, 16-bit mono PCM).

Pipeline Stages:
1. Audio Ingestion: Single-channel mono conversion, safe dtype handling
2. Resampling: Strict preservation of native 8,000 Hz; polyphase resampling if external
3. Physiological Bandpass Filter: 5th-order Butterworth bandpass (50 Hz - 2,000 Hz)
4. Time-Frequency Transform: Short-Time Fourier Transform (STFT) -> Log Mel-Spectrogram
5. Dynamic Range & Normalization: Power-to-dB conversion and linear [0, 1] normalization
6. Dimension Alignment: Fixed-duration windowing and 4D tensor formatting (B, H, W, C)
7. Event-Level Segmentation: Millisecond-to-sample mapping for annotated respiratory cycles
"""

import os
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import scipy.signal
import soundfile as sf
import librosa

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingConfig:
    """
    Unified preprocessing configuration contract.
    Defaults match the native acoustic properties of the SPRSound database.
    """
    sample_rate: int = 8000               # Native SPRSound sampling rate (Hz)
    n_mels: int = 128                     # Number of Mel frequency bands (height dimension)
    n_fft: int = 1024                     # FFT window length (128 ms at 8 kHz)
    hop_length: int = 512                 # Hop size between successive frames (64 ms at 8 kHz)
    f_min: float = 50.0                   # Low frequency cutoff (Hz) - eliminates heart sound artifacts
    f_max: float = 2000.0                 # High frequency cutoff (Hz) - physiological lung sound limit
    filter_order: int = 5                 # Butterworth bandpass filter order
    apply_bandpass: bool = True           # Whether to apply Butterworth bandpass prior to STFT
    top_db: float = 80.0                  # Dynamic range ceiling for decibel scaling
    normalization: str = "minmax_01"      # Normalization mode: 'minmax_01', 'standard', or 'none'
    record_duration_s: float = 8.192      # Duration for whole-record feature array (8.192s -> 128 frames)
    event_duration_s: float = 2.5         # Duration for event-level feature array (2.5s -> 40 frames)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "n_mels": self.n_mels,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "filter_order": self.filter_order,
            "apply_bandpass": self.apply_bandpass,
            "top_db": self.top_db,
            "normalization": self.normalization,
            "record_duration_s": self.record_duration_s,
            "event_duration_s": self.event_duration_s,
        }

    def validate(self) -> None:
        """Validates that parameters satisfy signal processing and Nyquist constraints."""
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        nyquist = self.sample_rate / 2.0
        if not (0.0 < self.f_min < self.f_max):
            raise ValueError(f"Invalid frequency range: f_min={self.f_min}, f_max={self.f_max}")
        if self.f_max > nyquist:
            raise ValueError(
                f"f_max ({self.f_max} Hz) exceeds Nyquist limit ({nyquist} Hz) "
                f"for sample_rate={self.sample_rate} Hz"
            )
        if self.n_mels <= 0 or self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError("STFT parameters (n_mels, n_fft, hop_length) must be positive integers.")
        if self.hop_length > self.n_fft:
            raise ValueError("hop_length cannot exceed n_fft.")


class RespiratoryAudioPreprocessor:
    """
    Stateless acoustic preprocessor implementing reproducible feature extraction
    for both whole-record and event-segmented pediatric respiratory audio.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.config.validate()

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Loads an audio file, converts multi-channel audio to mono via channel averaging,
        and resamples to target sample rate if needed.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file does not exist: {file_path}")

        audio, sr = sf.read(file_path, dtype='float32')

        if len(audio) == 0:
            raise ValueError(f"Audio file contains zero samples: {file_path}")

        # Convert multi-channel to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample only if necessary
        target_sr = self.config.sample_rate
        if sr != target_sr:
            # High-quality polyphase / sinc resampling
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type='soxr_hq')
            sr = target_sr

        return audio, sr

    def apply_butterworth_filter(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None,
        f_min: Optional[float] = None,
        f_max: Optional[float] = None,
        order: Optional[int] = None
    ) -> np.ndarray:
        """
        Applies a zero-phase 5th-order Butterworth bandpass filter using Second-Order Sections (SOS).
        Zero-phase forward-backward filtering ensures zero phase distortion in breathing waveforms.
        """
        sr = sr or self.config.sample_rate
        f_min = f_min or self.config.f_min
        f_max = f_max or self.config.f_max
        order = order or self.config.filter_order

        nyquist = sr / 2.0
        if f_max >= nyquist:
            f_max = nyquist - 1.0

        # Calculate normalized critical frequencies
        low = f_min / nyquist
        high = f_max / nyquist

        # Guard against signal shorter than filter pad requirements
        padlen = 3 * (2 * order)
        if len(audio) <= padlen:
            logger.warning("Audio duration too short for Butterworth SOS filtering; returning unfiltered signal.")
            return audio

        sos = scipy.signal.butter(order, [low, high], btype='bandpass', output='sos')
        filtered = scipy.signal.sosfiltfilt(sos, audio)
        return filtered.astype(np.float32)

    def compute_mel_spectrogram(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Generates 2D Log Mel-Spectrogram energy representation.
        Returns shape (n_mels, time_frames) with decibel scaling.
        """
        sr = sr or self.config.sample_rate

        # 1. Bandpass filter if configured
        if self.config.apply_bandpass:
            audio = self.apply_butterworth_filter(audio, sr=sr)

        # 2. Extract linear Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            fmin=self.config.f_min,
            fmax=self.config.f_max,
            power=2.0
        )

        # 3. Log-amplitude scaling (dB relative to peak)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=self.config.top_db)

        # 4. Normalization
        if self.config.normalization == "minmax_01":
            # Scale from [-top_db, 0] dB to [0.0, 1.0]
            mel_norm = np.clip((mel_db + self.config.top_db) / self.config.top_db, 0.0, 1.0)
            return mel_norm.astype(np.float32)
        elif self.config.normalization == "standard":
            mean = np.mean(mel_db)
            std = np.std(mel_db) + 1e-7
            mel_norm = (mel_db - mean) / std
            return mel_norm.astype(np.float32)
        else:
            return mel_db.astype(np.float32)

    def align_spectrogram_time_axis(
        self,
        spectrogram: np.ndarray,
        target_frames: int
    ) -> np.ndarray:
        """
        Aligns the width (time frames) of a spectrogram to target_frames
        via symmetric center-padding or center-cropping.
        """
        current_frames = spectrogram.shape[1]
        if current_frames == target_frames:
            return spectrogram

        if current_frames < target_frames:
            pad_total = target_frames - current_frames
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return np.pad(spectrogram, ((0, 0), (pad_left, pad_right)), mode='constant')
        else:
            crop_start = (current_frames - target_frames) // 2
            return spectrogram[:, crop_start:crop_start + target_frames]

    def format_cnn_tensor(
        self,
        spectrogram: np.ndarray,
        target_frames: Optional[int] = None
    ) -> np.ndarray:
        """
        Formats 2D spectrogram into 4D tensor (batch, height, width, channels)
        compatible with Keras 2D-CNN architectures: (1, n_mels, time_frames, 1).
        """
        if target_frames is not None:
            spectrogram = self.align_spectrogram_time_axis(spectrogram, target_frames)

        # Shape: (1, n_mels, time_frames, 1)
        tensor = np.expand_dims(spectrogram, axis=(0, -1)).astype(np.float32)
        return tensor

    def extract_whole_record_features(
        self,
        file_path: str,
        fixed_frames: Optional[int] = 128
    ) -> Dict[str, Any]:
        """
        Extracts whole-recording feature tensor and metadata.
        Target frames: 128 (resulting in standard (1, 128, 128, 1) CNN input).
        """
        audio, sr = self.load_audio(file_path)
        mel_spec = self.compute_mel_spectrogram(audio, sr=sr)
        tensor = self.format_cnn_tensor(mel_spec, target_frames=fixed_frames)

        return {
            "spectrogram": mel_spec,
            "tensor": tensor,
            "shape": list(tensor.shape),
            "sample_rate": sr,
            "raw_duration_seconds": len(audio) / sr,
            "time_frames_raw": mel_spec.shape[1],
            "time_frames_aligned": tensor.shape[2],
        }

    def extract_event_features(
        self,
        audio: np.ndarray,
        sr: int,
        start_ms: float,
        end_ms: float,
        target_duration_s: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Extracts feature tensor for a specific respiratory cycle event
        identified by millisecond timestamps [start_ms, end_ms].
        Duration is center-padded or center-cropped to target_duration_s (default: 2.5s -> 40 frames).
        """
        target_duration_s = target_duration_s or self.config.event_duration_s
        target_samples = int(round(target_duration_s * sr))

        start_sample = max(0, int(round(start_ms * sr / 1000.0)))
        end_sample = min(len(audio), int(round(end_ms * sr / 1000.0)))

        if start_sample >= end_sample:
            raise ValueError(f"Invalid timestamp range: start_sample={start_sample} >= end_sample={end_sample}")

        event_slice = audio[start_sample:end_sample]
        actual_event_dur = len(event_slice) / sr

        # Align audio slice length to target_samples via center-padding / center-cropping
        if len(event_slice) < target_samples:
            pad_total = target_samples - len(event_slice)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            aligned_audio = np.pad(event_slice, (pad_left, pad_right), mode='constant')
        else:
            crop_start = (len(event_slice) - target_samples) // 2
            aligned_audio = event_slice[crop_start:crop_start + target_samples]

        # Compute Mel-spectrogram
        mel_spec = self.compute_mel_spectrogram(aligned_audio, sr=sr)

        # Expected event frames: for 2.5s at 8kHz with n_fft=1024, hop=512 -> 40 frames
        target_frames = 1 + int(math.floor(target_samples / self.config.hop_length))
        tensor = self.format_cnn_tensor(mel_spec, target_frames=target_frames)

        return {
            "spectrogram": mel_spec,
            "tensor": tensor,
            "shape": list(tensor.shape),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "event_duration_seconds": actual_event_dur,
            "padded_duration_seconds": len(aligned_audio) / sr,
            "tensor_shape": list(tensor.shape),
        }
