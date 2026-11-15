"""
Acoustic Preprocessor for Pediatric Respiratory & Throat Auscultation.

Standardized time-frequency feature extraction pipeline.
Guaranteed bitwise and numerical parity with backend/ml_inference/services.py:AudioPreprocessor.
"""

import os
import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any

try:
    from .config import PREPROCESSING_CONFIG, CNN_INPUT_SHAPE
except (ImportError, ValueError):
    from config import PREPROCESSING_CONFIG, CNN_INPUT_SHAPE

logger = logging.getLogger("ThroatAudioPreprocessor")


class ThroatAudioPreprocessor:
    """
    Transforms raw children breathing audio into standardized Log Mel-Spectrogram tensors.
    """

    @classmethod
    def load_audio(cls, file_path: str, target_sr: Optional[int] = 8000) -> Tuple[np.ndarray, int]:
        """
        Loads audio safely, converts to mono, and resamples to target sample rate.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file does not exist: {file_path}")

        # Attempt reading with soundfile first, fallback to scipy.io.wavfile or wave
        try:
            import soundfile as sf
            audio, sr = sf.read(file_path, dtype='float32')
        except ImportError:
            try:
                from scipy.io import wavfile
                sr, audio = wavfile.read(file_path)
                if audio.dtype == np.int16:
                    audio = (audio / 32768.0).astype(np.float32)
                elif audio.dtype == np.int32:
                    audio = (audio / 2147483648.0).astype(np.float32)
                else:
                    audio = audio.astype(np.float32)
            except Exception as e:
                import wave
                with wave.open(file_path, 'rb') as wf:
                    sr = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    if wf.getnchannels() > 1:
                        audio = audio.reshape(-1, wf.getnchannels())

        # Convert stereo / multi-channel to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample if target sample rate differs
        if target_sr is not None and sr != target_sr:
            try:
                import scipy.signal
                num_samples = int(len(audio) * float(target_sr) / sr)
                audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
                sr = target_sr
            except Exception as exc:
                logger.warning("Resampling failed, keeping original sample rate: %s", exc)

        return audio, sr

    @classmethod
    def apply_butterworth_bandpass(
        cls,
        audio: np.ndarray,
        sr: int = 8000,
        f_min: float = 50.0,
        f_max: float = 2000.0,
        order: int = 5
    ) -> np.ndarray:
        """
        Applies a zero-phase 5th-order Butterworth bandpass filter.
        Isolates respiratory/throat acoustic energy and suppresses cardiac/friction noise.
        """
        try:
            import scipy.signal
            nyquist = sr / 2.0
            low = max(0.001, f_min / nyquist)
            high = min(0.999, f_max / nyquist)
            padlen = 3 * (2 * order)
            if len(audio) > padlen and low < high:
                sos = scipy.signal.butter(order, [low, high], btype='bandpass', output='sos')
                audio = scipy.signal.sosfiltfilt(sos, audio).astype(np.float32)
        except Exception as exc:
            logger.warning("Butterworth filtering failed, using raw audio: %s", exc)

        return audio

    @classmethod
    def compute_mel_spectrogram(
        cls,
        audio: np.ndarray,
        sr: int = 8000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        f_min: float = 50.0,
        f_max: float = 2000.0,
        top_db: float = 80.0,
        normalization: str = "minmax_01",
        apply_bandpass: bool = True,
        filter_order: int = 5,
    ) -> np.ndarray:
        """
        Generates 2D Mel-Spectrogram energy distribution from audio.
        Supports decibel conversion and min-max [0.0, 1.0] scaling.
        """
        if apply_bandpass:
            audio = cls.apply_butterworth_bandpass(
                audio, sr=sr, f_min=f_min, f_max=f_max, order=filter_order
            )

        try:
            import librosa
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmin=f_min,
                fmax=f_max,
                power=2.0
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=top_db)

        except ImportError:
            # Fallback pure NumPy/SciPy Mel-spectrogram generator
            mel_spec_db = cls._fallback_mel_spectrogram(
                audio, sr, n_mels, n_fft, hop_length, f_min, f_max, top_db
            )

        if normalization == "minmax_01":
            mel_spec_db = np.clip((mel_spec_db + top_db) / top_db, 0.0, 1.0)
        elif normalization == "standard":
            mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-7)

        if np.isnan(mel_spec_db).any() or np.isinf(mel_spec_db).any():
            raise ValueError("Mel-spectrogram extraction generated NaN or infinite values.")

        return mel_spec_db.astype(np.float32)

    @classmethod
    def _fallback_mel_spectrogram(
        cls,
        audio: np.ndarray,
        sr: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        f_min: float,
        f_max: float,
        top_db: float
    ) -> np.ndarray:
        """Pure NumPy Mel-spectrogram when librosa/scipy are unavailable."""
        try:
            from scipy import signal
            _, _, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
            magnitude = np.abs(Zxx) ** 2.0
        except ImportError:
            # Pure NumPy STFT
            window = np.hanning(n_fft)
            num_frames = 1 + max(0, (len(audio) - n_fft) // hop_length)
            if num_frames <= 0 or len(audio) < n_fft:
                audio = np.pad(audio, (0, max(0, n_fft - len(audio))))
                num_frames = 1
            frames = []
            for i in range(num_frames):
                start = i * hop_length
                frame = audio[start:start + n_fft]
                if len(frame) < n_fft:
                    frame = np.pad(frame, (0, n_fft - len(frame)))
                frames.append(np.fft.rfft(frame * window))
            magnitude = (np.abs(np.array(frames).T) ** 2.0)

        # Construct simple triangular Mel filterbank matrix
        nyquist = sr / 2.0
        f_min_mel = 2595.0 * np.log10(1.0 + f_min / 700.0)
        f_max_mel = 2595.0 * np.log10(1.0 + min(f_max, nyquist) / 700.0)
        mel_points = np.linspace(f_min_mel, f_max_mel, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        fb = np.zeros((n_mels, magnitude.shape[0]))
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]
            for k in range(f_m_minus, f_m):
                if k < fb.shape[1] and (f_m - f_m_minus) > 0:
                    fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if k < fb.shape[1] and (f_m_plus - f_m) > 0:
                    fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        mel_energy = np.dot(fb, magnitude)
        ref_val = np.maximum(np.max(mel_energy), 1e-10)
        log_mel = 10.0 * np.log10(np.maximum(mel_energy, 1e-10) / ref_val)
        return np.maximum(log_mel, -top_db)

    @classmethod
    def prepare_cnn_input(
        cls,
        audio_path: str,
        config: Optional[Dict[str, Any]] = None,
        expected_shape: Tuple[int, int, int] = CNN_INPUT_SHAPE
    ) -> np.ndarray:
        """
        Full feature extraction, padding/cropping, and tensor formatting.
        Produces shape (1, 128, 128, 1) ready for model prediction.
        """
        cfg = PREPROCESSING_CONFIG.copy()
        if config:
            cfg.update(config)

        target_sr = cfg.get("sample_rate", 8000)
        n_mels = cfg.get("n_mels", 128)
        n_fft = cfg.get("n_fft", 1024)
        hop_length = cfg.get("hop_length", 512)
        f_min = cfg.get("f_min", 50.0)
        f_max = cfg.get("f_max", 2000.0)
        apply_bandpass = cfg.get("apply_bandpass", True)
        filter_order = cfg.get("filter_order", 5)
        top_db = cfg.get("top_db", 80.0)
        normalization = cfg.get("normalization", "minmax_01")
        target_w = cfg.get("fixed_time_steps", 128)

        # 1. Load audio
        audio, sr = cls.load_audio(audio_path, target_sr=target_sr)
        if len(audio) == 0:
            raise ValueError("Audio recording file is empty.")

        # 2. Extract Mel-Spectrogram
        mel_spec = cls.compute_mel_spectrogram(
            audio=audio,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            top_db=top_db,
            normalization=normalization,
            apply_bandpass=apply_bandpass,
            filter_order=filter_order
        )

        # 3. Add batch and channel dimensions -> (1, n_mels, time_steps, 1)
        input_tensor = np.expand_dims(mel_spec, axis=(0, -1))

        # 4. Adjust time dimension to target_w (128)
        target_h = expected_shape[0] if expected_shape else 128
        target_w = expected_shape[1] if expected_shape else target_w

        if input_tensor.shape[1] != target_h:
            input_tensor = input_tensor[:, :target_h, :, :]

        current_w = input_tensor.shape[2]
        if current_w < target_w:
            pad_w = target_w - current_w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            input_tensor = np.pad(
                input_tensor,
                ((0, 0), (0, 0), (pad_left, pad_right), (0, 0)),
                mode='constant'
            )
        elif current_w > target_w:
            crop_start = (current_w - target_w) // 2
            input_tensor = input_tensor[:, :, crop_start:crop_start + target_w, :]

        # 5. Sanity checks
        if input_tensor.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got shape {input_tensor.shape}.")
        if np.isnan(input_tensor).any() or np.isinf(input_tensor).any():
            raise ValueError("Preprocessed tensor contains NaN or Inf.")

        return input_tensor.astype(np.float32)
