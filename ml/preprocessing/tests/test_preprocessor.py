"""
Unit Tests for Respiratory Audio Preprocessing & Feature Extraction
Acoustic Epidemiology Project

All tests in this suite utilize strictly synthetic audio waveforms
(sine waves, sweeps, chirps, noise). Synthetic audio is never treated as research data.
"""

import os
import io
import math
import tempfile
import unittest
import numpy as np
import soundfile as sf
from pathlib import Path

from ml.preprocessing.respiratory_preprocessor import (
    PreprocessingConfig,
    RespiratoryAudioPreprocessor
)


class TestRespiratoryPreprocessingConfig(unittest.TestCase):
    """Tests configuration schema validation and Nyquist constraints."""

    def test_default_config_validity(self):
        config = PreprocessingConfig()
        config.validate()
        self.assertEqual(config.sample_rate, 8000)
        self.assertEqual(config.n_mels, 128)
        self.assertEqual(config.f_min, 50.0)
        self.assertEqual(config.f_max, 2000.0)

    def test_fmax_exceeding_nyquist_raises_error(self):
        # Sample rate 8000 Hz has Nyquist 4000 Hz; f_max 5000 Hz is invalid
        config = PreprocessingConfig(sample_rate=8000, f_max=5000.0)
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("exceeds Nyquist limit", str(ctx.exception))

    def test_invalid_frequency_ordering_raises_error(self):
        config = PreprocessingConfig(sample_rate=8000, f_min=1000.0, f_max=500.0)
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("Invalid frequency range", str(ctx.exception))

    def test_negative_or_zero_sample_rate_raises_error(self):
        config = PreprocessingConfig(sample_rate=0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_hop_length_exceeding_nfft_raises_error(self):
        config = PreprocessingConfig(n_fft=512, hop_length=1024)
        with self.assertRaises(ValueError):
            config.validate()


class TestAudioIngestionAndResampling(unittest.TestCase):
    """Tests multi-channel to mono conversion and sample rate handling."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.preprocessor = RespiratoryAudioPreprocessor()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_wav(self, duration_s=1.0, sr=8000, channels=1):
        """Generates a synthetic stereo or mono WAV file for unit testing."""
        n_samples = int(duration_s * sr)
        t = np.linspace(0, duration_s, n_samples, endpoint=False)
        # Synthetic harmonic signal simulating breath sounds
        signal = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 600 * t)
        if channels > 1:
            signal = np.stack([signal, 0.8 * signal], axis=1)

        path = os.path.join(self.temp_dir.name, f"test_{sr}hz_{channels}ch.wav")
        sf.write(path, signal.astype(np.float32), sr)
        return path

    def test_native_8000hz_mono_audio(self):
        wav_path = self._create_synthetic_wav(duration_s=1.5, sr=8000, channels=1)
        audio, sr = self.preprocessor.load_audio(wav_path)
        self.assertEqual(sr, 8000)
        self.assertEqual(audio.ndim, 1)
        self.assertEqual(len(audio), int(1.5 * 8000))

    def test_stereo_to_mono_averaging(self):
        wav_path = self._create_synthetic_wav(duration_s=1.0, sr=8000, channels=2)
        audio, sr = self.preprocessor.load_audio(wav_path)
        self.assertEqual(sr, 8000)
        self.assertEqual(audio.ndim, 1)
        self.assertEqual(len(audio), 8000)

    def test_resampling_from_16000hz_to_8000hz(self):
        wav_path = self._create_synthetic_wav(duration_s=1.0, sr=16000, channels=1)
        audio, sr = self.preprocessor.load_audio(wav_path)
        self.assertEqual(sr, 8000)
        # 1.0 second at 8000 Hz should yield approximately 8000 samples
        self.assertTrue(abs(len(audio) - 8000) <= 2)

    def test_resampling_from_44100hz_to_8000hz(self):
        wav_path = self._create_synthetic_wav(duration_s=1.0, sr=44100, channels=1)
        audio, sr = self.preprocessor.load_audio(wav_path)
        self.assertEqual(sr, 8000)
        self.assertTrue(abs(len(audio) - 8000) <= 5)

    def test_nonexistent_audio_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.load_audio("/tmp/nonexistent_acoustic_file.wav")

    def test_empty_audio_file_raises_error(self):
        empty_path = os.path.join(self.temp_dir.name, "empty.wav")
        # Write empty file
        sf.write(empty_path, np.array([], dtype=np.float32), 8000)
        with self.assertRaises(ValueError):
            self.preprocessor.load_audio(empty_path)


class TestButterworthFiltering(unittest.TestCase):
    """Tests 5th-order Butterworth bandpass filter frequency response."""

    def setUp(self):
        self.preprocessor = RespiratoryAudioPreprocessor()

    def test_bandpass_attenuates_low_and_high_frequencies(self):
        fs = 8000
        duration = 1.0
        n_samples = int(duration * fs)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # 3 components: 20 Hz (below f_min), 500 Hz (in-band), 3500 Hz (above f_max)
        low_noise = 1.0 * np.sin(2 * np.pi * 20 * t)       # Sub-audible / heart sound
        in_band_signal = 1.0 * np.sin(2 * np.pi * 500 * t) # Lung sound
        high_noise = 1.0 * np.sin(2 * np.pi * 3500 * t)    # Stethoscope sensor hiss

        composite = low_noise + in_band_signal + high_noise
        filtered = self.preprocessor.apply_butterworth_filter(composite, sr=fs, f_min=50.0, f_max=2000.0)

        # In-band signal power should be largely preserved, out-of-band heavily attenuated
        fft_orig = np.abs(np.fft.rfft(composite))
        fft_filt = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(n_samples, 1 / fs)

        # Find bin indices
        idx_20 = np.argmin(np.abs(freqs - 20))
        idx_500 = np.argmin(np.abs(freqs - 500))
        idx_3500 = np.argmin(np.abs(freqs - 3500))

        # Check attenuation: 20 Hz and 3500 Hz should be attenuated by > 20 dB (factor of 10)
        self.assertTrue(fft_filt[idx_20] < 0.1 * fft_orig[idx_20])
        self.assertTrue(fft_filt[idx_3500] < 0.1 * fft_orig[idx_3500])
        # 500 Hz should be preserved within 10%
        self.assertTrue(fft_filt[idx_500] > 0.85 * fft_orig[idx_500])

    def test_short_signal_safe_fallback(self):
        # Signal shorter than filter pad requirement
        tiny_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        out = self.preprocessor.apply_butterworth_filter(tiny_audio, sr=8000)
        self.assertEqual(len(out), len(tiny_audio))


class TestMelSpectrogramExtraction(unittest.TestCase):
    """Tests STFT, Log Mel-Spectrogram, and 4D tensor formatting."""

    def setUp(self):
        self.preprocessor = RespiratoryAudioPreprocessor()

    def test_mel_spectrogram_dimensions_and_normalization(self):
        fs = 8000
        duration = 2.5
        n_samples = int(duration * fs)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        mel_spec = self.preprocessor.compute_mel_spectrogram(audio, sr=fs)

        # Frequency bands must equal n_mels (128)
        self.assertEqual(mel_spec.shape[0], 128)
        # Values must be bounded in [0.0, 1.0] for minmax_01
        self.assertTrue(mel_spec.min() >= 0.0)
        self.assertTrue(mel_spec.max() <= 1.0)
        self.assertFalse(np.isnan(mel_spec).any())
        self.assertFalse(np.isinf(mel_spec).any())

    def test_stft_exact_frame_count_regression(self):
        """
        Regression test: Verifies exact STFT frame count behavior under center=True.
        Formula: frames = 1 + floor(N / hop_length) for center=True with n_fft=1024, hop=512.
        """
        fs = 8000
        hop = 512

        # 1. Exactly 65024 samples (8.128s) -> exactly 128 frames (1 + 65024/512 = 128)
        audio_8128 = np.zeros(65024, dtype=np.float32)
        mel_8128 = self.preprocessor.compute_mel_spectrogram(audio_8128, sr=fs)
        self.assertEqual(mel_8128.shape, (128, 128))

        # 2. Exactly 65536 samples (8.192s = 128 * 512) -> exactly 129 frames (1 + 65536/512 = 129)
        audio_8192 = np.zeros(65536, dtype=np.float32)
        mel_8192 = self.preprocessor.compute_mel_spectrogram(audio_8192, sr=fs)
        self.assertEqual(mel_8192.shape, (128, 129))

        # 3. Alignment to target 128 frames via center-crop:
        aligned_8192 = self.preprocessor.align_spectrogram_time_axis(mel_8192, target_frames=128)
        self.assertEqual(aligned_8192.shape, (128, 128))

        # 4. SPRSound modal 9.216s audio (73728 samples) -> 145 frames (1 + 73728/512 = 145)
        audio_spr = np.zeros(73728, dtype=np.float32)
        mel_spr = self.preprocessor.compute_mel_spectrogram(audio_spr, sr=fs)
        self.assertEqual(mel_spr.shape, (128, 145))
        aligned_spr = self.preprocessor.align_spectrogram_time_axis(mel_spr, target_frames=128)
        self.assertEqual(aligned_spr.shape, (128, 128))

    def test_whole_record_tensor_shape_alignment(self):
        # Create 10-second synthetic audio (typical SPRSound recording length)
        temp_dir = tempfile.TemporaryDirectory()
        try:
            wav_path = os.path.join(temp_dir.name, "record_10s.wav")
            t = np.linspace(0, 10.0, 80000, endpoint=False)
            audio = 0.4 * np.sin(2 * np.pi * 350 * t).astype(np.float32)
            sf.write(wav_path, audio, 8000)

            result = self.preprocessor.extract_whole_record_features(wav_path, fixed_frames=128)
            tensor = result["tensor"]

            # Exact shape required by 2D-CNN: (1, 128, 128, 1)
            self.assertEqual(tensor.shape, (1, 128, 128, 1))
            self.assertEqual(tensor.dtype, np.float32)
            self.assertFalse(np.isnan(tensor).any())
        finally:
            temp_dir.cleanup()

    def test_event_feature_extraction_with_timestamps(self):
        fs = 8000
        total_dur = 10.0
        audio = 0.3 * np.random.randn(int(total_dur * fs)).astype(np.float32)

        # Event from 2000 ms to 3300 ms (1.3 seconds)
        start_ms = 2000.0
        end_ms = 3300.0

        event_res = self.preprocessor.extract_event_features(
            audio=audio,
            sr=fs,
            start_ms=start_ms,
            end_ms=end_ms,
            target_duration_s=2.5
        )

        tensor = event_res["tensor"]
        # Expected event tensor: (1, 128, 40, 1) for 2.5s window at 8kHz with hop=512
        self.assertEqual(tensor.shape[0], 1)
        self.assertEqual(tensor.shape[1], 128)
        self.assertEqual(tensor.shape[2], 40)
        self.assertEqual(tensor.shape[3], 1)
        self.assertAlmostEqual(event_res["event_duration_seconds"], 1.3, places=2)

    def test_event_invalid_timestamps_raises_error(self):
        audio = np.ones(80000, dtype=np.float32)
        # start_ms > end_ms
        with self.assertRaises(ValueError):
            self.preprocessor.extract_event_features(audio, sr=8000, start_ms=3000.0, end_ms=2000.0)


if __name__ == "__main__":
    unittest.main()
