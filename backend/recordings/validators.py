import os
import io
from django.core.exceptions import ValidationError
from django.utils.text import get_valid_filename

# Maximum upload limit: 25 MB
MAX_UPLOAD_SIZE = 25 * 1024 * 1024  
ALLOWED_EXTENSIONS = {'.wav', '.flac', '.ogg', '.mp3'}


def sanitize_filename(filename):
    """
    Sanitizes filename, strips path components, and prevents directory traversal.
    """
    base_name = os.path.basename(filename)
    clean_name = get_valid_filename(base_name)
    if not clean_name:
        clean_name = "recording.wav"
    return clean_name


def inspect_and_validate_audio(file_obj):
    """
    Deeply inspects the audio file content, validating magic headers and
    extracting acoustic metadata (sampling rate, duration, channels).
    Does NOT trust client-supplied MIME type.
    """
    if file_obj.size > MAX_UPLOAD_SIZE:
        raise ValidationError(
            f"File size exceeds maximum allowed limit of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
        )

    filename = getattr(file_obj, 'name', '')
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file extension '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    # Save current stream position
    initial_pos = file_obj.tell() if hasattr(file_obj, 'tell') else 0
    header = file_obj.read(16)
    file_obj.seek(initial_pos)

    # Verify magic bytes
    is_valid_format = False
    detected_format = 'wav'

    if header.startswith(b'RIFF') and b'WAVE' in header:
        is_valid_format = True
        detected_format = 'wav'
    elif header.startswith(b'fLaC'):
        is_valid_format = True
        detected_format = 'flac'
    elif header.startswith(b'OggS'):
        is_valid_format = True
        detected_format = 'ogg'
    elif header.startswith(b'ID3') or header.startswith(b'\xff\xfb') or header.startswith(b'\xff\xf3'):
        is_valid_format = True
        detected_format = 'mp3'

    if not is_valid_format:
        raise ValidationError(
            "File signature is invalid. The uploaded file does not match a valid audio format."
        )

    # Extract audio stream properties using soundfile
    sample_rate = None
    channels = 1
    duration = None

    try:
        import soundfile as sf
        file_obj.seek(0)
        # Read file info using SoundFile
        info = sf.info(file_obj)
        sample_rate = info.samplerate
        channels = info.channels
        duration = round(info.duration, 3)
        detected_format = info.format.lower() if info.format else detected_format
        file_obj.seek(0)
    except Exception as e:
        file_obj.seek(0)
        # Fallback inspection for WAV using wave module
        if ext == '.wav':
            try:
                import wave
                with wave.open(file_obj, 'rb') as wav_reader:
                    sample_rate = wav_reader.getframerate()
                    channels = wav_reader.getnchannels()
                    nframes = wav_reader.getnframes()
                    duration = round(nframes / float(sample_rate), 3) if sample_rate else None
                file_obj.seek(0)
            except Exception:
                file_obj.seek(0)
                raise ValidationError(f"Could not read audio stream: {str(e)}")
        else:
            raise ValidationError(f"Could not parse audio metadata: {str(e)}")

    if duration is not None and duration < 0.2:
        raise ValidationError("Audio duration is too short (< 0.2 seconds) for clinical respiratory analysis.")

    return {
        'format': detected_format,
        'sample_rate': sample_rate,
        'channels': channels,
        'duration_seconds': duration,
        'file_size_bytes': file_obj.size,
    }
