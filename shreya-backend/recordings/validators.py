import os

from django.conf import settings
from rest_framework import serializers

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg', 'flac'}
CONTENT_TYPES = {'wav': 'audio/wav', 'mp3': 'audio/mpeg', 'm4a': 'audio/mp4', 'ogg': 'audio/ogg', 'flac': 'audio/flac'}


def _looks_like(ext, head):
    """Check the first bytes ("magic bytes") so a renamed .exe cannot pass as .wav."""
    if ext == 'wav':
        return head[:4] == b'RIFF' and head[8:12] == b'WAVE'
    if ext == 'mp3':
        return head[:3] == b'ID3' or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0)
    if ext == 'm4a':
        return head[4:8] == b'ftyp'
    if ext == 'ogg':
        return head[:4] == b'OggS'
    if ext == 'flac':
        return head[:4] == b'fLaC'
    return False


def get_extension(name):
    return os.path.splitext(name)[1].lower().lstrip('.')


def validate_audio_file(upload):
    ext = get_extension(upload.name)
    if ext not in ALLOWED_EXTENSIONS:
        raise serializers.ValidationError(f'Unsupported file type. Allowed: {", ".join(sorted(ALLOWED_EXTENSIONS))}.')
    if upload.size == 0:
        raise serializers.ValidationError('The file is empty.')
    if upload.size > settings.MAX_AUDIO_UPLOAD_MB * 1024 * 1024:
        raise serializers.ValidationError(f'File is too large. Maximum size is {settings.MAX_AUDIO_UPLOAD_MB} MB.')
    head = upload.read(16)
    upload.seek(0)
    if not _looks_like(ext, head):
        raise serializers.ValidationError('The file content does not match its extension.')
