from django.db import transaction
from django.http import FileResponse
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated

from accounts.access import visible_to
from accounts.permissions import IsPatient
from assessments.services import create_assessment_for_recording

from .models import Recording
from .serializers import RecordingSerializer


def recordings_for(user):
    """Only recordings this user may see (own / assigned patients / all for admin)."""
    return visible_to(user, Recording.objects.prefetch_related('assessments'))


class RecordingListCreateView(generics.ListCreateAPIView):
    serializer_class = RecordingSerializer

    def get_permissions(self):
        # Reading: any signed-in user (filtered). Uploading: patients only, for themselves.
        return [IsAuthenticated(), IsPatient()] if self.request.method == 'POST' else [IsAuthenticated()]

    def get_queryset(self):
        qs = recordings_for(self.request.user)
        status_value = self.request.query_params.get('status')
        if status_value:
            qs = qs.filter(status=status_value)
        return qs

    def perform_create(self, serializer):
        with transaction.atomic():
            recording = serializer.save(patient=self.request.user)
            create_assessment_for_recording(recording)  # records "not_configured" unless a model is set up


class RecordingDetailView(generics.RetrieveAPIView):
    serializer_class = RecordingSerializer

    def get_queryset(self):
        return recordings_for(self.request.user)  # others' recordings -> 404


class RecordingFileView(generics.GenericAPIView):
    """Streams the audio file, but only to users allowed to see the recording."""

    def get_queryset(self):
        return recordings_for(self.request.user)

    def get(self, request, pk):
        recording = self.get_object()
        return FileResponse(recording.audio_file.open('rb'), content_type=recording.content_type,
                            as_attachment=True, filename=recording.original_filename)
