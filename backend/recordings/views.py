from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.shortcuts import get_object_or_404
from recordings.models import Recording
from recordings.serializers import RecordingSerializer, RecordingUploadSerializer
from accounts.permissions import IsOwnerOrClinician, IsOwnerOrReadOnly


class RecordingListCreateView(APIView):
    """
    List permitted recordings or upload a new pediatric respiratory recording.
    """
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        user = request.user
        profile = getattr(user, 'profile', None)

        if user.is_superuser or user.is_staff or (profile and (profile.is_clinician or profile.is_researcher)):
            queryset = Recording.objects.all().select_related('uploader')
        else:
            queryset = Recording.objects.filter(uploader=user).select_related('uploader')

        # Optional filters
        status_filter = request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(recording_status=status_filter)

        location_filter = request.query_params.get('location')
        if location_filter:
            queryset = queryset.filter(anatomical_location=location_filter)

        # Pagination
        from rest_framework.pagination import PageNumberPagination
        paginator = PageNumberPagination()
        paginator.page_size = 20
        page = paginator.paginate_queryset(queryset, request)
        if page is not None:
            serializer = RecordingSerializer(page, many=True)
            return paginator.get_paginated_response(serializer.data)

        serializer = RecordingSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        serializer = RecordingUploadSerializer(
            data=request.data,
            context={'request': request}
        )
        if serializer.is_valid():
            recording = serializer.save()
            return Response(
                {
                    "message": "Audio recording uploaded and validated successfully.",
                    "recording": RecordingSerializer(recording).data
                },
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class RecordingDetailView(APIView):
    """
    Retrieve or safely delete an authorized recording.
    """
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrClinician]

    def get(self, request, pk, *args, **kwargs):
        recording = get_object_or_404(Recording.objects.select_related('uploader'), pk=pk)
        self.check_object_permissions(request, recording)
        serializer = RecordingSerializer(recording)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def delete(self, request, pk, *args, **kwargs):
        recording = get_object_or_404(Recording, pk=pk)
        # Only uploader or admin can delete
        if not (request.user.is_superuser or recording.uploader == request.user):
            return Response(
                {"error": "You do not have permission to delete this recording."},
                status=status.HTTP_403_FORBIDDEN
            )
        recording.delete()
        return Response(
            {"message": "Recording and associated audio storage deleted successfully."},
            status=status.HTTP_200_OK
        )
