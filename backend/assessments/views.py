from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from ml_inference.models import Prediction
from assessments.models import Assessment
from assessments.serializers import AssessmentSerializer, AssessmentCreateUpdateSerializer
from accounts.permissions import IsClinician


class PredictionAssessmentCreateView(APIView):
    """
    Submits a clinician review/assessment for a specific prediction.
    Enforces clinician or administrator permissions.
    """
    permission_classes = [permissions.IsAuthenticated, IsClinician]

    def post(self, request, prediction_id, *args, **kwargs):
        prediction = get_object_or_404(
            Prediction.objects.select_related('recording'),
            pk=prediction_id
        )

        serializer = AssessmentCreateUpdateSerializer(
            data=request.data,
            context={'request': request, 'prediction': prediction}
        )
        if serializer.is_valid():
            assessment = serializer.save()
            return Response(
                {
                    "message": "Clinical assessment recorded successfully.",
                    "assessment": AssessmentSerializer(assessment).data
                },
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class AssessmentListView(APIView):
    """
    Lists assessments accessible to the authenticated clinician or researcher.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        user = request.user
        profile = getattr(user, 'profile', None)

        if user.is_superuser or (profile and (profile.is_clinician or profile.is_researcher)):
            queryset = Assessment.objects.all().select_related('reviewer', 'prediction')
        else:
            # Regular users only see assessments of their own recordings
            queryset = Assessment.objects.filter(
                prediction__recording__uploader=user
            ).select_related('reviewer', 'prediction')

        # Filter by review_status
        status_filter = request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(review_status=status_filter)

        from rest_framework.pagination import PageNumberPagination
        paginator = PageNumberPagination()
        paginator.page_size = 20
        page = paginator.paginate_queryset(queryset, request)
        if page is not None:
            serializer = AssessmentSerializer(page, many=True)
            return paginator.get_paginated_response(serializer.data)

        serializer = AssessmentSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class AssessmentDetailView(APIView):
    """
    Retrieve or update a clinical review.
    Updates are restricted to the authoring clinician or administrator.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk, *args, **kwargs):
        assessment = get_object_or_404(
            Assessment.objects.select_related('reviewer', 'prediction__recording'),
            pk=pk
        )
        user = request.user
        profile = getattr(user, 'profile', None)
        has_access = (
            user.is_superuser
            or assessment.reviewer == user
            or assessment.prediction.recording.uploader == user
            or (profile and (profile.is_clinician or profile.is_researcher))
        )
        if not has_access:
            return Response(
                {"error": "You do not have permission to view this assessment."},
                status=status.HTTP_403_FORBIDDEN
            )

        serializer = AssessmentSerializer(assessment)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def patch(self, request, pk, *args, **kwargs):
        assessment = get_object_or_404(Assessment, pk=pk)
        user = request.user
        if not (user.is_superuser or assessment.reviewer == user):
            return Response(
                {"error": "Only the authoring clinician or an administrator may modify this assessment."},
                status=status.HTTP_403_FORBIDDEN
            )

        serializer = AssessmentCreateUpdateSerializer(
            assessment,
            data=request.data,
            partial=True
        )
        if serializer.is_valid():
            updated = serializer.save()
            return Response(
                {
                    "message": "Assessment updated successfully.",
                    "assessment": AssessmentSerializer(updated).data
                },
                status=status.HTTP_200_OK
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
