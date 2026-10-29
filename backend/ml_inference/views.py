from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from recordings.models import Recording
from ml_inference.models import ModelVersion, Prediction, DiseasePrediction, ModelTaskType
from ml_inference.serializers import ModelVersionSerializer, PredictionSerializer, DiseasePredictionSerializer
from ml_inference.services import InferenceService
from ml_inference.disease_service import DiseaseInferenceService
from accounts.permissions import IsOwnerOrClinician


class ModelListView(APIView):
    """
    Lists all registered model versions, their architectures, and deployment statuses.
    """
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def get(self, request, *args, **kwargs):
        models = ModelVersion.objects.all()
        serializer = ModelVersionSerializer(models, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ActiveModelView(APIView):
    """
    Returns the currently active and verified model version.
    Returns HTTP 503 if no verified active model is loaded.
    """
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def get(self, request, *args, **kwargs):
        active_model = InferenceService.get_active_model()
        if not active_model:
            return Response(
                {
                    "error": "No Active Model Deployed",
                    "code": "NO_ACTIVE_MODEL",
                    "message": "There is currently no verified active CNN model deployed in the registry.",
                    "status": "UNAVAILABLE"
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        serializer = ModelVersionSerializer(active_model)
        return Response(serializer.data, status=status.HTTP_200_OK)


class PredictRecordingView(APIView):
    """
    Requests machine learning inference for a specific, authorized audio recording.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, recording_id, *args, **kwargs):
        recording = get_object_or_404(Recording, pk=recording_id)

        # Object-level permission check: user must own recording or be clinician/admin
        user = request.user
        profile = getattr(user, 'profile', None)
        has_access = (
            user.is_superuser 
            or recording.uploader == user 
            or (profile and (profile.is_clinician or profile.is_researcher))
        )
        if not has_access:
            return Response(
                {"error": "You do not have permission to run inference on this recording."},
                status=status.HTTP_403_FORBIDDEN
            )

        prediction = InferenceService.execute_prediction(recording, request_user=user)
        serializer = PredictionSerializer(prediction)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class PredictionListView(APIView):
    """
    Lists predictions permitted to the current user with pagination.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        user = request.user
        profile = getattr(user, 'profile', None)

        if user.is_superuser or user.is_staff or (profile and (profile.is_clinician or profile.is_researcher)):
            queryset = Prediction.objects.all().select_related('recording', 'model_version')
        else:
            queryset = Prediction.objects.filter(recording__uploader=user).select_related('recording', 'model_version')

        # Filter by status if provided
        status_filter = request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(prediction_status=status_filter)

        from rest_framework.pagination import PageNumberPagination
        paginator = PageNumberPagination()
        paginator.page_size = 20
        page = paginator.paginate_queryset(queryset, request)
        if page is not None:
            serializer = PredictionSerializer(page, many=True)
            return paginator.get_paginated_response(serializer.data)

        serializer = PredictionSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class PredictionDetailView(APIView):
    """
    Retrieves metadata, predicted class, and probability scores for a prediction.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk, *args, **kwargs):
        prediction = get_object_or_404(
            Prediction.objects.select_related('recording', 'model_version'),
            pk=pk
        )
        user = request.user
        profile = getattr(user, 'profile', None)
        has_access = (
            user.is_superuser 
            or prediction.recording.uploader == user 
            or (profile and (profile.is_clinician or profile.is_researcher))
        )
        if not has_access:
            return Response(
                {"error": "You do not have permission to view this prediction."},
                status=status.HTTP_403_FORBIDDEN
            )

        serializer = PredictionSerializer(prediction)
        return Response(serializer.data, status=status.HTTP_200_OK)


class RecordingPredictionsView(APIView):
    """
    Retrieves all prediction attempts for a specific recording.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, recording_id, *args, **kwargs):
        recording = get_object_or_404(Recording, pk=recording_id)
        user = request.user
        profile = getattr(user, 'profile', None)
        has_access = (
            user.is_superuser 
            or recording.uploader == user 
            or (profile and (profile.is_clinician or profile.is_researcher))
        )
        if not has_access:
            return Response(
                {"error": "You do not have permission to view predictions for this recording."},
                status=status.HTTP_403_FORBIDDEN
            )

        predictions = Prediction.objects.filter(recording=recording).select_related('model_version')
        serializer = PredictionSerializer(predictions, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)



# ---------------------------------------------------------------------------
# Disease Prediction Views (Model B — separate from sound classification)
# ---------------------------------------------------------------------------

class ActiveDiseaseModelView(APIView):
    """
    Returns the currently active DISEASE_CLASSIFICATION model version.
    Returns HTTP 503 if no disease model is deployed.

    Note: This endpoint is entirely separate from /models/active/ which returns
    the sound-classification model. Keeping them separate preserves independent
    model lifecycle management.
    """
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def get(self, request, *args, **kwargs):
        active_model = DiseaseInferenceService.get_active_disease_model()
        if not active_model:
            return Response(
                {
                    "error": "No Active Disease Model Deployed",
                    "code": "NO_ACTIVE_DISEASE_MODEL",
                    "message": (
                        "There is currently no verified DISEASE_CLASSIFICATION CNN model "
                        "deployed in the registry. Register a model with task_type="
                        "DISEASE_CLASSIFICATION and status=ACTIVE to enable disease screening."
                    ),
                    "status": "UNAVAILABLE",
                    "task_type": "DISEASE_CLASSIFICATION",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        serializer = ModelVersionSerializer(active_model)
        return Response(serializer.data, status=status.HTTP_200_OK)


class PredictDiseaseView(APIView):
    """
    Requests disease-model inference for a specific, authorized audio recording.

    POST /api/v1/recordings/<recording_id>/predict-disease/

    This endpoint is completely separate from /recordings/<id>/predict/ (sound
    classification). It uses the DISEASE_CLASSIFICATION model and returns a
    DiseasePrediction object, not a Prediction object.

    Response fields include:
    - predicted_disease: one of "Healthy Control", "Pneumonia", "Wheezing Disorders"
    - probability_scores: {class: score} from the softmax head
    - class_description: plain-English description of the predicted class
    - research_disclaimer: mandatory safety statement (never suppress this)
    - dataset_info: provenance metadata
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, recording_id, *args, **kwargs):
        recording = get_object_or_404(Recording, pk=recording_id)

        user = request.user
        profile = getattr(user, 'profile', None)
        has_access = (
            user.is_superuser
            or recording.uploader == user
            or (profile and (profile.is_clinician or profile.is_researcher))
        )
        if not has_access:
            return Response(
                {"error": "You do not have permission to run disease inference on this recording."},
                status=status.HTTP_403_FORBIDDEN,
            )

        disease_prediction = DiseaseInferenceService.execute_disease_prediction(
            recording, request_user=user
        )
        serializer = DiseasePredictionSerializer(disease_prediction)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class DiseasePredictionDetailView(APIView):
    """
    Retrieves metadata and scores for a specific disease prediction.

    GET /api/v1/disease-predictions/<pk>/
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk, *args, **kwargs):
        dp = get_object_or_404(
            DiseasePrediction.objects.select_related('recording', 'model_version'),
            pk=pk,
        )
        user = request.user
        profile = getattr(user, 'profile', None)
        has_access = (
            user.is_superuser
            or dp.recording.uploader == user
            or (profile and (profile.is_clinician or profile.is_researcher))
        )
        if not has_access:
            return Response(
                {"error": "You do not have permission to view this disease prediction."},
                status=status.HTTP_403_FORBIDDEN,
            )
        serializer = DiseasePredictionSerializer(dp)
        return Response(serializer.data, status=status.HTTP_200_OK)


class RecordingDiseasePredictionsView(APIView):
    """
    Retrieves all disease prediction attempts for a specific recording.

    GET /api/v1/recordings/<recording_id>/disease-predictions/
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, recording_id, *args, **kwargs):
        recording = get_object_or_404(Recording, pk=recording_id)
        user = request.user
        profile = getattr(user, 'profile', None)
        has_access = (
            user.is_superuser
            or recording.uploader == user
            or (profile and (profile.is_clinician or profile.is_researcher))
        )
        if not has_access:
            return Response(
                {"error": "You do not have permission to view disease predictions for this recording."},
                status=status.HTTP_403_FORBIDDEN,
            )
        dps = DiseasePrediction.objects.filter(recording=recording).select_related('model_version')
        serializer = DiseasePredictionSerializer(dps, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
