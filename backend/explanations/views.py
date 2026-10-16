from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from ml_inference.models import Prediction
from explanations.models import Explanation, ExplanationMethod, ExplanationStatus
from explanations.serializers import ExplanationSerializer
from explanations.services import GradCamService



class PredictionExplanationListCreateView(APIView):
    """
    List or request generation of explainability artifacts (Grad-CAM) for a prediction.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, prediction_id, *args, **kwargs):
        prediction = get_object_or_404(
            Prediction.objects.select_related('recording'),
            pk=prediction_id
        )

        # Check access to recording/prediction
        user = request.user
        profile = getattr(user, 'profile', None)
        has_access = (
            user.is_superuser
            or prediction.recording.uploader == user
            or (profile and (profile.is_clinician or profile.is_researcher))
        )
        if not has_access:
            return Response(
                {"error": "You do not have permission to view explanations for this prediction."},
                status=status.HTTP_403_FORBIDDEN
            )

        explanations = Explanation.objects.filter(prediction=prediction)
        serializer = ExplanationSerializer(explanations, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, prediction_id, *args, **kwargs):
        prediction = get_object_or_404(
            Prediction.objects.select_related('recording', 'model_version'),
            pk=prediction_id
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
                {"error": "You do not have permission to request an explanation for this prediction."},
                status=status.HTTP_403_FORBIDDEN
            )

        method = request.data.get('method', ExplanationMethod.GRAD_CAM)

        result = GradCamService.generate_explanation(prediction, method=method)
        if not result.get("success"):
            explanation = result["explanation"]
            return Response(
                {
                    "error": "Explanation Generation Unsupported",
                    "code": result.get("code", "GRADIENT_MODEL_UNAVAILABLE"),
                    "explanation_id": str(explanation.id),
                    "prediction_id": str(prediction.id),
                    "message": result.get("message", "Authentic saliency maps cannot be computed.")
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        explanation = result["explanation"]
        serializer = ExplanationSerializer(explanation)
        return Response(serializer.data, status=status.HTTP_202_ACCEPTED)



class ExplanationDetailView(APIView):
    """
    Retrieves a single authorized explanation record.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, pk, *args, **kwargs):
        explanation = get_object_or_404(
            Explanation.objects.select_related('prediction__recording'),
            pk=pk
        )
        prediction = explanation.prediction
        user = request.user
        profile = getattr(user, 'profile', None)
        has_access = (
            user.is_superuser
            or prediction.recording.uploader == user
            or (profile and (profile.is_clinician or profile.is_researcher))
        )
        if not has_access:
            return Response(
                {"error": "You do not have permission to view this explanation."},
                status=status.HTTP_403_FORBIDDEN
            )

        serializer = ExplanationSerializer(explanation)
        return Response(serializer.data, status=status.HTTP_200_OK)
