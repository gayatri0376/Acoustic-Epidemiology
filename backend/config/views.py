import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from django.utils import timezone
from django.db import connection
from django.conf import settings
from ml_inference.services import InferenceService


class HealthCheckView(APIView):
    """
    Liveness probe returning basic service metadata and timestamp.
    """
    permission_classes = [permissions.AllowAny]

    def get(self, request, *args, **kwargs):
        return Response(
            {
                "status": "healthy",
                "service": "Acoustic Epidemiology API",
                "version": "1.0.0",
                "timestamp": timezone.now().isoformat(),
            },
            status=status.HTTP_200_OK,
        )


class ReadinessCheckView(APIView):
    """
    Readiness probe evaluating database reachability, storage availability,
    and CNN model deployment status. Does NOT expose sensitive secrets.
    """
    permission_classes = [permissions.AllowAny]

    def get(self, request, *args, **kwargs):
        checks = {}
        all_ready = True
        http_status = status.HTTP_200_OK

        # 1. Database readiness
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1;")
                cursor.fetchone()
            db_vendor = connection.vendor
            checks["database"] = {
                "status": "connected",
                "engine": db_vendor,
                "reachable": True,
            }
        except Exception as exc:
            all_ready = False
            http_status = status.HTTP_503_SERVICE_UNAVAILABLE
            checks["database"] = {
                "status": "disconnected",
                "reachable": False,
                "error": "Database connectivity failed.",
            }

        # 2. Storage readiness
        try:
            media_root = settings.MEDIA_ROOT
            os.makedirs(media_root, exist_ok=True)
            test_file = os.path.join(media_root, ".storage_probe")
            with open(test_file, "w") as f:
                f.write("probe")
            if os.path.exists(test_file):
                os.remove(test_file)
            checks["storage"] = {
                "status": "writable",
                "available": True,
            }
        except Exception as exc:
            all_ready = False
            checks["storage"] = {
                "status": "unwritable",
                "available": False,
                "error": "Media storage directory is not writable.",
            }

        # 3. ML Model Deployment readiness
        try:
            active_model = InferenceService.get_active_model()
            if active_model:
                checks["ml_inference"] = {
                    "status": "deployed",
                    "version": active_model.version,
                    "architecture": active_model.architecture,
                    "available": True,
                }
            else:
                checks["ml_inference"] = {
                    "status": "unavailable",
                    "available": False,
                    "detail": "No verified CNN model weights are currently deployed in ml/models/.",
                }
        except Exception:
            checks["ml_inference"] = {
                "status": "uninitialized",
                "available": False,
                "detail": "Model registry uninitialized or unreachable.",
            }

        # Determine overall readiness
        # If DB is down, system is not ready. If model is missing, system is operational for audio ingestion and data collection.
        is_ready = checks.get("database", {}).get("reachable", False)

        return Response(
            {
                "ready": is_ready,
                "service": "Acoustic Epidemiology Backend",
                "timestamp": timezone.now().isoformat(),
                "components": checks,
            },
            status=http_status,
        )


class ApiIndexView(APIView):
    """
    Documented endpoint catalog for the Acoustic Epidemiology REST API.
    """
    permission_classes = [permissions.AllowAny]

    def get(self, request, *args, **kwargs):
        return Response(
            {
                "service": "Acoustic Epidemiology API",
                "version": "v1",
                "documentation": "Acoustic Epidemiology: Automated Classification of Pediatric Respiratory Diseases Using CNNs",
                "endpoints": {
                    "system": {
                        "health": "/api/v1/health/",
                        "readiness": "/api/v1/ready/",
                    },
                    "authentication": {
                        "register": "/api/v1/auth/register/",
                        "login": "/api/v1/auth/login/",
                        "logout": "/api/v1/auth/logout/",
                        "profile": "/api/v1/auth/me/",
                    },
                    "recordings": {
                        "list_and_upload": "/api/v1/recordings/",
                        "detail_and_delete": "/api/v1/recordings/{id}/",
                    },
                    "model_registry": {
                        "list_models": "/api/v1/models/",
                        "active_model": "/api/v1/models/active/",
                    },
                    "predictions": {
                        "request_prediction": "/api/v1/recordings/{id}/predict/",
                        "list_predictions": "/api/v1/predictions/",
                        "detail": "/api/v1/predictions/{id}/",
                        "recording_history": "/api/v1/recordings/{id}/predictions/",
                    },
                    "explanations": {
                        "request_or_list": "/api/v1/predictions/{id}/explanations/",
                        "detail": "/api/v1/explanations/{id}/",
                    },
                    "assessments": {
                        "create_assessment": "/api/v1/predictions/{id}/assessments/",
                        "list_assessments": "/api/v1/assessments/",
                        "detail_and_update": "/api/v1/assessments/{id}/",
                    },
                },
            },
            status=status.HTTP_200_OK,
        )
