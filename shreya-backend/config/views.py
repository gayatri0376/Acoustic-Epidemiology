from django.conf import settings
from django.db import connection
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from ml_inference.services import is_configured


@api_view(['GET'])
@permission_classes([AllowAny])
def health(request):
    """Liveness: the Django process is up. Does not touch the database."""
    return Response({'status': 'ok'})


@api_view(['GET'])
@permission_classes([AllowAny])
def health_ready(request):
    """Readiness: verifies the MySQL connection with a real query."""
    ml_status = 'configured' if is_configured() else 'not_configured'
    try:
        with connection.cursor() as cursor:
            cursor.execute('SELECT 1')
            cursor.fetchone()
    except Exception as exc:  # any DB failure means "not ready"
        body = {'status': 'unavailable', 'database': 'unavailable', 'ml_model': ml_status}
        if settings.DEBUG:
            body['detail'] = str(exc)  # error details only in development
        return Response(body, status=503)
    return Response({'status': 'ok', 'database': 'ok', 'ml_model': ml_status})
