from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from config.views import HealthCheckView, ReadinessCheckView, ApiIndexView

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # System probes and root index
    path('api/v1/', ApiIndexView.as_view(), name='api-index'),
    path('api/v1/health/', HealthCheckView.as_view(), name='health-check'),
    path('api/v1/ready/', ReadinessCheckView.as_view(), name='readiness-check'),

    # Modular domain APIs
    path('api/v1/auth/', include('accounts.urls')),
    path('api/v1/recordings/', include('recordings.urls')),
    path('api/v1/', include('ml_inference.urls')),
    path('api/v1/', include('explanations.urls')),
    path('api/v1/', include('assessments.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
