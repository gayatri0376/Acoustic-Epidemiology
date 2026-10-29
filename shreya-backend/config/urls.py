from django.contrib import admin
from django.urls import include, path

from .views import health, health_ready

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/health/', health, name='health'),
    path('api/v1/health/ready/', health_ready, name='health-ready'),
    path('api/v1/auth/', include('accounts.urls')),
    path('api/v1/recordings/', include('recordings.urls')),
    path('api/v1/assessments/', include('assessments.urls')),
    path('api/v1/explanations/', include('explanations.urls')),
]
