from django.urls import path

from .views import RecordingDetailView, RecordingFileView, RecordingListCreateView

urlpatterns = [
    path('', RecordingListCreateView.as_view(), name='recording-list'),
    path('<int:pk>/', RecordingDetailView.as_view(), name='recording-detail'),
    path('<int:pk>/file/', RecordingFileView.as_view(), name='recording-file'),
]
