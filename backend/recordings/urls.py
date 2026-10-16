from django.urls import path
from recordings.views import RecordingListCreateView, RecordingDetailView

urlpatterns = [
    path('', RecordingListCreateView.as_view(), name='recording-list-create'),
    path('<uuid:pk>/', RecordingDetailView.as_view(), name='recording-detail'),
]
