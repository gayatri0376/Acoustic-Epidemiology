from rest_framework import generics
from rest_framework.permissions import IsAuthenticated

from accounts.access import visible_to
from accounts.permissions import IsDoctorOrAdmin

from .models import Explanation
from .serializers import ExplanationSerializer


class ExplanationListCreateView(generics.ListCreateAPIView):
    serializer_class = ExplanationSerializer

    def get_permissions(self):
        return [IsDoctorOrAdmin()] if self.request.method == 'POST' else [IsAuthenticated()]

    def get_queryset(self):
        qs = visible_to(self.request.user, Explanation.objects.select_related('author'), patient_field='assessment__patient')
        assessment = self.request.query_params.get('assessment')
        return qs.filter(assessment_id=assessment) if assessment else qs

    def perform_create(self, serializer):
        serializer.save(author=self.request.user)
