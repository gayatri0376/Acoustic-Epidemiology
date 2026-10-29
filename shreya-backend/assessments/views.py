from django.utils import timezone
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated

from accounts.access import visible_to
from accounts.permissions import IsDoctorOrAdmin

from .models import Assessment
from .serializers import AssessmentSerializer


def assessments_for(user):
    return visible_to(user, Assessment.objects.all())


class AssessmentListView(generics.ListAPIView):
    serializer_class = AssessmentSerializer

    def get_queryset(self):
        qs = assessments_for(self.request.user)
        for param in ('status', 'recording'):
            value = self.request.query_params.get(param)
            if value:
                qs = qs.filter(**{param: value})
        return qs


class AssessmentDetailView(generics.RetrieveUpdateAPIView):
    """GET: any permitted user. PATCH: doctors/admins only (adds review notes)."""
    serializer_class = AssessmentSerializer
    http_method_names = ['get', 'patch', 'head', 'options']

    def get_permissions(self):
        return [IsDoctorOrAdmin()] if self.request.method == 'PATCH' else [IsAuthenticated()]

    def get_queryset(self):
        return assessments_for(self.request.user)

    def perform_update(self, serializer):
        serializer.save(status=Assessment.Status.REVIEWED, reviewed_by=self.request.user, reviewed_at=timezone.now())
