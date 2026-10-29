from django.conf import settings
from django.db import models


class Explanation(models.Model):
    assessment = models.ForeignKey('assessments.Assessment', on_delete=models.CASCADE, related_name='explanations')
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name='explanations')
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'Explanation {self.pk} for assessment {self.assessment_id}'
