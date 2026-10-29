from django.urls import path

from .views import ExplanationListCreateView

urlpatterns = [path('', ExplanationListCreateView.as_view(), name='explanation-list')]
