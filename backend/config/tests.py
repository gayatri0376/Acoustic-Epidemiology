from django.test import TestCase
from django.urls import reverse
from rest_framework import status


class SystemEndpointsTests(TestCase):
    def test_health_check_returns_200(self):
        url = reverse('health-check')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data.get('status'), 'healthy')
        self.assertEqual(data.get('service'), 'Acoustic Epidemiology API')
        self.assertIn('timestamp', data)

    def test_readiness_check_reports_components(self):
        url = reverse('readiness-check')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn('components', data)
        self.assertTrue(data['components']['database']['reachable'])
        self.assertTrue(data['components']['storage']['available'])
        self.assertFalse(data['components']['ml_inference']['available'])
        self.assertEqual(data['components']['ml_inference']['status'], 'unavailable')

    def test_api_index_catalog(self):
        url = reverse('api-index')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn('endpoints', data)
        self.assertIn('system', data['endpoints'])
        self.assertIn('recordings', data['endpoints'])
        self.assertIn('model_registry', data['endpoints'])
        self.assertIn('predictions', data['endpoints'])
        self.assertIn('explanations', data['endpoints'])
        self.assertIn('assessments', data['endpoints'])
