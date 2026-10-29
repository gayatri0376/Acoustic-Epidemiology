from unittest.mock import patch

from django.test import TestCase, override_settings


@override_settings(ML_MODEL_PATH='')
class HealthTests(TestCase):
    def test_health_is_public(self):
        response = self.client.get('/api/v1/health/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'ok')

    def test_ready_checks_database(self):
        response = self.client.get('/api/v1/health/ready/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['database'], 'ok')
        self.assertEqual(response.json()['ml_model'], 'not_configured')

    def test_ready_reports_503_when_database_is_down(self):
        with patch('config.views.connection') as fake_connection:
            fake_connection.cursor.side_effect = Exception('database down')
            response = self.client.get('/api/v1/health/ready/')
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()['database'], 'unavailable')
