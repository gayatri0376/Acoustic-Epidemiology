"""Small helpers shared by the test files."""
from django.core.cache import cache
from rest_framework.test import APITestCase

from accounts.models import Role, User

PASSWORD = 'Str0ng-Pass!word'


class BaseAPITestCase(APITestCase):
    def setUp(self):
        cache.clear()  # reset login rate-limit counters between tests

    def make_user(self, email, role=Role.PATIENT, **extra):
        return User.objects.create_user(email=email, password=PASSWORD, full_name=email.split('@')[0], role=role, **extra)

    def login_headers(self, email):
        """Log in through the real endpoint and return an Authorization header dict."""
        response = self.client.post('/api/v1/auth/login/', {'email': email, 'password': PASSWORD}, format='json')
        assert response.status_code == 200, response.content
        return {'HTTP_AUTHORIZATION': f"Bearer {response.data['access']}"}
