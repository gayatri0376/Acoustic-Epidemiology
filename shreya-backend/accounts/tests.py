from unittest.mock import patch

from rest_framework.throttling import ScopedRateThrottle

from config.testing import PASSWORD, BaseAPITestCase
from accounts.models import Role, User


class RegistrationTests(BaseAPITestCase):
    url = '/api/v1/auth/register/'
    payload = {'email': 'new@example.com', 'password': PASSWORD, 'full_name': 'New Patient', 'phone_number': '+91 98765 43210'}

    def test_register_creates_patient_with_hashed_password(self):
        response = self.client.post(self.url, self.payload, format='json')
        self.assertEqual(response.status_code, 201)
        user = User.objects.get(email='new@example.com')
        self.assertEqual(user.role, Role.PATIENT)
        self.assertNotEqual(user.password, PASSWORD)
        self.assertTrue(user.check_password(PASSWORD))

    def test_cannot_self_assign_admin_or_doctor(self):
        for role in ('admin', 'doctor'):
            email = f'{role}@example.com'
            self.client.post(self.url, {**self.payload, 'email': email, 'role': role, 'is_staff': True}, format='json')
            user = User.objects.get(email=email)
            self.assertEqual(user.role, Role.PATIENT)
            self.assertFalse(user.is_staff)

    def test_weak_password_rejected(self):
        response = self.client.post(self.url, {**self.payload, 'password': '12345678'}, format='json')
        self.assertEqual(response.status_code, 400)

    def test_duplicate_email_rejected(self):
        self.make_user('new@example.com')
        response = self.client.post(self.url, self.payload, format='json')
        self.assertEqual(response.status_code, 400)


class LoginTests(BaseAPITestCase):
    url = '/api/v1/auth/login/'

    def test_login_success_returns_tokens_and_role(self):
        self.make_user('doc@example.com', Role.DOCTOR)
        response = self.client.post(self.url, {'email': 'doc@example.com', 'password': PASSWORD}, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('access', response.data)
        self.assertIn('refresh', response.data)
        self.assertEqual(response.data['user']['role'], 'doctor')

    def test_invalid_credentials(self):
        self.make_user('p@example.com')
        response = self.client.post(self.url, {'email': 'p@example.com', 'password': 'wrong'}, format='json')
        self.assertEqual(response.status_code, 401)

    def test_inactive_user_cannot_login(self):
        self.make_user('off@example.com', is_active=False)
        response = self.client.post(self.url, {'email': 'off@example.com', 'password': PASSWORD}, format='json')
        self.assertEqual(response.status_code, 401)

    def test_login_is_rate_limited(self):
        # DRF reads rates when the class loads, so patch the rate dictionary directly.
        with patch.dict(ScopedRateThrottle.THROTTLE_RATES, {'auth': '2/min'}):
            codes = [self.client.post(self.url, {'email': 'x@example.com', 'password': 'bad'}, format='json').status_code
                     for _ in range(4)]
        self.assertIn(429, codes)


class ProtectedAccessTests(BaseAPITestCase):
    def test_protected_endpoints_require_token(self):
        for path in ('/api/v1/auth/me/', '/api/v1/recordings/', '/api/v1/assessments/', '/api/v1/explanations/'):
            self.assertEqual(self.client.get(path).status_code, 401, path)

    def test_me_returns_profile_and_role(self):
        self.make_user('p@example.com')
        response = self.client.get('/api/v1/auth/me/', **self.login_headers('p@example.com'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['email'], 'p@example.com')
        self.assertEqual(response.data['role'], 'patient')

    def test_refresh_and_logout_revoke_token(self):
        self.make_user('p@example.com')
        login = self.client.post('/api/v1/auth/login/', {'email': 'p@example.com', 'password': PASSWORD}, format='json')
        headers = {'HTTP_AUTHORIZATION': f"Bearer {login.data['access']}"}
        refresh = login.data['refresh']
        self.assertEqual(self.client.post('/api/v1/auth/logout/', {'refresh': refresh}, format='json', **headers).status_code, 200)
        self.assertEqual(self.client.post('/api/v1/auth/refresh/', {'refresh': refresh}, format='json').status_code, 401)

    def test_refresh_returns_new_access_token(self):
        self.make_user('p@example.com')
        login = self.client.post('/api/v1/auth/login/', {'email': 'p@example.com', 'password': PASSWORD}, format='json')
        response = self.client.post('/api/v1/auth/refresh/', {'refresh': login.data['refresh']}, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('access', response.data)
