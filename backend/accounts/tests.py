from django.test import TestCase
from django.urls import reverse
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APIClient
from accounts.models import UserProfile, UserRole


class AccountsAuthenticationTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='Password123!',
            first_name='Test',
            last_name='User'
        )

    def test_register_new_user_success(self):
        url = reverse('auth-register')
        payload = {
            'username': 'newclinician',
            'email': 'clinician@hospital.org',
            'password': 'SecurePassword123!',
            'password_confirm': 'SecurePassword123!',
            'organization': 'Children’s Hospital',
            'department': 'Pediatric Pulmonology'
        }
        response = self.client.post(url, payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('token', response.data)
        self.assertEqual(response.data['user']['username'], 'newclinician')
        
        # Verify that role defaults safely to UPLOADER and cannot be escalated via public registration
        new_user = User.objects.get(username='newclinician')
        self.assertEqual(new_user.profile.role, UserRole.UPLOADER)
        self.assertEqual(new_user.profile.organization, 'Children’s Hospital')

    def test_register_password_mismatch(self):
        url = reverse('auth-register')
        payload = {
            'username': 'mismatchuser',
            'email': 'mismatch@hospital.org',
            'password': 'SecurePassword123!',
            'password_confirm': 'DifferentPassword123!'
        }
        response = self.client.post(url, payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_login_success(self):
        url = reverse('auth-login')
        payload = {
            'username': 'testuser',
            'password': 'Password123!'
        }
        response = self.client.post(url, payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('token', response.data)

    def test_login_invalid_credentials(self):
        url = reverse('auth-login')
        payload = {
            'username': 'testuser',
            'password': 'WrongPassword!'
        }
        response = self.client.post(url, payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_current_user_profile(self):
        self.client.force_authenticate(user=self.user)
        url = reverse('auth-current-user')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['username'], 'testuser')
        self.assertIn('profile', response.data)

    def test_logout_deletes_token(self):
        login_url = reverse('auth-login')
        login_res = self.client.post(login_url, {'username': 'testuser', 'password': 'Password123!'}, format='json')
        token = login_res.data['token']

        self.client.credentials(HTTP_AUTHORIZATION=f'Token {token}')
        logout_url = reverse('auth-logout')
        logout_res = self.client.post(logout_url)
        self.assertEqual(logout_res.status_code, status.HTTP_200_OK)

        # After logout, accessing protected route with deleted token should fail
        me_url = reverse('auth-current-user')
        me_res = self.client.get(me_url)
        self.assertEqual(me_res.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_unauthenticated_user_cannot_access_profile(self):
        url = reverse('auth-current-user')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

