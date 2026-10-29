import re

from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

from .models import Role, User


class UserSerializer(serializers.ModelSerializer):
    """Read-only public profile returned by login and /me/."""
    class Meta:
        model = User
        fields = ['id', 'email', 'full_name', 'phone_number', 'role', 'is_active', 'created_at']
        read_only_fields = fields


class RegisterSerializer(serializers.ModelSerializer):
    """Public registration. There is NO role field here, so a caller can never choose a role."""
    password = serializers.CharField(write_only=True, style={'input_type': 'password'})

    class Meta:
        model = User
        fields = ['email', 'password', 'full_name', 'phone_number']
        extra_kwargs = {'full_name': {'required': True}, 'phone_number': {'required': False}}

    def validate_email(self, value):
        value = value.lower()
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError('An account with this email already exists.')
        return value

    def validate_phone_number(self, value):
        if value and not re.fullmatch(r'\+?[0-9 ()\-]{7,20}', value):
            raise serializers.ValidationError('Enter a valid phone number.')
        return value

    def validate(self, attrs):
        # Run Django's password rules; similarity checks need the other fields, so build a temp user.
        validate_password(attrs['password'], User(email=attrs.get('email', ''), full_name=attrs.get('full_name', '')))
        return attrs

    def create(self, validated_data):
        return User.objects.create_user(role=Role.PATIENT, **validated_data)


class LoginSerializer(TokenObtainPairSerializer):
    """Accepts {email, password}. Returns {access, refresh, user}."""

    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token['role'] = user.role  # informational only: the frontend should use `user.role`, the API re-checks the DB
        return token

    def validate(self, attrs):
        data = super().validate(attrs)  # raises 401 for wrong password or inactive account
        data['user'] = UserSerializer(self.user).data
        return data
