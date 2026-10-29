from rest_framework import serializers
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from accounts.models import UserProfile, UserRole


class UserProfileSerializer(serializers.ModelSerializer):
    role_display = serializers.CharField(source='get_role_display', read_only=True)

    class Meta:
        model = UserProfile
        fields = [
            'id',
            'role',
            'role_display',
            'organization',
            'department',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'role', 'role_display', 'created_at', 'updated_at']


class UserSerializer(serializers.ModelSerializer):
    profile = UserProfileSerializer(read_only=True)
    role = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = [
            'id',
            'username',
            'email',
            'first_name',
            'last_name',
            'role',
            'profile',
        ]
        read_only_fields = ['id']

    def get_role(self, obj):
        profile = getattr(obj, 'profile', None)
        if not profile:
            return 'admin' if (obj.is_staff or obj.is_superuser) else 'patient'
        r = profile.role
        if r in (UserRole.CLINICIAN, UserRole.RESEARCHER):
            return 'doctor'
        elif r == UserRole.ADMINISTRATOR or obj.is_superuser:
            return 'admin'
        return 'patient'


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    password_confirm = serializers.CharField(write_only=True, required=True)
    organization = serializers.CharField(write_only=True, required=False, allow_blank=True, default='')
    department = serializers.CharField(write_only=True, required=False, allow_blank=True, default='')

    class Meta:
        model = User
        fields = [
            'username',
            'email',
            'password',
            'password_confirm',
            'first_name',
            'last_name',
            'organization',
            'department',
        ]

    def validate(self, attrs):
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError({"password": "Password fields didn't match."})
        return attrs

    def create(self, validated_data):
        validated_data.pop('password_confirm')
        organization = validated_data.pop('organization', '')
        department = validated_data.pop('department', '')

        # Create user through built-in helper to hash password securely
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data.get('email', ''),
            password=validated_data['password'],
            first_name=validated_data.get('first_name', ''),
            last_name=validated_data.get('last_name', '')
        )

        # Profile is created via post_save signal with default role UPLOADER;
        # update metadata without elevating privileges
        profile = user.profile
        profile.organization = organization
        profile.department = department
        profile.save()

        return user


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField(required=False, allow_blank=True)
    email = serializers.CharField(required=False, allow_blank=True)
    password = serializers.CharField(required=True, write_only=True)

    def validate(self, attrs):
        username = attrs.get('username') or attrs.get('email')
        if not username:
            raise serializers.ValidationError({"email": "Username or email is required."})
        attrs['username'] = username
        return attrs
