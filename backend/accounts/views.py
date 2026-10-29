from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from accounts.serializers import (
    UserSerializer, 
    RegisterSerializer, 
    LoginSerializer,
    UserProfileSerializer
)


class RegisterView(APIView):
    """
    Public registration endpoint.
    Users cannot self-assign privileged clinician or administrator roles.
    """
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            token, _ = Token.objects.get_or_create(user=user)
            return Response(
                {
                    "message": "User registered successfully.",
                    "user": UserSerializer(user).data,
                    "token": token.key,
                },
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(APIView):
    """
    Authentication endpoint returning an API Auth Token.
    Accepts username or email.
    """
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = LoginSerializer(data=request.data)
        if serializer.is_valid():
            ident = serializer.validated_data['username'].strip()
            password = serializer.validated_data['password']
            
            # 1. Try standard Django authenticate
            user = authenticate(request=request, username=ident, password=password)

            # 2. If not authenticated directly, find user by email or username
            if not user:
                from django.contrib.auth.models import User
                matched_user = (
                    User.objects.filter(email__iexact=ident).first() or
                    User.objects.filter(username__iexact=ident).first()
                )
                if matched_user:
                    if matched_user.check_password(password) or matched_user.check_password(password.strip()):
                        user = matched_user

            if user:
                if not user.is_active:
                    return Response(
                        {"error": "Account is disabled."},
                        status=status.HTTP_403_FORBIDDEN,
                    )
                token, _ = Token.objects.get_or_create(user=user)
                return Response(
                    {
                        "message": "Authentication successful.",
                        "token": token.key,
                        "auth_token": token.key,
                        "user": UserSerializer(user).data,
                    },
                    status=status.HTTP_200_OK,
                )
            return Response(
                {"error": "Invalid username, email, or password."},
                status=status.HTTP_401_UNAUTHORIZED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LogoutView(APIView):
    """
    Terminates the session and deletes the user's active auth token.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        try:
            request.user.auth_token.delete()
        except Exception:
            pass
        return Response(
            {"message": "Logged out successfully."},
            status=status.HTTP_200_OK,
        )


class CurrentUserView(APIView):
    """
    Returns the profile and permissions of the currently authenticated user.
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        serializer = UserSerializer(request.user)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def patch(self, request, *args, **kwargs):
        profile = getattr(request.user, 'profile', None)
        if not profile:
            return Response({"error": "Profile not found."}, status=status.HTTP_404_NOT_FOUND)
        
        # Allow updating organization and department only
        serializer = UserProfileSerializer(
            profile, 
            data=request.data, 
            partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response(UserSerializer(request.user).data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
