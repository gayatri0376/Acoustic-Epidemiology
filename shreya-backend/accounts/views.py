from rest_framework import generics, status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from .serializers import LoginSerializer, RegisterSerializer, UserSerializer


class ThrottledMixin:
    """Rate-limits an endpoint using the "auth" rate from settings (default 10/min per IP)."""
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'auth'


class RegisterView(ThrottledMixin, generics.CreateAPIView):
    permission_classes = [AllowAny]
    authentication_classes = []
    serializer_class = RegisterSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response({'user': UserSerializer(user).data}, status=status.HTTP_201_CREATED)


class LoginView(ThrottledMixin, TokenObtainPairView):
    serializer_class = LoginSerializer


class RefreshView(ThrottledMixin, TokenRefreshView):
    pass


class MeView(APIView):
    """Returns the authenticated user's profile. The role comes from the database, not from the token."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(UserSerializer(request.user).data)


class LogoutView(APIView):
    """Revokes (blacklists) the given refresh token. The short-lived access token expires on its own."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        refresh = request.data.get('refresh')
        if not refresh:
            return Response({'detail': 'The "refresh" token is required.'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            token = RefreshToken(refresh)
            if str(token['user_id']) != str(request.user.pk):  # can't revoke someone else's token
                return Response({'detail': 'Invalid refresh token.'}, status=status.HTTP_400_BAD_REQUEST)
            token.blacklist()
        except TokenError:
            return Response({'detail': 'Invalid or expired refresh token.'}, status=status.HTTP_400_BAD_REQUEST)
        return Response({'detail': 'Logged out.'})
