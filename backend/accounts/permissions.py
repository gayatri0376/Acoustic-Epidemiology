from rest_framework import permissions
from accounts.models import UserRole


class IsClinician(permissions.BasePermission):
    """
    Allows access only to authenticated clinicians or administrators.
    """
    def has_permission(self, request, view):
        if not (request.user and request.user.is_authenticated):
            return False
        if request.user.is_superuser or request.user.is_staff:
            return True
        profile = getattr(request.user, 'profile', None)
        return bool(profile and profile.is_clinician)


class IsResearcher(permissions.BasePermission):
    """
    Allows access to researchers, clinicians, and administrators.
    """
    def has_permission(self, request, view):
        if not (request.user and request.user.is_authenticated):
            return False
        if request.user.is_superuser or request.user.is_staff:
            return True
        profile = getattr(request.user, 'profile', None)
        return bool(profile and (profile.is_researcher or profile.is_clinician))


class IsUploader(permissions.BasePermission):
    """
    Allows access to authenticated users who have upload rights.
    """
    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated)


class IsOwnerOrClinician(permissions.BasePermission):
    """
    Object-level permission allowing access to the object owner, clinicians, or admins.
    """
    def has_object_permission(self, request, view, obj):
        if not (request.user and request.user.is_authenticated):
            return False
        if request.user.is_superuser or request.user.is_staff:
            return True
        
        # Check if user is clinician, researcher, or admin
        profile = getattr(request.user, 'profile', None)
        if profile and (profile.is_clinician or profile.is_researcher):
            return True


        # Check ownership on different entity types
        uploader = getattr(obj, 'uploader', None)
        if uploader and uploader == request.user:
            return True

        reviewer = getattr(obj, 'reviewer', None)
        if reviewer and reviewer == request.user:
            return True

        # If obj has a recording or prediction relationship
        recording = getattr(obj, 'recording', None)
        if recording and getattr(recording, 'uploader', None) == request.user:
            return True

        return False


class IsOwnerOrReadOnly(permissions.BasePermission):
    """
    Object-level permission to allow read-only for permitted users, and edit for owners.
    """
    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        if not (request.user and request.user.is_authenticated):
            return False
        if request.user.is_superuser:
            return True
        uploader = getattr(obj, 'uploader', None)
        return uploader == request.user
