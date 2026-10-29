from rest_framework.permissions import BasePermission

from .models import Role


class HasRole(BasePermission):
    """Allows access only to active users whose role (read from the database) is listed."""
    allowed_roles = ()

    def has_permission(self, request, view):
        user = request.user
        return bool(user and user.is_authenticated and user.is_active and user.role in self.allowed_roles)


class IsPatient(HasRole):
    allowed_roles = (Role.PATIENT,)


class IsDoctorOrAdmin(HasRole):
    allowed_roles = (Role.DOCTOR, Role.ADMIN)


class IsAdminRole(HasRole):
    allowed_roles = (Role.ADMIN,)
