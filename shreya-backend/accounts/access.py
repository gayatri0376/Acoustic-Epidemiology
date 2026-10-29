"""The ONE place that decides which rows a user may see. Every list/detail view uses it."""
from .models import Role


def visible_to(user, queryset, patient_field='patient'):
    """Filter `queryset` to rows the user may access.
    patient_field is the path from the row to its patient, e.g. 'patient' or 'assessment__patient'."""
    if not user or not user.is_authenticated or not user.is_active:
        return queryset.none()
    if user.role == Role.ADMIN:
        return queryset
    if user.role == Role.DOCTOR:  # only assigned patients
        return queryset.filter(**{f'{patient_field}__doctor_links__doctor': user})
    if user.role == Role.PATIENT:  # only their own records
        return queryset.filter(**{patient_field: user})
    return queryset.none()
