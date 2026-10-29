from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin

from .models import DoctorPatientAssignment, User


@admin.register(User)
class UserAdmin(DjangoUserAdmin):
    """Doctors and admins are created or promoted here (trusted process), never through public sign-up."""
    ordering = ('email',)
    list_display = ('email', 'full_name', 'role', 'is_active', 'is_staff')
    list_filter = ('role', 'is_active', 'is_staff')
    search_fields = ('email', 'full_name')
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Profile', {'fields': ('full_name', 'phone_number', 'role')}),
        ('Status', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Dates', {'fields': ('last_login',)}),
    )
    add_fieldsets = (
        (None, {'classes': ('wide',), 'fields': ('email', 'full_name', 'role', 'password1', 'password2')}),
    )


@admin.register(DoctorPatientAssignment)
class AssignmentAdmin(admin.ModelAdmin):
    list_display = ('doctor', 'patient', 'created_at')
