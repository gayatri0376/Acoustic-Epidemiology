import uuid
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserRole(models.TextChoices):
    CLINICIAN = 'CLINICIAN', 'Clinician / Pulmonologist'
    RESEARCHER = 'RESEARCHER', 'Biomedical Researcher'
    UPLOADER = 'UPLOADER', 'Acoustic Data Uploader'
    ADMINISTRATOR = 'ADMINISTRATOR', 'System Administrator'


class UserProfile(models.Model):
    """
    Extends Django built-in User with role-based access control,
    institutional affiliation, and UUID identifier.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        User, 
        on_delete=models.CASCADE, 
        related_name='profile'
    )
    role = models.CharField(
        max_length=32,
        choices=UserRole.choices,
        default=UserRole.UPLOADER,
        db_index=True,
        help_text="Role determining clinical assessment and administration permissions."
    )
    organization = models.CharField(
        max_length=255, 
        blank=True, 
        help_text="Hospital, clinic, or research institution affiliation."
    )
    department = models.CharField(
        max_length=128, 
        blank=True, 
        help_text="Medical department, e.g., Pediatric Pulmonology."
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'user_profiles'
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} ({self.get_role_display()})"

    @property
    def is_clinician(self):
        return self.role in (UserRole.CLINICIAN, UserRole.ADMINISTRATOR)

    @property
    def is_researcher(self):
        return self.role in (UserRole.RESEARCHER, UserRole.ADMINISTRATOR)

    @property
    def is_administrator(self):
        return self.role == UserRole.ADMINISTRATOR or self.user.is_staff or self.user.is_superuser


@receiver(post_save, sender=User)
def create_or_save_user_profile(sender, instance, created, **kwargs):
    """
    Automatically creates a UserProfile for new users.
    Existing profiles are saved on user updates.
    """
    if created:
        UserProfile.objects.create(
            user=instance,
            role=UserRole.ADMINISTRATOR if instance.is_superuser else UserRole.UPLOADER
        )
    else:
        if hasattr(instance, 'profile'):
            instance.profile.save()
