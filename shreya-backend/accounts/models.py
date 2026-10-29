from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.core.exceptions import ValidationError
from django.db import models


class Role(models.TextChoices):
    PATIENT = 'patient', 'Patient'
    DOCTOR = 'doctor', 'Doctor'
    ADMIN = 'admin', 'Admin'


class UserManager(BaseUserManager):
    use_in_migrations = True

    def get_by_natural_key(self, email):
        return self.get(email__iexact=email)  # case-insensitive login

    def _create(self, email, password, **extra):
        if not email:
            raise ValueError('An email address is required.')
        user = self.model(email=self.normalize_email(email).lower(), **extra)
        user.set_password(password)  # hashes the password; plaintext is never stored
        user.save(using=self._db)
        return user

    def create_user(self, email, password=None, **extra):
        extra.setdefault('role', Role.PATIENT)
        extra.setdefault('is_staff', False)
        extra.setdefault('is_superuser', False)
        return self._create(email, password, **extra)

    def create_superuser(self, email, password=None, **extra):
        extra.update(role=Role.ADMIN, is_staff=True, is_superuser=True)
        return self._create(email, password, **extra)


class User(AbstractBaseUser, PermissionsMixin):
    """Custom user. Email is the login name. `role` decides what the API allows."""
    email = models.EmailField(unique=True)
    full_name = models.CharField(max_length=150)
    phone_number = models.CharField(max_length=20, blank=True)
    role = models.CharField(max_length=10, choices=Role.choices, default=Role.PATIENT, db_index=True)
    is_active = models.BooleanField(default=True)  # False = account disabled / awaiting approval
    is_staff = models.BooleanField(default=False)  # may log in to /admin/
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = UserManager()
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['full_name']

    def __str__(self):
        return f'{self.email} ({self.role})'


class DoctorPatientAssignment(models.Model):
    """Access policy: a doctor may see only the patients assigned to them. Managed by admins."""
    doctor = models.ForeignKey(User, on_delete=models.CASCADE, related_name='patient_assignments',
                               limit_choices_to={'role': Role.DOCTOR})
    patient = models.ForeignKey(User, on_delete=models.CASCADE, related_name='doctor_links',
                                limit_choices_to={'role': Role.PATIENT})
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=['doctor', 'patient'], name='unique_doctor_patient')]

    def clean(self):
        if self.doctor_id and self.doctor.role != Role.DOCTOR:
            raise ValidationError({'doctor': 'This user is not a doctor.'})
        if self.patient_id and self.patient.role != Role.PATIENT:
            raise ValidationError({'patient': 'This user is not a patient.'})

    def __str__(self):
        return f'{self.doctor.email} -> {self.patient.email}'
