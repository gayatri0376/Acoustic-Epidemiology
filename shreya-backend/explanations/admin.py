from django.contrib import admin

from .models import Explanation


@admin.register(Explanation)
class ExplanationAdmin(admin.ModelAdmin):
    list_display = ('id', 'assessment', 'author', 'created_at')
