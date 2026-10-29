"""
Migration 0003: Add ModelTaskType to ModelVersion and create DiseasePrediction table.

This migration:
1. Adds task_type field to ml_inference_modelversions to distinguish between
   SOUND_CLASSIFICATION (existing Model A) and DISEASE_CLASSIFICATION (new Model B).
2. Creates the disease_predictions table for SPRSound 3-class disease model outputs.

Dataset: SPRSound (SJTU Paediatric Respiratory Sound Database) CC BY 4.0
Classes: Healthy Control, Pneumonia, Wheezing Disorders
Population: 100% pediatric patients (0–16.2 years)
"""
# Generated for Acoustic-Epidemiology project
import uuid
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('recordings', '0001_initial'),
        ('ml_inference', '0002_add_task_type_to_model_version'),
    ]

    operations = [
        # 0002 already added task_type — confirm it is reflected in this state.
        # Create DiseasePrediction table.
        migrations.CreateModel(
            name='DiseasePrediction',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('predicted_disease', models.CharField(
                    blank=True,
                    max_length=128,
                    help_text=(
                        "Highest-probability disease screening class. "
                        "Valid values: 'Healthy Control', 'Pneumonia', 'Wheezing Disorders'. "
                        "This is a research output; not a clinical diagnosis."
                    )
                )),
                ('probability_scores', models.JSONField(
                    blank=True,
                    default=dict,
                    help_text="Dictionary mapping verified disease classes to softmax probability values."
                )),
                ('prediction_status', models.CharField(
                    choices=[
                        ('PENDING', 'Pending Inference'),
                        ('SUCCESS', 'Inference Successful'),
                        ('FAILED', 'Inference Failed'),
                        ('UNAVAILABLE', 'Model Artifact Unavailable'),
                    ],
                    db_index=True,
                    default='PENDING',
                    max_length=32
                )),
                ('error_details', models.TextField(
                    blank=True,
                    help_text="Safe technical error diagnostic if disease prediction failed or was unavailable."
                )),
                ('inference_duration_ms', models.IntegerField(
                    blank=True,
                    null=True,
                    help_text="Time taken for audio preprocessing and model forward pass in milliseconds."
                )),
                ('created_at', models.DateTimeField(auto_now_add=True, db_index=True)),
                ('recording', models.ForeignKey(
                    db_index=True,
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='disease_predictions',
                    to='recordings.recording'
                )),
                ('model_version', models.ForeignKey(
                    blank=True,
                    db_index=True,
                    null=True,
                    on_delete=django.db.models.deletion.PROTECT,
                    related_name='disease_predictions',
                    to='ml_inference.modelversion'
                )),
            ],
            options={
                'verbose_name': 'Disease Prediction',
                'verbose_name_plural': 'Disease Predictions',
                'db_table': 'disease_predictions',
                'ordering': ['-created_at'],
            },
        ),
        migrations.AddIndex(
            model_name='diseaseprediction',
            index=models.Index(fields=['recording', '-created_at'], name='disease_pred_recording_idx'),
        ),
        migrations.AddIndex(
            model_name='diseaseprediction',
            index=models.Index(fields=['prediction_status', '-created_at'], name='disease_pred_status_idx'),
        ),
    ]

