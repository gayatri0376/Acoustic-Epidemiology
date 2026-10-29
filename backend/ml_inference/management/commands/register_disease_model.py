"""
Management command to register the disease-classification CNN model in the Django registry.

Usage:
    python manage.py register_disease_model \\
        --model-version sprsound-disease-cnn-v1 \\
        --artifact-path ml/models/saved_models/sprsound-disease-cnn-v1.keras \\
        --classes "Healthy Control" Pneumonia "Wheezing Disorders" \\
        --status ACTIVE \\
        --architecture 2D-CNN-Spectrogram-4Block \\
        --target-conv-layer last_conv_layer \\
        --set-active

Safety guards (identical to register_model):
    1. Directory traversal protection.
    2. Storage convention guard (must be inside ml/models/).
    3. Physical existence guard.
    4. Format guard (.keras, .h5, .hdf5, or SavedModel directory).
    5. Dimension validation (input shape and output neuron count).
    6. Conv layer validation.
    7. Duplicate protection (require --overwrite).
    8. task_type is forced to DISEASE_CLASSIFICATION.
"""
import os
import sys
import json
import logging
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from django.core.exceptions import ValidationError
from django.conf import settings
from ml_inference.models import ModelVersion, ModelStatus, ModelTaskType

logger = logging.getLogger(__name__)

# Re-use the same path-validation helpers as register_model
from ml_inference.management.commands.register_model import (
    resolve_and_validate_artifact_path,
    inspect_and_validate_cnn_artifact,
)


class Command(BaseCommand):
    help = "Register a trained DISEASE_CLASSIFICATION CNN model in the Django model registry."

    def add_arguments(self, parser):
        parser.add_argument(
            "--model-version",
            required=True,
            help="Unique version identifier (e.g., 'sprsound-disease-cnn-v1')."
        )
        parser.add_argument(
            "--artifact-path",
            required=True,
            help="Relative path from project root to the trained .keras artifact."
        )
        parser.add_argument(
            "--classes",
            nargs="+",
            required=True,
            help="Ordered list of verified disease class labels (e.g., 'Healthy Control' Pneumonia 'Wheezing Disorders')."
        )
        parser.add_argument(
            "--status",
            choices=["DEVELOPMENT", "ACTIVE", "RETIRED"],
            default="DEVELOPMENT",
            help="Initial model deployment status. Default: DEVELOPMENT."
        )
        parser.add_argument(
            "--architecture",
            default="2D-CNN-Spectrogram-4Block",
            help="Architecture descriptor string."
        )
        parser.add_argument(
            "--target-conv-layer",
            default="last_conv_layer",
            help="Name of the convolutional layer used for Grad-CAM."
        )
        parser.add_argument(
            "--evaluation-json",
            default=None,
            help="Path to a JSON file containing evaluation metrics to store."
        )
        parser.add_argument(
            "--set-active",
            action="store_true",
            help="If set, retire all other DISEASE_CLASSIFICATION models and activate this one."
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite an existing model version with the same name."
        )

    def handle(self, *args, **options):
        version = options["model_version"].strip()
        artifact_path_str = options["artifact_path"].strip()
        verified_classes = [c.strip() for c in options["classes"]]
        desired_status = options["status"]
        architecture = options["architecture"].strip()
        target_conv_layer = options.get("target_conv_layer", "last_conv_layer")
        set_active = options["set_active"]
        overwrite = options["overwrite"]

        self.stdout.write(self.style.NOTICE(f"\n=== Registering Disease Model: {version} ==="))

        # 1. Path validation
        try:
            resolved_path = resolve_and_validate_artifact_path(artifact_path_str)
        except CommandError as e:
            raise e

        # 2. Artifact inspection and CNN validation
        self.stdout.write("  Inspecting CNN artifact (loading with TensorFlow/Keras)...")
        try:
            inspect_and_validate_cnn_artifact(resolved_path, verified_classes, target_conv_layer)
            self.stdout.write(self.style.SUCCESS("  CNN artifact validated successfully."))
        except CommandError as e:
            raise e

        # 3. Preprocessing config (matches SPRSound contract)
        preprocessing_config = {
            "sample_rate": 8000,
            "n_mels": 128,
            "n_fft": 1024,
            "hop_length": 512,
            "f_min": 50.0,
            "f_max": 2000.0,
            "apply_bandpass": True,
            "top_db": 80.0,
            "normalization": "minmax_01",
        }

        # 4. Load evaluation metadata if provided
        evaluation_metadata = {}
        if options.get("evaluation_json"):
            eval_path = Path(options["evaluation_json"])
            if not eval_path.exists():
                raise CommandError(f"Evaluation JSON not found: {eval_path}")
            with open(eval_path, "r", encoding="utf-8") as f:
                evaluation_metadata = json.load(f)
            self.stdout.write(f"  Loaded evaluation metadata from {eval_path}")

        # 5. Duplicate check
        existing = ModelVersion.objects.filter(version=version).first()
        if existing:
            if not overwrite:
                raise CommandError(
                    f"A model version named '{version}' already exists (id={existing.id}). "
                    "Use --overwrite to replace it."
                )
            self.stdout.write(self.style.WARNING(f"  Overwriting existing model version: {version}"))
            existing.delete()

        # 6. Retire other DISEASE_CLASSIFICATION models if --set-active
        if set_active and desired_status == "ACTIVE":
            retired_count = ModelVersion.objects.filter(
                task_type=ModelTaskType.DISEASE_CLASSIFICATION,
                status=ModelStatus.ACTIVE,
            ).update(status=ModelStatus.RETIRED)
            if retired_count:
                self.stdout.write(self.style.WARNING(
                    f"  Retired {retired_count} previously active DISEASE_CLASSIFICATION model(s)."
                ))

        # 7. Use relative path for storage (from project root)
        project_root = Path(settings.BASE_DIR.parent).resolve()
        try:
            relative_path = str(resolved_path.relative_to(project_root))
        except ValueError:
            relative_path = str(resolved_path)

        # 8. Create the model version record
        model_version = ModelVersion.objects.create(
            version=version,
            architecture=architecture,
            artifact_path=relative_path,
            verified_classes=verified_classes,
            preprocessing_config=preprocessing_config,
            status=desired_status,
            task_type=ModelTaskType.DISEASE_CLASSIFICATION,
            evaluation_metadata=evaluation_metadata,
        )

        self.stdout.write(self.style.SUCCESS(
            f"\n  Disease model registered successfully!"
            f"\n  ID          : {model_version.id}"
            f"\n  Version     : {model_version.version}"
            f"\n  Task Type   : {model_version.task_type}"
            f"\n  Status      : {model_version.status}"
            f"\n  Artifact    : {model_version.artifact_path}"
            f"\n  Classes     : {model_version.verified_classes}"
        ))

        if desired_status == "ACTIVE":
            self.stdout.write(self.style.SUCCESS(
                "\n  Disease model is now ACTIVE. The Django REST API will route "
                "POST /api/v1/recordings/<id>/predict-disease/ requests to this model."
            ))
        else:
            self.stdout.write(self.style.NOTICE(
                f"\n  Model status is '{desired_status}'. To activate:\n"
                f"  python manage.py register_disease_model "
                f"--model-version {version} --artifact-path {artifact_path_str} "
                f"--classes {' '.join(repr(c) for c in verified_classes)} "
                f"--status ACTIVE --set-active --overwrite"
            ))

