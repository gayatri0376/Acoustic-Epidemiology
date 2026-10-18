import os
import sys
import json
import logging
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from django.core.exceptions import ValidationError
from django.conf import settings
from ml_inference.models import ModelVersion, ModelStatus

logger = logging.getLogger(__name__)


def resolve_and_validate_artifact_path(artifact_path_str):
    """
    Validates that the artifact path:
    1. Does not escape project repository root (directory traversal protection).
    2. Resides within the designated 'ml/models/' storage hierarchy.
    3. Physically exists on disk.
    4. Has a recognized extension (.keras, .h5, .hdf5) or is a SavedModel directory.
    """
    if not artifact_path_str or not artifact_path_str.strip():
        raise CommandError("Artifact path cannot be empty.")

    project_root = Path(settings.BASE_DIR.parent).resolve()
    raw_path = Path(artifact_path_str.strip())

    if raw_path.is_absolute():
        resolved = raw_path.resolve()
    else:
        resolved = (project_root / raw_path).resolve()

    # 1. Directory traversal guard
    try:
        resolved.relative_to(project_root)
    except ValueError:
        raise CommandError(
            f"Security violation: Artifact path '{artifact_path_str}' resolves outside "
            f"the project repository root ({project_root}). Path traversal is strictly forbidden."
        )

    # 2. Convention guard: must be inside ml/models/
    allowed_model_dir = (project_root / "ml" / "models").resolve()
    try:
        resolved.relative_to(allowed_model_dir)
    except ValueError:
        raise CommandError(
            f"Model storage convention violation: Artifact path '{artifact_path_str}' must be located "
            f"inside the 'ml/models/' hierarchy (e.g., 'ml/models/saved_models/<filename>.keras')."
        )

    # 3. Physical existence guard
    if not resolved.exists():
        raise CommandError(
            f"Artifact file not found: '{resolved}' does not exist on disk. "
            "Please ensure the trained model artifact has been physically delivered before registration."
        )

    # 4. Format guard
    if resolved.is_dir():
        if not (resolved / "saved_model.pb").exists():
            raise CommandError(
                f"Invalid directory artifact: '{resolved}' is a directory but does not contain 'saved_model.pb'. "
                "TensorFlow SavedModel directories must contain 'saved_model.pb' and variables."
            )
    else:
        valid_extensions = (".keras", ".h5", ".hdf5")
        if resolved.suffix.lower() not in valid_extensions:
            raise CommandError(
                f"Unsupported model artifact extension '{resolved.suffix}'. "
                f"Supported formats: {', '.join(valid_extensions)} or a TensorFlow SavedModel directory."
            )

    return resolved


def inspect_and_validate_cnn_artifact(resolved_path, verified_classes, target_conv_layer=None):
    """
    Safely loads the genuine CNN artifact using TensorFlow/Keras and validates:
    - Input tensor shape.
    - Output tensor shape and neuron count against verified_classes.
    - Output activation (Softmax vs Logits).
    - Presence of target_conv_layer if specified.

    Fails gracefully if TensorFlow/Keras is unavailable.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise CommandError(
            "ML Runtime Dependency Error: TensorFlow/Keras is not installed in the active environment "
            f"({sys.executable}). "
            "The model artifact cannot be loaded, inspected, or verified. "
            "TensorFlow must be installed before a model can be verified and registered."
        )

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    try:
        model = tf.keras.models.load_model(str(resolved_path), compile=False)
    except Exception as exc:
        raise CommandError(f"Failed to load model artifact with tf.keras.models.load_model: {exc}")

    # 1. Validate Input Shape
    raw_input_shape = getattr(model, "input_shape", None)
    if isinstance(raw_input_shape, list) and len(raw_input_shape) > 0:
        raw_input_shape = raw_input_shape[0]

    if not raw_input_shape:
        raise CommandError("Could not inspect model input shape: model.input_shape is undefined.")

    # 2. Validate Output Shape and Classes
    raw_output_shape = getattr(model, "output_shape", None)
    if isinstance(raw_output_shape, list) and len(raw_output_shape) > 0:
        raw_output_shape = raw_output_shape[0]

    if not raw_output_shape or len(raw_output_shape) < 2:
        raise CommandError(
            f"Invalid model output shape: {raw_output_shape}. Expected 2D tensor (batch_size, num_classes)."
        )

    num_output_neurons = raw_output_shape[-1]
    if num_output_neurons != len(verified_classes):
        raise CommandError(
            f"Output dimension mismatch: Model output layer has {num_output_neurons} neurons, "
            f"but {len(verified_classes)} verified class labels were supplied: {verified_classes}. "
            "The number of class labels must match the model's output dimensionality exactly."
        )

    # 3. Validate Output Activation (Softmax vs Logits)
    output_activation = "unknown"
    try:
        layers = getattr(model, "layers", [])
        if layers:
            last_layer = layers[-1]
            act = getattr(last_layer, "activation", None)
            if act is not None:
                act_name = getattr(act, "__name__", str(act)).lower()
                if "softmax" in act_name:
                    output_activation = "softmax"
                elif "linear" in act_name or act_name == "none":
                    output_activation = "logits"
                else:
                    output_activation = act_name
    except Exception:
        output_activation = "unverified"

    # 4. Validate Target Conv Layer for Grad-CAM
    if target_conv_layer:
        layer_names = [l.name for l in getattr(model, "layers", [])]
        if target_conv_layer not in layer_names:
            raise CommandError(
                f"Target Conv2D layer '{target_conv_layer}' not found in model topology. "
                f"Available layers: {layer_names}"
            )

    return {
        "input_shape": list(raw_input_shape) if raw_input_shape else None,
        "output_shape": list(raw_output_shape) if raw_output_shape else None,
        "output_activation": output_activation,
    }


class Command(BaseCommand):
    help = (
        "Safely registers a verified CNN model artifact into the ModelVersion registry. "
        "Validates filesystem storage conventions, model topology, and preprocessing contracts "
        "without fabricating metrics or activating unverified weights."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--model-version",
            dest="model_version",
            type=str,
            required=True,
            help="Unique model version identifier (e.g. 'pediatric-cnn-v1.0.0').",
        )
        parser.add_argument(
            "--artifact-path",
            type=str,
            required=True,
            help="Path to trained weights file relative to project root (e.g. 'ml/models/saved_models/cnn_v1.keras').",
        )
        parser.add_argument(
            "--classes",
            nargs="+",
            required=True,
            help="Ordered list of class labels matching model output neurons (e.g. --classes Normal Wheeze Crackle).",
        )
        parser.add_argument(
            "--architecture",
            type=str,
            default="2D-CNN-Spectrogram",
            help="Neural network architecture descriptor (default: '2D-CNN-Spectrogram').",
        )
        parser.add_argument(
            "--status",
            type=str,
            choices=[s.value for s in ModelStatus],
            default=ModelStatus.DEVELOPMENT.value,
            help="Registration status: 'DEVELOPMENT', 'ACTIVE', or 'RETIRED' (default: 'DEVELOPMENT').",
        )
        parser.add_argument(
            "--sample-rate",
            type=int,
            default=16000,
            help="Audio sampling rate in Hz (default: 16000).",
        )
        parser.add_argument(
            "--n-mels",
            type=int,
            default=128,
            help="Number of Mel frequency filterbanks (default: 128).",
        )
        parser.add_argument(
            "--n-fft",
            type=int,
            default=1024,
            help="STFT FFT window length (default: 1024).",
        )
        parser.add_argument(
            "--hop-length",
            type=int,
            default=512,
            help="STFT hop length in samples (default: 512).",
        )
        parser.add_argument(
            "--f-min",
            type=float,
            default=50.0,
            help="Minimum frequency cutoff in Hz (default: 50.0).",
        )
        parser.add_argument(
            "--f-max",
            type=float,
            default=2000.0,
            help="Maximum frequency cutoff in Hz (default: 2000.0).",
        )
        parser.add_argument(
            "--preprocessing-config",
            type=str,
            default=None,
            help="Optional path to a JSON file or raw JSON string with full preprocessing settings.",
        )
        parser.add_argument(
            "--target-conv-layer",
            type=str,
            help="Target 2D Convolutional layer name for Grad-CAM explainability (e.g. 'conv2d_final').",
        )
        parser.add_argument(
            "--evaluation-metadata",
            type=str,
            help="Optional path to JSON file or raw JSON string with genuine test metrics (e.g. test_accuracy).",
        )
        parser.add_argument(
            "--replace",
            action="store_true",
            help="Allow updating an existing ModelVersion record with the same version string.",
        )
        parser.add_argument(
            "--deactivate-others",
            action="store_true",
            help="When setting status to ACTIVE, retire any currently active models.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Perform all validations without saving anything to the database.",
        )

    def handle(self, *args, **options):
        version_str = options["model_version"].strip()
        artifact_path_str = options["artifact_path"].strip()
        raw_classes = options["classes"]
        architecture = options["architecture"].strip()
        status_choice = options["status"]
        replace = options["replace"]
        deactivate_others = options["deactivate_others"]
        dry_run = options["dry_run"]
        target_conv_layer = options.get("target_conv_layer")
        if target_conv_layer:
            target_conv_layer = target_conv_layer.strip()

        # Parse classes: handle both comma-separated strings inside list and space-separated list
        verified_classes = []
        for item in raw_classes:
            if "," in item:
                verified_classes.extend([c.strip() for c in item.split(",") if c.strip()])
            elif item.strip():
                verified_classes.append(item.strip())

        if not verified_classes:
            raise CommandError("At least one class label must be supplied via --classes.")

        self.stdout.write(self.style.NOTICE(f"Initiating registration verification for '{version_str}'..."))

        # 1. Path and storage convention validation
        resolved_path = resolve_and_validate_artifact_path(artifact_path_str)
        project_root = Path(settings.BASE_DIR.parent).resolve()
        relative_path_for_storage = str(resolved_path.relative_to(project_root))

        self.stdout.write(f"  [✓] Artifact physically verified at: {resolved_path}")
        self.stdout.write(f"  [✓] Storage path relative to project: {relative_path_for_storage}")

        # 2. Inspect CNN topology using TensorFlow
        inspection_results = inspect_and_validate_cnn_artifact(
            resolved_path=resolved_path,
            verified_classes=verified_classes,
            target_conv_layer=target_conv_layer,
        )

        input_shape = inspection_results["input_shape"]
        output_shape = inspection_results["output_shape"]
        output_activation = inspection_results["output_activation"]

        self.stdout.write(f"  [✓] Model loaded. Input shape: {input_shape}")
        self.stdout.write(f"  [✓] Output shape: {output_shape} ({len(verified_classes)} classes)")
        self.stdout.write(f"  [✓] Output activation mode: {output_activation}")

        # 3. Assemble Preprocessing Configuration
        preprocessing_config = {
            "sample_rate": options["sample_rate"],
            "n_mels": options["n_mels"],
            "n_fft": options["n_fft"],
            "hop_length": options["hop_length"],
            "f_min": options["f_min"],
            "f_max": options["f_max"],
            "to_db": True,
            "expected_input_shape": input_shape,
            "output_shape": output_shape,
            "output_activation": output_activation,
        }
        if target_conv_layer:
            preprocessing_config["target_conv_layer"] = target_conv_layer

        # Merge optional JSON preprocessing override
        custom_prep = options.get("preprocessing_config")
        if custom_prep:
            try:
                if os.path.isfile(custom_prep):
                    with open(custom_prep, "r", encoding="utf-8") as f:
                        loaded_prep = json.load(f)
                else:
                    loaded_prep = json.loads(custom_prep)
                if not isinstance(loaded_prep, dict):
                    raise ValueError("JSON content must be a dictionary.")
                preprocessing_config.update(loaded_prep)
                self.stdout.write("  [✓] Merged custom preprocessing configuration.")
            except Exception as exc:
                raise CommandError(f"Invalid --preprocessing-config argument: {exc}")

        # 4. Handle Evaluation Metadata (strictly avoid fabrication)
        evaluation_metadata = {}
        eval_arg = options.get("evaluation_metadata")
        if eval_arg:
            try:
                if os.path.isfile(eval_arg):
                    with open(eval_arg, "r", encoding="utf-8") as f:
                        evaluation_metadata = json.load(f)
                else:
                    evaluation_metadata = json.loads(eval_arg)
                if not isinstance(evaluation_metadata, dict):
                    raise ValueError("Evaluation metadata must be a dictionary of verified metrics.")
                self.stdout.write(f"  [✓] Parsed {len(evaluation_metadata)} verified evaluation metrics.")
            except Exception as exc:
                raise CommandError(f"Invalid --evaluation-metadata argument: {exc}")
        else:
            self.stdout.write("  [-] No evaluation metadata provided (empty dictionary preserved).")

        # 5. Database Duplicate Check
        existing = ModelVersion.objects.filter(version=version_str).first()
        if existing and not replace:
            raise CommandError(
                f"ModelVersion '{version_str}' already exists in database (current status: {existing.status}). "
                "Specify --replace if you explicitly intend to overwrite this model registration."
            )

        # 6. Activation Safety Handling
        if status_choice == ModelStatus.ACTIVE.value:
            if deactivate_others and not dry_run:
                active_count = ModelVersion.objects.filter(status=ModelStatus.ACTIVE.value).count()
                if active_count > 0:
                    ModelVersion.objects.filter(status=ModelStatus.ACTIVE.value).update(
                        status=ModelStatus.RETIRED.value
                    )
                    self.stdout.write(f"  [✓] Retired {active_count} previously active model(s).")

        # 7. Model Instance Construction & Django full_clean() Validation
        if existing:
            model_instance = existing
            model_instance.architecture = architecture
            model_instance.artifact_path = relative_path_for_storage
            model_instance.verified_classes = verified_classes
            model_instance.preprocessing_config = preprocessing_config
            model_instance.status = status_choice
            model_instance.evaluation_metadata = evaluation_metadata
        else:
            model_instance = ModelVersion(
                version=version_str,
                architecture=architecture,
                artifact_path=relative_path_for_storage,
                verified_classes=verified_classes,
                preprocessing_config=preprocessing_config,
                status=status_choice,
                evaluation_metadata=evaluation_metadata,
            )

        try:
            model_instance.full_clean()
        except ValidationError as val_err:
            raise CommandError(f"ModelVersion validation rejected: {val_err.message_dict}")

        if dry_run:
            self.stdout.write(
                self.style.SUCCESS(
                    f"DRY RUN SUCCESS: ModelVersion '{version_str}' passed all artifact, topology, "
                    "and schema validations. No database changes were made."
                )
            )
            return

        model_instance.save()

        self.stdout.write(
            self.style.SUCCESS(
                f"SUCCESS: ModelVersion '{version_str}' successfully registered with status '{status_choice}'.\n"
                f"  ID: {model_instance.id}\n"
                f"  Artifact: {relative_path_for_storage}\n"
                f"  Classes ({len(verified_classes)}): {verified_classes}\n"
                f"  Activation: {output_activation}\n"
            )
        )
