"""
Backend Synchronization & Model Registration Bridge.

Registers trained CNN models into Django database and validates backend readiness.
Works directly with Gayatri's Django backend architecture.
"""

import os
import sys
import shutil
import logging
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure imports work
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

try:
    from .config import (
        DISEASE_CLASSES,
        SOUND_CLASSES_4CLASS,
        BACKEND_SAVED_MODELS_DIR,
        LOCAL_SAVED_MODELS_DIR,
        TARGET_CONV_LAYER_NAME,
    )
    from .predict import find_latest_model
except (ImportError, ValueError):
    from config import (
        DISEASE_CLASSES,
        SOUND_CLASSES_4CLASS,
        BACKEND_SAVED_MODELS_DIR,
        LOCAL_SAVED_MODELS_DIR,
        TARGET_CONV_LAYER_NAME,
    )
    from predict import find_latest_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BackendSync")


def sync_and_register_model(
    model_path: Optional[str] = None,
    task: str = "disease",
    model_version: Optional[str] = None,
    set_active: bool = True
) -> Dict[str, Any]:
    """
    1. Copies model to ml/models/saved_models/.
    2. Runs Django register_disease_model or register_model.
    3. Confirms model is marked ACTIVE in Django database.
    """
    # 1. Locate model artifact
    if model_path:
        source_artifact = Path(model_path)
    else:
        source_artifact = find_latest_model(task=task)

    if not source_artifact or not source_artifact.exists():
        raise FileNotFoundError(
            f"No model artifact found to sync for task '{task}'. Train a model first."
        )

    # 2. Copy to backend storage convention (ml/models/saved_models/)
    filename = source_artifact.name
    target_artifact = BACKEND_SAVED_MODELS_DIR / filename
    BACKEND_SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if source_artifact.resolve() != target_artifact.resolve():
        shutil.copy2(source_artifact, target_artifact)
        logger.info("Copied model artifact: %s -> %s", source_artifact, target_artifact)

    rel_artifact_path = f"ml/models/saved_models/{filename}"

    # 3. Model parameters
    if not model_version:
        model_version = source_artifact.stem

    classes = DISEASE_CLASSES if task == "disease" else SOUND_CLASSES_4CLASS
    cmd_name = "register_disease_model" if task == "disease" else "register_model"
    arch = "3Block-ThroatDisease-2DCNN" if task == "disease" else "4Block-ThroatSound-2DCNN"

    # 4. Construct Django management command
    manage_py = BACKEND_DIR / "manage.py"
    if not manage_py.exists():
        raise FileNotFoundError(f"Django manage.py not found at {manage_py}")

    cmd = [
        sys.executable,
        str(manage_py),
        cmd_name,
        "--model-version", model_version,
        "--artifact-path", rel_artifact_path,
        "--classes", *classes,
        "--status", "ACTIVE" if set_active else "DEVELOPMENT",
        "--architecture", arch,
        "--target-conv-layer", TARGET_CONV_LAYER_NAME,
        "--overwrite"
    ]
    if set_active:
        cmd.append("--set-active")

    logger.info("Executing Django command:\n%s", " ".join(cmd))

    # Run the registration subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd,
        cwd=str(BACKEND_DIR),
        capture_output=True,
        text=True,
        env=env
    )

    if result.returncode != 0:
        logger.error("Django registration failed:\nSTDOUT:\n%s\nSTDERR:\n%s", result.stdout, result.stderr)
        raise RuntimeError(f"Registration command failed: {result.stderr or result.stdout}")

    logger.info("Django registration succeeded:\n%s", result.stdout.strip())

    return {
        "status": "REGISTERED",
        "model_version": model_version,
        "artifact_path": rel_artifact_path,
        "classes": classes,
        "task": task,
        "active": set_active,
        "backend_output": result.stdout.strip()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync and register CNN model with Django backend")
    parser.add_argument("--model", type=str, default=None, help="Path to .keras model")
    parser.add_argument("--task", choices=["disease", "sound"], default="disease")
    parser.add_argument("--version-name", type=str, default=None)
    parser.add_argument("--inactive", action="store_true", help="Register as DEVELOPMENT rather than ACTIVE")

    args = parser.parse_args()

    try:
        res = sync_and_register_model(
            model_path=args.model,
            task=args.task,
            model_version=args.version_name,
            set_active=not args.inactive
        )
        print("\nModel successfully registered and active in Django backend!")
    except Exception as exc:
        logger.error("Sync failed: %s", exc)
        sys.exit(1)
