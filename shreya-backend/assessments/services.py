from ml_inference.services import run_inference

from .models import Assessment


def create_assessment_for_recording(recording):
    """Runs inference through the service layer and stores exactly what it returned (no invented values)."""
    result = run_inference(recording)
    return Assessment.objects.create(
        recording=recording,
        patient=recording.patient,
        status=result.status,
        result_label=result.label or '',
        confidence=result.confidence,
        summary=result.message,
    )
