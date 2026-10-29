from django.test import SimpleTestCase, override_settings

from ml_inference.services import InferenceResult, run_inference


class InferenceTests(SimpleTestCase):
    @override_settings(ML_MODEL_PATH='')
    def test_not_configured_returns_no_prediction(self):
        result = run_inference(recording=None)
        self.assertIsInstance(result, InferenceResult)
        self.assertEqual(result.status, 'not_configured')
        self.assertIsNone(result.label)
        self.assertIsNone(result.confidence)
