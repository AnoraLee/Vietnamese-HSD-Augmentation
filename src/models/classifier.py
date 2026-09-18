"""Backward-compatible name for the shared inference service."""

from src.services.inference import HSDInferenceService


class HSDClassifier(HSDInferenceService):
    """Compatibility wrapper retained for the temporary Streamlit demo."""
