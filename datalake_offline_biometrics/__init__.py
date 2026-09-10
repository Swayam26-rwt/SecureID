"""Offline facial recognition and liveness primitives for NHAI Datalake 3.0."""

__version__ = "1.0.0"

from .engine import AuthenticationResult, OfflineBiometricEngine
from .image_ops import synthetic_face
from .liveness import LivenessDetector, LivenessResult
from .recognition import FaceTemplate, LBPHFaceRecognizer, RecognitionResult
from .storage import TemplateStore

__all__ = [
    "__version__",
    "AuthenticationResult",
    "FaceTemplate",
    "LBPHFaceRecognizer",
    "LivenessDetector",
    "LivenessResult",
    "OfflineBiometricEngine",
    "RecognitionResult",
    "TemplateStore",
    "synthetic_face",
]
