"""SecureID — Offline Facial Recognition and Liveness Detection Engine.

v2.0.0 — Advanced ML capstone release:
  - Multi-modal score-level fusion (ml_fusion)
  - Operational session analytics (analytics)
  - CLAHE adaptive equalization + Gabor + LPQ descriptors (image_ops)
  - Cosine similarity + PCA whitening + adaptive EER threshold (recognition)
  - Multi-scale LBP + optical flow + symmetry + attack classification (liveness)
  - PBKDF2-HMAC storage + template clustering (storage)
"""

__version__ = "2.0.0"

from .analytics import AuthEvent, SessionAnalytics, SessionReport
from .engine import AuditEntry, AuthenticationResult, OfflineBiometricEngine
from .image_ops import (
    blur_score,
    brightness_score,
    clahe_equalize,
    gabor_features,
    image_quality_score,
    lpq_features,
    occlusion_ratio,
    synthetic_face,
)
from .liveness import LivenessDetector, LivenessResult
from .ml_fusion import FusionConfig, FusionResult, ScoreFusion
from .recognition import (
    AdaptiveThreshold,
    FaceTemplate,
    LBPHFaceRecognizer,
    OnlinePCAWhitener,
    RecognitionResult,
)
from .storage import TemplateStore

__all__ = [
    "__version__",
    # Engine
    "OfflineBiometricEngine",
    "AuthenticationResult",
    "AuditEntry",
    # Recognition
    "LBPHFaceRecognizer",
    "FaceTemplate",
    "RecognitionResult",
    "AdaptiveThreshold",
    "OnlinePCAWhitener",
    # Liveness
    "LivenessDetector",
    "LivenessResult",
    # Fusion
    "ScoreFusion",
    "FusionConfig",
    "FusionResult",
    # Analytics
    "SessionAnalytics",
    "SessionReport",
    "AuthEvent",
    # Storage
    "TemplateStore",
    # Image ops
    "synthetic_face",
    "gabor_features",
    "lpq_features",
    "clahe_equalize",
    "blur_score",
    "brightness_score",
    "occlusion_ratio",
    "image_quality_score",
]
