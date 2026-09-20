"""Domain layer: โมเดล, ข้อมูล และข้อผิดพลาด (ไม่ยุ่งกับ UI)."""

from .exceptions import (
    InvalidInputError,
    ModelLoadError,
    ModelNotFoundError,
    PredictionError,
    WhatHorseError,
)
from .model_registry import ModelRegistry
from .prediction import HorseFeatures, PredictionResult
from .predictors import BasePredictor, FeaturePredictor, ImagePredictor

__all__ = [
    "WhatHorseError",
    "ModelNotFoundError",
    "ModelLoadError",
    "InvalidInputError",
    "PredictionError",
    "ModelRegistry",
    "HorseFeatures",
    "PredictionResult",
    "BasePredictor",
    "FeaturePredictor",
    "ImagePredictor",
]
