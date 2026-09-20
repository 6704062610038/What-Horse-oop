"""ชั้นโมเดล (Model layer) — หัวใจของการทำ OOP ในโปรเจกต์นี้.

* ``BasePredictor``  : abstract base class กำหนด "สัญญา" ร่วมของตัวทำนายทุกตัว
                       และคุม flow ด้วย Template Method ``predict()``
* ``FeaturePredictor``: ทำนายจากคุณลักษณะ (Ensemble Voting Classifier)
* ``ImagePredictor``  : ทำนายจากรูปภาพ (MobileNetV2)

เพราะทั้งสองคลาสสืบทอด interface เดียวกัน หน้าเว็บจึงเรียก ``predictor.predict(x)``
ได้เหมือนกันหมดโดยไม่ต้องรู้ว่าเบื้องหลังเป็น scikit-learn หรือ Keras (Polymorphism)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .exceptions import InvalidInputError, PredictionError
from .model_registry import ModelRegistry
from .prediction import HorseFeatures, PredictionResult


class BasePredictor(ABC):
    """Abstract base class ของตัวทำนายทุกชนิด."""

    #: ชื่อที่แสดงบนหน้าจอ (subclass เขียนทับ)
    display_name: str = "Predictor"

    def __init__(self, model_path: Path, threshold: float) -> None:
        self._model_path = Path(model_path)
        self._threshold = float(threshold)

    # ---------- Properties (Encapsulation) ----------
    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def is_available(self) -> bool:
        """ไฟล์โมเดลมีอยู่จริงหรือไม่."""
        return self._model_path.exists()

    # ---------- Template Method ----------
    def predict(self, raw_input: Any) -> PredictionResult:
        """ขั้นตอนมาตรฐาน: โหลด → เตรียมข้อมูล → ทำนาย → ห่อเป็น PredictionResult."""
        self._validate(raw_input)
        model = self._load_model()
        prepared = self._preprocess(raw_input)
        try:
            label, confidence = self._infer(model, prepared)
        except Exception as exc:  # noqa: BLE001
            raise PredictionError(str(exc)) from exc
        return PredictionResult(label=label, confidence=float(confidence),
                                threshold=self._threshold)

    # ---------- Hooks ที่ subclass ต้องทำ ----------
    @abstractmethod
    def _load_model(self) -> Any:
        ...

    @abstractmethod
    def _preprocess(self, raw_input: Any) -> Any:
        ...

    @abstractmethod
    def _infer(self, model: Any, prepared: Any) -> Tuple[str, float]:
        ...

    # ---------- Hook ที่ override ได้ (default: ผ่านตลอด) ----------
    def _validate(self, raw_input: Any) -> None:
        if raw_input is None:
            raise InvalidInputError("No input was provided.")

    def __repr__(self) -> str:
        return f"<{type(self).__name__} path={self._model_path.name} thr={self._threshold}>"


class FeaturePredictor(BasePredictor):
    """ทำนายสายพันธุ์จากคุณลักษณะ 3 อย่าง (has_horn / has_wings / is_striped)."""

    display_name = "Machine Learning (feature-based)"

    def __init__(self, model_path: Path, threshold: float,
                 feature_columns: Sequence[str]) -> None:
        super().__init__(model_path, threshold)
        self._feature_columns = list(feature_columns)

    def _load_model(self):
        return ModelRegistry.load_sklearn(self._model_path)

    def _preprocess(self, raw_input: HorseFeatures) -> pd.DataFrame:
        if not isinstance(raw_input, HorseFeatures):
            raise InvalidInputError("FeaturePredictor expects a HorseFeatures object.")
        return pd.DataFrame([raw_input.as_tuple()], columns=self._feature_columns)

    def _infer(self, model, prepared: pd.DataFrame) -> Tuple[str, float]:
        probabilities = model.predict_proba(prepared)
        label = str(model.predict(prepared)[0])
        return label, float(np.max(probabilities))


class ImagePredictor(BasePredictor):
    """ทำนายสายพันธุ์จากรูปภาพด้วย MobileNetV2 (Transfer Learning)."""

    display_name = "Neural Network (image-based)"

    def __init__(self, model_path: Path, threshold: float,
                 classes: Sequence[str], image_size: Tuple[int, int]) -> None:
        super().__init__(model_path, threshold)
        self._classes = list(classes)
        self._image_size = tuple(image_size)

    @property
    def classes(self):
        return tuple(self._classes)

    def _validate(self, raw_input: Any) -> None:
        if not isinstance(raw_input, Image.Image):
            raise InvalidInputError("Please upload an image before analyzing.")

    def _preprocess(self, raw_input: Image.Image) -> np.ndarray:
        import tensorflow as tf

        image = raw_input.convert("RGB").resize(self._image_size)
        array = np.expand_dims(np.array(image), axis=0)
        return tf.keras.applications.mobilenet_v2.preprocess_input(array)

    def _load_model(self):
        return ModelRegistry.load_keras(self._model_path)

    def _infer(self, model, prepared: np.ndarray) -> Tuple[str, float]:
        predictions = model.predict(prepared, verbose=0)
        class_index = int(np.argmax(predictions))
        return self._classes[class_index], float(np.max(predictions))
