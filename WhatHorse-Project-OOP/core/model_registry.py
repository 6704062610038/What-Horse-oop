"""ตัวโหลดโมเดลพร้อมแคช."""

from __future__ import annotations

from pathlib import Path

import joblib
import streamlit as st

from .exceptions import ModelLoadError, ModelNotFoundError


class ModelRegistry:
    """Utility class สำหรับโหลดไฟล์โมเดลแต่ละชนิด."""

    @staticmethod
    @st.cache_resource(show_spinner="Loading feature model...")
    def _load_sklearn(path_str: str):
        with open(path_str, "rb") as f:
            return joblib.load(f)

    @staticmethod
    @st.cache_resource(show_spinner="Loading image model...")
    def _load_keras(path_str: str):
        import tensorflow as tf  # import ตอนใช้จริง เพื่อให้แอปเปิดเร็วขึ้น

        return tf.keras.models.load_model(path_str)

    @classmethod
    def load_sklearn(cls, path: Path):
        cls._ensure_exists(path)
        try:
            return cls._load_sklearn(str(path))
        except Exception as exc:  # noqa: BLE001 - แปลงเป็น error ของโดเมน
            raise ModelLoadError(f"Cannot load feature model: {exc}") from exc

    @classmethod
    def load_keras(cls, path: Path):
        cls._ensure_exists(path)
        try:
            return cls._load_keras(str(path))
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"Cannot load image model: {exc}") from exc

    @staticmethod
    def _ensure_exists(path: Path) -> None:
        if not Path(path).exists():
            raise ModelNotFoundError(f"The system could not find the file at {path}")
