"""Central configuration for the What Horse Project.

ทุกค่าคงที่ (path, threshold, ชื่อคลาส) ถูกรวมไว้ในอ็อบเจ็กต์เดียว
แทนที่จะกระจายเป็นตัวแปร global ทั่วไฟล์เหมือนเวอร์ชันเดิม
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


BASE_DIR = Path(__file__).parent.resolve()


@dataclass(frozen=True)
class AppConfig:
    """Immutable settings shared by every object in the app."""

    # --- General ---
    page_title: str = "Horse Family AI Project"
    layout: str = "wide"

    # --- Model files ---
    base_dir: Path = BASE_DIR
    feature_model_file: str = "horse_feature_model.pkl"
    image_model_file: str = "horse_image_model.h5"

    # --- Domain ---
    classes: Tuple[str, ...] = ("Alicorn", "Horse", "Pegasus", "Unicorn", "Zebra")
    feature_columns: Tuple[str, ...] = ("has_horn", "has_wings", "is_striped")

    # --- Decision logic ---
    feature_threshold: float = 0.50
    image_threshold: float = 0.70
    image_size: Tuple[int, int] = (224, 224)

    @property
    def feature_model_path(self) -> Path:
        return self.base_dir / self.feature_model_file

    @property
    def image_model_path(self) -> Path:
        return self.base_dir / self.image_model_file


#: อ็อบเจ็กต์ config ที่ใช้ร่วมกันทั้งแอป
CONFIG = AppConfig()
