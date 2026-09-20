"""Value objects: ข้อมูลเข้าและผลลัพธ์การทำนาย.

โค้ดเดิมส่งค่าดิบ ๆ (tuple ของ 0/1, float) ไปมาระหว่างส่วนต่าง ๆ
เวอร์ชันนี้ห่อไว้เป็นคลาสที่มีความหมายชัดเจน และย้าย "ตรรกะการตัดสินใจ"
(เช่น ความมั่นใจถึงเกณฑ์หรือไม่) มาไว้กับข้อมูลโดยตรง (Encapsulation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class HorseFeatures:
    """คุณลักษณะทางกายภาพที่ผู้ใช้เลือกในหน้า ML Test."""

    has_horn: int = 0
    has_wings: int = 0
    is_striped: int = 0

    def as_row(self) -> Dict[str, int]:
        return {
            "has_horn": int(self.has_horn),
            "has_wings": int(self.has_wings),
            "is_striped": int(self.is_striped),
        }

    def as_tuple(self) -> Tuple[int, int, int]:
        return (int(self.has_horn), int(self.has_wings), int(self.is_striped))


@dataclass(frozen=True)
class PredictionResult:
    """ผลการทำนายหนึ่งครั้ง พร้อมตรรกะตัดสินว่า 'มั่นใจพอ' หรือไม่."""

    label: str
    confidence: float
    threshold: float

    @property
    def is_confident(self) -> bool:
        """True เมื่อค่าความมั่นใจผ่านเกณฑ์ (Confidence Thresholding)."""
        return self.confidence >= self.threshold

    @property
    def confidence_text(self) -> str:
        return f"{self.confidence:.2%}"

    def __str__(self) -> str:  # ใช้ตอน debug / log
        status = "confident" if self.is_confident else "uncertain"
        return f"<PredictionResult {self.label} {self.confidence_text} ({status})>"
