"""Exception hierarchy ของโปรเจกต์ """


class WhatHorseError(Exception):
    """Base class ของข้อผิดพลาดทุกชนิดในแอปนี้."""


class ModelNotFoundError(WhatHorseError):
    """ไม่พบไฟล์โมเดลบนดิสก์."""


class ModelLoadError(WhatHorseError):
    """พบไฟล์โมเดล แต่โหลดไม่สำเร็จ."""


class InvalidInputError(WhatHorseError):
    """ข้อมูลที่ผู้ใช้ส่งเข้ามาไม่ถูกต้อง เช่น ยังไม่ได้อัปโหลดรูป."""


class PredictionError(WhatHorseError):
    """เกิดข้อผิดพลาดระหว่างการทำนาย."""
