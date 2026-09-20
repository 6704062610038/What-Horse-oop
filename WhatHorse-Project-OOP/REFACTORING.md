# สรุปการรีแฟกเตอร์เป็น OOP — What Horse Project

## โครงสร้างไฟล์

**ก่อน**
```
app.py            268 บรรทัด — ทุกอย่างอยู่ไฟล์เดียว (if/elif สลับหน้า)
```

**หลัง**
```
app.py                    WhatHorseApp — ประกอบร่าง + รันแอป
config.py                 AppConfig — ค่าคงที่ทั้งหมด
core/
  exceptions.py           WhatHorseError และลูก ๆ
  prediction.py           HorseFeatures, PredictionResult
  model_registry.py       ModelRegistry — โหลดโมเดลพร้อมแคช
  predictors.py           BasePredictor (ABC) → FeaturePredictor, ImagePredictor
ui/
  router.py               Router — ทะเบียนหน้า + นำทาง
  base_page.py            BasePage (ABC), InfoPage
  home_page.py            HomePage
  ml_pages.py             MLInfoPage, MLTestPage
  nn_pages.py             NNInfoPage, NNTestPage
```

## คลาสทั้งหมดและหลักการ OOP ที่ใช้

| คลาส | หน้าที่ | หลักการ |
|---|---|---|
| `AppConfig` | เก็บ path / threshold / ชื่อคลาส | Encapsulation, immutability |
| `HorseFeatures` | ห่อ input 3 ฟีเจอร์ | Value object |
| `PredictionResult` | ผลทำนาย + `is_confident` | Encapsulation (ข้อมูล + ตรรกะอยู่ด้วยกัน) |
| `ModelRegistry` | โหลดและแคชไฟล์โมเดล | Single Responsibility |
| `BasePredictor` | สัญญาร่วมของตัวทำนาย | Abstraction (ABC), Template Method |
| `FeaturePredictor` / `ImagePredictor` | โมเดล ML / NN | Inheritance, Polymorphism |
| `BasePage` | โครงหน้าเว็บ + helper ที่ใช้ซ้ำ | Abstraction, DRY |
| `InfoPage` | หน้าเนื้อหาที่จบด้วยปุ่มไปหน้าทดสอบ | Inheritance ชั้นที่ 2 |
| `HomePage`, `MLInfoPage`, `MLTestPage`, `NNInfoPage`, `NNTestPage` | 5 หน้าของเว็บ | Polymorphism |
| `Router` | จัดการ session state + สลับหน้า | Composition |
| `WhatHorseApp` | ประกอบทุกอย่างเข้าด้วยกัน | Composition, Dependency Injection |

## บั๊กที่แก้ไปพร้อมกัน

1. **ต้องกดปุ่ม 2 ครั้งหน้าถึงเปลี่ยน** — เดิม `navigate_to()` แค่ตั้งค่า session state
   โดยไม่ rerun ตอนนี้ `Router.go_to()` เรียก `st.rerun()` ให้
2. **`get_ml_model()` ประกาศไว้แต่ไม่ถูกใช้** — หน้า ML Test เปิดไฟล์ `.pkl` ใหม่ทุกครั้ง
   ที่กด Predict ตอนนี้โหลดผ่าน `ModelRegistry` ที่แคชด้วย `st.cache_resource`
3. **`NameError` ตอนกด Analyze Image โดยยังไม่อัปโหลดรูป** — เดิม `img` ยังไม่ถูกสร้าง
   แล้วถูก `except:` เปล่า ๆ จับไว้ ทำให้ขึ้นข้อความว่า "ไม่พบไฟล์โมเดล" ซึ่งผิด
   ตอนนี้ตรวจก่อนและแจ้งว่า "Please upload an image first."
4. **`except:` เปล่า** — เปลี่ยนเป็นจับ `WhatHorseError` เฉพาะเจาะจง
5. **import ซ้ำซ้อน** — `import joblib` 2 ครั้ง, `import tensorflow` 2 ครั้ง,
   `load_model = tf.keras.models.load_model` ที่ไม่ถูกใช้, `CLASSES` ประกาศ 2 รอบ
6. **TensorFlow ถูก import ตั้งแต่เปิดแอป** — ย้ายไป import ตอนใช้จริง หน้าแรกจึงโหลดเร็วขึ้น
7. **โค้ดแถบปุ่มท้ายหน้าซ้ำกัน** — รวมเป็น `BasePage._render_bottom_nav()`
8. **ตรรกะแสดงผลซ้ำกัน 2 ที่** — รวมเป็น `BasePage._show_result()`
9. **`model.predict()` ของ Keras พ่น progress bar** — เพิ่ม `verbose=0`

## เนื้อหาที่ "ไม่ได้" เปลี่ยน

ข้อความบนหน้าเว็บ, ลำดับปุ่ม, สัดส่วนคอลัมน์, ค่า threshold (ML 0.5 / NN 0.7),
ขนาดรูป 224x224 และการ preprocess ด้วย `mobilenet_v2.preprocess_input`
ทั้งหมดเหมือนเดิมทุกประการ

## วิธีรัน

```bash
streamlit run app.py
```
โครงสร้างไฟล์ประกอบ (`requirements.txt`, `.devcontainer`) ใช้ได้เหมือนเดิม
ไม่ต้องแก้ เพราะ entry point ยังชื่อ `app.py`
