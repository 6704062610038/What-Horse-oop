"""หน้าฝั่ง Neural Network (image-based)."""

from __future__ import annotations

from typing import Optional

import streamlit as st
from PIL import Image, UnidentifiedImageError

from config import AppConfig
from core.exceptions import WhatHorseError
from core.predictors import BasePredictor

from .base_page import BasePage, InfoPage
from .router import Router


class NNInfoPage(InfoPage):
    name = "NN_Info"
    title = "📚 Details on Neural Network Development"
    next_page = "NN_Test"
    next_label = "➡️ Go to Neural Network test"

    WHY = """
    ### 1. Why Neural Network?
    To push the boundaries of this project, I implemented a Deep Learning approach using Transfer Learning (MobileNetV2). The goal was to see if a neural network could 'see' the subtle visual differences that define each creature. By training the model on hundreds of images, I aimed to create a system capable of autonomous visual recognition, identifying an Alicorn from a Pegasus with the same visual intuition as a human expert.
    """

    IMAGE_FEATURE_ENGINEERING = """
    ### 2. Image Feature Engineering (NN)
    * **Automated Image Resizing:**
        * **Concept:** I developed a preprocessing pipeline to automatically transform all raw images into a standardized **224x224 pixel** resolution.
        * **Why I use it:** Neural Networks like **MobileNetV2** require a fixed input shape. This automation ensures that every image—regardless of its original size—is perfectly formatted for the model's first layer, preventing spatial distortion during analysis.
    * **Min-Max Pixel Scaling:**
        * **Concept:** I implemented a scaling script to divide all pixel values (0-255) by **255**, effectively normalizing the data into a floating-point range between **0.0 and 1.0**.
        * **Why I use it:** Raw pixels are too large for efficient weight updates. Scaling prevents **Gradient Explosion** and ensures the model converges faster during training, making the mathematical operations within the network more stable and accurate.
    * **Dynamic Image Augmentation:**
        * **Concept:** I utilized a real-time augmentation script to apply random **Rotations, Horizontal Flips, and Zooms** to the training images.
        * **Why I use it:** Since our dataset is specialized (490+ images), this technique "simulates" a much larger dataset. It teaches the model to recognize a **Unicorn** or **Pegasus** from any angle or perspective, significantly improving **Generalization** and reducing **Overfitting**.
    * **Label Encoding for Classes:**
        * **Concept:** I mapped the 5 equine categories (Horse, Zebra, Unicorn, Pegasus, and Alicorn) into a numeric format that the **Softmax** layer can interpret.
        * **Why I use it:** Deep Learning models cannot process text strings. By converting species names into a **Probability Vector**, the model can calculate the specific **Confidence Score** for each category, which is essential for our system's final decision logic.
    """

    THEORY = """
    ### 3. Deep Learning Theory & Architecture (NN)
    * **MobileNetV2 (CNN Architecture):**
        * **Concept:** A specialized **Convolutional Neural Network (CNN)** designed for mobile and web-based efficiency. It uses **Depthwise Separable Convolutions** to process image pixels (edges, shapes, and textures) with significantly fewer parameters than standard CNNs.
        * **Why I use it:** Since I am deploying this on a web interface (Streamlit), I need a model that is **Lightweight and Fast**. MobileNetV2 provides high accuracy for image classification while maintaining low latency, ensuring users get instant predictions without heavy server loads.
    * **Transfer Learning & Freezing Layers:**
        * **Concept:** A technique where a model developed for one task (ImageNet) is reused as the starting point for a second task. **Freezing** means locking the weights of the initial layers so they don't change during our specific training.
        * **Why I use it:** Our dataset is relatively small (approx. 500 images). Training a deep network from scratch would lead to **Overfitting**. By freezing the "knowledge" of 1.4 million images from ImageNet, I retain the ability to recognize basic visual features (like animal legs or fur) and only need to train the final layers to distinguish our 5 specific equine classes.
    * **Softmax Activation Function:**
        * **Concept:** A mathematical function used in the final **Output Layer** that turns raw numerical scores (logits) into a **Probability Distribution** (ranging from 0 to 1).
        * **Why I use it:** I need to know not just "what" the animal is, but "how sure" the model is. Softmax allows us to see the **Confidence Percentage** for all 5 classes (Horse, Zebra, Unicorn, Pegasus, and Alicorn). This is crucial for our **70% Threshold** logic—if the highest Softmax probability is too low, the system can safely say it "doesn't know," preventing false guesses.
    """

    def render_body(self) -> None:
        self._spacer()
        st.markdown(self.WHY)
        st.markdown(self.IMAGE_FEATURE_ENGINEERING)
        st.markdown(self.THEORY)


class NNTestPage(BasePage):
    """หน้าทดสอบโมเดล NN — รับ predictor เข้ามาทาง constructor."""

    name = "NN_Test"
    title = "🧪 Neural Network model testing (Image-based)"

    UNCERTAIN_MESSAGE = (
        "No matching species found. The system is unable to categorize the input "
        "based on the existing Horse, Zebra, Unicorn, Pegasus, and Alicorn datasets."
    )

    def __init__(self, router: Router, config: AppConfig,
                 predictor: BasePredictor) -> None:
        super().__init__(router, config)
        self._predictor = predictor

    def render_body(self) -> None:
        uploaded_file = st.file_uploader(
            "Upload an image to perform a prediction...",
            type=["jpg", "jpeg", "png"],
        )
        self._spacer()

        _, _, center, _, _ = st.columns([1, 1.5, 1.5, 1.5, 1])
        with center:
            image = self._show_preview(uploaded_file)
            if st.button("Analyze Image", use_container_width=True):
                self._run_prediction(image)

    # ---------- ส่วนย่อยของหน้า ----------
    def _show_preview(self, uploaded_file) -> Optional[Image.Image]:
        if uploaded_file is None:
            return None
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except (UnidentifiedImageError, OSError):
            st.error("The image format is incorrect or the file is corrupted.")
            return None
        st.image(image, caption="Uploaded images")
        return image

    def _run_prediction(self, image: Optional[Image.Image]) -> None:
        if image is None:
            st.warning("Please upload an image first.")
            return
        try:
            result = self._predictor.predict(image)
        except WhatHorseError as exc:
            st.error(f"❌ {exc}")
            return
        self._show_result(result, uncertain_message=self.UNCERTAIN_MESSAGE,
                          success_prefix="This is")

    def render_footer(self) -> None:
        self._render_bottom_nav("🏠 Go to Homepage", "Home",
                                "🤖 Go to Machine Learning", "ML_Info")
