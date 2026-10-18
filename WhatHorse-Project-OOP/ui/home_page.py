"""หน้าแรกของเว็บ."""

from __future__ import annotations

import streamlit as st

from .base_page import BasePage


class HomePage(BasePage):
    name = "Home"
    title = "🐴 Horse Family Classification Project"

    SUBTITLE = (
        "This project aims to test how AI models—both feature-based (ML) and "
        "image-based (NN)—can accurately differentiate between these closely "
        "related biological and mythical forms."
    )

    INSPIRATION = """
    **The inspiration for this project** began with a fundamental challenge in Computer Vision: object classification. While classifying distinct objects is a standard task, I was intrigued by the challenge of Fine-Grained Classification—distinguishing between entities with highly similar physical structures.
    """

    SCOPE = """
    **Starting from a base of common equines** like Horses and Zebras, I expanded the scope into the mythical realm. This led to a fascinating progression: if a horse has a horn, it's a Unicorn; if it has wings, it's a Pegasus. But what if it possesses both? This curiosity introduced the Alicorn into our dataset. This project aims to test how AI models—both feature-based (ML) and image-based (NN)—can accurately differentiate between these closely related biological and mythical forms.
    """

    PROCESS = """
    ### Model Development Process
    * **Phase 1: Problem Definition & Scope**
        * **Objective:** To design a system capable of distinguishing between 5 closely related equine species: Horse, Zebra, Unicorn, Pegasus, and Alicorn.
        * **Approach:** Implementing a hybrid solution using Feature-based ML and Image-based Neural Networks.
    * **Phase 2: Data Acquisition & Preprocessing**
        * **Machine Learning Data Generation & Simulation:** I developed a script to extract physical attributes (Horns, Wings, Stripes) into a structured CSV format. To demonstrate robust data handling, I intentionally introduced missing values via a randomization function, simulating real-world incomplete data scenarios, which were then resolved using Mode Imputation.
        * **Neural Network Data Curation:** A specialized image dataset of over 490+ samples was curated. I implemented Image Augmentation techniques (such as rotation and flipping) to diversify the training data and enhance the model's ability to generalize to new images.
        * **Normalization & Encoding:** All inputs were standardized. I applied Pixel Scaling for image data and Label Encoding for categorical targets, ensuring the data was in the optimal format for model consumption.
    * **Phase 3: Model Architecture & Training**
        * **Machine Learning Implementation:** Developed an Ensemble Voting Classifier (Random Forest, XGBoost, and Logistic Regression) to ensure robust logic-based predictions.
        * **Neural Network Implementation:** Leveraged Transfer Learning with the MobileNetV2 architecture, fine-tuning the top layers for our specific 5-class classification task.
    * **Phase 4: Evaluation & Optimization**
        * **Performance Metrics:** Evaluated models based on Accuracy and Confidence Scores.
        * **Confidence Thresholding:** Implemented a 70% Threshold to filter out uncertain predictions (Out-of-Distribution data), significantly reducing false positives.
    * **Phase 5: Web Integration & Deployment**
        * **UI/UX Design:** Built an interactive web application using Streamlit to provide a seamless user experience for both ML and NN testing.
        * **Cloud Deployment:** Uploaded the project to GitHub and deployed via Streamlit Cloud for global accessibility.
    """

    REFERENCES = """
    ### References
    * **Some horses and zebra Image :** Curated from [Donkeys, Horses & Zebra Images Dataset](https://www.kaggle.com/datasets/ifeanyinneji/donkeys-horses-zebra-images-dataset?resource=download).
    * **My little pony characters :** Curated from [My little pony character](https://mlp.fandom.com/wiki/Characters#Main_characters).
    * **Other Data Sourcing:** Supplementary images were curated via [Google Images](https://images.google.com) under Creative Commons licenses, specifically searching for diverse equine variations.
    """

    def render_body(self) -> None:
        st.subheader(self.SUBTITLE)
        st.markdown(self.INSPIRATION)
        st.markdown(self.SCOPE)
        self._spacer()
        st.markdown(self.PROCESS)
        self._spacer(2)

        self._render_model_cards()

        self._spacer(2)
        st.markdown(self.REFERENCES)

    def _render_model_cards(self) -> None:
        """การ์ด 2 ใบสำหรับเข้าหน้า ML และ NN."""
        col_ml, col_nn = st.columns([2, 2])

        with col_ml:
            st.info("### 🤖  Machine Learning (feature-based)")
            _, center, _ = st.columns([0.6, 1.5, 0.2])
            self._nav_button("Go to Machine Learning model", "ML_Info",
                             container=center, use_container_width=False)

        with col_nn:
            st.success("### 🧠  Neural Network (image-based)")
            _, center, _ = st.columns([0.6, 1.5, 0.2])
            self._nav_button("Go to Neural Network model", "NN_Info",
                             container=center, use_container_width=False)
