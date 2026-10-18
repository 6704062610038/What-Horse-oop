"""หน้าฝั่ง Machine Learning (feature-based)."""

from __future__ import annotations

import streamlit as st

from config import AppConfig
from core.exceptions import WhatHorseError
from core.predictors import BasePredictor
from core.prediction import HorseFeatures

from .base_page import BasePage, InfoPage
from .router import Router


class MLInfoPage(InfoPage):
    name = "ML_Info"
    title = "📚 Details on Machine Learning Development"
    next_page = "ML_Test"
    next_label = "➡️ Go to Machine Learning test"

    WHY = """
    ### 1. Why Machine Learning?
    While images provide a rich data source, I wanted to explore a logic-based approach. By extracting specific physical attributes—such as the presence of horns, wings, or stripes I developed an Ensemble Learning model. This allows for a lightweight and interpretable system that can classify species based on descriptive data rather than pixels, mimicking how a taxonomist might categorize a new discovery.
    """

    FEATURE_ENGINEERING = """
    ### 2. Feature Engineering
    * **Automated Dataset Construction:** I developed a specialized script to automate the extraction of key physical attributes, including has_horn, has_wings, and is_striped, accurately mapping them to each of the 5 equine categories.
    * **Missing Data Simulation:** To demonstrate advanced data handling, I intentionally introduced randomized missing values (NaN) within the feature matrix. This process was designed to simulate the challenges of real-world incomplete datasets.
    * **Data Integrity & Cleaning:** I resolved these simulated gaps using Mode Imputation, replacing null values with the most frequent data points to maintain consistent logic and ensure high-quality input for the models.
    * **Data Serialization:** The final cleaned feature set was structured and exported in CSV format, serving as the primary source of truth for our Machine Learning ensemble.
    """

    ENSEMBLE = """
    ### 3. Ensemble Learning Approach
    For this project, I implemented an **Ensemble Learning** technique specifically using a **Voting Classifier (Soft Voting)**. This approach acts like a "committee of experts," where three distinct models collaborate to provide the most accurate prediction by leveraging their individual strengths.

    * **Random Forest:** 
        * **Concept:** An ensemble of many **Decision Trees** operating as a committee. It uses a technique called **Bagging** (Bootstrap Aggregating).
        * **Why I use it:** Since our equine dataset relies on specific binary features (e.g., *has_horn*, *has_wings*), Random Forest is excellent for **reducing variance**. By training multiple trees on different subsets of data and averaging their results, it prevents **Overfitting** and ensures the model doesn't get confused by "noise" or outliers in the data.
    * **XGBoost:**
        * **Concept:** A powerful implementation of **Gradient Boosting** that builds trees sequentially. Each new tree focuses on **correcting the errors** made by the previous ones.
        * **Why I use it:** For complex mythological creatures like the **Alicorn** (which shares features with both Unicorns and Pegasi), XGBoost excels at capturing subtle patterns. It "squeezes" out maximum performance and optimization, making it highly effective at distinguishing between classes that have overlapping physical traits.
    * **Logistic Regression:**
        * **Concept:** Despite its name, this is a fundamental **Classification** algorithm that uses a Sigmoid function to establish a "Decision Boundary" between classes.
        * **Why I use it:** I included Logistic Regression as a **Stability Anchor**. While tree-based models (like RF and XGB) are great for non-linear patterns, they can sometimes become overly complex. Logistic Regression provides a simple, linear perspective that balances the ensemble, ensuring the overall prediction remains grounded and reliable.
    """

    def render_body(self) -> None:
        self._spacer()
        st.markdown(self.WHY)
        st.markdown(self.FEATURE_ENGINEERING)
        st.markdown(self.ENSEMBLE)


class MLTestPage(BasePage):
    """หน้าทดสอบโมเดล ML — รับ predictor เข้ามาทาง constructor (Dependency Injection)."""

    name = "ML_Test"
    title = "🧪 Machine Learning model testing (Feature-based)"

    def __init__(self, router: Router, config: AppConfig,
                 predictor: BasePredictor) -> None:
        super().__init__(router, config)
        self._predictor = predictor

    def render_body(self) -> None:
        self._spacer()
        features = self._collect_features()
        self._spacer()

        _, center, _ = st.columns([1.5, 2, 1.5])
        with center:
            if st.button("Predict", use_container_width=True):
                self._run_prediction(features)

    # ---------- ส่วนย่อยของหน้า ----------
    def _collect_features(self) -> HorseFeatures:
        yes_no = lambda value: "Yes" if value == 1 else "No"  # noqa: E731
        col1, col2, col3 = st.columns(3)
        with col1:
            horn = st.selectbox("Has horn ?", [0, 1], format_func=yes_no)
        with col2:
            wings = st.selectbox("Has wings ?", [0, 1], format_func=yes_no)
        with col3:
            stripes = st.selectbox("Is striped ?", [0, 1], format_func=yes_no)
        return HorseFeatures(has_horn=horn, has_wings=wings, is_striped=stripes)

    def _run_prediction(self, features: HorseFeatures) -> None:
        try:
            result = self._predictor.predict(features)
        except WhatHorseError as exc:  # จับเฉพาะ error ของโดเมน
            st.error(f"❌ {exc}")
            return
        self._show_result(result, uncertain_message="Unknown species in our dataset.")

    def render_footer(self) -> None:
        self._render_bottom_nav("🏠 Go to Homepage", "Home",
                                "🧠 Go to Neural Network", "NN_Info")
