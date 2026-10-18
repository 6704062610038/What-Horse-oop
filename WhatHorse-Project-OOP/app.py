"""What Horse Project (Streamlit entry point).
ไฟล์นี้ทำหน้าที่ประกอบร่างอ็อบเจ็กต์ต่าง ๆ เข้าด้วยกัน แล้วสั่งให้แอปทำงาน
ตรรกะจริงอยู่ในแพ็กเกจ core (โมเดล) และ ui (หน้าเว็บ)"""

from __future__ import annotations

import streamlit as st

from config import CONFIG, AppConfig
from core.predictors import FeaturePredictor, ImagePredictor
from ui.home_page import HomePage
from ui.ml_pages import MLInfoPage, MLTestPage
from ui.nn_pages import NNInfoPage, NNTestPage
from ui.router import Router


class WhatHorseApp:

    def __init__(self, config: AppConfig = CONFIG) -> None:
        self._config = config
        self._configure_page()
        self._feature_predictor = self._build_feature_predictor()
        self._image_predictor = self._build_image_predictor()
        self._router = self._build_router()

    # ---------- ขั้นตอนประกอบร่าง ----------
    def _configure_page(self) -> None:
        st.set_page_config(page_title=self._config.page_title,
                           layout=self._config.layout)

    def _build_feature_predictor(self) -> FeaturePredictor:
        return FeaturePredictor(
            model_path=self._config.feature_model_path,
            threshold=self._config.feature_threshold,
            feature_columns=self._config.feature_columns,
        )

    def _build_image_predictor(self) -> ImagePredictor:
        return ImagePredictor(
            model_path=self._config.image_model_path,
            threshold=self._config.image_threshold,
            classes=self._config.classes,
            image_size=self._config.image_size,
        )

    def _build_router(self) -> Router:
        router = Router(default_page=HomePage.name)
        router.register(HomePage(router, self._config))
        router.register(MLInfoPage(router, self._config))
        router.register(MLTestPage(router, self._config, self._feature_predictor))
        router.register(NNInfoPage(router, self._config))
        router.register(NNTestPage(router, self._config, self._image_predictor))
        return router

    # ---------- รันแอป ----------
    def run(self) -> None:
        self._router.render()


def main() -> None:
    WhatHorseApp().run()


if __name__ == "__main__":
    main()
