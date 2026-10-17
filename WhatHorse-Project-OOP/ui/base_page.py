"""BasePage: abstract class ของทุกหน้าในเว็บ + ส่วนประกอบ UI ที่ใช้ซ้ำ."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import streamlit as st

from config import AppConfig
from core.prediction import PredictionResult

if TYPE_CHECKING:
    from .router import Router


class BasePage(ABC):

    #: คีย์ที่ใช้ใน session state (ต้องไม่ซ้ำกัน)
    name: str = "Page"
    title: str = ""

    def __init__(self, router: "Router", config: AppConfig) -> None:
        self._router = router
        self._config = config

    # ---------- API หลัก ----------
    @abstractmethod
    def render_body(self) -> None:
        """เนื้อหาเฉพาะของแต่ละหน้า."""

    def render(self) -> None:
        if self.title:
            st.title(self.title)
        self.render_body()
        self.render_footer()

    def render_footer(self) -> None:

    # ---------- Helper ที่ทุกหน้าใช้ร่วมกัน ----------
    def _spacer(self, times: int = 1) -> None:
        for _ in range(times):
            st.markdown(""" """)

    def _nav_button(self, label: str, target: str, *, container=None,
                    use_container_width: bool = True) -> None:
        """ปุ่มนำทางหนึ่งปุ่ม (รวมโค้ดที่เดิมเขียนซ้ำหลายที่)"""
        holder = container if container is not None else st
        if holder.button(label, key=f"{self.name}->{target}",
                         use_container_width=use_container_width):
            self._router.go_to(target)

    def _render_bottom_nav(self, left_label: str, left_target: str,
                           right_label: str, right_target: str) -> None:
        """แถบปุ่ม 2 ปุ่มท้ายหน้า"""
        st.divider()
        _, col_left, _, col_right, _ = st.columns([1, 0.8, 0.2, 0.8, 1])
        self._nav_button(left_label, left_target, container=col_left)
        self._nav_button(right_label, right_target, container=col_right)

    def _show_result(self, result: PredictionResult,
                     uncertain_message: str,
                     success_prefix: str = "The result is") -> None:
        """แสดงผลการทำนายตามเกณฑ์ความมั่นใจ — ใช้ร่วมกันทั้งหน้า ML และ NN."""
        if result.is_confident:
            st.balloons()
            st.success(f"{success_prefix}: **{result.label}** "
                       f"(accuracy: {result.confidence_text})")
        else:
            st.warning(f"⚠️ Low accuracy level: {result.confidence_text}")
            st.error(uncertain_message)


class InfoPage(BasePage):

    #: หน้าถัดไปที่ปุ่มกลางจะพาไป
    next_page: Optional[str] = None
    next_label: str = "➡️ Continue"

    def render_footer(self) -> None:
        if not self.next_page:
            return
        self._spacer(2)
        _, center, _ = st.columns([1, 2, 1])
        self._nav_button(self.next_label, self.next_page, container=center)
