"""Router: จัดการ session state และสลับหน้า."""

from __future__ import annotations

from typing import Dict

import streamlit as st

from .base_page import BasePage


class Router:
    """เก็บทะเบียนหน้าเว็บ และตัดสินใจว่าจะ render หน้าไหน."""

    STATE_KEY = "page"

    def __init__(self, default_page: str = "Home") -> None:
        self._pages: Dict[str, BasePage] = {}
        self._default_page = default_page
        if self.STATE_KEY not in st.session_state:
            st.session_state[self.STATE_KEY] = default_page

    # ---------- ทะเบียนหน้า ----------
    def register(self, page: BasePage) -> "Router":
        """เพิ่มหน้าเข้าทะเบียน (คืนค่า self เพื่อให้ต่อ method chaining ได้)."""
        self._pages[page.name] = page
        return self

    @property
    def current_page_name(self) -> str:
        return st.session_state.get(self.STATE_KEY, self._default_page)

    # ---------- การนำทาง ----------
    def go_to(self, page_name: str) -> None:
        if page_name not in self._pages:
            raise KeyError(f"Unknown page: {page_name}")
        st.session_state[self.STATE_KEY] = page_name
        st.rerun()

    # ---------- วาดหน้า ----------
    def render(self) -> None:
        page = self._pages.get(self.current_page_name)
        if page is None:
            st.session_state[self.STATE_KEY] = self._default_page
            page = self._pages[self._default_page]
        page.render()