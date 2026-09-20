"""Presentation layer: หน้าเว็บทั้งหมดและตัวจัดการนำทาง."""

from .base_page import BasePage, InfoPage
from .home_page import HomePage
from .ml_pages import MLInfoPage, MLTestPage
from .nn_pages import NNInfoPage, NNTestPage
from .router import Router

__all__ = [
    "BasePage",
    "InfoPage",
    "HomePage",
    "MLInfoPage",
    "MLTestPage",
    "NNInfoPage",
    "NNTestPage",
    "Router",
]
