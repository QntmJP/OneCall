"""API 路由模块"""

from .health import router as health_router
from .chat import router as chat_router
from .aiops import router as aiops_router
from .file import router as file_router

__all__ = [
    "health_router",
    "chat_router",
    "aiops_router",
    "file_router",
]