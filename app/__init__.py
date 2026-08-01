"""
@Author: QntmJP
@Desc: OneCall，基于 LangChain 的智能业务代理系统
"""

"""应用包初始化

模块导入时自动加载日志配置，确保所有子模块都能使用 loguru
"""

from app.utils.logger import logger

# 导出 logger 供其他模块使用
__all__ = ["logger"]