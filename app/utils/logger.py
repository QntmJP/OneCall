"""
@Author: QntmJP
@Desc: 日志配置模块
"""

"""日志配置模块

使用 Loguru 实现结构化日志，自动按天轮转写入文件
"""

import sys
from pathlib import Path
from loguru import logger

def setup_logger():
    """配置日志系统

    - 控制台输出：彩色格式，方便开发调试
    - 文件输出：按天轮转，保留 30 天，UTF-8 编码
    """
    # 先移除默认配置（否则会重复输出）
    logger.remove()

    # 1. 控制台输出（开发时看终端）
    logger.add(
        sys.stdout,
        level="DEBUG",              # 控制台显示 DEBUG 及以上级别
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True,
    )

    # 2. 文件输出（持久化，按天轮转）
    # 确保日志目录存在
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",  # 文件名含日期
        level="INFO",                # 文件记录 INFO 及以上
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="00:00",            # 每天零点轮转（生成新文件）
        retention="30 days",         # 保留 30 天
        encoding="utf-8",
    )

    return logger

# 模块加载时就配置好，其他文件 import logger 即可直接用
logger = setup_logger()