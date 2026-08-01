"""
@Author: QntmJP
@Desc: 配置管理模块
"""

"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",           # 从项目根目录的 .env 文件读取
        env_file_encoding="utf-8", # 文件编码
        case_sensitive=False,      # 环境变量名不区分大小写
        extra="ignore",            # .env 里多余的字段忽略，不报错
    )

    # ---- 应用配置 ----
    app_name: str = "MyOnCallAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # ---- DashScope（阿里云大模型）配置 ----
    dashscope_api_key: str = ""                    # 默认空，实际从 .env 读
    dashscope_model: str = "qwen-max"              # 对话/诊断用的模型
    dashscope_embedding_model: str = "text-embedding-v4"  # 向量化模型

    # ---- Milvus（向量数据库）配置 ----
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # ---- RAG 配置 ----
    rag_top_k: int = 3        # 检索返回前 3 个最相关文档
    rag_model: str = "qwen-max"

    # ---- 文档分块配置 ----
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # ---- MCP 服务配置 ----
    # transport: stdio | sse | streamable-http
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    # ---- Prometheus（告警查询用）----
    prometheus_base_url: str = "http://127.0.0.1:9090"
    prometheus_request_timeout: float = 10.0

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置（动态拼装成字典）"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        }

# 全局配置实例（整个项目共享这一个）
config = Settings()