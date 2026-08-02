"""
@Author: QntmJP
@Desc: 向量嵌入服务模块
"""

"""向量嵌入服务模块 - 基于 LangChain Embeddings 标准接口

使用阿里云 DashScope text-embedding-v4 模型，将文本转换为 1024 维向量
"""

from typing import List

from langchain_core.embeddings import Embeddings
from openai import OpenAI
from loguru import logger

from app.config import config

class DashScopeEmbeddings(Embeddings):
    """阿里云 DashScope Text Embedding（OpenAI 兼容模式）

    实现 LangChain 标准 Embeddings 接口：
    - embed_documents(texts) → 批量嵌入文档（入库时用）
    - embed_query(text) → 嵌入单个查询（检索时用）

    为什么继承 Embeddings？
    因为 LangChain 的 Milvus VectorStore 要求传入一个 Embeddings 对象，
    它会自动调用这两个方法完成向量化。就像 Java 里实现标准接口，
    框架就能自动调用你的实现。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
    ):
        """
        初始化 DashScope Embeddings

        Args:
            api_key: DashScope API Key
            model: 嵌入模型名称（默认 text-embedding-v4）
            dimensions: 向量维度（默认 1024）
        """
        # 校验 API Key 不能为空或占位符
        if not api_key or api_key == "your-api-key-here":
            raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

        # 使用 OpenAI SDK 调用 DashScope（因为 DashScope 兼容 OpenAI 接口）
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        self.dimensions = dimensions

        # 打印初始化信息（Key 做脱敏处理，安全打印）
        masked_key = self._mask_api_key(api_key)
        logger.info(
            f"DashScope Embeddings 初始化完成 - "
            f"模型: {model}, 维度: {dimensions}, API Key: {masked_key}"
        )

    @staticmethod
    def _mask_api_key(api_key: str) -> str:
        """掩码 API Key，只显示前8位和后4位，用于安全日志"""
        if len(api_key) > 8:
            return f"{api_key[:8]}...{api_key[-4:]}"
        return "***"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档列表（LangChain 标准接口）

        用途：文档入库时，把多个文本块一次性转成向量

        Args:
            texts: 文本列表

        Returns:
            向量列表，每个向量是 1024 个浮点数
        """
        if not texts:
            return []

        try:
            logger.info(f"批量嵌入 {len(texts)} 个文档")

            # 调用 DashScope embeddings 接口（OpenAI 兼容格式）
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
                encoding_format="float"
            )

            # 提取向量
            embeddings = [item.embedding for item in response.data]
            logger.debug(f"批量嵌入完成, 维度: {len(embeddings[0])}")

            return embeddings

        except Exception as e:
            logger.error(f"批量嵌入失败: {e}")
            raise RuntimeError(f"批量嵌入失败: {e}") from e

    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询文本（LangChain 标准接口）

        用途：用户提问时，把问题转成向量，再去 Milvus 里找相似文档

        Args:
            text: 查询文本

        Returns:
            单个向量（1024 个浮点数）
        """
        if not text or not text.strip():
            raise ValueError("查询文本不能为空")

        try:
            logger.debug(f"嵌入查询, 长度: {len(text)} 字符")

            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
                encoding_format="float"
            )

            embedding = response.data[0].embedding
            logger.debug(f"查询嵌入完成, 维度: {len(embedding)}")

            return embedding

        except Exception as e:
            logger.error(f"查询嵌入失败: {e}")
            raise RuntimeError(f"查询嵌入失败: {e}") from e

# 全局单例
vector_embedding_service = DashScopeEmbeddings(
    api_key=config.dashscope_api_key,
    model=config.dashscope_embedding_model,
    dimensions=1024
)