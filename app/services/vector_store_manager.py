"""
@Author: QntmJP
@Desc: 向量存储管理器
"""

"""向量存储管理器 - 封装 Milvus VectorStore 操作

职责：
  - 封装 LangChain Milvus VectorStore 的增删查操作
  - add_documents(): 批量添加文档（自动向量化）
  - delete_by_source(): 按文件路径删除旧数据
  - similarity_search(): 相似度搜索（RAG 检索时用）
"""

import time
import uuid
from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service

# 统一使用 biz collection（和 milvus_client.py 中的 COLLECTION_NAME 一致）
COLLECTION_NAME = "biz"

class VectorStoreManager:
    """向量存储管理器

    封装 LangChain 的 Milvus VectorStore，提供文档的增删查接口。
    LangChain Milvus 会自动调用 embedding_function 将文本转为向量。
    """

    def __init__(self):
        """初始化向量存储管理器"""
        self.vector_store = None
        self.collection_name = COLLECTION_NAME
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """初始化 Milvus VectorStore

        创建 LangChain Milvus 实例，映射字段名到我们在 milvus_client.py 中定义的 schema：
          - text_field="content"     → 文本内容存到 content 字段
          - vector_field="vector"    → 向量存到 vector 字段
          - primary_field="id"       → 主键
          - metadata_field="metadata"→ 元数据（JSON 格式）
        """
        try:
            # 必须先建立 Milvus 连接，否则 LangChain Milvus 内部访问 Collection 时会报错
            # ConnectionNotExistException: should create connection first.
            _ = milvus_manager.connect()

            connection_args = {
                "host": config.milvus_host,
                "port": config.milvus_port,
            }

            # 创建 LangChain Milvus VectorStore
            # 这就像 Spring Data JPA 中创建 JpaRepository 实例
            # embedding_function 类似 JPA 的 EntityMapper，负责对象↔数据库转换
            self.vector_store = Milvus(
                embedding_function=vector_embedding_service,  # 自动调用 embed_documents
                collection_name=self.collection_name,          # 使用 biz collection
                connection_args=connection_args,
                auto_id=False,        # 使用自定义 id（uuid），不用 Milvus 自动生成
                drop_old=False,       # 不删除已有数据
                text_field="content",  # 文本内容存储到 content 字段
                vector_field="vector", # 向量存储到 vector 字段
                primary_field="id",    # 主键字段
                metadata_field="metadata",  # 元数据字段
            )

            logger.info(
                f"VectorStore 初始化成功: {config.milvus_host}:{config.milvus_port}, "
                f"collection: {self.collection_name}"
            )

        except Exception as e:
            logger.error(f"VectorStore 初始化失败: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> List[str]:
        """批量添加文档到向量存储（自动批量向量化）

        LangChain Milvus 的 add_documents 会自动：
        1. 调用 embedding_function.embed_documents() 把文本转成向量
        2. 将 id、content、vector、metadata 四个字段写入 Milvus

        Args:
            documents: 文档列表（每个 Document 包含 page_content 和 metadata）

        Returns:
            List[str]: 文档 ID 列表
        """
        try:
            start_time = time.time()

            # 为每个文档生成唯一 id（因为 auto_id=False）
            ids = [str(uuid.uuid4()) for _ in documents]

            # add_documents 内部会自动调用 embedding_function 进行批量向量化
            # 相当于：embedding_function.embed_documents([doc.page_content for doc in documents])
            result_ids = self.vector_store.add_documents(documents, ids=ids)

            elapsed = time.time() - start_time
            logger.info(
                f"批量添加 {len(documents)} 个文档到 VectorStore 完成, "
                f"耗时: {elapsed:.2f}秒, 平均: {elapsed/len(documents):.2f}秒/个"
            )
            return result_ids
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

    def delete_by_source(self, file_path: str) -> int:
        """删除指定文件的所有文档

        通过 metadata 中的 _source 字段过滤
        （_source 是在 document_splitter_service.py 中设置的）

        Args:
            file_path: 文件路径（与 metadata._source 的值匹配）

        Returns:
            int: 删除的文档数量
        """
        try:
            # 使用 milvus_manager 获取已连接的 collection（底层 pymilvus Collection）
            collection = milvus_manager.get_collection()

            # metadata 是 JSON 字段，使用 JSON 路径查询语法
            # _source 是文档的来源文件路径（在 document_splitter_service 中设置）
            expr = f'metadata["_source"] == "{file_path}"'

            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0

            logger.info(f"删除文件旧数据: {file_path}, 删除数量: {deleted_count}")
            return deleted_count

        except Exception as e:
            # 首次索引时没有旧数据，删除会失败，这是正常的
            logger.warning(f"删除旧数据失败 (可能是首次索引): {e}")
            return 0

    def get_vector_store(self) -> Milvus:
        """获取 VectorStore 实例（供 RAG Agent 使用）"""
        return self.vector_store

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """相似度搜索

        LangChain Milvus 的 similarity_search 会自动：
        1. 调用 embedding_function.embed_query() 把查询文本转成向量
        2. 在 Milvus 中做 L2 距离搜索
        3. 返回最相似的 k 个文档

        Args:
            query: 查询文本
            k: 返回结果数量（默认 3，对应 config.rag_top_k）

        Returns:
            List[Document]: 相关文档列表
        """
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.debug(f"相似度搜索完成: query='{query}', 结果数={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []

# 全局单例（全项目共享这一个实例）
# Java 类比：类似于 @Component 单例 Bean
vector_store_manager = VectorStoreManager()