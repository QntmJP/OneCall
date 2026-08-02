"""
@Author: QntmJP
@Desc: 文档分割服务
"""

"""文档分割服务模块 - 基于 LangChain 的智能文档分割

两阶段分割策略：
  阶段1：按 Markdown 标题切分（# 和 ##）
  阶段2：按字符数二次切分（chunk_size=1600）
  阶段3：合并太小的碎片（< 300 字符）
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

from app.config import config

class DocumentSplitterService:
    """文档分割服务 - 使用 LangChain 的分割器"""

    def __init__(self):
        """初始化分割器"""
        self.chunk_size = config.chunk_max_size       # 800
        self.chunk_overlap = config.chunk_overlap      # 100

        # 分割器1：按 Markdown 标题切分（只按 # 和 ## 切，不按 ### 切，避免过度碎片化）
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),    # 一级标题作为元数据
                ("##", "h2"),   # 二级标题作为元数据
                # 不按三级标题切，避免碎片太碎
            ],
            strip_headers=False,  # 保留标题在内容中（检索时自带标题上下文）
        )

        # 分割器2：递归字符分割器（用于二次分割，chunk_size 加倍 = 1600）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2,   # 800*2=1600，减少分片数
            chunk_overlap=self.chunk_overlap,  # 相邻块重叠 100 字符，保证上下文连续
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            f"文档分割服务初始化完成, "
            f"chunk_size={self.chunk_size}, secondary_chunk_size={self.chunk_size * 2}, "
            f"overlap={self.chunk_overlap}"
        )

    def split_markdown(self, content: str, file_path: str = "") -> List[Document]:
        """
        分割 Markdown 文档（两阶段分割 + 合并小片段）

        Args:
            content: Markdown 文本内容
            file_path: 文件路径（写入元数据，用于后续按文件删除）

        Returns:
            文档分片列表，每个分片是一个 Document 对象
        """
        if not content or not content.strip():
            logger.warning(f"Markdown 文档内容为空: {file_path}")
            return []

        try:
            # ===== 阶段1：按标题分割 =====
            md_docs = self.markdown_splitter.split_text(content)

            # ===== 阶段2：按字符数二次分割 =====
            docs_after_split = self.text_splitter.split_documents(md_docs)

            # ===== 阶段3：合并太小的碎片 =====
            final_docs = self._merge_small_chunks(docs_after_split, min_size=300)

            # 给每个分片添加文件来源元数据
            for doc in final_docs:
                doc.metadata["_source"] = file_path
                doc.metadata["_extension"] = ".md"
                doc.metadata["_file_name"] = Path(file_path).name

            logger.info(f"Markdown 分割完成: {file_path} -> {len(final_docs)} 个分片")
            return final_docs

        except Exception as e:
            logger.error(f"Markdown 分割失败: {file_path}, 错误: {e}")
            raise

    def split_text(self, content: str, file_path: str = "") -> List[Document]:
        """
        分割普通文本文档（TXT，只做字符分割，不按标题切）

        Args:
            content: 文本内容
            file_path: 文件路径

        Returns:
            文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"文本文档内容为空: {file_path}")
            return []

        try:
            docs = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[
                    {
                        "_source": file_path,
                        "_extension": Path(file_path).suffix,
                        "_file_name": Path(file_path).name,
                    }
                ],
            )

            logger.info(f"文本分割完成: {file_path} -> {len(docs)} 个分片")
            return docs

        except Exception as e:
            logger.error(f"文本分割失败: {file_path}, 错误: {e}")
            raise

    def split_document(self, content: str, file_path: str = "") -> List[Document]:
        """
        智能分割文档（根据文件后缀自动选择分割器）

        - .md 文件 → 走两阶段分割
        - 其他文件 → 走纯字符分割

        Args:
            content: 文档内容
            file_path: 文件路径

        Returns:
            文档分片列表
        """
        if file_path.endswith(".md"):
            return self.split_markdown(content, file_path)
        else:
            return self.split_text(content, file_path)

    def _merge_small_chunks(
        self, documents: List[Document], min_size: int = 300
    ) -> List[Document]:
        """
        合并太小的分片

        策略：如果当前片段 < 300 字符，且合并后不超过 chunk_size*2，
        就把它拼到前一个片段后面，用空行分隔。

        Args:
            documents: 文档列表
            min_size: 最小分片大小（字符数）

        Returns:
            合并后的文档列表
        """
        if not documents:
            return []

        merged_docs = []
        current_doc = None

        for doc in documents:
            doc_size = len(doc.page_content)

            if current_doc is None:
                # 第一个文档
                current_doc = doc
            elif doc_size < min_size and len(current_doc.page_content) < self.chunk_size * 2:
                # 当前文档太小且合并后不会太大 → 合并
                current_doc.page_content += "\n\n" + doc.page_content
            else:
                # 保存当前文档，开始新文档
                merged_docs.append(current_doc)
                current_doc = doc

        # 添加最后一个文档
        if current_doc is not None:
            merged_docs.append(current_doc)

        return merged_docs

# 全局单例
document_splitter_service = DocumentSplitterService()