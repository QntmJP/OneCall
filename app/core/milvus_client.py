"""
@Author: QntmJP
@Desc: Milvus 客户端封装
"""

"""Milvus 客户端工厂模块

负责连接 Milvus 向量数据库，创建 collection 和索引
"""

from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
    MilvusException,
)

from app.config import config

def _patch_pymilvus_milvus_client_orm_alias() -> None:
    """
    Fix: langchain_milvus 内部创建的 MilvusClient 使用了未注册的别名，
    导致 ORM Collection 操作报 ConnectionNotExistException。
    这里强制让 MilvusClient 使用 "default" 别名，与 ORM 一致。
    """
    if getattr(_patch_pymilvus_milvus_client_orm_alias, "_done", False):
        return
    try:
        from pymilvus.milvus_client.milvus_client import MilvusClient
    except ImportError:
        return

    _orig_init = MilvusClient.__init__

    def _wrapped_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._using = "default"

    MilvusClient.__init__ = _wrapped_init
    setattr(_patch_pymilvus_milvus_client_orm_alias, "_done", True)

class MilvusClientManager:
    """Milvus 客户端管理器

    负责连接管理、collection 创建、索引创建、加载
    全项目通过全局单例 milvus_manager 使用
    """

    # 常量定义
    COLLECTION_NAME: str = "biz"       # 集合名（相当于表名）
    VECTOR_DIM: int = 1024             # 向量维度（text-embedding-v4 输出 1024 维）
    ID_MAX_LENGTH: int = 100           # id 字段最大长度
    CONTENT_MAX_LENGTH: int = 8000     # content 字段最大长度
    DEFAULT_SHARD_NUMBER: int = 2      # 分片数

    def __init__(self) -> None:
        """初始化，此时还没连接"""
        self._client: MilvusClient | None = None
        self._collection: Collection | None = None

    def connect(self) -> MilvusClient:
        """
        连接到 Milvus 并初始化 collection

        幂等设计：已连接则直接返回，不重复初始化

        Returns:
            MilvusClient: Milvus 客户端实例

        Raises:
            RuntimeError: 连接或初始化失败
        """
        # 幂等：如果已经连过，跳过
        if self._collection is not None and self._client is not None:
            logger.debug("Milvus 已连接，跳过重复 connect")
            return self._client

        try:
            _patch_pymilvus_milvus_client_orm_alias()

            logger.info(f"正在连接到 Milvus: {config.milvus_host}:{config.milvus_port}")

            # 1. 建立 ORM 连接（pymilvus 的 connections 是全局连接池）
            connections.connect(
                alias="default",
                host=config.milvus_host,
                port=str(config.milvus_port),
                timeout=config.milvus_timeout / 1000,  # 转为秒
            )

            # 2. 创建客户端实例（用于高级操作）
            uri = f"http://{config.milvus_host}:{config.milvus_port}"
            self._client = MilvusClient(uri=uri)

            logger.info("成功连接到 Milvus")

            # 3. 检查并创建 collection
            if not self._collection_exists():
                logger.info(f"collection '{self.COLLECTION_NAME}' 不存在，正在创建...")
                self._create_collection()
                logger.info(f"成功创建 collection '{self.COLLECTION_NAME}'")
            else:
                logger.info(f"collection '{self.COLLECTION_NAME}' 已存在")
                self._collection = Collection(self.COLLECTION_NAME)

                # 检查向量维度是否匹配
                schema = self._collection.schema
                vector_field = None
                existing_dim = None
                for field in schema.fields:
                    if field.name == "vector":
                        vector_field = field
                        break

                if vector_field and hasattr(vector_field, 'params') and 'dim' in vector_field.params:
                    existing_dim = vector_field.params['dim']
                    if existing_dim != self.VECTOR_DIM:
                        logger.warning(
                            f"向量维度不匹配！现有: {existing_dim}, 配置: {self.VECTOR_DIM}"
                        )
                        logger.info(f"删除旧 collection 重建...")
                        _ = utility.drop_collection(self.COLLECTION_NAME)
                        self._create_collection()
                        logger.info(f"重建完成，维度: {self.VECTOR_DIM}")
                    else:
                        logger.info(f"向量维度匹配: {self.VECTOR_DIM}")

            # 4. 加载 collection 到内存（查询前必须 load）
            self._load_collection()

            return self._client

        except MilvusException as e:
            logger.error(f"Milvus 操作失败: {e}")
            self.close()
            raise RuntimeError(f"Milvus 操作失败: {e}") from e
        except ConnectionError as e:
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e

    def _collection_exists(self) -> bool:
        """检查 collection 是否存在"""
        result = utility.has_collection(self.COLLECTION_NAME)
        return bool(result)

    def _create_collection(self) -> None:
        """创建 biz collection（相当于建表）"""
        # 定义 4 个字段
        fields = [
            # 字段1：id（主键，字符串类型）
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=self.ID_MAX_LENGTH,
                is_primary=True,
            ),
            # 字段2：vector（向量字段，1024 维浮点向量）
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.VECTOR_DIM,
            ),
            # 字段3：content（原文内容）
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=self.CONTENT_MAX_LENGTH,
            ),
            # 字段4：metadata（元数据，JSON 格式）
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
            ),
        ]

        # 创建 schema（相当于表结构定义）
        schema = CollectionSchema(
            fields=fields,
            description="Business knowledge collection",
            enable_dynamic_field=False,
        )

        # 创建 collection
        self._collection = Collection(
            name=self.COLLECTION_NAME,
            schema=schema,
            num_shards=self.DEFAULT_SHARD_NUMBER,
        )

        # 创建索引
        self._create_index()

    def _create_index(self) -> None:
        """为 vector 字段创建索引"""
        if self._collection is None:
            raise RuntimeError("Collection 未初始化")

        index_params = {
            "metric_type": "L2",        # 欧氏距离（值越小越相似）
            "index_type": "IVF_FLAT",   # 倒排文件索引
            "params": {"nlist": 128},   # 聚类中心数
        }

        _ = self._collection.create_index(
            field_name="vector",
            index_params=index_params,
        )

        logger.info("成功为 vector 字段创建索引")

    def _load_collection(self) -> None:
        """加载 collection 到内存（Milvus 查询前必须先 load）"""
        if self._collection is None:
            self._collection = Collection(self.COLLECTION_NAME)

        try:
            load_state = utility.load_state(self.COLLECTION_NAME)
            state_name = getattr(load_state, "name", str(load_state))
            if state_name != "Loaded":
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            else:
                logger.info(f"Collection '{self.COLLECTION_NAME}' 已加载")
        except AttributeError:
            try:
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            except MilvusException as e:
                error_msg = str(e).lower()
                if "already loaded" in error_msg or "loaded" in error_msg:
                    logger.info(f"Collection '{self.COLLECTION_NAME}' 已加载")
                else:
                    raise
        except Exception as e:
            logger.error(f"加载 collection 失败: {e}")
            raise

    def get_collection(self) -> Collection:
        """获取 collection 实例"""
        if self._collection is None:
            raise RuntimeError("Collection 未初始化，请先调用 connect()")
        return self._collection

    def health_check(self) -> bool:
        """健康检查"""
        try:
            if self._client is None:
                return False
            _ = connections.list_connections()
            return True
        except (MilvusException, ConnectionError) as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False
        except Exception as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False

    def close(self) -> None:
        """关闭连接"""
        errors = []

        try:
            if self._collection is not None:
                self._collection.release()
                self._collection = None
        except Exception as e:
            errors.append(f"释放 collection 失败: {e}")

        try:
            if connections.has_connection("default"):
                connections.disconnect("default")
        except Exception as e:
            errors.append(f"断开连接失败: {e}")

        self._client = None

        if errors:
            logger.error(f"关闭 Milvus 连接时出现错误: {'; '.join(errors)}")
        else:
            logger.info("已关闭 Milvus 连接")

# 全局单例（全项目共享这一个实例）
milvus_manager = MilvusClientManager()