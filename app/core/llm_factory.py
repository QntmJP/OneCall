"""
@Author: QntmJP
@Desc: LLM 工厂类

"""

"""LLM 工厂类

使用 LangChain ChatOpenAI 通过 OpenAI 兼容模式调用阿里云 DashScope
这种方式便于后续切换到其他支持 OpenAI API 的模型提供商

支持的模型提供商（只需修改 base_url 和 api_key）：
- 阿里云 DashScope: https://dashscope.aliyuncs.com/compatible-mode/v1
- OpenAI: https://api.openai.com/v1
- Azure OpenAI: https://{resource}.openai.azure.com
- 其他兼容 OpenAI API 的服务
"""

from langchain_openai import ChatOpenAI
from app.config import config
from loguru import logger

class LLMFactory:
    """LLM 工厂类 - 使用 OpenAI 兼容模式"""

    # 阿里云 DashScope OpenAI 兼容模式 URL
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @staticmethod
    def create_chat_model(
        model: str | None = None,
        temperature: float = 0.7,
        streaming: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> ChatOpenAI:
        """
        创建聊天模型实例

        核心设计：通过 OpenAI 兼容接口调用阿里云通义千问
        只需要改 base_url 和 api_key 就能切换到其他模型提供商

        Args:
            model: 模型名称，默认从配置读取（qwen-max）
            temperature: 温度值（0=确定性，1=创造性），默认 0.7
            streaming: 是否流式输出，默认 True
            base_url: API 地址，默认用 DashScope 地址
            api_key: API 密钥，默认从配置读取

        Returns:
            ChatOpenAI: LangChain 聊天模型实例
        """
        # 参数缺省时从全局配置读取
        model = model or config.dashscope_model
        base_url = base_url or LLMFactory.DASHSCOPE_BASE_URL
        api_key = api_key or config.dashscope_api_key

        # 构建 extra_body 参数（DashScope 需要显式指定 stream）
        # 参考：https://help.aliyun.com/zh/model-studio/getting-started/models
        extra_body = {}
        extra_body["stream"] = streaming

        # 创建并返回 ChatOpenAI 实例
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            base_url=base_url,
            api_key=api_key,
            extra_body=extra_body if extra_body else None,
        )

        logger.info(f"LLM 创建完成: model={model}, temperature={temperature}, streaming={streaming}")
        return llm

# 全局 LLM 工厂实例
llm_factory = LLMFactory()