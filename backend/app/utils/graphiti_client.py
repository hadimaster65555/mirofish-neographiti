"""
Graphiti 客户端单例 + 异步运行助手

使用持久化后台事件循环（persistent background event loop）确保：
- 同一个 Graphiti 实例和 Neo4j 连接池被所有 Flask 线程共享
- 异步 coroutine 总是在同一个事件循环上运行，避免跨 loop 使用连接的问题
- Flask (同步) 线程通过 run_async() 提交任务并阻塞等待结果
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

from neo4j import GraphDatabase  # 同步驱动，供 Cypher 直查

if TYPE_CHECKING:
    from graphiti_core import Graphiti as GraphitiType

# --------------------------------------------------------------------------
# 持久化后台事件循环
# --------------------------------------------------------------------------

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """确保后台事件循环正在运行，返回该循环。"""
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is None or not _loop.is_running():
            _loop = asyncio.new_event_loop()
            _loop_thread = threading.Thread(
                target=_loop.run_forever,
                daemon=True,
                name="graphiti-event-loop",
            )
            _loop_thread.start()
    return _loop


def run_async(coro, timeout: float | None = None):
    """
    在持久化后台事件循环中运行异步协程，阻塞当前线程直到完成。
    适用于 Flask 同步线程调用 Graphiti 的异步 API。
    timeout 默认 900s：add_episode 内部会依次调用多个 LLM 接口
    （实体提取、边提取、去重等），慢速模型累计耗时可能超过 5 分钟。
    """
    if timeout is None:
        from ..config import Config
        timeout = Config.GRAPHITI_EPISODE_TIMEOUT
    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


# --------------------------------------------------------------------------
# Graphiti 单例
# --------------------------------------------------------------------------

_graphiti: "GraphitiType | None" = None
_graphiti_lock = threading.Lock()


def get_graphiti() -> "GraphitiType":
    """获取全局 Graphiti 单例（懒加载）。"""
    global _graphiti
    if _graphiti is None:
        with _graphiti_lock:
            if _graphiti is None:
                _graphiti = _build_graphiti()
    return _graphiti


def _build_graphiti() -> "GraphitiType":
    """构建 Graphiti 实例（driver 在后台 loop 首次使用时建连）。"""
    import httpx
    from openai import AsyncOpenAI
    from graphiti_core import Graphiti
    from graphiti_core.llm_client.openai_client import OpenAIClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

    from ..config import Config

    # 每个 HTTP 请求超时；防止单次 API 调用挂死
    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(Config.GRAPHITI_HTTP_TIMEOUT, connect=10.0)
    )

    class _SafeEmbedder(OpenAIEmbedder):
        """Replaces empty strings with a space to prevent 400 'Input is empty' errors."""

        async def create(self, input_data) -> list:
            if isinstance(input_data, str):
                input_data = input_data if input_data and input_data.strip() else " "
            return await super().create(input_data)

        async def create_batch(self, input_data_list: list) -> list:
            if not input_data_list:
                return []
            safe = [t if (t and t.strip()) else " " for t in input_data_list]
            return await super().create_batch(safe)

    llm_config = LLMConfig(
        api_key=Config.LLM_API_KEY,
        model=Config.LLM_MODEL_NAME,
        base_url=Config.LLM_BASE_URL,
    )

    # Inject the http_client with timeout directly into the underlying AsyncOpenAI client
    llm_client = OpenAIClient(llm_config)
    llm_client.client = AsyncOpenAI(
        api_key=Config.LLM_API_KEY,
        base_url=Config.LLM_BASE_URL,
        http_client=_http_client,
    )

    embedder = _SafeEmbedder(
        OpenAIEmbedderConfig(
            api_key=Config.EMBEDDING_API_KEY,
            base_url=Config.EMBEDDING_BASE_URL,
            embedding_model=Config.EMBEDDING_MODEL_NAME,
        )
    )
    embedder.client = AsyncOpenAI(
        api_key=Config.EMBEDDING_API_KEY,
        base_url=Config.EMBEDDING_BASE_URL,
        http_client=_http_client,
    )

    cross_encoder = OpenAIRerankerClient(config=llm_config)

    return Graphiti(
        uri=Config.NEO4J_URI,
        user=Config.NEO4J_USER,
        password=Config.NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )


# --------------------------------------------------------------------------
# 同步 Neo4j 驱动（用于直接 Cypher 查询，避免 async 复杂性）
# --------------------------------------------------------------------------

_sync_driver = None
_sync_driver_lock = threading.Lock()


def get_sync_driver():
    """获取同步 Neo4j 驱动单例，供 zep_paging.py 等直接 Cypher 查询使用。"""
    global _sync_driver
    if _sync_driver is None:
        with _sync_driver_lock:
            if _sync_driver is None:
                from ..config import Config
                _sync_driver = GraphDatabase.driver(
                    Config.NEO4J_URI,
                    auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
                )
    return _sync_driver
