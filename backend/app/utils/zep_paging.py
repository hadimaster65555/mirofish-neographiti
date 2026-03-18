"""图谱节点/边分页读取工具（基于 Graphiti + Neo4j）。

替代原 Zep Cloud 的分页 API，改为直接对本地 Neo4j 执行 Cypher 查询。
返回的代理对象（_NodeProxy / _EdgeProxy）暴露与原 Zep 节点/边完全相同的属性，
确保所有下游调用方（graph_builder、zep_entity_reader、zep_tools）无需修改。
"""

from __future__ import annotations

import json
from typing import Any

from .graphiti_client import get_sync_driver
from .logger import get_logger

logger = get_logger('mirofish.zep_paging')

_DEFAULT_PAGE_SIZE = 100
_MAX_NODES = 2000


# --------------------------------------------------------------------------
# 代理类：与原 Zep SDK 返回对象保持相同的属性接口
# --------------------------------------------------------------------------

class _NodeProxy:
    """节点代理：暴露与 Zep NodeData 相同的属性。"""

    def __init__(self, node_data: dict, labels: list):
        self._data = node_data
        self._labels = labels

    @property
    def uuid_(self) -> str:
        return self._data.get('uuid', '')

    @property
    def uuid(self) -> str:
        return self._data.get('uuid', '')

    @property
    def name(self) -> str:
        return self._data.get('name', '') or ''

    @property
    def labels(self) -> list:
        return self._labels

    @property
    def summary(self) -> str:
        return self._data.get('summary', '') or ''

    @property
    def attributes(self) -> dict:
        raw = self._data.get('attributes', {})
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except Exception:
                return {}
        return raw if isinstance(raw, dict) else {}

    @property
    def created_at(self):
        return self._data.get('created_at', None)


class _EdgeProxy:
    """边代理：暴露与 Zep EdgeData 相同的属性。"""

    def __init__(self, edge_data: dict, source_uuid: str, target_uuid: str):
        self._data = edge_data
        self._source_uuid = source_uuid
        self._target_uuid = target_uuid

    @property
    def uuid_(self) -> str:
        return self._data.get('uuid', '')

    @property
    def uuid(self) -> str:
        return self._data.get('uuid', '')

    @property
    def name(self) -> str:
        return self._data.get('name', '') or ''

    @property
    def fact(self) -> str:
        return self._data.get('fact', '') or ''

    @property
    def source_node_uuid(self) -> str:
        return self._source_uuid or self._data.get('source_node_uuid', '')

    @property
    def target_node_uuid(self) -> str:
        return self._target_uuid or self._data.get('target_node_uuid', '')

    @property
    def attributes(self) -> dict:
        raw = self._data.get('attributes', {})
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except Exception:
                return {}
        return raw if isinstance(raw, dict) else {}

    @property
    def created_at(self):
        return self._data.get('created_at', None)

    @property
    def valid_at(self):
        return self._data.get('valid_at', None)

    @property
    def invalid_at(self):
        return self._data.get('invalid_at', None)

    @property
    def expired_at(self):
        return self._data.get('expired_at', None)

    @property
    def fact_type(self) -> str:
        return self._data.get('fact_type', self._data.get('name', ''))

    @property
    def episodes(self) -> list:
        return self._data.get('episodes', [])

    @property
    def episode_ids(self) -> list:
        return self._data.get('episode_ids', [])


# --------------------------------------------------------------------------
# 公共分页函数（签名兼容原 Zep 版本）
# --------------------------------------------------------------------------

def fetch_all_nodes(
    graphiti_instance,  # 保留参数以兼容调用方，内部不直接使用
    group_id: str,
    page_size: int = _DEFAULT_PAGE_SIZE,
    max_items: int = _MAX_NODES,
    **kwargs: Any,
) -> list:
    """分页获取图谱所有节点（最多 max_items 条）。"""
    driver = get_sync_driver()
    all_nodes: list = []
    skip = 0

    query = """
    MATCH (n:Entity {group_id: $group_id})
    RETURN n, labels(n) AS node_labels
    ORDER BY n.created_at
    SKIP $skip LIMIT $limit
    """

    with driver.session() as session:
        while True:
            records = list(session.run(query, group_id=group_id, skip=skip, limit=page_size))
            if not records:
                break

            for record in records:
                node_data = dict(record['n'])
                node_labels = list(record['node_labels'])
                all_nodes.append(_NodeProxy(node_data, node_labels))

            if len(all_nodes) >= max_items:
                all_nodes = all_nodes[:max_items]
                logger.warning(
                    f"Node count reached limit ({max_items}), stopping pagination for group {group_id}"
                )
                break

            if len(records) < page_size:
                break

            skip += page_size

    logger.info(f"Fetched {len(all_nodes)} nodes for group {group_id}")
    return all_nodes


def fetch_all_edges(
    graphiti_instance,  # 保留参数以兼容调用方，内部不直接使用
    group_id: str,
    page_size: int = _DEFAULT_PAGE_SIZE,
    **kwargs: Any,
) -> list:
    """分页获取图谱所有边（无数量上限）。"""
    driver = get_sync_driver()
    all_edges: list = []
    skip = 0

    query = """
    MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity)
    WHERE e.group_id = $group_id
    RETURN e, s.uuid AS source_uuid, t.uuid AS target_uuid
    ORDER BY e.created_at
    SKIP $skip LIMIT $limit
    """

    with driver.session() as session:
        while True:
            records = list(session.run(query, group_id=group_id, skip=skip, limit=page_size))
            if not records:
                break

            for record in records:
                edge_data = dict(record['e'])
                source_uuid = record.get('source_uuid') or edge_data.get('source_node_uuid', '')
                target_uuid = record.get('target_uuid') or edge_data.get('target_node_uuid', '')
                all_edges.append(_EdgeProxy(edge_data, source_uuid, target_uuid))

            if len(records) < page_size:
                break

            skip += page_size

    logger.info(f"Fetched {len(all_edges)} edges for group {group_id}")
    return all_edges
