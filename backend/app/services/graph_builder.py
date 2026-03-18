"""
图谱构建服务
使用 Graphiti + 本地 Neo4j 构建知识图谱（替代 Zep Cloud）
"""

import uuid
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from ..utils.graphiti_client import get_graphiti, get_sync_driver, run_async
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges
from .text_processor import TextProcessor


@dataclass
class GraphInfo:
    """图谱信息"""
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class GraphBuilderService:
    """
    图谱构建服务
    负责调用 Graphiti API 构建知识图谱
    """

    def __init__(self):
        self.graphiti = get_graphiti()
        self.task_manager = TaskManager()
        # 存储每个 graph_id 对应的实体类型（Pydantic 模型字典）
        self._entity_types: Dict[str, Dict] = {}

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3
    ) -> str:
        """
        异步构建图谱

        Args:
            text: 输入文本
            ontology: 本体定义（来自接口1的输出）
            graph_name: 图谱名称
            chunk_size: 文本块大小
            chunk_overlap: 块重叠大小
            batch_size: 每批发送的块数量

        Returns:
            任务ID
        """
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            }
        )

        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size)
        )
        thread.daemon = True
        thread.start()

        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int
    ):
        """图谱构建工作线程"""
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message="开始构建图谱..."
            )

            # 1. 创建图谱（生成 group_id）
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id,
                progress=10,
                message=f"图谱已创建: {graph_id}"
            )

            # 2. 解析本体（存储实体类型供后续 add_episode 使用）
            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(
                task_id,
                progress=15,
                message="本体已解析"
            )

            # 3. 文本分块
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id,
                progress=20,
                message=f"文本已分割为 {total_chunks} 个块"
            )

            # 4. 分批发送数据（同步完成，无需等待）
            self.add_text_batches(
                graph_id, chunks, batch_size,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.70),  # 20-90%
                    message=msg
                )
            )

            # 5. 获取图谱信息
            self.task_manager.update_task(
                task_id,
                progress=90,
                message="获取图谱信息..."
            )

            graph_info = self._get_graph_info(graph_id)

            # 完成
            self.task_manager.complete_task(task_id, {
                "graph_id": graph_id,
                "graph_info": graph_info.to_dict(),
                "chunks_processed": total_chunks,
            })

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)

    def create_graph(self, name: str) -> str:
        """创建图谱，返回 group_id（无需 API 调用）"""
        return f"mirofish_{uuid.uuid4().hex[:16]}"

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """
        解析本体定义，将实体类型构建为 Pydantic 模型并存储。
        Graphiti 不需要全局注册本体；每次 add_episode 时传入 entity_types 即可。
        """
        import warnings
        from typing import Optional
        from pydantic import BaseModel, Field, create_model

        warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

        RESERVED_NAMES = {'uuid', 'name', 'group_id', 'name_embedding', 'summary', 'created_at'}

        def safe_attr_name(attr_name: str) -> str:
            if attr_name.lower() in RESERVED_NAMES:
                return f"entity_{attr_name}"
            return attr_name

        entity_types: Dict[str, Any] = {}
        for entity_def in ontology.get("entity_types", []):
            ent_name = entity_def["name"]
            description = entity_def.get("description", f"A {ent_name} entity.")

            fields: Dict[str, Any] = {}
            for attr_def in entity_def.get("attributes", []):
                attr_name = safe_attr_name(attr_def["name"])
                attr_desc = attr_def.get("description", attr_name)
                fields[attr_name] = (Optional[str], Field(None, description=attr_desc))

            model = create_model(ent_name, **fields, __base__=BaseModel)
            model.__doc__ = description
            entity_types[ent_name] = model

        self._entity_types[graph_id] = entity_types

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None
    ) -> List[str]:
        """
        将文本块逐一发送到 Graphiti（add_episode），
        Graphiti 在线 LLM 提取实体/关系，无需轮询。
        返回空列表（兼容原接口，调用方不再需要 episode UUIDs）。
        """
        from graphiti_core.nodes import EpisodeType
        from uuid import uuid4

        chunks = [c for c in chunks if c and c.strip()]
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            batch_num = i + 1
            if progress_callback:
                progress = (i + 1) / total_chunks
                progress_callback(
                    f"处理第 {batch_num}/{total_chunks} 个文本块...",
                    progress
                )

            try:
                run_async(
                    self.graphiti.add_episode(
                        name=f"episode_{uuid4().hex[:8]}",
                        episode_body=chunk,
                        source_description="MiroFish seed document",
                        reference_time=datetime.now(),
                        source=EpisodeType.text,
                        group_id=graph_id,
                        entity_types=self._entity_types.get(graph_id, {}),
                    )
                )
            except Exception as e:
                if progress_callback:
                    progress_callback(f"块 {batch_num} 处理失败: {str(e)}", 0)
                raise

        return []

    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        """获取图谱统计信息"""
        nodes = fetch_all_nodes(self.graphiti, graph_id)
        edges = fetch_all_edges(self.graphiti, graph_id)

        entity_types = set()
        for node in nodes:
            if node.labels:
                for label in node.labels:
                    if label not in ["Entity", "Node"]:
                        entity_types.add(label)

        return GraphInfo(
            graph_id=graph_id,
            node_count=len(nodes),
            edge_count=len(edges),
            entity_types=list(entity_types)
        )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """
        获取完整图谱数据（包含详细信息）

        Returns:
            包含 nodes 和 edges 的字典，包括时间信息、属性等
        """
        nodes = fetch_all_nodes(self.graphiti, graph_id)
        edges = fetch_all_edges(self.graphiti, graph_id)

        node_map = {}
        for node in nodes:
            node_map[node.uuid_] = node.name or ""

        nodes_data = []
        for node in nodes:
            created_at = getattr(node, 'created_at', None)
            if created_at:
                created_at = str(created_at)

            nodes_data.append({
                "uuid": node.uuid_,
                "name": node.name,
                "labels": node.labels or [],
                "summary": node.summary or "",
                "attributes": node.attributes or {},
                "created_at": created_at,
            })

        edges_data = []
        for edge in edges:
            created_at = getattr(edge, 'created_at', None)
            valid_at = getattr(edge, 'valid_at', None)
            invalid_at = getattr(edge, 'invalid_at', None)
            expired_at = getattr(edge, 'expired_at', None)

            episodes = getattr(edge, 'episodes', None) or getattr(edge, 'episode_ids', None)
            if episodes and not isinstance(episodes, list):
                episodes = [str(episodes)]
            elif episodes:
                episodes = [str(e) for e in episodes]

            fact_type = getattr(edge, 'fact_type', None) or edge.name or ""

            edges_data.append({
                "uuid": edge.uuid_,
                "name": edge.name or "",
                "fact": edge.fact or "",
                "fact_type": fact_type,
                "source_node_uuid": edge.source_node_uuid,
                "target_node_uuid": edge.target_node_uuid,
                "source_node_name": node_map.get(edge.source_node_uuid, ""),
                "target_node_name": node_map.get(edge.target_node_uuid, ""),
                "attributes": edge.attributes or {},
                "created_at": str(created_at) if created_at else None,
                "valid_at": str(valid_at) if valid_at else None,
                "invalid_at": str(invalid_at) if invalid_at else None,
                "expired_at": str(expired_at) if expired_at else None,
                "episodes": episodes or [],
            })

        return {
            "graph_id": graph_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }

    def delete_graph(self, graph_id: str):
        """删除图谱（从 Neo4j 中移除所有相关节点和关系）"""
        driver = get_sync_driver()
        with driver.session() as session:
            # 先删关系，再删节点
            session.run(
                "MATCH ()-[e:RELATES_TO {group_id: $gid}]-() DELETE e",
                gid=graph_id
            )
            session.run(
                "MATCH (n {group_id: $gid}) DETACH DELETE n",
                gid=graph_id
            )
