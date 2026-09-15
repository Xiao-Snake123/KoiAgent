"""记忆系统的 SQLite 存储层。

三张表，对应三类需要持久化的记忆：

=============  ==============================================================
表              用途
=============  ==============================================================
``memories``   长期记忆：从对话中抽取的事实/偏好/异议/承诺，跨会话保留
``user_profiles`` 用户画像：结构化属性（预算区间、意向等级、议价风格…）
``tool_calls`` 工具记忆：工具调用历史（含缓存命中标记），用于回放与统计
=============  ==============================================================

设计与 ``koiagent.storage.context`` 的差异
------------------------------------------
- **独立数据库文件**（``MEMORY_DB_PATH``，默认 ``data/memory.db``）：记忆写入频繁，
  与业务对话历史分离可避免 SQLite 的库级锁互相拖累。
- **启用 WAL**：允许「一写多读」并发，后台记忆抽取任务写入时不会阻塞主流程读取。
- **每操作独立连接**：与 ``ChatContextManager`` 保持一致，避免跨线程共享连接
  （sqlite3 连接默认不可跨线程使用）。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

DEFAULT_DB_PATH = "data/memory.db"

# 读操作不需要全局写锁；用一个始终可获取的锁占位，让 _execute 的分支保持统一
_NULL_LOCK = threading.Lock()

_SCHEMA = [
    # ---- 长期记忆 ----
    """
    CREATE TABLE IF NOT EXISTS memories (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        scope        TEXT    NOT NULL,
        scope_id     TEXT    NOT NULL,
        kind         TEXT    NOT NULL,
        content      TEXT    NOT NULL,
        fingerprint  TEXT    NOT NULL,
        weight       REAL    NOT NULL DEFAULT 1.0,
        hits         INTEGER NOT NULL DEFAULT 0,
        created_at   REAL    NOT NULL,
        last_used_at REAL
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_fp
        ON memories (scope, scope_id, fingerprint)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_memory_scope
        ON memories (scope, scope_id, created_at DESC)
    """,
    # ---- 用户画像 ----
    """
    CREATE TABLE IF NOT EXISTS user_profiles (
        user_id    TEXT PRIMARY KEY,
        data       TEXT NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    # ---- 工具记忆 ----
    """
    CREATE TABLE IF NOT EXISTS tool_calls (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id    TEXT,
        tool_name  TEXT    NOT NULL,
        args       TEXT,
        result     TEXT,
        cache_hit  INTEGER NOT NULL DEFAULT 0,
        created_at REAL    NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tool_calls_chat
        ON tool_calls (chat_id, created_at DESC)
    """,
]


@dataclass
class MemoryRecord:
    """一条长期记忆。"""

    content: str
    kind: str = "fact"          # fact | preference | objection | commitment
    weight: float = 1.0
    hits: int = 0
    created_at: float = field(default_factory=time.time)
    last_used_at: Optional[float] = None
    fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "kind": self.kind,
            "weight": round(self.weight, 3),
            "hits": self.hits,
        }


class MemoryStore:
    """记忆持久化（线程安全：每次操作独立连接 + 全局写锁）。"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("MEMORY_DB_PATH", DEFAULT_DB_PATH)
        self._write_lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------ #
    # 基础设施
    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            with self._write_lock:
                conn = self._connect()
                try:
                    # WAL：允许读写并发，后台记忆写入不阻塞主流程读取
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    for statement in _SCHEMA:
                        conn.execute(statement)
                    conn.commit()
                finally:
                    conn.close()
        except Exception as e:  # 记忆是增强能力，初始化失败不应阻断启动
            logger.warning(f"记忆数据库初始化失败（记忆功能将不可用）: {e}")
            return
        logger.debug(f"记忆数据库就绪: {self.db_path}")

    def _execute(self, sql: str, params: tuple = (), *, write: bool) -> List[sqlite3.Row]:
        cmd = sql.lstrip().split()[0].upper()
        if cmd not in ("SELECT", "WITH"):
            write = True
        lock = self._write_lock if write else _NULL_LOCK
        try:
            with lock:
                conn = self._connect()
                try:
                    cursor = conn.execute(sql, params)
                    rows = cursor.fetchall() if cmd in ("SELECT", "WITH") else []
                    if write:
                        conn.commit()
                    return rows
                finally:
                    conn.close()
        except Exception as e:
            logger.warning(f"记忆数据库操作失败: {e} (sql={sql[:60]}...)")
            return []

    # ------------------------------------------------------------------ #
    # 长期记忆
    # ------------------------------------------------------------------ #
    def upsert_memory(
        self,
        scope: str,
        scope_id: str,
        fingerprint: str,
        content: str,
        kind: str = "fact",
        weight: float = 1.0,
    ) -> bool:
        """写入一条记忆；已存在（同 scope + fingerprint）时**提升权重**而不是重复插入。

        返回 ``True`` 表示是新增，``False`` 表示命中去重（已存在）。
        """
        now = time.time()
        rows = self._execute(
            "SELECT id, weight FROM memories WHERE scope=? AND scope_id=? AND fingerprint=?",
            (scope, scope_id, fingerprint),
            write=False,
        )
        if rows:
            # 同一事实被重复提及 → 说明它更重要，做权重累积（上限 5.0 防止单条主导）
            new_weight = min(5.0, float(rows[0]["weight"]) + 0.5)
            self._execute(
                "UPDATE memories SET weight=?, last_used_at=? WHERE id=?",
                (new_weight, now, rows[0]["id"]),
                write=True,
            )
            return False

        self._execute(
            "INSERT INTO memories (scope, scope_id, kind, content, fingerprint, weight, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (scope, scope_id, kind, content, fingerprint, weight, now),
            write=True,
        )
        return True

    def list_memories(
        self, scope: str, scope_id: str, limit: int = 200
    ) -> List[MemoryRecord]:
        rows = self._execute(
            "SELECT content, kind, weight, hits, created_at, last_used_at, fingerprint"
            " FROM memories WHERE scope=? AND scope_id=?"
            " ORDER BY weight DESC, created_at DESC LIMIT ?",
            (scope, scope_id, limit),
            write=False,
        )
        return [
            MemoryRecord(
                content=row["content"],
                kind=row["kind"],
                weight=float(row["weight"]),
                hits=int(row["hits"]),
                created_at=float(row["created_at"]),
                last_used_at=row["last_used_at"],
                fingerprint=row["fingerprint"],
            )
            for row in rows
        ]

    def touch_memories(self, fingerprints: List[str], scope: str, scope_id: str) -> None:
        """记录被召回的记忆（命中计数 + 最近使用时间），供衰减淘汰使用。"""
        if not fingerprints:
            return
        now = time.time()
        for fingerprint in fingerprints:
            self._execute(
                "UPDATE memories SET hits=hits+1, last_used_at=?"
                " WHERE scope=? AND scope_id=? AND fingerprint=?",
                (now, scope, scope_id, fingerprint),
                write=True,
            )

    def delete_memories(self, scope: str, scope_id: str, fingerprints: List[str]) -> int:
        if not fingerprints:
            return 0
        removed = 0
        for fingerprint in fingerprints:
            self._execute(
                "DELETE FROM memories WHERE scope=? AND scope_id=? AND fingerprint=?",
                (scope, scope_id, fingerprint),
                write=True,
            )
            removed += 1
        return removed

    def prune_memories(
        self, scope: str, scope_id: str, max_facts: int, decay_days: float
    ) -> int:
        """淘汰记忆：① 超期且从未命中的低价值项 ② 超出容量后权重最低的项。"""
        removed = 0
        now = time.time()

        if decay_days > 0:
            cutoff = now - decay_days * 86400
            self._execute(
                "DELETE FROM memories WHERE scope=? AND scope_id=?"
                " AND hits=0 AND weight < 1.5 AND created_at < ?",
                (scope, scope_id, cutoff),
                write=True,
            )

        rows = self._execute(
            "SELECT COUNT(*) AS n FROM memories WHERE scope=? AND scope_id=?",
            (scope, scope_id),
            write=False,
        )
        total = int(rows[0]["n"]) if rows else 0
        overflow = total - max_facts
        if overflow > 0:
            self._execute(
                "DELETE FROM memories WHERE id IN ("
                "  SELECT id FROM memories WHERE scope=? AND scope_id=?"
                "  ORDER BY weight ASC, COALESCE(last_used_at, created_at) ASC LIMIT ?"
                ")",
                (scope, scope_id, overflow),
                write=True,
            )
            removed += overflow

        return removed

    def count_memories(self, scope: Optional[str] = None) -> int:
        if scope:
            rows = self._execute(
                "SELECT COUNT(*) AS n FROM memories WHERE scope=?", (scope,), write=False
            )
        else:
            rows = self._execute("SELECT COUNT(*) AS n FROM memories", (), write=False)
        return int(rows[0]["n"]) if rows else 0

    # ------------------------------------------------------------------ #
    # 用户画像
    # ------------------------------------------------------------------ #
    def load_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        rows = self._execute(
            "SELECT data FROM user_profiles WHERE user_id=?", (user_id,), write=False
        )
        if not rows:
            return None
        try:
            return json.loads(rows[0]["data"])
        except (json.JSONDecodeError, TypeError):
            return None

    def save_profile(self, user_id: str, data: Dict[str, Any]) -> None:
        self._execute(
            "INSERT INTO user_profiles (user_id, data, updated_at) VALUES (?, ?, ?)"
            " ON CONFLICT(user_id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at",
            (user_id, json.dumps(data, ensure_ascii=False), time.time()),
            write=True,
        )

    def count_profiles(self) -> int:
        rows = self._execute("SELECT COUNT(*) AS n FROM user_profiles", (), write=False)
        return int(rows[0]["n"]) if rows else 0

    # ------------------------------------------------------------------ #
    # 工具记忆
    # ------------------------------------------------------------------ #
    def record_tool_call(
        self,
        chat_id: Optional[str],
        tool_name: str,
        args: Dict[str, Any],
        result: str,
        cache_hit: bool = False,
    ) -> None:
        # 结果截断存储：工具记忆用于回放与统计，不需要完整长文本
        self._execute(
            "INSERT INTO tool_calls (chat_id, tool_name, args, result, cache_hit, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                chat_id,
                tool_name,
                json.dumps(args, ensure_ascii=False)[:500],
                (result or "")[:1000],
                1 if cache_hit else 0,
                time.time(),
            ),
            write=True,
        )

    def tool_call_stats(self, limit: int = 2000) -> Dict[str, Any]:
        """统计最近的工具调用：总数、缓存命中数、按工具的分布。"""
        rows = self._execute(
            "SELECT tool_name, cache_hit FROM tool_calls ORDER BY id DESC LIMIT ?",
            (limit,),
            write=False,
        )
        stats: Dict[str, Any] = {"total": len(rows), "cache_hits": 0, "by_tool": {}}
        for row in rows:
            name = row["tool_name"]
            stats["by_tool"][name] = stats["by_tool"].get(name, 0) + 1
            stats["cache_hits"] += int(row["cache_hit"])
        return stats

    def prune_tool_calls(self, keep: int = 5000) -> None:
        """只保留最近 ``keep`` 条工具调用记录。"""
        self._execute(
            "DELETE FROM tool_calls WHERE id NOT IN ("
            "  SELECT id FROM tool_calls ORDER BY id DESC LIMIT ?"
            ")",
            (keep,),
            write=True,
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "db": self.db_path,
            "memories": self.count_memories(),
            "profiles": self.count_profiles(),
            "tool_calls": self.tool_call_stats()["total"],
        }
