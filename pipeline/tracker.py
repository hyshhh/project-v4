"""
TrackManager — 跟踪状态管理

维护 YOLO track ID 与弦号识别结果的映射关系。
一旦某个 track ID 完成识别，后续帧沿用该结果，无需重复调用 Agent。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrackInfo:
    """单个 track 的状态信息。"""
    track_id: int
    hull_number: str = ""
    description: str = ""
    recognized: bool = False       # 是否已完成识别
    pending: bool = False          # 是否正在异步识别中
    first_seen_frame: int = 0      # 首次出现的帧号
    last_seen_frame: int = 0       # 最近一次出现的帧号
    last_recognized_frame: int = 0 # 上次完成识别的帧号（用于定时刷新判断）
    db_match_id: str = ""          # 数据库匹配到的弦号（确认后）
    db_match_desc: str = ""        # 数据库匹配到的描述
    db_matched: bool = False       # 是否在数据库中精确匹配到
    semantic_match_ids: list[str] = field(default_factory=list)  # 语义检索匹配到的弦号列表


class TrackManager:
    """
    管理所有活跃的跟踪目标。

    功能：
    - 注册新 track ID
    - 查询 track 是否已识别
    - 绑定识别结果到 track ID
    - 清理长时间未出现的 track

    线程安全：所有公开方法都通过锁保护，支持并发访问。
    """

    def __init__(self, max_stale_frames: int = 300):
        """
        Args:
            max_stale_frames: 超过此帧数未出现的 track 将被清理。
        """
        self._tracks: dict[int, TrackInfo] = {}
        self._max_stale_frames = max_stale_frames
        self._lock = threading.Lock()

    def get_or_create(self, track_id: int, frame_id: int) -> TrackInfo:
        """获取或创建 track 记录。"""
        with self._lock:
            if track_id not in self._tracks:
                self._tracks[track_id] = TrackInfo(
                    track_id=track_id,
                    first_seen_frame=frame_id,
                    last_seen_frame=frame_id,
                )
                logger.debug("新 track 注册: id=%d, frame=%d", track_id, frame_id)
            else:
                self._tracks[track_id].last_seen_frame = frame_id
            return self._tracks[track_id]

    def needs_recognition(self, track_id: int) -> bool:
        """
        判断该 track 是否需要进行 Agent 识别。

        已识别 → False
        正在异步识别中 → False
        其他 → True
        """
        with self._lock:
            if track_id not in self._tracks:
                return True
            info = self._tracks[track_id]
            return not info.recognized and not info.pending

    def needs_refresh(self, track_id: int, frame_id: int, gap_num: int) -> bool:
        """
        判断该 track 是否需要定时刷新识别。

        条件：
        - 已识别 (recognized=True)
        - 非 pending 状态
        - 距上次识别已超过 gap_num 帧

        Args:
            track_id: 跟踪 ID。
            frame_id: 当前帧号。
            gap_num: 刷新间隔帧数。

        Returns:
            是否需要刷新。
        """
        with self._lock:
            if track_id not in self._tracks:
                return False
            info = self._tracks[track_id]
            if not info.recognized or info.pending:
                return False
            if info.last_recognized_frame == 0:
                # last_recognized_frame 未设置（兼容旧数据），用 first_seen_frame
                return frame_id - info.first_seen_frame >= gap_num
            return frame_id - info.last_recognized_frame >= gap_num

    def mark_pending(self, track_id: int) -> None:
        """标记 track 正在异步识别中。"""
        with self._lock:
            if track_id in self._tracks:
                self._tracks[track_id].pending = True

    def cancel_pending(self, track_id: int) -> None:
        """取消 pending 标记（推理失败或队列满时调用）。"""
        with self._lock:
            if track_id in self._tracks:
                self._tracks[track_id].pending = False

    def bind_result(
        self,
        track_id: int,
        hull_number: str,
        description: str,
        frame_id: int = 0,
    ) -> None:
        """将识别结果绑定到 track ID。"""
        with self._lock:
            if track_id not in self._tracks:
                logger.warning("尝试绑定结果到不存在的 track: %d", track_id)
                return

            info = self._tracks[track_id]
            info.hull_number = hull_number
            info.description = description
            info.recognized = True
            info.pending = False
            info.last_recognized_frame = frame_id

        logger.info(
            "Track %d 识别完成: 弦号=%s, 描述=%s",
            track_id,
            hull_number or "(未识别)",
            description[:50] if description else "",
        )

    def bind_db_match(
        self,
        track_id: int,
        db_match_id: str,
        db_match_desc: str,
    ) -> None:
        """绑定数据库精确匹配结果。"""
        with self._lock:
            if track_id in self._tracks:
                info = self._tracks[track_id]
                info.db_match_id = db_match_id
                info.db_match_desc = db_match_desc
                info.db_matched = True

    def bind_semantic_matches(
        self,
        track_id: int,
        match_ids: list[str],
    ) -> None:
        """绑定语义检索匹配结果（非精确匹配时的候选列表）。"""
        with self._lock:
            if track_id in self._tracks:
                self._tracks[track_id].semantic_match_ids = match_ids

    def get_display_text(self, track_id: int) -> str:
        """
        获取用于在画面上显示的文字。

        已识别 + 数据库匹配 → "(库内确定id：XXX)"
        已识别 + 未匹配 → "(未知id：XXX - 描述)"
        未识别 → "(识别中...)"
        """
        with self._lock:
            if track_id not in self._tracks:
                return "(等待识别...)"

            info = self._tracks[track_id]

            if not info.recognized:
                return "(识别中...)" if info.pending else "(等待识别...)"

            if info.db_matched:
                return f"(库内确定id：{info.db_match_id})"

            # 未匹配数据库
            label = info.hull_number or "未知"
            desc_short = info.description[:20] if info.description else ""
            if desc_short:
                return f"(未知id：{label} - {desc_short})"
            return f"(未知id：{label})"

    def cleanup_stale(self, current_frame: int) -> int:
        """
        清理长时间未出现的 track。

        Returns:
            清理的数量。
        """
        with self._lock:
            stale_ids = [
                tid for tid, info in self._tracks.items()
                if current_frame - info.last_seen_frame > self._max_stale_frames
            ]
            for tid in stale_ids:
                del self._tracks[tid]

        if stale_ids:
            logger.info("清理 %d 个过期 track: %s", len(stale_ids), stale_ids)

        return len(stale_ids)

    def get(self, track_id: int) -> TrackInfo | None:
        """只读查询 track，不存在返回 None（不创建）。"""
        with self._lock:
            return self._tracks.get(track_id)

    @property
    def active_tracks(self) -> dict[int, TrackInfo]:
        """返回当前所有 track 的快照（线程安全副本）。"""
        with self._lock:
            return dict(self._tracks)

    def __len__(self) -> int:
        with self._lock:
            return len(self._tracks)
