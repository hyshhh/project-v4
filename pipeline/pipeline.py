"""
ShipPipeline — 主流水线编排

级联模式（concurrent_mode=false）：
  YOLO 检测 → Agent 三步链路（识别→精确查找→语义检索） → 绑定结果 → 绘制输出

并发模式（concurrent_mode=true）：
  YOLO 检测 → crop 送入队列 → Agent 独立线程异步推理
  → 结果按帧时间戳严格顺序出队 → 匹配到对应帧绘制输出

双层并发架构：
  外层：帧级任务队列（max_queued_frames 限制深度，防 OOM）
  内层：crop 级 API 并发（max_concurrent 控制）
"""

from __future__ import annotations

import base64
import logging
import queue
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np

from agent import AgentResult
from pipeline.detector import ShipDetector, Detection
from pipeline.demo import DemoRenderer
from pipeline.output import ScreenshotSaver
from pipeline.fps import FPSMeter
from pipeline.tracker import TrackManager
from pipeline.video_input import InputSource

logger = logging.getLogger(__name__)


class ShipPipeline:
    """
    船弦号识别视频处理流水线。

    整合 YOLO 检测、Agent 推理、跟踪管理，支持级联/并发双模式。
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Args:
            config: 全局配置字典。None 则从 config.yaml 加载。
        """
        if config is None:
            from config import load_config
            config = load_config()

        self._config = config

        # 读取 pipeline 相关配置
        pipe_cfg = config.get("pipeline", {})
        self._concurrent_mode: bool = pipe_cfg.get("concurrent_mode", False)
        self._max_queued_frames: int = pipe_cfg.get("max_queued_frames", 30)
        self._process_every_n: int = max(1, pipe_cfg.get("process_every_n_frames", 1))
        self._demo_enabled: bool = pipe_cfg.get("demo", False)
        self._use_agent: bool = pipe_cfg.get("use_agent", False)

        # 读取 Agent 数据库配置
        from database import ShipDatabase
        self._db = ShipDatabase(config=config)

        # Agent 模式：初始化 LangChain ReAct Agent
        self._agent = None
        if self._use_agent:
            from agent import create_agent
            self._agent = create_agent(config=config)
            logger.info("Agent 模式已启用：使用 LangChain ReAct Agent 三步工具链")

        # 初始化组件
        self._detector = ShipDetector(
            model_path=pipe_cfg.get("yolo_model", "yolov8n.pt"),
            device=pipe_cfg.get("device", ""),
            conf_threshold=pipe_cfg.get("conf_threshold", 0.25),
            tracker_type=pipe_cfg.get("tracker", "bytetrack"),
            tracker_params=pipe_cfg.get("tracker_params"),
            classes=pipe_cfg.get("detect_classes", [8]),  # COCO: 8=boat
        )

        self._tracker = TrackManager(
            max_stale_frames=pipe_cfg.get("max_stale_frames", 300),
        )

        self._fps = FPSMeter(window_seconds=10.0)

        # Demo 渲染器
        self._renderer = DemoRenderer(
            show_fps=True,
            show_track_id=True,
        )

        # 截图保存器
        output_dir = pipe_cfg.get("output_dir", "./output")
        self._saver = ScreenshotSaver(output_dir=output_dir)

        # 并发模式相关
        self._task_queue: queue.Queue = queue.Queue(
            maxsize=self._max_queued_frames
        )
        self._result_queue: queue.Queue = queue.Queue()
        self._agent_workers: list[threading.Thread] = []
        self._stop_event = threading.Event()

        # Agent 运行链路日志（限制最大条数防内存泄漏）
        self._agent_trace: list[dict[str, Any]] = []
        self._trace_lock = threading.Lock()
        self._max_trace_entries = 500

        logger.info(
            "ShipPipeline 初始化: mode=%s, inference=%s, process_every=%d",
            "concurrent" if self._concurrent_mode else "cascade",
            "agent" if self._use_agent else "hardcoded",
            self._process_every_n,
        )

    # ── Agent 链路日志 ──────────────────────────

    def _log_agent_trace(
        self,
        event_type: str,
        track_id: int,
        frame_id: int,
        content: str = "",
        **extra: Any,
    ) -> None:
        """记录 Agent 运行链路到 trace 日志。"""
        entry = {
            "type": event_type,
            "track_id": track_id,
            "frame_id": frame_id,
            "content": content,
            "timestamp": time.time(),
            **extra,
        }
        with self._trace_lock:
            self._agent_trace.append(entry)
            # 超过上限时截断，保留最近一半
            if len(self._agent_trace) > self._max_trace_entries:
                self._agent_trace = self._agent_trace[-(self._max_trace_entries // 2):]

        logger.info(
            "[AgentTrace] %s | track=%d frame=%d | %s%s",
            event_type.upper(),
            track_id,
            frame_id,
            content[:100] if content else "",
            f" | {extra}" if extra else "",
        )

    # ── 工具方法 ──────────────────────────────────

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        """将 BGR 图像编码为 base64 字符串。"""
        success, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            raise RuntimeError("图像编码失败")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _run_three_step_chain(self, crop: np.ndarray) -> AgentResult:
        """
        执行三步链路：VLM识别 → 精确查找 → 语义检索。

        第1步: _vlm_infer(crop) → 弦号 + 描述
        第2步: db.lookup(hull_number) → 有弦号时精确查找
        第3步: db.semantic_search_filtered(description) → 弦号未匹配或无弦号时语义检索
        """
        from tools import _vlm_infer

        # 第一步：VLM 识别
        crop_b64 = self._encode_image(crop)
        vlm_result = _vlm_infer(crop_b64)
        hull_number = vlm_result.get("hull_number", "")
        description = vlm_result.get("description", "")

        if not hull_number and not description:
            return AgentResult(answer="VLM 未返回结果")

        # 第二步：精确查找（有弦号时）
        exact_matched = False
        semantic_ids: list[str] = []

        if hull_number:
            desc_in_db = self._db.lookup(hull_number)
            if desc_in_db is not None:
                exact_matched = True
                description = description or desc_in_db
            elif description:
                # 第三步：语义检索（弦号未匹配时，用描述检索）
                results = self._db.semantic_search_filtered(description)
                semantic_ids = [r["hull_number"] for r in results if r.get("hull_number")]
        elif description:
            # 无弦号，直接第三步：语义检索
            results = self._db.semantic_search_filtered(description)
            semantic_ids = [r["hull_number"] for r in results if r.get("hull_number")]

        return AgentResult(
            hull_number=hull_number,
            description=description,
            match_type="exact" if exact_matched else ("semantic" if semantic_ids else "none"),
            semantic_match_ids=semantic_ids,
        )

    def _run_agent_chain(self, crop: np.ndarray) -> AgentResult:
        """
        使用外层 Agent 的三步工具链执行识别。

        直接调用 Agent 的工具函数（recognize_ship → lookup → retrieve），
        而非走 ReAct 循环（base64 文本化会导致 token 超限）。

        工具函数内部正确地以 image_url 格式发送图片给 VLM API。
        """
        if self._agent is None:
            raise RuntimeError("Agent 模式未初始化（use_agent=True 但 agent 为 None）")

        from tools import build_tools

        # 获取 Agent 的三个工具函数
        tools_list = build_tools(self._db)
        tool_map = {t.name: t for t in tools_list}

        # 第一步：recognize_ship — 调用 VLM 识别弦号+描述
        crop_b64 = self._encode_image(crop)
        recognize_tool = tool_map["recognize_ship"]
        recognize_result_str = recognize_tool.invoke({"image_base64": crop_b64})

        try:
            import json as _json
            vlm_data = _json.loads(recognize_result_str)
        except Exception:
            vlm_data = {"hull_number": "", "description": ""}

        hull_number = vlm_data.get("hull_number", "")
        description = vlm_data.get("description", "")

        self._log_agent_trace(
            "agent_tool_recognize",
            track_id=0,
            frame_id=0,
            content=f"VLM 识别: 弦号={hull_number or '(无)'} 描述={description[:50]}",
        )

        if not hull_number and not description:
            return AgentResult(answer="VLM 未返回结果")

        # 第二步：lookup_by_hull_number — 有弦号时精确查找
        exact_matched = False
        semantic_ids: list[str] = []

        if hull_number:
            lookup_tool = tool_map["lookup_by_hull_number"]
            lookup_result_str = lookup_tool.invoke({"hull_number": hull_number})
            try:
                lookup_data = _json.loads(lookup_result_str)
            except Exception:
                lookup_data = {}

            self._log_agent_trace(
                "agent_tool_lookup",
                track_id=0,
                frame_id=0,
                content=f"精确查找: found={lookup_data.get('found', False)}",
            )

            if lookup_data.get("found"):
                exact_matched = True
                description = lookup_data.get("description", description)
            elif description:
                # 第三步：retrieve_by_description — 弦号未匹配，语义检索
                retrieve_tool = tool_map["retrieve_by_description"]
                retrieve_result_str = retrieve_tool.invoke({"target_description": description})
                try:
                    retrieve_data = _json.loads(retrieve_result_str)
                    results = retrieve_data.get("results", [])
                    semantic_ids = [r["hull_number"] for r in results if r.get("hull_number")]
                except Exception:
                    pass

                self._log_agent_trace(
                    "agent_tool_retrieve",
                    track_id=0,
                    frame_id=0,
                    content=f"语义检索: 候选={semantic_ids}",
                )
        elif description:
            # 无弦号，直接第三步：语义检索
            retrieve_tool = tool_map["retrieve_by_description"]
            retrieve_result_str = retrieve_tool.invoke({"target_description": description})
            try:
                retrieve_data = _json.loads(retrieve_result_str)
                results = retrieve_data.get("results", [])
                semantic_ids = [r["hull_number"] for r in results if r.get("hull_number")]
            except Exception:
                pass

            self._log_agent_trace(
                "agent_tool_retrieve",
                track_id=0,
                frame_id=0,
                content=f"语义检索: 候选={semantic_ids}",
            )

        return AgentResult(
            hull_number=hull_number,
            description=description,
            match_type="exact" if exact_matched else ("semantic" if semantic_ids else "none"),
            semantic_match_ids=semantic_ids,
        )

    def _run_recognition(self, crop: np.ndarray) -> AgentResult:
        """
        统一识别调度：根据 use_agent 配置选择硬编码链路或 Agent 工具链。

        - use_agent=False → _run_three_step_chain（直接调用 VLM HTTP API + 查库 + 语义检索）
        - use_agent=True  → _run_agent_chain（调用外层 Agent 的 3 个工具函数：recognize_ship → lookup → retrieve）
        """
        if self._use_agent:
            return self._run_agent_chain(crop)
        return self._run_three_step_chain(crop)

    # ── 推理结果处理 ────────────────────────────

    def _handle_agent_result(
        self,
        track_id: int,
        frame_id: int,
        agent_result: AgentResult,
    ) -> None:
        """处理 Agent 三步链路结果：绑定到 track。"""
        self._log_agent_trace(
            "agent_result",
            track_id=track_id,
            frame_id=frame_id,
            content=(
                f"弦号={agent_result.hull_number or '(未识别)'} "
                f"匹配={agent_result.match_type} "
                f"语义候选={agent_result.semantic_match_ids}"
            ),
        )

        # 绑定识别结果
        self._tracker.bind_result(
            track_id,
            agent_result.hull_number,
            agent_result.description,
        )

        if agent_result.match_type == "exact":
            # 精确匹配 → 绿色
            desc = self._db.lookup(agent_result.hull_number) or ""
            self._tracker.bind_db_match(track_id, agent_result.hull_number, desc)
        elif agent_result.semantic_match_ids:
            # 有语义匹配候选 → 黄色或红色
            self._tracker.bind_semantic_matches(track_id, agent_result.semantic_match_ids)

    def _handle_agent_error(
        self,
        track_id: int,
        frame_id: int,
        error: str,
    ) -> None:
        """处理 Agent 推理错误。"""
        logger.warning("Agent 推理出错 (track=%d, frame=%d): %s", track_id, frame_id, error)
        self._tracker.cancel_pending(track_id)

    # ── 级联模式 ────────────────────────────────

    def _cascade_process(
        self,
        detections: list[Detection],
        frame_id: int,
    ) -> None:
        """级联模式：同步处理每个需要识别的检测目标。"""
        for det in detections:
            if not self._tracker.needs_recognition(det.track_id):
                continue

            if det.crop is None or det.crop.size == 0:
                continue

            self._tracker.mark_pending(det.track_id)

            self._log_agent_trace(
                "cascade_infer_start",
                track_id=det.track_id,
                frame_id=frame_id,
                content="同步推理开始",
            )

            try:
                agent_result = self._run_recognition(det.crop)
                self._log_agent_trace(
                    "recognition_result",
                    track_id=det.track_id,
                    frame_id=frame_id,
                    content=(
                        f"弦号={agent_result.hull_number or '(无)'} "
                        f"匹配={agent_result.match_type} "
                        f"语义候选={agent_result.semantic_match_ids}"
                    ),
                )
                self._handle_agent_result(det.track_id, frame_id, agent_result)
            except Exception as e:
                self._handle_agent_error(det.track_id, frame_id, str(e))

    # ── 并发模式 ────────────────────────────────

    def _concurrent_process(
        self,
        detections: list[Detection],
        frame_id: int,
    ) -> None:
        """并发模式：将 crop 送入队列，Agent 异步推理。"""
        for det in detections:
            if not self._tracker.needs_recognition(det.track_id):
                continue

            if det.crop is None or det.crop.size == 0:
                continue

            # 标记为 pending
            self._tracker.mark_pending(det.track_id)

            task = {
                "frame_id": frame_id,
                "timestamp": time.time(),
                "track_id": det.track_id,
                "crop": det.crop.copy(),
            }

            try:
                self._task_queue.put_nowait(task)
                self._log_agent_trace(
                    "concurrent_enqueue",
                    track_id=det.track_id,
                    frame_id=frame_id,
                    content=f"送入异步队列 (队列深度: {self._task_queue.qsize()})",
                )
            except queue.Full:
                logger.warning(
                    "任务队列已满 (%d)，丢弃 frame=%d track=%d",
                    self._max_queued_frames, frame_id, det.track_id,
                )
                # 取消 pending 状态
                self._tracker.cancel_pending(det.track_id)

    def _agent_worker_loop(self) -> None:
        """Agent 工作线程：从队列取任务并推理。"""
        while not self._stop_event.is_set():
            try:
                task = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            track_id = task["track_id"]
            frame_id = task["frame_id"]
            crop = task["crop"]

            self._log_agent_trace(
                "concurrent_infer_start",
                track_id=track_id,
                frame_id=frame_id,
                content="异步推理开始",
            )

            try:
                agent_result = self._run_recognition(crop)
            except Exception as e:
                logger.exception("Agent 推理异常 (track=%d, frame=%d)", track_id, frame_id)
                agent_result = AgentResult(answer=str(e))

            self._result_queue.put({
                "frame_id": frame_id,
                "track_id": track_id,
                "agent_result": agent_result,
            })

    def _drain_results(self) -> int:
        """排空结果队列，处理所有已完成的异步推理结果。返回处理数量。"""
        count = 0
        while True:
            try:
                pending = self._result_queue.get_nowait()
                track_id = pending["track_id"]
                frame_id = pending["frame_id"]
                agent_result = pending["agent_result"]

                if agent_result.hull_number or agent_result.semantic_match_ids or agent_result.match_type != "none":
                    self._handle_agent_result(track_id, frame_id, agent_result)
                else:
                    self._handle_agent_error(track_id, frame_id, agent_result.answer or "Agent 无结果")
                count += 1
            except queue.Empty:
                break
        return count

    def _start_agent_workers(self) -> None:
        """启动 Agent 工作线程池。"""
        self._stop_event.clear()
        self._agent_workers.clear()
        for i in range(self._max_concurrent):
            worker = threading.Thread(
                target=self._agent_worker_loop,
                name=f"agent-worker-{i}",
                daemon=True,
            )
            worker.start()
            self._agent_workers.append(worker)
        logger.info("启动 %d 个 Agent 工作线程", self._max_concurrent)

    def _stop_agent_workers(self) -> None:
        """停止 Agent 工作线程，等待全部完成。"""
        self._stop_event.set()
        for worker in self._agent_workers:
            worker.join(timeout=10.0)
            if worker.is_alive():
                logger.warning("工作线程 %s 未在超时内退出", worker.name)
        self._agent_workers.clear()

        # workers 全部退出后再排空，避免竞态丢结果
        remaining_tasks = 0
        while not self._task_queue.empty():
            try:
                self._task_queue.get_nowait()
                remaining_tasks += 1
            except queue.Empty:
                break
        if remaining_tasks:
            logger.info("清理 %d 个未处理任务", remaining_tasks)

        remaining_results = self._drain_results()
        if remaining_results:
            logger.info("处理 %d 个残留结果", remaining_results)

        logger.info("Agent 工作线程已停止")

    # ── 渲染 ────────────────────────────────────

    def _render_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        frame_id: int,
    ) -> np.ndarray:
        """通过 DemoRenderer 在帧上绘制检测框、识别结果和 HUD。"""
        return self._renderer.render(
            frame=frame,
            detections=detections,
            tracks=self._tracker.active_tracks,
            fps_info=self._fps.get_all_fps(),
            frame_id=frame_id,
            queue_depth=self._task_queue.qsize(),
            max_queue=self._max_queued_frames,
        )

    # ── 主流程 ──────────────────────────────────

    def process(
        self,
        source: str | int,
        output_path: str | None = None,
        display: bool = False,
        max_frames: int = 0,
        frame_callback: Callable[[np.ndarray, int], None] | None = None,
    ) -> dict[str, Any]:
        """
        运行完整的视频处理流水线。

        Args:
            source: 视频输入源（文件路径/相机号/RTSP URL）。
            output_path: 输出视频路径（可选）。
            display: 是否实时显示窗口（仅本地有显示器时有效）。
            max_frames: 最大处理帧数，0 表示不限制。
            frame_callback: 每帧处理完成后的回调函数 callback(frame, frame_id)。

        Returns:
            统计信息字典。
        """
        input_src = InputSource(source)
        video_writer = None
        last_detections: list[Detection] = []

        try:
            # 初始化视频写入器
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    output_path, fourcc,
                    input_src.source_fps,
                    (input_src.width, input_src.height),
                )
                if not video_writer.isOpened():
                    logger.error("无法创建输出视频: %s", output_path)
                    video_writer = None
                else:
                    logger.info("输出视频: %s", output_path)

            # 启动并发 worker
            if self._concurrent_mode:
                self._start_agent_workers()

            frame_id = 0
            total_detections = 0
            start_time = time.time()

            logger.info(
                "开始处理: source=%s, mode=%s, inference=%s, demo=%s",
                source,
                "concurrent" if self._concurrent_mode else "cascade",
                "agent" if self._use_agent else "hardcoded",
                self._demo_enabled,
            )

            while True:
                ret, frame = input_src.read()
                if not ret:
                    logger.info("视频源结束或读取失败")
                    break

                frame_id += 1
                if max_frames > 0 and frame_id > max_frames:
                    break

                # FPS 统计
                self._fps.tick("stream")

                # ── 每帧都进行 YOLO 检测 + track 注册，保持 ByteTrack 连续 ──
                try:
                    detections = self._detector.detect(frame, frame_id)
                except Exception as e:
                    logger.error("YOLO 检测异常 (frame=%d): %s", frame_id, e)
                    detections = []

                last_detections = detections
                total_detections += len(detections)

                # 注册/更新 track（每帧执行，保持跟踪状态）
                for det in detections:
                    self._tracker.get_or_create(det.track_id, frame_id)

                # ── process_every_n_frames 控制 Agent 推理频率 ──
                should_process = (frame_id % self._process_every_n == 0)

                if should_process:
                    # Agent 推理（级联或并发）
                    if self._concurrent_mode:
                        self._concurrent_process(detections, frame_id)
                    else:
                        self._cascade_process(detections, frame_id)

                    # 并发模式下排空已完成的结果
                    if self._concurrent_mode:
                        self._drain_results()

                # 清理过期 track
                self._tracker.cleanup_stale(frame_id)

                # 渲染输出
                if self._demo_enabled or output_path or display:
                    display_frame = self._render_frame(frame, last_detections, frame_id)
                else:
                    display_frame = frame

                # 每 N 帧：有已识别的 track 就保存截图
                if should_process:
                    has_recognized = any(
                        t.hull_number for t in self._tracker.active_tracks.values() if t.recognized
                    )
                    if has_recognized:
                        self._saver.save(display_frame, frame_id)

                # 写入输出视频
                if video_writer:
                    video_writer.write(display_frame)

                # 回调
                if frame_callback:
                    frame_callback(display_frame, frame_id)

                # 实时显示
                if display:
                    cv2.imshow("Ship Pipeline", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("用户按下 q，停止处理")
                        break

                # 处理 FPS
                self._fps.tick("process")

                # 定期打印 FPS（每 5 秒）
                if self._fps.should_print("stream"):
                    elapsed = time.time() - start_time
                    self._fps.print_fps(
                        "stream",
                        extra=f"frames={frame_id}, elapsed={elapsed:.0f}s, tracks={len(self._tracker)}",
                    )
                    self._fps.print_fps("process")

                # 定期打印 Agent 运行链路摘要（每 100 帧）
                if frame_id % 100 == 0:
                    self._print_trace_summary()

            # ── 处理完成，收集统计 ──

            # 并发模式下最终排空结果
            if self._concurrent_mode:
                self._drain_results()

            elapsed = time.time() - start_time
            tracks = self._tracker.active_tracks
            total_recognized = sum(1 for t in tracks.values() if t.recognized)

            stats = {
                "total_frames": frame_id,
                "total_detections": total_detections,
                "total_tracks": len(tracks),
                "recognized_tracks": total_recognized,
                "elapsed_seconds": round(elapsed, 1),
                "avg_fps": round(frame_id / elapsed, 1) if elapsed > 0 else 0,
                "mode": "concurrent" if self._concurrent_mode else "cascade",
                "inference": "agent" if self._use_agent else "hardcoded",
                "screenshots_saved": self._saver.saved_count,
            }

            logger.info("=" * 50)
            logger.info("处理完成统计:")
            logger.info("  总帧数: %d", stats["total_frames"])
            logger.info("  总检测数: %d", stats["total_detections"])
            logger.info("  跟踪目标数: %d", stats["total_tracks"])
            logger.info("  已识别: %d", stats["recognized_tracks"])
            logger.info("  耗时: %.1fs", stats["elapsed_seconds"])
            logger.info("  平均 FPS: %.1f", stats["avg_fps"])
            logger.info("  模式: %s", stats["mode"])
            logger.info("=" * 50)

            return stats

        except KeyboardInterrupt:
            logger.info("用户中断处理")
            elapsed = time.time() - start_time
            return {
                "total_frames": frame_id,
                "total_detections": total_detections,
                "total_tracks": len(self._tracker),
                "recognized_tracks": 0,
                "elapsed_seconds": round(elapsed, 1),
                "avg_fps": round(frame_id / elapsed, 1) if elapsed > 0 else 0,
                "mode": "concurrent" if self._concurrent_mode else "cascade",
                "interrupted": True,
            }

        finally:
            if self._concurrent_mode:
                self._stop_agent_workers()
            input_src.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()

    # ── 链路摘要 ────────────────────────────────

    def _print_trace_summary(self) -> None:
        """打印 Agent 运行链路摘要。"""
        with self._trace_lock:
            if not self._agent_trace:
                return
            recent = self._agent_trace[-20:]
            logger.info("=== Agent 运行链路摘要 (最近 %d 条) ===", len(recent))
            for entry in recent:
                logger.info(
                    "  [%s] track=%d frame=%d: %s",
                    entry["type"],
                    entry["track_id"],
                    entry["frame_id"],
                    entry.get("content", "")[:80],
                )

    @property
    def agent_trace(self) -> list[dict[str, Any]]:
        """获取完整的 Agent 运行链路日志。"""
        with self._trace_lock:
            return list(self._agent_trace)

    # ── 运行时控制 ──────────────────────────────

    def set_demo(self, enabled: bool) -> None:
        """设置 demo 开关。"""
        self._demo_enabled = enabled
        logger.info("Demo 模式: %s", "开启" if enabled else "关闭")

    def set_use_agent(self, enabled: bool) -> None:
        """设置 Agent 模式开关。"""
        if enabled and self._agent is None:
            from agent import create_agent
            self._agent = create_agent(config=self._config)
            logger.info("Agent 模式已启用：初始化 LangChain ReAct Agent")
        self._use_agent = enabled
        logger.info("推理模式: %s", "Agent (LangChain)" if enabled else "硬编码 (直接调用)")

    def switch_to_concurrent(self, enabled: bool) -> None:
        """动态切换级联/并发模式。"""
        self._concurrent_mode = enabled
        logger.info("切换为 %s 模式", "并发" if enabled else "级联")
