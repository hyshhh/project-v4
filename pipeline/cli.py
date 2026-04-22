"""
Pipeline CLI — 视频处理命令行入口

用法:
    python -m pipeline.cli <source> [options]

示例:
    python -m pipeline.cli video.mp4
    python -m pipeline.cli 0                          # USB 相机
    python -m pipeline.cli rtsp://192.168.1.100/stream
    python -m pipeline.cli video.mp4 --demo --output result.mp4
    python -m pipeline.cli video.mp4 --concurrent --max-concurrent 8
"""

from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ship-pipeline",
        description="🚢 船弦号识别视频处理流水线",
    )

    parser.add_argument(
        "source",
        help="视频输入源：文件路径 / 相机号(0,1,...) / RTSP URL",
    )

    parser.add_argument(
        "--output", "-o",
        help="输出视频路径（如 result.mp4）",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="开启 demo 模式（在输出视频上绘制检测框和识别结果）",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="实时显示窗口（需要有显示器）",
    )

    parser.add_argument(
        "--concurrent", "-c",
        action="store_true",
        help="使用并发模式（默认级联模式）",
    )

    parser.add_argument(
        "--agent",
        action="store_true",
        default=None,
        help="使用 LangChain Agent 模式（三步工具链编排）",
    )

    parser.add_argument(
        "--no-agent",
        action="store_true",
        default=None,
        help="使用硬编码模式（直接调用 VLM + 查库 + 语义检索）",
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="最大并发 Agent 推理数（默认 4）",
    )

    parser.add_argument(
        "--max-queued-frames",
        type=int,
        default=30,
        help="最大队列深度（默认 30，防止 OOM）",
    )

    parser.add_argument(
        "--process-every",
        type=int,
        default=1,
        help="每 N 帧处理一次（默认 1 = 每帧都处理）",
    )

    parser.add_argument(
        "--enable-refresh",
        action="store_true",
        default=None,
        help="开启定时刷新（每隔 gap_num 帧重新识别已跟踪的船只）",
    )

    parser.add_argument(
        "--no-refresh",
        action="store_true",
        default=None,
        help="关闭定时刷新（仅新 track 时识别，保持原有逻辑）",
    )

    parser.add_argument(
        "--gap-num",
        type=int,
        default=None,
        help="定时刷新间隔帧数（默认 150，仅 --enable-refresh 时生效）",
    )

    parser.add_argument(
        "--prompt-mode",
        choices=["detailed", "brief"],
        default="detailed",
        help="提示词模式：detailed（详细）或 brief（简略）",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="最大处理帧数（0 = 不限制）",
    )

    parser.add_argument(
        "--yolo-model",
        default="yolov8n.pt",
        help="YOLO 模型路径（默认 yolov8n.pt）",
    )

    parser.add_argument(
        "--device",
        default="",
        help="推理设备（默认自动选择，'cpu' 强制 CPU）",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="检测置信度阈值（默认 0.25）",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from config import load_config
    from pipeline.pipeline import ShipPipeline

    config = load_config()

    # 合并命令行参数到配置
    config.setdefault("pipeline", {})
    config["pipeline"]["concurrent_mode"] = args.concurrent
    config["pipeline"]["max_concurrent"] = args.max_concurrent
    config["pipeline"]["max_queued_frames"] = args.max_queued_frames
    config["pipeline"]["process_every_n_frames"] = args.process_every
    config["pipeline"]["prompt_mode"] = args.prompt_mode
    config["pipeline"]["demo"] = args.demo
    config["pipeline"]["yolo_model"] = args.yolo_model
    config["pipeline"]["device"] = args.device
    config["pipeline"]["conf_threshold"] = args.conf

    # 处理 --agent / --no-agent 开关（命令行优先于 config）
    if args.agent is not None:
        config["pipeline"]["use_agent"] = args.agent
    elif args.no_agent is not None:
        config["pipeline"]["use_agent"] = not args.no_agent

    # 处理 --enable-refresh / --no-refresh 开关（命令行优先于 config）
    if args.enable_refresh is not None:
        config["pipeline"]["enable_refresh"] = args.enable_refresh
    elif args.no_refresh is not None:
        config["pipeline"]["enable_refresh"] = not args.no_refresh

    # 处理 --gap-num
    if args.gap_num is not None:
        config["pipeline"]["gap_num"] = max(1, args.gap_num)

    # 读取最终配置
    use_agent = config["pipeline"].get("use_agent", False)
    enable_refresh = config["pipeline"].get("enable_refresh", False)
    gap_num = config["pipeline"].get("gap_num", 150)

    # 显示启动信息
    console.print(Panel(
        f"[bold]🚢 船弦号识别视频流水线[/bold]\n\n"
        f"输入源: [cyan]{args.source}[/cyan]\n"
        f"模式: [{'green' if args.concurrent else 'yellow'}]"
        f"{'并发' if args.concurrent else '级联'}[/]\n"
        f"推理: [{'magenta' if use_agent else 'blue'}]"
        f"{'Agent (LangChain)' if use_agent else '硬编码 (直接调用)'}[/]\n"
        f"并发数: {args.max_concurrent}\n"
        f"定时刷新: {'[green]开启[/green] (每%d帧)' % gap_num if enable_refresh else '[dim]关闭[/dim]'}\n"
        f"提示词: {args.prompt_mode}\n"
        f"Demo: {'[green]开启[/green]' if args.demo else '[dim]关闭[/dim]'}\n"
        f"YOLO: {args.yolo_model}",
        title="启动配置",
    ))

    # 创建并运行流水线
    try:
        pipeline = ShipPipeline(config=config)
        stats = pipeline.process(
            source=args.source,
            output_path=args.output,
            display=args.display,
            max_frames=args.max_frames,
        )

        # 显示统计结果
        table = Table(title="📊 处理统计")
        table.add_column("指标", style="cyan")
        table.add_column("值", style="white")

        for key, value in stats.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        console.print(table)

    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断[/yellow]")
    except Exception as e:
        console.print(f"\n[red]错误: {e}[/red]")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
