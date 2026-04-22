# QUICKSTART — 常用命令速查

## 启动服务

### 1. LLM 推理服务（Qwen3-VL-4B-AWQ）

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /media/ddc/新加卷/hys/hysnew/Qwen3.5-2B-AWQ \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 10240 \
  --port 7890 \
  --gpu-memory-utilization 0.15 \
  --max-num-seqs 10 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

### 2. Embedding 服务（Qwen3-Embedding-0.6B）

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen3-Embedding-0.6B \
  --api-key abc123 \
  --served-model-name Qwen3-Embedding-0.6B \
  --convert embed \
  --gpu-memory-utilization 0.15 \
  --max-model-len 2048 \
  --port 7891
```

---

## 视频处理

### 基本用法

```bash
# 处理视频 + 输出结果（默认硬编码模式）
python -m pipeline.cli /media/ddc/新加卷/hys/hysnew2/学习/1.mp4 \
  --demo \
  --output /media/ddc/新加卷/hys/hysnew2/学习/result.mp4

# 开启定时刷新（每150帧重新识别已跟踪船只）
python -m pipeline.cli /media/ddc/新加卷/hys/hysnew2/学习/1.mp4 \
  --demo --enable-refresh --output result.mp4

# 定时刷新 + 自定义间隔（每100帧刷新一次）
python -m pipeline.cli /media/ddc/新加卷/hys/hysnew2/学习/1.mp4 \
  --demo --enable-refresh --gap-num 100 --output result.mp4
```

### Agent 模式（LangChain 三步工具链）

```bash
# 使用 Agent 模式（recognize_ship → lookup → retrieve）
python -m pipeline.cli /media/ddc/新加卷/hys/hysnew2/学习/1.mp4 \
  --agent --demo --output result.mp4

# 强制硬编码模式
python -m pipeline.cli /media/ddc/新加卷/hys/hysnew2/学习/1.mp4 \
  --no-agent --demo --output result.mp4
```

### 实时显示

```bash
# 弹窗实时看检测效果
python -m pipeline.cli /media/ddc/新加卷/hys/hysnew2/学习/1.mp4 \
  --demo --display -v
```

### 并发模式（更快）

```bash
python -m pipeline.cli /media/ddc/新加卷/hys/hysnew2/学习/1.mp4 \
  --demo \
  --output result.mp4 \
  --concurrent \
  --max-concurrent 8 \
  --prompt-mode brief
```

### 快速测试（只跑50帧）

```bash
python -m pipeline.cli /media/ddc/新加卷/hys/hysnew2/学习/1.mp4 \
  --demo --output result.mp4 \
  --max-frames 50 \
  --prompt-mode brief -v
```

### 摄像头 / RTSP

```bash
# USB 摄像头
python -m pipeline.cli 0 --demo --display

# RTSP 流
python -m pipeline.cli rtsp://192.168.1.100/stream --demo --display
```

---

## 常用参数速查

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--demo` | 开启可视化（画检测框） | 关 |
| `--display` | 实时弹窗显示 | 关 |
| `-o / --output` | 输出视频路径 | 无 |
| `-c / --concurrent` | 并发模式 | 级联模式 |
| `--agent` | Agent 模式（LangChain 三步工具链） | 关 |
| `--no-agent` | 硬编码模式（直接调用 VLM+查库+检索） | — |
| `--enable-refresh` | 开启定时刷新（间隔帧数重新识别已跟踪船只） | 关 |
| `--gap-num N` | 定时刷新间隔帧数 | 150 |
| `--max-concurrent N` | 并发 Agent 数 | 4 |
| `--max-frames N` | 最大处理帧数（0=不限） | 0 |
| `--process-every N` | 每N帧处理一次 | 1 |
| `--prompt-mode` | 提示词：`detailed` / `brief` | detailed |
| `--yolo-model` | YOLO 模型路径 | yolov8n.pt |
| `--device` | 推理设备（`cpu` / `0`） | 自动 |
| `--conf` | 检测置信度阈值 | 0.25 |
| `-v` | 详细日志 | 关 |

---

## config.yaml 关键配置

### Agent/硬编码模式切换

```yaml
pipeline:
  # false = 硬编码模式（直接调用 VLM + 查库 + 语义检索）
  # true  = Agent 模式（LangChain ReAct Agent 编排 3 个工具）
  use_agent: false

  # 定时刷新：true 则每隔 gap_num 帧重新识别已跟踪的船
  # false 则仅新 track 出现时识别一次（现有逻辑不变）
  enable_refresh: false
  gap_num: 150
```

### Tracker 调参

```yaml
pipeline:
  tracker: "bytetrack"          # 或 "botsort"
  tracker_params:
    track_high_thresh: 0.5      # 高置信度匹配阈值
    track_low_thresh: 0.1       # 低置信度匹配阈值
    new_track_thresh: 0.6       # 新轨迹创建阈值
    track_buffer: 30            # 丢失后保留帧数
    match_thresh: 0.8           # IoU 匹配阈值
```

### 检测配置

```yaml
pipeline:
  conf_threshold: 0.25          # 检测置信度
  detect_classes: [8]           # COCO 8=船, null=所有
  max_stale_frames: 300         # 过期 track 清理帧数
```

### Embedding 配置

```yaml
embed:
  model: "Qwen3-Embedding-0.6B"
  api_key: "abc123"
  base_url: "http://localhost:7891/v1"
```

---

## 数据库

```bash
# 重建向量库（换了 embedding 模型后必须做）
rm -rf vector_store/

# 查看船只数据
cat data/ships.csv
```

---

## 推理模式对比

| | 硬编码模式 (默认) | Agent 模式 |
|---|---|---|
| 调用方式 | 直接调 VLM → 查库 → 语义检索 | LangChain ReAgent 编排 3 个工具 |
| 优势 | 快速、可控、无额外 LLM 调用 | 灵活、可扩展、Agent 自动决策跳步 |
| 适用 | 固定流程、追求速度 | 需要 Agent 灵活编排的场景 |
| config | `use_agent: false` | `use_agent: true` |
| CLI | `--no-agent` 或默认 | `--agent` |

---

## 运行时快捷键

显示窗口下：
- **`q`** — 退出
- **`d`** — 切换 detailed / brief 提示词
- **`p`** — 暂停 / 继续
- **`s`** — 截图
