# sam3.1_head_mask

Privacy-preserving **head / face occlusion** for robot demonstration data
using **Meta SAM 3.1** (the [Object-Multiplex](https://ai.meta.com/blog/segment-anything-model-3/) update of SAM 3).

Two pipelines are shipped:

| 入口 | 输入 → 输出 | 典型用途 |
|---|---|---|
| `scripts/run_video.py`<br>`scripts/batch_video.py` | `*.mp4` → `*.mp4` | 普通视频（ARIO 等）整段做人头打码 |
| `scripts/run_h5.py`<br>`scripts/batch_h5.py` | `*.hdf5` → `*.hdf5` | 机器人数据集，每个相机里存 **JPEG 字节流拼接**（`images_dict/<cam>/rgb` + `rgb_size`） |

仓库本身**不包含任何对象存储 / 公司内部存储的代码** —— 数据请自行下载到本地后再调用。

> 单文本提示词、视频级追踪、多目标共享一次 forward —— 这些都是 SAM 3.1 直接提供的能力，仓库只是把它**包装成"把这片像素糊掉"** 这一个具体功能。

---

## 1. 安装

依赖 Python 3.10+ 与 CUDA-compatible GPU（CUDA 12.x，PyTorch ≥ 2.7）。

```bash
# 1. 创建环境（推荐隔离的 venv 或 conda 环境）
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 2. 安装本仓库 + 依赖
git clone https://github.com/MrLuckyFive/sam3.1_head_mask.git
cd sam3.1_head_mask
pip install -e .

# 3. 安装上游 SAM 3 / 3.1 代码（这一步独立于本仓库；权重也由它指定）
pip install "git+https://github.com/facebookresearch/sam3.git"

# 4.（可选）加速：FlashAttention-3 + torch.compile
pip install einops ninja
pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
```

如果你的 Python 是 PEP 668 保护的 Debian/Ubuntu 系统（`error: externally-managed-environment`）：
建议用 `python -m venv --system-site-packages .venv` 继承系统级 torch / opencv，再 `pip install -e .` 到 venv 内。

---

## 2. 准备 SAM 3.1 权重

SAM 3.1 的官方权重在 Hugging Face 上是 **gated**（`facebook/sam3.1`），需要先在 Meta 那边申请。

如果你已经申请通过：

```bash
hf auth login   # 输入 token
python -c "from huggingface_hub import hf_hub_download; \
    p = hf_hub_download('facebook/sam3.1', 'sam3.1_multiplex.pt'); \
    print('saved to:', p)"
```

如果你只是想跑通本仓库的功能，社区里有若干第三方 mirror 重新上传了**字节完全一致**的权重，例如：

* `jetjodh/sam3.1` — 包含 `sam3.1_multiplex.pt` + tokenizer assets
* `research21/sam3.1`、`AEmotionStudio/sam3.1`、`Comfy-Org/sam3.1` 等

```bash
huggingface-cli download jetjodh/sam3.1 \
    sam3.1_multiplex.pt --local-dir checkpoints/
```

> ⚠️ 转发权重应当遵守 Meta 的 SAM License；本仓库不附带任何权重文件。

---

## 3. 单 mp4 视频

```bash
python scripts/run_video.py \
    --video        path/to/input.mp4 \
    --out          path/to/output_head_blur.mp4 \
    --checkpoint   checkpoints/sam3.1_multiplex.pt \
    --prompt       "human head" \
    --mode         blur
```

参数：

| Flag | 默认 | 说明 |
|---|---|---|
| `--prompt` | `"human head"` | 文本提示：`"human head"` / `"head"` / `face` / `"human face"` / `person` 都可以；默认的 `"human head"` 既能盖住头发→下巴，又能避免把毛绒玩具/卡通公仔的"头"也圈进来 |
| `--mode` | `blur` | `blur` / `mosaic` / `fill` |
| `--blur_ksize` | 51 | 高斯核尺寸（奇数） |
| `--mosaic_block` | 24 | 马赛克块大小 |
| `--mask_expand` | 10 | mask 形态学膨胀像素数（保险地盖住头发/下巴边缘） |
| `--max_objects` | 8 | 同一帧最多追踪几个人 |
| `--restrict_region` | — | `x1,y1,x2,y2`，只在这个矩形内的 mask 才生效（拼接视频里非常实用） |
| `--debug_overlay` | False | 在输出里再画一圈绿色轮廓，方便目视 QA |

---

## 4. 单 hdf5 文件

适用于 `images_dict/<camera>/rgb`（JPEG 字节流）+ `rgb_size` 这种布局（Astribot
/ 整理餐盘 / 12moon 系列等）：

```
images_dict/
  head/   rgb (uint8 1-D)   rgb_size (int32, N)   rgb_timestamp (float64, N)
  left/   rgb …              rgb_size …            rgb_timestamp …
  right/  rgb …              rgb_size …            rgb_timestamp …
<all other groups / datasets are copied through verbatim>
```

```bash
python scripts/run_h5.py \
    --src          path/to/episode.hdf5 \
    --dst          path/to/episode_blur.hdf5 \
    --checkpoint   checkpoints/sam3.1_multiplex.pt \
    --cameras      "head:human head" "left:human head" "right:human head"
```

`--cameras` 用 `<camera_name>:<text_prompt>` 列表来精细控制：

```bash
# 第一视角不打码，只对左右相机做 face 打码
--cameras head:   "left:face"   "right:face"

# 全部用 human head 提示词（默认）
--cameras "head:human head" "left:human head" "right:human head"

# 同时盖整个人（更激进）
--cameras head:person left:person right:person
```

`<prompt>` 为空字符串时表示**直通**（仍会被写入新 hdf5，但 JPEG 字节不变）。

输出 hdf5 的结构与输入完全一致：
- `images_dict/<cam>/rgb` 与 `rgb_size` 被替换为打码后的 JPEG 串
- `rgb_timestamp` 原样保留
- 其它所有 group / dataset（`joints_dict/`, `poses_dict/`, `command_poses_dict/`, `time` 等）**字节级直拷**

---

## 5. 批处理（一次加载模型，处理 N 个文件）

加载模型大约要 30 秒，所以**绝对不要为每个文件起一次进程**。

`batch_video.py` / `batch_h5.py` 自带：

* manifest（`--manifest file.txt`，一行一个路径）或递归扫描（`--input_dir dir/`）
* 状态文件断点续跑（`--state state.json`）
* 失败重试（`--retries`）
* `SIGTERM` 优雅退出（当前文件处理完就停）
* 多卡：自己切片，多个进程指向同一 `--state` 即可

```bash
# mp4 批跑示例
python scripts/batch_video.py \
    --manifest     manifest/episodes.txt \
    --output_dir   output/clear_table_head \
    --state        state/clear_table_head.json \
    --checkpoint   checkpoints/sam3.1_multiplex.pt \
    --prompt "human head" --mode blur --suffix head_blur

# hdf5 批跑示例
python scripts/batch_h5.py \
    --input_dir    data/12moon_1231/ \
    --output_dir   output/12moon_1231_head_blur/ \
    --state        state/12moon_1231_head_blur.json \
    --checkpoint   checkpoints/sam3.1_multiplex.pt \
    --cameras "head:human head" "left:human head" "right:human head" \
    --mode blur --suffix head_blur
```

多卡示例（按 manifest 切片）：

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python scripts/batch_video.py \
      --manifest manifest/all.txt \
      --start $((i*50)) --end $(((i+1)*50)) \
      --output_dir output/all_head_blur \
      --state state/all_head_blur.json \
      --checkpoint checkpoints/sam3.1_multiplex.pt &
done
wait
```

四个 worker 共用同一份 `state/*.json`；靠 `--start/--end` 互不重叠，避免抢占。

---

## 6. Python API

```python
from sam31_head_mask import (
    build_predictor, OcclusionCfg, H5CameraSpec,
    process_video, process_hdf5,
)

predictor = build_predictor("checkpoints/sam3.1_multiplex.pt")
cfg = OcclusionCfg(mode="blur", blur_ksize=61, mask_expand=12)

# 单 mp4
result = process_video(
    predictor, "in.mp4", "out.mp4",
    prompt="human head", occlusion=cfg,
)
print(result)

# 单 hdf5
res = process_hdf5(
    predictor, "in.hdf5", "out.hdf5",
    cameras=[
        H5CameraSpec("head",  prompt="human head"),
        H5CameraSpec("left",  prompt="human head"),
        H5CameraSpec("right", prompt="human head"),
    ],
    occlusion=cfg,
)
for s in res.cameras:
    print(s)
```

---

## 7. 性能参考

测试机：单卡 NVIDIA B200（180 GB）。

| 任务 | 帧数 / 分辨率 | 端到端耗时 | 速度 |
|---|---|---|---|
| mp4：1280×1080，1232 帧 | 1232 帧 | ~30 s | ≈ 41 fps（1× 实时） |
| hdf5：head 1280×720 + left/right 640×360 共 1173 帧 | 1173 帧 | ~25 s | ≈ 47 fps |
| 模型加载 | — | ~30 s | one-time |

数字里包含视频解码 / JPEG 重编码；纯推理本身按 Meta 官方在 H100 上 ~32 fps。
开 `--use_fa3 --compile` 可以再快 1.5–2 倍。

---

## 8. Prompt 选择经验

| Prompt | 行为 | 漏检 | 过度遮挡 |
|---|---|---|---|
| `face` | 只圈脸 | 较高（特别是侧脸/远景） | 低 |
| `human face` | 同上但稍稳 | 中 | 低 |
| `head` | 头发到颈部 | 低 | 略（会被毛绒/卡通公仔的"头"误触发） |
| `human head` ✅ 默认 | 头发到颈部，但限定真人 | 低 | 低 |
| `person` | 整个人 | 极低 | 高（连衣服一起糊） |

实际场景里**先用 `human head` 跑一遍 → 抽帧 review → 漏检多再升级到 `person`** 是最划算的策略。
还可以两轮跑取 mask 并集（自己加一行：先用 `human head` 写到中间 mp4，再用 `face` 跑一次）。

---

## 9. 仓库结构

```
sam3.1_head_mask/
├── sam31_head_mask/                # Python package
│   ├── __init__.py
│   ├── sam_runtime.py              # build_predictor + upstream-bug monkey-patch
│   ├── occlusion.py                # blur / mosaic / fill renderer
│   ├── video_pipeline.py           # mp4 -> mp4
│   ├── h5_pipeline.py              # hdf5 -> hdf5 (JPEG-concat layout)
│   └── _stream.py                  # shared predictor-session helpers
├── scripts/
│   ├── run_video.py                # CLI: single mp4
│   ├── run_h5.py                   # CLI: single hdf5
│   ├── batch_video.py              # CLI: batch mp4
│   └── batch_h5.py                 # CLI: batch hdf5
├── tests/
│   └── test_occlusion.py
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## 10. 已知坑（已在代码里规避）

1. **`Sam3MultiplexTrackingWithInteractivity.init_state(offload_state_to_cpu=…)`** 不接受这个 kwarg，但 `Sam3BasePredictor.start_session` 总会传 — `sam_runtime.py` 里 monkey-patch 修复。
2. **`sam3` 是 namespace package**，`pkg_resources.resource_filename` 在某些 editable install 下会拿到 `None`；本仓库自己定位 BPE 文件并传给 builder。
3. **第三方 mirror checkpoint** 的 key 命名与官方完全一致，`demo_model.load_state_dict(strict=False)` 即可全量加载（中间 `tracker_model.load_state_dict` 打印的 missing/unexpected 是构建过程产物，不影响最终模型）。

---

## 11. 致谢

* [Meta SAM 3 / 3.1](https://github.com/facebookresearch/sam3) — model & inference code.
* 上游 paper: *SAM 3: Segment Anything with Concepts*（Carion et al., 2025）。
