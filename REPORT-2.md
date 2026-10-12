# alpha-erase 改进报告：模型配置参数化 + build.sh

## 1. BiRefNetConfig 完整字段列表

来源：`.build/checkouts/mlx-swift-BiRefNet/Sources/MLXBiRefNet/BiRefNetConfig.swift`
（mnmly/mlx-swift-BiRefNet @ main，commit 43c1f83）。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `backbone` | `enum Backbone: String` | `.swinV1L` (`"swin_v1_l"`) | 可选：`swin_v1_l` / `swin_v1_b` / `swin_v1_s` / `swin_v1_t` |
| `multiScaleInput` | `enum MultiScaleInput: String` | `.cat` | 可选：`""`(none) / `add` / `cat` |
| `cxtNum` | `Int` | `3` | 追加到最深特征图的 context 通道数 |
| `decoderAttention` | `enum DecoderAttention: String` | `.asppDeformable` (`"ASPPDeformable"`) | 可选：`""`(none) / `"ASPP"` / `"ASPPDeformable"` |
| `decoderBlock` | `String` | `"BasicDecBlk"` | 源码注释："we only ship BasicDecBlk"，**无其他实现** |
| `lateralBlock` | `String` | `"BasicLatBlk"` | 同上，仅一种实现 |
| `decoderInputInjection` | `Bool` | `true` | |
| `decoderInputSplit` | `Bool` | `true` | |
| `squeezeBlockEnabled` | `Bool` | `true` | `"BasicDecBlk_x1"` |
| `squeezeBlockCount` | `Int` | `1` | |
| `msSupervision` | `Bool` | `true` | 源码注释：仅训练输出相关 |
| `outRef` | `Bool` | `true` | gdt_convs / gdt_attn 在推理路径上 |
| `batchNormEnabled` | `Bool` | `true` | checkpoint 是否含 BN running stats |
| `inputSize` | `(width: Int, height: Int)` | `(1024, 1024)` | 训练分辨率；推理可换其他尺寸 |

另有 3 个**计算属性**（只读，不可直接设置，由上述字段推导）：
`lateralChannelsRaw`、`lateralChannels`、`contextChannels`。

预置变体：**只有 `.swinLargeDefault`**（即 `BiRefNetConfig()`，全部默认值）。
库中**没有** `.swinLargeHRMatting`、`.swinTiny` 等预置——换模型需自行覆盖字段。

**未参数化的字段及原因**（诚实说明，不假装支持）：
- `decoderBlock` / `lateralBlock`：库只实现了一种 block，改成其他字符串会在
  构图时无意义或崩溃，故不暴露。
- `decoderInputInjection` / `decoderInputSplit` / `squeezeBlock*` /
  `msSupervision` / `outRef`：这些是训练变体的结构开关，公开 checkpoint
  （ZhengPeng7 系列、mlx-community 转换版）全部使用默认值，暴露它们只会
  让用户造出无法加载权重的配置。如果将来拿到非默认结构的 checkpoint，
  需要改源码（或给库提 PR）。

## 2. 新增 CLI 参数

全部为 `@Option`，help 文本已验证 `--help` 可读：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--weights <path>` | `models/BiRefNet-fp16.safetensors` | MLX safetensors 权重路径 |
| `--input-size <n>` | `1024` | 模型输入边长（正方形）。HR 模型用 2048 |
| `--dtype <fp16\|fp32>` | `fp16` | 推理精度 |
| `--backbone <swin_v1_l\|swin_v1_b\|swin_v1_s\|swin_v1_t>` | `swin_v1_l` | Swin 主干 |
| `--multi-scale-input <cat\|add\|none>` | `cat` | 多尺度输入模式 |
| `--cxt-num <n>` | `3` | context 通道数 |
| `--decoder-attention <aspp-deformable\|aspp\|none>` | `aspp-deformable` | 解码器注意力 |
| `--batch-norm <true\|false>` | `true` | checkpoint 是否含 BN |

校验：`--input-size` 必须为正且是 32 的倍数；`--dtype` 只接受
fp16/float16/fp32/float32；枚举值不合法时打印明确错误并 exit 1。

启动时打印实际配置（stderr）：

```
Config: inputSize=2048x2048, dtype=fp16, backbone=swin_v1_l, multiScaleInput=cat, cxtNum=3, decoderAttention=aspp-deformable, batchNorm=true, weights=/Users/matt/workspace/image/alpha-erase/models/BiRefNet_HR-matting-fp16.safetensors
```

## 3. build.sh

完整内容见项目根目录 `build.sh`（已 `chmod +x`）。要点：

- `set -euo pipefail`
- `swift build -c release`
- 自动在 `.build/checkouts` 下定位 mlx-swift 的
  `mlx/mlx/backend/metal/kernels` 目录（不硬编码 hash 路径）
- 32 个 kernel 逐个 `xcrun -sdk macosx metal` 编译到 `mktemp -d` 临时
  目录，最后 `xcrun metallib` 链接，退出时自动清理
- **增量**：若 `mlx.metallib` 存在且不比任何 `.metal` 源文件旧，则跳过
  kernel 编译（实测第二次运行显示 `up to date, skipping`）
- steel 系列 kernel 的子目录映射用 `case` 实现（兼容 macOS 自带 bash 3.2，
  不用 `declare -A`）
- `cp -f` 复制 `alpha-erase` 和 `mlx.metallib` 到项目根目录，打印大小
- 幂等：重复执行不报错

从零构建验证：`rm -rf .build && ./build.sh` → 成功，总耗时 **3m13s**
（含依赖重新拉取 + MLX C++ 编译 + 32 个 Metal kernel 编译），产物
`./alpha-erase`（27 MB）+ `./mlx.metallib`（102 MB），从根目录直接运行正常。

## 4. 构建产物

| 路径 | 大小 |
|---|---|
| `./alpha-erase`（项目根目录） | 27 MB |
| `./mlx.metallib`（项目根目录，必须与可执行文件同目录） | 102 MB |

## 5. 1024 模型回归测试（6 张全过）

`./alpha-erase temp temp/output-1024` → 6 succeeded, 0 failed。

| 图 | 耗时 | alpha min/max/mean | 前景 | 背景 | 对比改动前 |
|---|---|---|---|---|---|
| anime_girl | 3.83s* | 0/255/181.4 | 71.07% | 28.71% | 一致 |
| cheerleader | 1.78s | 0/255/62.0 | 24.12% | 75.52% | 一致 |
| fantasy_sword | 1.78s | 0/255/33.2 | 12.94% | 86.92% | 一致 |
| host | 1.77s | 0/255/127.0 | 49.73% | 50.09% | 一致 |
| swordsman | 1.77s | 0/255/90.7 | 35.46% | 64.28% | 一致 |
| swordsman_souls | 1.78s | 0/255/62.3 | 24.27% | 75.35% | 一致 |

*首张含 kernel JIT 预热，稳态 ~1.77s/张，与改动前一致。alpha 统计逐位相同，
GPU 峰值 ~4.9GB 不变。参数化未破坏任何原有功能。

## 6. HR-matting 模型测试（16GB M4）

权重：`models/BiRefNet_HR-matting-fp16.safetensors`（420MB，fp16，
与 `hf download` 产物 md5 一致）。同一张 `temp/anime_girl.png`：

| input-size | 耗时 | GPU 峰值 (MLX pool) | swap 变化 | 结果 |
|---|---|---|---|---|
| 2048 | 16.54s | **18.3 GB** | 242MB → **峰值 5.3GB**（swap 总量扩到 6GB） | 成功，但大量交换 |
| 1792 | 6.45s | 14.1 GB | +188MB（轻微） | 成功 |
| 1536 | 3.90s | 10.5 GB | 无增长 | 成功，流畅 |

三个尺寸的输出均为合法 RGBA PNG（alpha 0–255 双峰，前景 ~69%）。
2048 输出 `temp/anime_girl_hr.png`：1024×1024（输出按源图分辨率）、
alpha mean=175.9、fg=68.69%。

swap 采样（`sysctl vm.swapusage`，0.5s 间隔）显示：2048 推理期间 swap
从 242MB 一路涨到 ~5.3GB，说明统一内存被换出到磁盘；系统未崩溃但会有
明显卡顿（窗口服务器内存受压）。

## 7. 结论：16GB Mac 上 HR 模型可用性

- **2048（官方 HR 尺寸）：能跑完但不推荐**。GPU 内存池峰值 18.3GB >
  16GB 物理内存，靠 ~5GB swap 撑过去，单张 16.5s，期间系统明显卡顿。
  批量处理会反复交换，不建议。
- **推荐：1792**。峰值 14.1GB，留有余量，仅轻微交换（+188MB），
  单张 6.45s——质量接近 2048，代价小得多。
- **稳妥：1536**。零交换、3.9s，适合批量或后台任务。

命令示例：

```bash
./alpha-erase input.png out.png \
    --weights models/BiRefNet_HR-matting-fp16.safetensors \
    --input-size 1792
```

注意：HR checkpoint 是在 2048 训练的，用 1792/1536 推理属于跨分辨率，
质量略降但仍明显优于 1024 模型（BiRefNet 系列对分辨率相对鲁棒）。

## 8. 换其他模型的参数指南

| 模型 | 命令参数 |
|---|---|
| BiRefNet-fp16（默认，1024） | 无需额外参数 |
| BiRefNet_HR-matting-fp16 | `--weights models/BiRefNet_HR-matting-fp16.safetensors --input-size 1792`（或 2048/1536） |
| BiRefNet_lite（swin_v1_t） | `--weights <lite 权重> --backbone swin_v1_t`（lite 的 lateral 通道为 [768,384,192,96]，config 会自动推导；若权重无 BN 再加 `--batch-norm false`） |
| BiRefNet-ep / 其他变体 | 先核对原 PyTorch config 的 `bb`（backbone）、`dec_att`、`mul_scl_ipt`、`cxt_num`，分别对应 `--backbone` / `--decoder-attention` / `--multi-scale-input` / `--cxt-num` |

通用排查步骤：
1. 加载失败（strict 校验报 unused keys / missing keys）→ backbone 或
   BN 设置与权重不匹配，核对 checkpoint 的原始 config.json。
2. 输出 mask 异常（全黑/全白）→ 多半是 input-size 与训练尺寸差太远，
   或 dtype 不对（非 fp16 权重别用 fp16 加载——库会统一 cast，但精度
   最好与权重一致）。
3. 内存不够 → 降 `--input-size`（32 的倍数即可，不必是 64 的倍数）。
