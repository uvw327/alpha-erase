# alpha-erase — 交付报告

BiRefNet 背景移除 CLI（Swift + MLX，Apple Silicon Metal 加速）。

## 1. 环境搭建

| 项目 | 检查结果 | 动作 |
|---|---|---|
| Xcode CLT | `/Applications/Xcode.app/Contents/Developer`（随完整 Xcode） | 无需安装 |
| Xcode 完整版 | Xcode 16.3 (16E140) | 无需安装 |
| Homebrew | 6.0.22 | 无需安装 |
| Python3 + PIL | Pillow 11.3.0 | 无需安装 |
| Swift | 6.1 (swiftlang-6.1.0.110.21)，arm64-apple-macosx15.0 | 满足 ≥5.9 |
| huggingface CLI | 2.2.0 | 无需安装 |
| 硬件 | Apple M4，16 GB 统一内存，macOS 15.7.9 | — |

本机环境完整，未执行任何安装命令。

## 2. 库调研与选型

### 2.1 xocialize/mlx-birefnet-swift —— 不可用

- README + `Package.swift` 确认：`platforms: [.macOS(.v26)]`，swift-tools 6.2，
  依赖 `mlx-engine-swift`（MLXEngine ModelPackage 体系）。
- **本机 macOS 15.7.9，无法满足 macOS 26+ 要求**，且权重由 engine
  自行从 `mlx-community/BiRefNet-fp16` 下载到 model store，不方便直接挂
  本地 `models/BiRefNet-fp16.safetensors`。

### 2.2 替代方案：mnmly/mlx-swift-BiRefNet —— 最终采用

该库正是 xocialize 版 vendored 的核心来源（MIT，Hiroaki Yamane），
要求 macOS 14+，与本机匹配。读源码确认的 API（非凭记忆）：

- `BiRefNetPipeline.fromPretrained(_ path: String, dtype: DType = .float16,
  config: BiRefNetConfig = .swinLargeDefault) throws -> BiRefNetPipeline`
  — 权重加载 `strict: true`（`.noUnusedKeys` 校验）。
- 预处理：CGImage → 高质量双线性 resize 到 1024×1024 → ImageNet 归一化
  `(0.485,0.456,0.406)/(0.229,0.224,0.225)` → NHWC fp16。
  （见 `Sources/MLXBiRefNet/Pipeline.swift` `cgImageToNHWC`）
- 输出：`pipeline(cgImage) -> BiRefNetPrediction`，内部已做
  `sigmoid`；`prediction.mask` 为模型分辨率 alpha（0=背景,1=前景），
  `prediction.maskAtSource` 为双线性重采样回原图分辨率的 alpha。
- RGBA 合成库不自带，由本项目在 `compositeRGBA` 中完成
  （CoreGraphics 绘制源图 RGB + 直接写入 straight-alpha 通道）。

### 2.3 权重格式确认

`models/BiRefNet-fp16.safetensors`：687 个 F16 张量，420 MB。
抽查 4D 卷积权重形状（如 `bb.patch_embed.proj.weight [192,4,4,3]`
= `[O,H,W,I]` NHWC），确认已是 MLX 布局（mlx-community 转换产物），
与 mnmly 库 `loadWeights`（要求 Conv2d 权重 NHWC）匹配，加载
`strict: true` 通过、零未用键。

## 3. 构建

```bash
swift package resolve    # mlx-swift 0.31.4, mlx-swift-BiRefNet@main(43c1f83),
                         # swift-argument-parser 1.8.2, swift-numerics 1.1.1
swift build -c release   # → .build/release/alpha-erase (27 MB)
```

### 关键坑：metallib 不会由 SwiftPM 自动产出

首次运行报错：

```
MLX error: Failed to load the default metallib. library not found ...
  at mlx-c/mlx/c/stream.cpp:115
```

原因：mlx-c 的 `Device::Device()` 强制 `load_default_library()`，按顺序找
二进制同目录的 `mlx.metallib` → SwiftPM bundle → …，全部失败即抛异常
（**没有静默回退 CPU 的路径**）。mlx-swift 的 Package.swift 把
`mlx/mlx/backend/metal/kernels` 排除出编译，metallib 需按上游
CMakeLists 的 kernel 清单自行编译。

解决：按 `kernels/CMakeLists.txt` 复刻编译流程（Metal 3.2，不含 NAX 变体）：

```bash
# 32 个 .metal → .air → mlx.metallib（102 MB，未 strip）
xcrun -sdk macosx metal -x metal -Wall -fno-fast-math -c <kernel>.metal \
    -I<mlx-root> -o <kernel>.air
xcrun -sdk macosx metallib *.air -o mlx.metallib
cp mlx.metallib .build/release/   # 必须与可执行文件同目录
```

kernel 清单：arg_reduce, conv, gemv, layer_norm, random, rms_norm, rope,
scaled_dot_product_attention, arange, binary, binary_two, copy, fft, reduce,
quantized, fp_quantized, scan, softmax, logsumexp, sort, ternary, unary,
steel_conv{,_3d,_general}, steel_gemm_{fused,gather,masked,splitk,segmented},
gemv_masked, steel_attention。

最终产物：

| 文件 | 大小 |
|---|---|
| `.build/release/alpha-erase` | 27 MB |
| `.build/release/mlx.metallib` | 102 MB |

**部署要求：`mlx.metallib` 必须与可执行文件放同一目录。**

## 4. 测试

### 4.1 批量（temp/ 6 张，全部成功）

```
.build/release/alpha-erase temp temp/output
```

| 图 | 尺寸 | 耗时 | 输出大小 | alpha min/max/mean | 前景(>200) | 背景(<50) |
|---|---|---|---|---|---|---|
| anime_girl.png | 1024×1024 | 1.80s | 1729 KB | 0/255/181.4 | 71.07% | 28.71% |
| cheerleader.png | 1024×1024 | 1.77s | 1228 KB | 0/255/62.0 | 24.12% | 75.52% |
| fantasy_sword.png | 1024×1024 | 1.76s | 783 KB | 0/255/33.2 | 12.94% | 86.92% |
| host.png | 1024×1024 | 1.77s | 1090 KB | 0/255/127.0 | 49.73% | 50.09% |
| swordsman.png | 1024×1024 | 1.77s | 1097 KB | 0/255/90.7 | 35.46% | 64.28% |
| swordsman_souls.png | 1024×1024 | 1.79s | 1876 KB | 0/255/62.3 | 24.27% | 75.35% |

- `file` 确认全部为 `PNG image data, 8-bit/color RGBA, non-interlaced`
- PIL 断言 `shape[2]==4` 且 `alpha.min()==0, alpha.max()==255` 通过
- 单张稳态耗时 ~1.77s（首张含 JIT 预热 ~1.8-5.3s）

### 4.2 格式与尺寸

- JPG / WebP 输入 → 输出同名 RGBA PNG ✔
- 640×480、1536×864 非正方形输入 → 输出保持原分辨率
  （mask 经 `maskAtSource` 重采样回源尺寸）✔

### 4.3 错误恢复

- 截断 PNG（`head -c 100`）：打印 `FAIL ... cannot decode image`，
  exit=1，不产生输出文件，进程不崩溃 ✔
- 批量中混入 corrupt.png + 假 junk.jpg：`2 succeeded, 2 failed`，
  正常图照常输出 ✔
- 单张模式失败 exit 1；批量模式任一失败 exit 1；全部成功 exit 0 ✔

## 5. Metal 加速验证 —— 生效

**结论：Metal GPU 加速已生效，无 CPU 回退。** 判断依据（多证据交叉）：

1. **启动日志**：`MLX default device: gpu`；
   `Metal GPU: applegpu_g16g maxBufferSize=8192MB maxWorkingSet=10GB`
   （applegpu_g16g = Apple M4 GPU）。
2. **强制 Metal、无回退路径**：mlx-c `Device()` 构造函数直接
   `MTL::CopyAllDevices()` + `load_default_library()`；删除
   `mlx.metallib` 后程序**立刻崩溃**（见 §3 报错），证明 Metal 后端
   是硬依赖，不存在静默 CPU 回退。
3. **GPU 内存池统计**：推理后 `Memory.activeMemory=426MB,
   peakMemory≈4893MB`——MLX Metal allocator 的 buffer 池峰值 ~4.9GB，
   与 fp16 BiRefNet 权重+激活的 Metal 占用吻合（CPU 路径不会走这个池）。
4. **CPU 占用**：推理期间 `ps` 采样为 15–53%（单核满载应为 ~100%），
   进程 RSS ~590MB 但 GPU 池峰值 4.9GB——重活在 GPU。
5. **自定义 Metal kernel**：BiRefNet 解码器的 DCNv2 可变形卷积由
   `MLXFast.metalKernel` JIT 实现（mlx-swift-BiRefNet
   `Modules/DCNv2Kernel.swift`），这类 kernel 只能在 Metal 设备上执行；
   前向成功本身即 Metal 生效的证据。
6. `otool -L`：链接 `Metal.framework`、`Accelerate.framework`。
7. 未能执行 `sudo powermetrics`（非交互环境无法输 sudo 密码）与
   `MLX_METAL_DISABLE` 对比（该版本 C++ 源码中无此环境变量开关），
   以上 6 项证据已足够。

## 6. 踩的坑

1. **xocialize/mlx-birefnet-swift 要求 macOS 26+** —— 换用其核心来源
   mnmly/mlx-swift-BiRefNet（macOS 14+）。
2. **mlx.metallib 缺失导致启动崩溃** —— SwiftPM 构建不编译 Metal
   kernels；按上游 CMakeLists 手动 `xcrun metal`/`metallib` 编译 32 个
   kernel 并放到二进制同目录（详见 §3）。
3. **`MLX.defaultDevice()` 不存在** —— 正确 API 是 `Device.defaultDevice()`
   （mlx-swift 0.31.4 `Device.swift:113`）。
4. **`metal.deviceInfo()` 不存在** —— 是顶层 `GPU.deviceInfo()`
   （`GPU+Metal.swift:224`）；内存统计是顶层 `Memory.activeMemory/
   peakMemory`，不是 `GPU.Memory.*`。
5. **alpha 合成的 premultiplied 陷阱** —— 直接把 mask 写进
   premultipliedLast 上下文会导致 PNG 编码时被错误反预乘；最终方案是
   从 CG context 拷 RGB 后组装 straight-alpha（`CGImageAlphaInfo.last`）
   的独立缓冲区再建 CGImage。
6. zsh 下 `==`/`*.air` 无匹配时会中断脚本 —— 编译脚本改用绝对路径 +
   变量展开。

## 7. 已知限制

- **输入尺寸**：模型固定 1024×1024 推理，超大图（如 4K+）的细节
  边缘可能因下采样略糊；`maskAtSource` 会重采样回原分辨率。
- **内存**：单次推理 GPU 池峰值 ~4.9GB；16GB 机器批量串行处理安全，
  并发多进程推理会争抢统一内存。
- **metallib 部署**：可执行文件必须与 `mlx.metallib` 同目录分发
  （102MB，未 strip，可用 `metal-strip` 进一步缩小）。
- **fp16 数值**：库 README 指出 fp16 下 DCNv2 Metal kernel 与
  PyTorch 有 ~5e-3 meanAbs 漂移（fp32 模式可对齐但更慢更耗内存）；
  对抠图结果影响可忽略（IoU 0.9905）。
- **未处理**：EXIF 方向、多帧 TIFF/动图（只取第 0 帧）、
  输入本身带 alpha 通道的图（会被忽略，仅用 RGB 推理）。
- 批量模式输出目录若与输入目录相同，同名 `.png` 会被覆盖。

## 8. 用法

```bash
# 单张
.build/release/alpha-erase input.jpg output.png
# 批量
.build/release/alpha-erase temp temp/output
# 自定义权重路径
.build/release/alpha-erase in.jpg out.png --weights /path/to/BiRefNet-fp16.safetensors
```

默认权重：`models/BiRefNet-fp16.safetensors`（相对 CWD 解析）。
