import ArgumentParser
import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXBiRefNet
import UniformTypeIdentifiers

struct AlphaErase: ParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "alpha-erase",
        abstract: "Remove image backgrounds with BiRefNet (MLX/Metal) and write RGBA PNGs."
    )

    @Argument(help: "Input image file, or a directory of images (batch mode).")
    var input: String

    @Argument(help: "Output PNG path (single file), or output directory (batch mode).")
    var output: String

    @Option(name: .long, help: "Path to BiRefNet MLX safetensors weights.")
    var weights: String = "models/BiRefNet-fp16.safetensors"

    @Option(name: .long, help: "Model input square size in pixels. Use 2048 with HR-matting checkpoints.")
    var inputSize: Int = 1024

    @Option(name: .long, help: "Inference precision: fp16 or fp32.")
    var dtype: String = "fp16"

    @Option(name: .long, help: "Swin backbone: swin_v1_l, swin_v1_b, swin_v1_s, or swin_v1_t.")
    var backbone: String = "swin_v1_l"

    @Option(name: .long, help: "Multi-scale input mode: cat, add, or none.")
    var multiScaleInput: String = "cat"

    @Option(name: .long, help: "Number of context channels (cxtNum).")
    var cxtNum: Int = 3

    @Option(name: .long, help: "Decoder attention: aspp-deformable, aspp, or none.")
    var decoderAttention: String = "aspp-deformable"

    @Option(name: .customLong("batch-norm"), help: "Whether the checkpoint has BatchNorm layers: true or false.")
    var batchNormEnabled: String = "true"

    private static let supportedExtensions: Set<String> = ["png", "jpg", "jpeg", "webp"]

    func run() throws {
        let cwd = FileManager.default.currentDirectoryPath
        func resolve(_ p: String) -> String {
            if p.hasPrefix("/") { return (p as NSString).standardizingPath }
            return ((cwd as NSString).appendingPathComponent(p) as NSString).standardizingPath
        }

        let weightsPath = resolve(weights)
        let inputPath = resolve(input)
        let outputPath = resolve(output)

        let device = Device.defaultDevice()
        let info = GPU.deviceInfo()
        FileHandle.standardError.write(
            "MLX default device: \(device.deviceType.map { "\($0)" } ?? "unknown")\n"
                .data(using: .utf8)!)
        FileHandle.standardError.write(
            "Metal GPU: \(info.architecture)  maxBufferSize=\(info.maxBufferSize / 1024 / 1024)MB  maxWorkingSet=\(info.maxRecommendedWorkingSetSize / 1024 / 1024 / 1024)GB\n"
                .data(using: .utf8)!)

        // Parse and validate config overrides.
        guard inputSize > 0, inputSize % 32 == 0 else {
            FileHandle.standardError.write(
                "error: --input-size must be a positive multiple of 32 (got \(inputSize))\n"
                    .data(using: .utf8)!)
            throw ExitCode.failure
        }
        let mlxDtype: DType
        switch dtype.lowercased() {
        case "fp16", "float16": mlxDtype = .float16
        case "fp32", "float32": mlxDtype = .float32
        default:
            FileHandle.standardError.write(
                "error: --dtype must be fp16 or fp32 (got \(dtype))\n".data(using: .utf8)!)
            throw ExitCode.failure
        }
        let cfgBackbone: BiRefNetConfig.Backbone
        switch backbone.lowercased() {
        case "swin_v1_l", "swinl": cfgBackbone = .swinV1L
        case "swin_v1_b", "swinb": cfgBackbone = .swinV1B
        case "swin_v1_s", "swins": cfgBackbone = .swinV1S
        case "swin_v1_t", "swint": cfgBackbone = .swinV1T
        default:
            FileHandle.standardError.write(
                "error: --backbone must be swin_v1_l|swin_v1_b|swin_v1_s|swin_v1_t (got \(backbone))\n"
                    .data(using: .utf8)!)
            throw ExitCode.failure
        }
        let cfgMsi: BiRefNetConfig.MultiScaleInput
        switch multiScaleInput.lowercased() {
        case "cat": cfgMsi = .cat
        case "add": cfgMsi = .add
        case "none": cfgMsi = .none
        default:
            FileHandle.standardError.write(
                "error: --multi-scale-input must be cat|add|none (got \(multiScaleInput))\n"
                    .data(using: .utf8)!)
            throw ExitCode.failure
        }
        let cfgDecAtt: BiRefNetConfig.DecoderAttention
        switch decoderAttention.lowercased() {
        case "aspp-deformable", "aspp_deformable", "asppdeformable": cfgDecAtt = .asppDeformable
        case "aspp": cfgDecAtt = .aspp
        case "none": cfgDecAtt = .none
        default:
            FileHandle.standardError.write(
                "error: --decoder-attention must be aspp-deformable|aspp|none (got \(decoderAttention))\n"
                    .data(using: .utf8)!)
            throw ExitCode.failure
        }
        let cfgBn: Bool
        switch batchNormEnabled.lowercased() {
        case "true", "yes", "1": cfgBn = true
        case "false", "no", "0": cfgBn = false
        default:
            FileHandle.standardError.write(
                "error: --batch-norm must be true or false (got \(batchNormEnabled))\n"
                    .data(using: .utf8)!)
            throw ExitCode.failure
        }

        var cfg = BiRefNetConfig.swinLargeDefault
        cfg.inputSize = (inputSize, inputSize)
        cfg.backbone = cfgBackbone
        cfg.multiScaleInput = cfgMsi
        cfg.cxtNum = cxtNum
        cfg.decoderAttention = cfgDecAtt
        cfg.batchNormEnabled = cfgBn

        FileHandle.standardError.write(
            "Config: inputSize=\(inputSize)x\(inputSize), dtype=\(dtype.lowercased()), backbone=\(backbone), multiScaleInput=\(multiScaleInput), cxtNum=\(cxtNum), decoderAttention=\(decoderAttention), batchNorm=\(batchNormEnabled), weights=\(weightsPath)\n"
                .data(using: .utf8)!)

        // Load model once.
        let loadStart = Date()
        let pipeline: BiRefNetPipeline
        do {
            pipeline = try BiRefNetPipeline.fromPretrained(
                weightsPath, dtype: mlxDtype, config: cfg)
        } catch {
            FileHandle.standardError.write(
                "fatal: failed to load weights at \(weightsPath): \(error)\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        FileHandle.standardError.write(
            String(format: "Model loaded in %.2fs\n", Date().timeIntervalSince(loadStart))
                .data(using: .utf8)!)

        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: inputPath, isDirectory: &isDir) else {
            FileHandle.standardError.write("error: input not found: \(inputPath)\n".data(using: .utf8)!)
            throw ExitCode.failure
        }

        var failures = 0
        if isDir.boolValue {
            // Batch mode.
            if fm.fileExists(atPath: outputPath) {
                var outIsDir: ObjCBool = false
                guard fm.fileExists(atPath: outputPath, isDirectory: &outIsDir), outIsDir.boolValue else {
                    FileHandle.standardError.write(
                        "error: output must be a directory in batch mode: \(outputPath)\n"
                            .data(using: .utf8)!)
                    throw ExitCode.failure
                }
            } else {
                try fm.createDirectory(atPath: outputPath, withIntermediateDirectories: true)
            }

            let items = try fm.contentsOfDirectory(atPath: inputPath).sorted()
            var processed = 0
            for item in items {
                let ext = (item as NSString).pathExtension.lowercased()
                guard Self.supportedExtensions.contains(ext) else { continue }
                let src = (inputPath as NSString).appendingPathComponent(item)
                let dstName = ((item as NSString).deletingPathExtension) + ".png"
                let dst = (outputPath as NSString).appendingPathComponent(dstName)
                do {
                    let t0 = Date()
                    try processImage(inputPath: src, outputPath: dst, pipeline: pipeline)
                    let dt = Date().timeIntervalSince(t0)
                    print("ok   \(item) -> \(dstName)  (\(String(format: "%.2f", dt))s)")
                    processed += 1
                } catch {
                    print("FAIL \(item): \(error)")
                    failures += 1
                }
            }
            FileHandle.standardError.write(
                "Done: \(processed) succeeded, \(failures) failed.\n".data(using: .utf8)!)
        } else {
            // Single file mode.
            do {
                let t0 = Date()
                try processImage(inputPath: inputPath, outputPath: outputPath, pipeline: pipeline)
                let dt = Date().timeIntervalSince(t0)
                print("ok   \(inputPath) -> \(outputPath)  (\(String(format: "%.2f", dt))s)")
            } catch {
                print("FAIL \(inputPath): \(error)")
                failures += 1
            }
        }

        if failures > 0 { throw ExitCode.failure }
    }

    private func processImage(inputPath: String, outputPath: String, pipeline: BiRefNetPipeline) throws {
        let url = URL(fileURLWithPath: inputPath)
        guard let image = loadCGImage(url: url) else {
            throw NSError(
                domain: "alpha-erase", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "cannot decode image"])
        }

        let prediction = pipeline(image)
        guard let mask = prediction.maskAtSource else {
            throw NSError(
                domain: "alpha-erase", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "model produced no mask"])
        }
        // GPU memory pool stats — proof the Metal backend allocated buffers.
        let peak = Memory.peakMemory
        let active = Memory.activeMemory
        FileHandle.standardError.write(
            "GPU mem after inference: active=\(active / 1024 / 1024)MB peak=\(peak / 1024 / 1024)MB\n"
                .data(using: .utf8)!)

        let composited = try compositeRGBA(image: image, mask: mask)

        let outURL = URL(fileURLWithPath: outputPath)
        let outDir = outURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: outDir, withIntermediateDirectories: true)

        guard let dest = CGImageDestinationCreateWithURL(
            outURL as CFURL, UTType.png.identifier as CFString, 1, nil)
        else {
            throw NSError(
                domain: "alpha-erase", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "cannot create PNG destination"])
        }
        CGImageDestinationAddImage(dest, composited, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw NSError(
                domain: "alpha-erase", code: 4,
                userInfo: [NSLocalizedDescriptionKey: "PNG encode failed"])
        }
    }

    /// Composite source RGB with BiRefNet alpha into a straight-alpha RGBA CGImage.
    private func compositeRGBA(image: CGImage, mask: MLXArray) throws -> CGImage {
        let w = image.width
        let h = image.height

        // Source RGB bytes via CoreGraphics (premultiplied RGBA draw, alpha is 255).
        let cs = CGColorSpaceCreateDeviceRGB()
        let drawInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w * 4, space: cs, bitmapInfo: drawInfo)
        else {
            throw NSError(
                domain: "alpha-erase", code: 5,
                userInfo: [NSLocalizedDescriptionKey: "CGContext creation failed"])
        }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let srcData = ctx.data?.assumingMemoryBound(to: UInt8.self) else {
            throw NSError(
                domain: "alpha-erase", code: 6,
                userInfo: [NSLocalizedDescriptionKey: "CGContext data unavailable"])
        }

        // Alpha floats: (1, h, w, 1) in [0, 1].
        let floats: [Float] = mask.asArray(Float.self)
        guard floats.count == w * h else {
            throw NSError(
                domain: "alpha-erase", code: 7,
                userInfo: [NSLocalizedDescriptionKey:
                    "mask size mismatch: \(floats.count) vs \(w * h)"])
        }

        // Straight-alpha RGBA output.
        var out = [UInt8](repeating: 255, count: w * h * 4)
        for i in 0..<(w * h) {
            out[i * 4 + 0] = srcData[i * 4 + 0]
            out[i * 4 + 1] = srcData[i * 4 + 1]
            out[i * 4 + 2] = srcData[i * 4 + 2]
            var a = floats[i]
            if a < 0 { a = 0 } else if a > 1 { a = 1 }
            out[i * 4 + 3] = UInt8(a * 255.0 + 0.5)
        }

        let straightInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.last.rawValue
        guard let provider = CGDataProvider(data: Data(out) as CFData),
              let cg = CGImage(
                width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: w * 4, space: cs,
                bitmapInfo: CGBitmapInfo(rawValue: straightInfo),
                provider: provider, decode: nil, shouldInterpolate: false,
                intent: .defaultIntent)
        else {
            throw NSError(
                domain: "alpha-erase", code: 8,
                userInfo: [NSLocalizedDescriptionKey: "RGBA CGImage creation failed"])
        }
        return cg
    }
}

AlphaErase.main()
