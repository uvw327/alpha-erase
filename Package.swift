// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "alpha-erase",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.3"),
        .package(url: "https://github.com/mnmly/mlx-swift-BiRefNet.git", branch: "main"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "alpha-erase",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXBiRefNet", package: "mlx-swift-BiRefNet"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
