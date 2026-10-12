#!/usr/bin/env bash
# build.sh — build alpha-erase, compile the MLX Metal kernels into mlx.metallib,
# and copy both artifacts to the project root.
#
# Idempotent: safe to re-run. mlx.metallib is only rebuilt when a .metal source
# is newer than the existing metallib (or when it is missing).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-release}"
BUILD_DIR="$ROOT/.build/$CONFIG"

echo "==> swift build -c $CONFIG"
swift build -c "$CONFIG"

# ---------------------------------------------------------------------------
# Locate the mlx-swift checkout and its Metal kernel sources.
# ---------------------------------------------------------------------------
KERNELS_ROOT="$(find "$ROOT/.build/checkouts" -type d -path '*/mlx/mlx/backend/metal/kernels' 2>/dev/null | head -n1 || true)"
if [[ -z "$KERNELS_ROOT" ]]; then
    echo "error: could not find mlx-swift Metal kernels under .build/checkouts" >&2
    echo "hint: run 'swift package resolve' first" >&2
    exit 1
fi
# The -I include root for xcrun metal is four levels up from kernels/
# (.../mlx/mlx/backend/metal/kernels -> .../mlx)
MLX_INCLUDE_ROOT="$(cd "$KERNELS_ROOT/../../../.." && pwd)"
echo "==> Metal kernels: $KERNELS_ROOT"

METALLIB="$BUILD_DIR/mlx.metallib"

# ---------------------------------------------------------------------------
# Incremental check: rebuild only if a source is newer than the metallib.
# ---------------------------------------------------------------------------
needs_build=1
if [[ -f "$METALLIB" ]]; then
    needs_build=0
    while IFS= read -r src; do
        if [[ "$src" -nt "$METALLIB" ]]; then
            needs_build=1
            break
        fi
    done < <(find "$KERNELS_ROOT" -name '*.metal')
fi

if [[ "$needs_build" -eq 0 ]]; then
    echo "==> mlx.metallib is up to date, skipping kernel compilation"
else
    echo "==> Compiling Metal kernels"
    KERNELS=(
        arg_reduce conv gemv layer_norm random rms_norm rope
        scaled_dot_product_attention
        arange binary binary_two copy fft reduce quantized fp_quantized
        scan softmax logsumexp sort ternary unary
        steel_conv steel_conv_3d steel_conv_general
        steel_gemm_fused steel_gemm_gather steel_gemm_masked
        steel_gemm_splitk steel_gemm_segmented
        gemv_masked steel_attention
    )
    # steel kernels live in subdirectories; map via case (bash 3.2 compatible).
    kernel_rel_path() {
        case "$1" in
            steel_conv)         echo "steel/conv/kernels/steel_conv" ;;
            steel_conv_3d)      echo "steel/conv/kernels/steel_conv_3d" ;;
            steel_conv_general) echo "steel/conv/kernels/steel_conv_general" ;;
            steel_gemm_fused)   echo "steel/gemm/kernels/steel_gemm_fused" ;;
            steel_gemm_gather)  echo "steel/gemm/kernels/steel_gemm_gather" ;;
            steel_gemm_masked)  echo "steel/gemm/kernels/steel_gemm_masked" ;;
            steel_gemm_splitk)  echo "steel/gemm/kernels/steel_gemm_splitk" ;;
            steel_gemm_segmented) echo "steel/gemm/kernels/steel_gemm_segmented" ;;
            steel_attention)    echo "steel/attn/kernels/steel_attention" ;;
            *)                  echo "$1" ;;
        esac
    }

    AIR_DIR="$(mktemp -d)"
    trap 'rm -rf "$AIR_DIR"' EXIT

    for k in "${KERNELS[@]}"; do
        rel="$(kernel_rel_path "$k")"
        src="$KERNELS_ROOT/$rel.metal"
        if [[ ! -f "$src" ]]; then
            echo "error: kernel source not found: $src" >&2
            exit 1
        fi
        echo "    metal $k"
        xcrun -sdk macosx metal -x metal -Wall -Wextra -fno-fast-math \
            -Wno-c++17-extensions -Wno-c++20-extensions \
            -c "$src" -I"$MLX_INCLUDE_ROOT" -o "$AIR_DIR/$k.air"
    done

    echo "==> Linking mlx.metallib"
    xcrun -sdk macosx metallib "$AIR_DIR"/*.air -o "$METALLIB"
fi

# ---------------------------------------------------------------------------
# Copy artifacts to the project root.
# ---------------------------------------------------------------------------
echo "==> Copying artifacts to project root"
cp -f "$BUILD_DIR/alpha-erase" "$ROOT/alpha-erase"
cp -f "$METALLIB" "$ROOT/mlx.metallib"

echo "==> Done"
ls -lh "$ROOT/alpha-erase" "$ROOT/mlx.metallib"
