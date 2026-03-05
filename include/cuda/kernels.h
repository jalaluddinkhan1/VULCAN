#pragma once

/// @file kernels.h
/// @brief CUDA kernel declarations for VULCAN inference engine.
///
/// All CUDA kernels are declared here as host-callable wrapper functions.
/// Actual kernel implementations live in cuda/*.cu files.

#include <cstddef>

namespace vulcan {
namespace cuda {

// ─── Verification ──────────────────────────────────────────────────────────

/// Launch element-wise vector addition: c[i] = a[i] + b[i]
/// @param a     Input array A (device pointer)
/// @param b     Input array B (device pointer)
/// @param c     Output array C (device pointer)
/// @param n     Number of elements
void launch_vector_add(const float* a, const float* b, float* c, int n);

// ─── Basic Operators ────────────────────────────────────────────────────────

/// Launch tiled matrix multiplication: C = A * B  (scalar CUDA cores)
/// @param A     Input matrix A [M x K] (device, row-major)
/// @param B     Input matrix B [K x N] (device, row-major)
/// @param C     Output matrix C [M x N] (device, row-major)
/// @param M     Rows of A
/// @param K     Shared dimension
/// @param N     Columns of B
void launch_matmul(const float* A, const float* B, float* C,
                   int M, int K, int N);

/// Launch Tensor Core GEMM: C = A * B  (FP16 compute, FP32 accumulate, SM 7.0+)
///
/// Uses nvcuda::wmma 16×16×16 fragments to route work through Tensor Cores on
/// Volta/Turing/Ampere/Ada GPUs.  2–4× faster than launch_matmul for
/// transformer-scale matrices.  Automatically falls back to launch_matmul when
/// any dimension is < 16.
///
/// @param A     Input matrix A [M x K] (device, row-major, float32)
/// @param B     Input matrix B [K x N] (device, row-major, float32)
/// @param C     Output matrix C [M x N] (device, row-major, float32)
/// @param M     Rows of A
/// @param K     Shared dimension
/// @param N     Columns of B
void launch_matmul_wmma(const float* A, const float* B, float* C,
                        int M, int K, int N);

/// Launch SiLU (Swish) activation: y[i] = x[i] * sigmoid(x[i])
/// @param input   Input array (device pointer)
/// @param output  Output array (device pointer)
/// @param n       Number of elements
void launch_silu(const float* input, float* output, int n);

/// Launch ReLU activation: y[i] = max(0, x[i])
/// @param input   Input array (device pointer)
/// @param output  Output array (device pointer)
/// @param n       Number of elements
void launch_relu(const float* input, float* output, int n);

/// Launch RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
/// @param input   Input tensor [n] (device pointer)
/// @param weight  Learnable scale [n] (device pointer)
/// @param output  Output tensor [n] (device pointer)
/// @param n       Dimension
/// @param eps     Epsilon for numerical stability
void launch_rmsnorm(const float* input, const float* weight, float* output,
                    int n, float eps);

/// Launch numerically stable softmax: out[i] = exp(x[i]-max) / sum(exp(x[j]-max))
/// @param input   Input logits [n] (device pointer)
/// @param output  Output probabilities [n] (device pointer, sums to 1)
/// @param n       Number of elements
void launch_softmax(const float* input, float* output, int n);

/// Launch Rotary Position Embedding (RoPE).
/// Rotates pairs of dimensions by position-dependent angles.
/// @param input      Input vector [head_dim] (device pointer)
/// @param output     Output vector [head_dim] (device pointer)
/// @param pos        Token position index
/// @param head_dim   Head dimension (must be even)
/// @param theta_base Frequency base (default: 10000.0)
void launch_rope(const float* input, float* output,
                 int pos, int head_dim, float theta_base = 10000.0f);

// ─── Attention ──────────────────────────────────────────────────────────────

/// Launch scaled dot-product attention.
/// @param Q       Query tensor [batch, heads, seq, head_dim]
/// @param K       Key tensor [batch, heads, seq, head_dim]
/// @param V       Value tensor [batch, heads, seq, head_dim]
/// @param output  Output tensor [batch, heads, seq, head_dim]
/// @param batch   Batch size
/// @param heads   Number of attention heads
/// @param seq_len Sequence length
/// @param head_dim Head dimension
void launch_attention(const float* Q, const float* K, const float* V,
                      float* output,
                      int batch, int heads, int seq_len, int head_dim);

// ─── Element-wise Ops ──────────────────────────────────────────────────────

/// Launch residual add: c[i] = a[i] + b[i]
void launch_residual_add(const float* a, const float* b, float* c, int n);

/// Launch element-wise multiply: c[i] = a[i] * b[i]
void launch_elementwise_mul(const float* a, const float* b, float* c, int n);

/// Launch embedding lookup: copies one row per token from the embedding table.
/// @param table      Embedding table [vocab_size, dim] (device)
/// @param token_ids  Token IDs [seq_len] (device, int*)
/// @param output     Output embeddings [seq_len, dim] (device)
/// @param seq_len    Number of tokens
/// @param dim        Embedding dimension
void launch_embedding_lookup(const float* table, const int* token_ids,
                             float* output, int seq_len, int dim);

// ─── Quantization ──────────────────────────────────────────────────────────

/// Launch INT4 dequantization: converts packed 4-bit weights to float.
/// @param input       Packed INT4 weights (device pointer)
/// @param scales      Per-group scale factors (device pointer)
/// @param output      Dequantized float output (device pointer)
/// @param n           Number of output elements
/// @param group_size  Quantization group size
void launch_dequantize_int4(const uint8_t* input, const float* scales,
                            float* output, int n, int group_size);

// ─── Fused Kernels ─────────────────────────────────────────────────────────

/// Fused SiLU(gate) * up — eliminates intermediate SiLU output buffer.
/// out[i] = silu(gate[i]) * up[i]
void launch_fused_silu_mul(const float* gate, const float* up,
                           float* output, int n);

/// Fused RMSNorm + Linear projection — eliminates normalized buffer.
/// output = rmsnorm(input, norm_weight, eps) @ proj_weight
void launch_fused_rmsnorm_linear(
    const float* input, const float* norm_weight,
    const float* proj_weight, float* output,
    int seq_len, int dim, int out_dim, float eps);

/// Quantized MatMul: C = A * dequant(B_int4)
/// Dequantizes INT4 weights in-register — never writes decompressed to HBM.
void launch_quantized_matmul(
    const float* A, const uint8_t* B_packed, const float* scales,
    float* C, int M, int K, int N, int group_size);

/// Fused Residual + RMSNorm: output = rmsnorm(residual + hidden, weight, eps)
/// Optionally saves the residual sum to residual_out.
void launch_fused_residual_rmsnorm(
    const float* residual, const float* hidden, const float* weight,
    float* output, float* residual_out, int n, float eps);

} // namespace cuda
} // namespace vulcan

