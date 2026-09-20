# Custom kernel additions on the decode development branch

`_C.rms_norm_gated_decode(out, input, gate, weight, epsilon)` writes gated
RMSNorm into `out`. Inputs are contiguous XPU FP16 tensors: input/gate/output
have shape `[rows,128]`, weight is `[128]`. The operation computes RMSNorm
before SiLU gating, with FP32 intermediate arithmetic and one final FP16 cast.
Inputs are unchanged; callers supply a separate output buffer. This entry
uses the existing `_C` SYCL build and adds no dependency.

The initial vLLM dispatch is limited to the validated GDN decode shape
`[16,128]` and `[24,128]` in inference/eager execution; other configurations retain fallback.

## Existing Qwen MoE decode path

`moe_qwen35_sp::fused_forward` is the native implementation behind the
optional `XpuFusedMoeV4` adapter imported from commit
`a8a169f5096a335bf1acebd5580f30cb149261af`. Enable its guarded dispatch with
`VLLM_XPU_MOE_QWEN35_FASTPATH=1`. The public `XpuFusedMoe` call signature
and output-buffer usage are retained. The adapter passes no shared-expert
weights; the model's shared-expert pipeline remains separate.

The shape guard targets hidden size 2048, local intermediate size 256,
256 experts, top-k 8, FP16 activations, per-expert scalar FP8 E4M3 weights,
SiLU, no bias, and no expert parallelism. This development iteration
validates batch size 1; the imported implementation's larger-batch domain
is not additional validation evidence. Both kernels use the existing
SYCL-TLA grouped GEMM build and dependency pins.
