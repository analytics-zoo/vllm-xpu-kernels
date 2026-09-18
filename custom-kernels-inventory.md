# Gemma26B 小 M SYCL / SYCL-TLA 算子登记

本提交基于 vllm-xpu-kernels v0.1.11.1，覆盖 Gemma-4-26B-A4B TP2、
FP8 权重 / FP16 activation、M=1..8。以下是新增接口及内部调度，
不是全仓算子清单。未提交的 V1 model fusion 与普通 SYCL 实验继续保留在工作区。

| 接口 | 模块 / 源码 | 范围与行为 |
| --- | --- | --- |
| `_xpu_C.fp8_gemm_w8a16_tla` | `_linear_C`；`csrc/xpu/fp8_linear_tla/` | BMG-G31、六个 Gemma TP2 Linear 形状、M1..8、FP8 E4M3FN NT 权重、FP32 单 scale、无 bias；QKV Split16，其余 Split8，Ktile32；FP32 累加与归约，两个 kernel。范围外回退原算子。 |
| `_xpu_C.fp16_nt_gemm` | `_fp16_C`；`csrc/xpu/fp16_linear/` | BMG-G31、router N128/K2816 与 LM head N131072/K2816、M1..8、FP16 NT 权重；router Split16，LM head 直接 TLA。范围外回退 `at::linear`；未对齐输入先 clone，此开销属于调用本身。 |
| `_xpu_C.rms_norm_no_weight` | `_rms_C`；`csrc/xpu/rms/` | FP16、H256/512/2816、rank2..4、首维 M1..8、总 rows1..64、末维 stride1；删除全1权重创建及读取，保持原归一化舍入。支持 Meta。 |
| `_moe_C.gemma4_small_m_topk` | `_moe_C`；`csrc/moe/gemma4_small_m_topk.cpp` | FP16/FP32 logits、M1..8、E128、topk8；匹配 Triton 的排序与 tie-break，融合可选 per-expert scale，输出 FP32 weights 与 int32 IDs。 |
| 原 `_xpu_C.cutlass_grouped_gemm_interface` | `csrc/xpu/grouped_gemm/xe_2/` | E128、R1..64、FP16/FP8、FP32 per-expert scale、两个 Gemma expert 形状：compact map + direct-grid，每次 GEMM 两个 kernel；保留 xe_gemm 数学循环。其余形状继续既有调度，小 M 采用 m8 policy。 |

接口使用当前 Torch XPU stream，不持有跨调用 scratch；元数据检查在 GPU 提交前完成。
FP8/FP16 Linear 提供 Meta 实现。Routing 当前仅注册 XPU，实现不读取 host 数据。

## 部分编译与接线

独立 target `_linear_C`、`_fp16_C`、`_rms_C` 分别由
`LINEAR_TLA_KERNELS_ENABLED`、`FP16_LINEAR_KERNELS_ENABLED`、
`RMS_KERNELS_ENABLED` 控制；前两者同时要求 TLA / XE2。
Routing 使用原 `_moe_C` target，compact GEMM 使用原 `_xpu_C` target。
可选模块缺失时保留其他调用路径；模块存在但 ABI 损坏时正常报错。
本提交不包含 vLLM 侧调用分派；新接口需要对应 XPU adapter 接线才能生效。

## 已完成验证与限制

验证环境：PyTorch 2.12.0+xpu、oneAPI 2025.3、BMG-G31、GPU4/5。
已有独立部分编译、安装及源码/库哈希核对；M1..8 的组件正确性记录、
32 组构建开关组合和 TP2 服务 dispatch trace 均已保留。
现场证据位于 `/llm/models/test/test1/gemma_upstream_port_v2_20260917/`：
`source_integration/kernel_repo/source_verification.json`、
`audit/downstream_ablation_20260917/upstreamable_dispatch_audit.json`。
这些现场测试不是仓内可移植测试套件；完整评测集质量尚未验证。
当前生产选择未启用 V1 跨 model fusion；减少 kernel 数仍有优化空间。
