# 本次新增/调整算子登记

此文件登记本次 ESIMD 迁移涉及的接口，不是全仓既有算子的完整清单。

以下按时间保留实验记录；当前策略以最后相关章节为准。2026-09-15 的
`_C.norm_add_norm`、旧 GELU 候选及其 benchmark 属于历史草稿，未纳入本次
两笔提交，也不表示当前默认库注册了这些接口。未采用的 ALU 候选及备份
同样保留在工作目录，不属于本次交付。


| 算子 | 模块 | 实现 | 变化 |
| --- | --- | --- | --- |
| `gelu_tanh_and_mul(out, input)` | `_C` | `csrc/activation.cpp` | 保持已有 ABI；连续 FP16、1024≤D≤4096、D%64=0、M≤8 使用列分块 SYCL 路径，其余走已有实现。 |
| `norm_add_norm(out, input, residual, weight1, weight2, epsilon1, epsilon2)` | `_C` | `csrc/norm_add_norm.cpp` | 新增双 RMSNorm 与残差相加融合，原地更新 residual、写 out；两次归一化和乘权重之间保留 FP16 舍入。 |

## 双 norm 契约

输入/残差/输出为同形状连续 FP16 XPU `[..., D]`（rank≥2、D>0）；
权重为连续 FP16 `[D]`，所有张量在同一设备。epsilon 必须有限且为正。
写入的 out、residual 不得互相重叠或与输入/权重重叠；不满足契约时在提交前报错。
支持空 batch 和尾部 hidden size。算子遵守当前 XPU stream，不持有跨调用 scratch。

Python 包装：`vllm_xpu_kernels.fused_norm.norm_add_norm`，分配并返回 out、更新传入 residual；
该模块同时注册 fake tensor 实现。调用方应在局部 XPU adapter 中核对上述契约，
不支持的形状/类型可沿用原模型分步路径。当前没有修改模型 forward 或默认启用该融合。

测试位于 `tests/test_activation.py`、`tests/test_layernorm.py`；
性能对照脚本为 `benchmark/benchmark_esimd_migration.py`。

## 2026-09-15 实测范围

容器 `wj-test-new-0814`，BMG `0xe223`，GPU 0，PyTorch `2.12.0+xpu`，oneAPI `2025.3`。
仅部分编译 `_C`，未编译 SYCL-TLA GEMM/attention 全集。
一次预热、三次正式测量；157 项相关回归通过。

M=1、D=2816 的设备 kernel 中位耗时：

| 项目 | 同配置旧 SYCL | ESIMD | 新 SYCL |
| --- | ---: | ---: | ---: |
| GELU-tanh×up | 2.702 µs | 7.662 µs | 1.608 µs |
| Norm→Add→Norm | 4.072 µs（三个 kernel） | 6.875 µs | 6.308 µs |

双 norm 的新实现保持 Gemma 分步舍入；原 ESIMD 有不同的混合舍入顺序，不能称为逐位等价迁移。
新双 norm 在该形状慢于旧 SYCL 分步执行，其他宽度也未普遍超过 ESIMD，尚不建议默认启用。
上述为驻留工作集单算子实验，未测整模型性能或评测集准确率。

## 2026-09-17：Gemma decode 独立扩展候选

新增可单独编译的 `_decode_C`，由 `DECODE_KERNELS_ENABLED` 控制。
通过 `TORCH_LIBRARY_FRAGMENT(_xpu_C, ...)` 添加以下接口，保留原 `_xpu_C`
和其中既有算子的 ABI；这不是替换已有 oneDNN/TLA 算子。

| 新接口（`torch.ops._xpu_C`） | 返回值 | 实现 |
| --- | --- | --- |
| `gemma_scaled_add_rms_norm(input, residual, weight, epsilon, scalar)` | normalized、residual 两个新张量 | `csrc/xpu/gemma_norm_fusions.cpp` |
| `gemma_scaled_add_rms_norm_tensor(input, residual, weight, epsilon, scalar_tensor)` | 同上；scalar 为同设备连续 FP16 单元素张量 | 同上 |
| `gemma_norm_add_norm(input, residual, weight1, weight2, epsilon1, epsilon2)` | normalized、residual 两个新张量 | 同上 |
| `gemma_qkv_norm_rope(qkv, q_weight, k_weight, positions, cos_sin_cache, q_eps, k_eps, v_eps)` | query、key、value 新张量 | `csrc/xpu/gemma_decode_fusions.cpp` |
| `gemma_norm_router_norm(input, router_scale, root_size, projection, moe_weight, router_eps, moe_eps)` | FP32 router logits、FP16 MoE input | 同上 |

这些接口保持输入不变，有 XPU 与 Meta 实现；Meta 只分配输出 shape，不提交 GPU 工作。
当前核函数有显式 shape/dtype/device/stride 检查，主要支持 Gemma26B TP2 的 FP16
小 M decode；QKV 当前限定 Q8/KV4/head_dim256。调用方需在模型 XPU 路径中匹配
元数据，其他条件继续原有路径。Tensor scalar 版本不在 host 上读取 scalar。
当前包不自动导入 `_decode_C`；模型融合实验需显式加载该可选扩展，不能仅凭编译成功推断调用已接入。
默认安装尚未覆盖，实验使用独立候选包与显式上层开关。


## Gemma26B V2：独立 XPU 入口与 kernel 调度（2026-09-17）

本节追加于既有清单，不表示前述 V1 model 融合已启用。当前任务关闭 V1 跨算子/model 融合，仅另外授权 routing 的最小分派；是否实际调用以服务 trace 为准。

| Torch 接口 / 内部实现 | 模块 / 源码 | 当前契约与实现 |
|---|---|---|
| `_xpu_C.fp8_gemm_w8a16_tla(A, B, B_scale_, bias_=None)` | 可选 `_linear_C`；`csrc/xpu/fp8_linear_tla/` | Gemma TP2 六形状、M1..8、FP16 activation、FP8 E4M3FN NT权重、FP32单scale、无bias、BMG-G31；两个QKV Split16，其余Split8，全部Ktile32；FP32累加/归约，两个kernel，无repack/current stream；scope外调用原op。 |
| `_xpu_C.rms_norm_no_weight(input, epsilon)` | 可选 `_rms_C`；`csrc/xpu/rms/` | FP16，H256/512/2816，rank2..4，首维M1..8、总rows1..64、末维stride1；允许外维stride；删除全1权重创建/读取，保留upstream归一化舍入；新输出及Meta。 |
| `_xpu_C.rms_norm_small_m(input, weight?, epsilon)` / `.out(output, input, weight?, epsilon)` | 可选 `_rms_C`；`csrc/xpu/rms/small_m_rms.cpp` | 新写的普通 SYCL 小 M RMSNorm：FP16，H256/512/2816，rank2..4，M1..8、rows1..128；支持外维stride及未对齐输入。向量化加载、寄存器保留输入、每行一个工作组；归一化后先舍入FP16，再乘权重。functional分配输出；out要求输出连续且不与输入/权重重叠；当前stream、XPU/Meta注册。 |
| 原 `_xpu_C.cutlass_grouped_gemm_interface` 内部 compact map + direct-grid | `csrc/xpu/grouped_gemm/xe_2/compact_moe_tile_map.hpp` 与 `grouped_gemm_xe2_interface.hpp` | 不新增公开ABI；E128、R1..64、FP16/FP8、FP32 per-expert scale；仅Gemma两个expert shape；每GEMM包含map+计算两kernel，原xe_gemm数学主循环不改，guard不符继续既有V1 m8路径。 |

构建分别由 `LINEAR_TLA_KERNELS_ENABLED`（同时要求TLA/XE2）、`RMS_KERNELS_ENABLED` 和原 MoE 选项控制。新的 target 配置在主 CMake 作用域 include，确保 `cmake --install --component` 实际安装；旧/精简包缺少可选模块时保留现有调用路径，存在但损坏的模块正常报加载错误。

证据与本轮源仓修改前备份：`/llm/models/test/test1/gemma_upstream_port_v2_20260917/source_integration/kernel_repo/`。本次仅同步源码，不以此推断实际安装库或运行服务已经更新。多数大FP8 Linear形状的纯设备时间尚未超过oneDNN；六形状采用TLA不等于全部取得性能优势。

### V2 FP16 NT Linear 补充

| Torch 接口 | 模块 / 源码 | 实际范围与回退 |
|---|---|---|
| `_xpu_C.fp16_nt_gemm(input, weight)` | 可选 `_fp16_C`；`csrc/xpu/fp16_linear/` | Gemma router N128/K2816 与 LM head N131072/K2816，FP16、M1..8、BMG-G31、NT原始weight[N,K]；Ktile32，router Split16，LM head直接TLA。无bias，FP32积累后FP16结果；其余输入回退 `at::linear`。 |

此处合入的是 `fp16_linear/production_v3`，不是旧 `production` 或 lm-head K64 候选。正常target路径无repack；unsupported且A/W未按64B对齐时，先将对应张量clone到对齐连续存储再调用 `at::linear`，以避开已复现的原生未对齐输入错误；该特殊回退copy属于实际调用边界，不能计时外隐藏。由 `FP16_LINEAR_KERNELS_ENABLED` 与TLA/XE2开关控制；现有XPU专用Linear调用入口由vLLM侧负责接线。

## Gemma26B 算子优化补充（2026-09-17，服务 E2E 暂缓）

| 接口 | 实现与范围 |
|---|---|
| `_xpu_C.fp8_moe_decode` | 可选 `_moe_decode_C`，BMG-G31、M1..8、E128/topk8/H2816/I352、FP8 per-expert scale、FP16激活。M1..4直接route的双累加器GEMM+GELU，experts三kernel；M5..8 prepare生成共享map、原TLA GEMM、GELU、GEMM、gather，experts五kernel。不含routing，当前stream，无全局scratch，不改模型或权重布局。Meta返回false，让编译路径继续已有experts；本轮验证为eager算子。 |
| `_xpu_C.moe_prepare_small_m` | 普通SYCL：一次提交产生counts、packed activation和reverse；无EP、FP16、M1..8/H2816/topk8/E128；辅助fallback。 |
| `_xpu_C.moe_gather_small_m` | 普通SYCL向量gather，FP32路由权重，逐route FP32累加，FP16输出。 |
| `_xpu_C.gelu_tanh_and_mul_small_m` | 普通SYCL向量GELU，宽352/1056、rows1..64；保持FP16 GELU中间舍入、处理尾部和大幅输入。 |
| `_xpu_C.rotary_embedding_small_m` | 普通SYCL head并行NeoX RoPE，M1..8、FP16、D256/512，支持partial rotary和外维stride；原地Q/K。 |

算子证据与部分编译备份：`gemma_upstream_port_v2_20260917/{aux_v4,moe_v5}`。
源码已接入XPU入口；已在host_dispatch_v7中备份并同步默认安装。本轮按用户要求不运行服务E2E，不宣称已在server中命中新融合路径。Flash Attention排除；KV写入暂无对应独立ESIMD版本，保持原实现。

### 算子完整调用时间优化补充（2026-09-17）

- FP8 NT Linear：M1滑窗QKV/O、共享gate/up/down采用单工作组Split4+归约，一个kernel；全局QKV/O及M2..8保留原Split16/8策略。各split的循环次数相同，K尾部使用2D block I/O补零，工作组内global fence保证分片可见性。48种shape/M验证通过。
- `gelu_tanh_and_mul_decode`：在C++完成输出分配、支持条件和旧GELU回退；vLLM只修改GeluAndMul.forward_xpu。
- `rms_norm_decode_dispatch`：在既有xpu_kernels IR provider内下移支持条件，不改IR优先级；依赖rms_norm_small_m，旧包走原provider。
- `try_rotary_embedding_small_m`：支持检查在任何enqueue之前；RotaryEmbedding.forward_xpu缓存入口，未命中仍调用旧RoPE。
- 采用依据为完整算子E2E；单kernel不必然更快。三kernel MoE方案、单kernel ALU Linear、全局projection融合候选未采用。
- 默认安装已同步并保留旧SO，未启动服务。证据：`host_dispatch_v7`、`linear_v5`、`moe_v5`；服务TPOT尚未验证。

### M=1优先的Linear分支补充（2026-09-17，m1_v8）

- 本节替代前节“全局QKV/O保留两kernel”的当前策略；旧实现和SO已保留。
- 六种FP8 NT Linear在M=1均使用单kernel工作组内Split-K+归约：全局QKV采用Ntile32/Split2，全局O采用Ntile16/Split4，其他四种保持Ntile64/Split4；不改FP8权重布局。
- M=2..8继续原两kernel分支。48种shape/M的CPU参考、真实kernel trace、非默认stream和动态scale检查通过。
- 同一torch接口内的融合不新增vLLM改动；默认安装已同步。M1 MoE继续含routing共4 kernels；尝试的3-kernel并行专家归约未证明稳定完整调用收益，保留隔离参考。
- 原始证据：gemma_upstream_port_v2_20260917/m1_v8；消费者组合不含attention/collective，不替代完整服务TPOT；本轮未运行服务E2E。

### 原有ABI的库内转接（2026-09-17，legacy_bridge_v11）

- 原 `_xpu_C.fp8_gemm_w8a16` 在支持的小M/Gemma TP2形状下调用新的内部 `try_fp8_gemm_w8a16_tla`，不支持时仍由原入口执行fallback。try入口先完整检查，返回Tensor或None，不调用原op，避免双向转接递归。
- 原 `_C.rms_norm` 转到 `rms_norm_small_m.out`；原 `_C.gelu_tanh_and_mul` 转到 `gelu_tanh_and_mul_small_m`；原 `_C.rotary_embedding` 对NeoX且有K的调用尝试 `try_rotary_embedding_small_m`。没有新增输出copy kernel。
- 保持原公开schema和vLLM源码不变。dtype/shape/stride/输出别名等契约不满足或可选扩展缺失时仍使用原实现；不通过Python monkey patch或重复Torch注册实现转接。
- 无weight RMSNorm在vLLM里已经创建全1权重，该提交无法在库内部消除；FP16 F.linear及Gemma Triton routing绕过此库，仍需vLLM的XPU接线。
- 已完成部分编译，原公开接口123项数值/实际kernel检查、缺扩展23项回退检查通过；三份SO已同步wj-test-new-0.26.0默认安装，原件保留。证据为 `legacy_bridge_v11/{build_manifest,validation_candidate,validation_missing,deployment}.json`；完整服务trace/四版本E2E随后核对。原有M1/M2..8算法策略不变。

### QKV步长输入的默认Norm转接修正（2026-09-17，legacy_bridge_v12）

- v11真实服务trace发现：M=1已全部命中新RMS；M>1的Q/K/V切片因overlap分析返回TooHard而被旧入口的严格No条件误拒，回到原RMS。新SmallMRms本身支持该stride，无需改vLLM或新算法。
- 原入口先检查是否共享storage；不同storage无需重叠分析，共享storage仍只接受明确No的情况。保留原地别名回退，不增加copy或kernel。
- 只重编layernorm.cpp并复用v11其余_C对象重新链接；源码、对象及安装SO哈希和原件均保留于legacy_bridge_v12。
- 48个真实Q/K/V切片（M1..8、D256/512）数值、原地别名回退、非默认stream通过。旧安装trace未命中新RMS，候选及默认安装均命中；默认_C已同步。两种接线方式的M1/M8均完成双rank真实decode trace：FP8、RMS、RoPE、GELU及MoE全部命中新kernel；无接线版每rank四step的1324次旧RMS调用全部转到新RMS，旧RMS设备kernel为0。有接线版另命中新FP16 Linear/routing。两版acc.sh各5/5，四server×B1..8的32组已核对；性能以v12主表为准。

### Linear 分派整理（2026-09-18）

- 新增可选 `_fp16_C` 接口 `_xpu_C.unquantized_gemm(input, weight, bias=None)`，保持 Linear 输入/输出语义；具体支持条件由 native 检查。仅无 bias、FP16、M1..8、K2816/N128、连续且64字节对齐、BMG-G31 的 Router 使用已有 TLA。
- LM head 及其他输入走 `at::linear`；沿用已验证的未对齐 XPU 输入/权重对齐回退。原 `fp16_nt_gemm` 保留历史 Router/LM head 实验能力，算法不变。
- vLLM `layers/utils.py` 提供薄 XPU 分派，旧包缺少新接口时直接 `F.linear`；不再依赖独立 `layers/xpu_linear.py`，不修改模型 forward、IR 注册或 CUDA/CPU 分派。
- 本条源码与验证证据位于 `gemma_upstream_port_v2_20260917/linear_dispatch_review_20260918/`；不继承旧 TPOT 作为本次重测结果。
