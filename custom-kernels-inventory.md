# Qwen3.8 普通 SYCL 算子与安装清单

`QWEN38_KERNELS_ENABLED` 控制独立 `_qwen38_C` 扩展，可只构建/安装
此目标，不重编已有扩展。计算采用普通 SYCL、subgroup 与 SYCL-TLA，
不包含 `sycl::ext::intel::esimd`，不链接 `custom_esimd_kernels_vllm`。
Torch dispatcher 与 direct Pybind 共用这一 DSO，不混载独立测试库。

| 基础入口 | 契约 |
| --- | --- |
| `q4_0_quantize` | XPU FP16/BF16 连续 `[N,K]`，K为128倍数；输出 unsigned nibble 的 int32 `[N,K/8]` 和有符号 FP16 scale。保持 GGML Q4_0 位值。 |
| `int4_linear` | FP16 `[M,K]`、uint8 `[N,K/2]` 与 signed FP16 scale；M1按 FP32 dequant，M>1按 FP16 dequant 后 FP32 累加。M2–8 适配形状走 XMX，其余走普通 SYCL。 |
| `int4_linear_fused2` | M1两个独立投影、两个 caller-owned 输出，一次提交。 |
| `ngram_decode_ids` | int64 `[M]`、历史 `[M,2]`；uint64 wrapping hash＋signed remainder，输出 `[M,16]`。 |
| `ngram_host_lookup` / `chunked` | XPU读取 pinned CPU FP16/BF16 `[V,160]` 的原始位值；TP局部词表外输出零。chunked保留TP4的分块加载布局。 |

| 模块族 | 统一模块入口与 provider |
| --- | --- |
| HC | `_qwen38_C` 的 norm/combine/gate/down/up；`qwen38_hc_sycl` workspace 和同一 DSO 的 direct class。TP4维度10240/2560、低秩320，M1/M2–8。另有 `hc_prefill_grouped_norm`、`hc_prefill_combine`、`hc_prefill_combine_norm`、`hc_prefill_gate_mix` 四个 M9–4096 入口，只融合原FP16边界/FP32数学，不改变GEMM；caller-owned输出、alias预检和当前in-order stream生命周期保留。 |
| PLE | 13个 `ple_*` dispatcher：NGram、embedding、norm、score/gate、融合 norm、residual 和 decode/prefill/spec short-conv。trusted 元数据只接受生产者证明，不把普通调用伪装 trusted。 |
| GDN | `gdn_decode_sycl`、`gdn_spec_v2_sycl`、`gdn_norm_gate_sycl`、`gdn_norm_int4_sycl`；`qwen38_gdn.py` 保留模型侧参数顺序。TP4/TP8 sequential 几何；不宣告未实现的 interleaved 布局。 |
| MoE | `qwen38_moe_sycl` dispatcher/classes、同一 DSO 的 direct workspace；`qwen38_moe.py` 提供 compact80/160 tuple与prefill DOWN。router是Q4_0，routed权重是signed-S4，不能混淆。 |
| QSA | direct Pybind 的 compression、selection、token-split attention、QKV/norm/RoPE、cache-store/M1 transaction；`qwen38_qsa.py` 保留旧 positional ABI并传真实 host max_seq_len。M≤4096，C++分块复用显式 scratch。BMG G31、localQ6、M≥128 的 FP16 prefill 默认采用三头 QK-DPAS，P/PV仍FP32，空tile跳过；小M和不支持布局保持旧后端，可用 `QWEN38_QSA_PREFILL_QK_DPAS=0` 回切。 |
| Host preflight | 私有 Pybind `tensors_disjoint_host` 检查实际地址与带 pitch 的内存行；`TensorBindingSnapshotV1` 批量核对活动 tensor 绑定。均无设备提交。Python 调用方必须先过滤自定义 tensor/Torch modes；inference 内容 reload 仍需显式清除缓存。 |
| GDN M1 host transaction | 私有 `GDNM1WorkspaceDirectV1` 合并输入 INT4 投影、GDN core、norm+输出 INT4 的主机检查/提交，复用原三个设备实现，不是单设备 kernel。仅 TP4/TP8 FP16 M1 sequential；其余在提交前返回 None。活动参数、别名与 stream 检查保留，转换失败清除旧绑定见证，reload 调用 invalidate。 |
| GDN speculative host transaction | 私有 `GDNSpecWorkspaceDirectV1` 合并两次 INT4 输入投影、spec GDN、norm 和 INT4 输出投影的主机检查/提交，仍是多个设备 kernel。仅 TP4/TP8、FP16、单个 speculative 序列、M2–8、sequential；接受 padded state/token indices，并保留 FP32 A_log。Python 侧 sigmoid norm 仅允许原权重 FP16，其余提交前回退。实时绑定、alias、current-stream 与 reload 失效规则同 M1；scratch 分配全部成功后才发布，分配失败可重试。 |

所有 native try/fallback 在首次提交前校验支持条件；提交后的异常传播，
不能重放会写同一状态的 fallback。使用输入设备的 current stream，
记录 allocator 生命周期；recordStream 不是跨 stream 数据依赖。
host lookup 同时记录 pinned host allocation，提前释放 Python owner
也不得在队列完成前回收。workspace 按 stream/M 持有，不能跨飞行调用共享。

配套安装文件为 `_qwen38_C.cpython-*.so`、`qwen38.py`、
`qwen38_gdn.py`、`qwen38_moe.py`、`qwen38_qsa.py`。
Pybind direct Tensor ABI需要对应 CPython/Torch，不能标成abi3扩展。
部分构建示例：`cmake --build <build> --target _qwen38_C -j2`；
部分安装用 `cmake --install <build> --component _qwen38_C --prefix <site-packages>`。
安装前备份/hash核对，通过新进程确认真实导入路径，不把build-only环境变量
当作默认安装完成。完整模型 async/MTP、多模态、准确性和 E2E 仍以运行记录
为准，代码/算子单测存在不代表整链验收。
