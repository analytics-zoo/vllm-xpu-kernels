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
| 私有 greedy argmax | `try_local_argmax_packet` 接受单行FP16词表分片；`try_global_argmax_token` / `try_global_argmax_token_int32` 接受 `[4,64]` 或 `[8,64]` FP32运输包，分别输出int64/int32 token。保留NaN优先、低rank/低token同分及padding语义；`greedy_argmax_sycl_tp_sizes=(4,8)` 让调用方在提交前检查旧库能力，ABI1的旧八卡接口保留。 |
| `qwen38_fp8.try_qwen38_fp8_linear` | 独立 `_qwen38_fp8_C` 的普通 SYCL-TLA W8A16 投影，FP16 activation、FP8 E4M3FN `[N,K]`、FP32 scalar scale；TP4/TP8 M1–8 原生宽度，包括 N12 和 K80 尾部。提交前拒绝不支持的布局/设备，提交后异常传播。 |
| `qwen38_fp8.try_qwen4_dense_fp8_linear` / `try_qwen4_dense_fp8_pair` | B70 TP2 dense Qwen4 的 M1 W8A16 SYCL-TLA 投影；复用既有 FP8 GEMV body，按实测 shape 选择 K 分片。pair 将 GDN `[8192,5120]` 与 `[48,5120]` 两个真实投影合为一次提交；两套 scalar scale 独立读取，无额外 padding。模型标记限制调用范围，旧 DSO 缺少可选入口时保留原路径。 |
| `qwen38_fp8.Workspace.try_run` | TP4/TP8 M1–8 的 router/top10、routed/shared gate+up、DOWN、归约；routed 使用 contiguous KN、shared 使用 NK，I80/I160 不填充。每 stream/shape scratch；活动权重每次传入，输出独立拥有。支持动态输入 FULL graph replay。 |
| `qwen38_fp8.try_gdn_norm` | TP4/TP8 `[M,12/6,128]` FP16 激活，FP16/FP32 norm weight 按原精度读取，FP32 RMS norm 和 sigmoid/SiLU，FP16 输出；M1–4096。默认 GDN 输出链使用普通 SYCL norm，再交给现有最快 SYCL-TLA FP8 投影。 |
| `ngram_decode_ids` | int64 `[M]`、历史 `[M,2]`；uint64 wrapping hash＋signed remainder，输出 `[M,16]`。 |
| `ngram_host_lookup` / `chunked` | XPU读取 pinned CPU FP16/BF16 `[V,160]` 的原始位值；TP局部词表外输出零。chunked保留TP4的分块加载布局。 |

| 模块族 | 统一模块入口与 provider |
| --- | --- |
| HC | `_qwen38_C` 的 norm/combine/gate/down/up；`qwen38_hc_sycl` workspace 和同一 DSO 的 direct class。TP4维度10240/2560、低秩320，M1/M2–8。另有 `hc_prefill_grouped_norm`、`hc_prefill_combine`、`hc_prefill_combine_norm`、`hc_prefill_gate_mix` 四个 M9–4096 入口，只融合原FP16边界/FP32数学，不改变GEMM；caller-owned输出、alias预检和当前in-order stream生命周期保留。 |
| Qwen4 dense HC | `hc_dense_grouped_norm`、`hc_dense_combine_norm`、`hc_dense_gate_mix` 覆盖 FP16 `[M,20480]`、4路各5120、M1–8；combine 接受带行间 padding 的 `[M,4]` injection。合并残差先舍入FP16，再做FP32 grouped Gemma RMSNorm；gate沿用FP32 sigmoid/mean。保留原FP16投影和SSM状态精度；旧DSO没有新入口时模型保持原回退。 |
| Qwen4 dense HC down | `hc_dense_down_silu` 接受 TP2 的 FP16 `[320/336,20480]` 活动权重；M1 的 40 路 K 分片与 SiLU 融合。merged 只计算 320 个低秩和 4 个 injection 输出，跳过 12 行无效 padding；保留投影、除以 4 和 SiLU 的 FP16 舍入边界。支持当前 stream 和动态 graph replay，不缓存权重副本。 |
| PLE | 13个 `ple_*` dispatcher：NGram、embedding、norm、score/gate、融合 norm、residual 和 decode/prefill/spec short-conv。trusted 元数据只接受生产者证明，不把普通调用伪装 trusted。 |
| GDN | `gdn_decode_sycl`、`gdn_spec_v2_sycl`、`gdn_norm_gate_sycl`、`gdn_norm_int4_sycl`；`qwen38_gdn.py` 保留模型侧参数顺序。TP4/TP8 sequential 几何；不宣告未实现的 interleaved 布局。 |
| MoE | `qwen38_moe_sycl` dispatcher/classes、同一 DSO 的 direct workspace；`qwen38_moe.py` 提供 compact80/160 tuple与prefill DOWN。router是Q4_0，routed权重是signed-S4，不能混淆。 |
| QSA | direct Pybind 的 compression、selection、token-split attention、QKV/norm/RoPE、cache-store/M1 transaction；`qwen38_qsa.py` 保留旧 positional ABI并传真实 host max_seq_len。M1 selection的4096–256000 capture复用四块原scratch并行FP32打分，以SG32固定网络排序，再按32→8→1归并；page64/128共用，graph拓扑固定，M>1保持原路径。M≤4096，C++分块复用显式 scratch。BMG G31、localQ6、M≥128 的 FP16 prefill 默认采用三头 QK-DPAS，P/PV仍FP32，空tile跳过；小M和不支持布局保持旧后端，可用 `QWEN38_QSA_PREFILL_QK_DPAS=0` 回切。 |
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

## Qwen3.8 M1 QSA attention 提前读取

对齐的 FP16 M1 attention 在每个 SG32 worker 内提前读取两个选中 token。
每个 head 的 dot、subgroup 归约、online softmax 和 partial 合并
保持原有 FP32 运算顺序；无效 slot、缺页和越界 request 在读取 KV 前屏蔽。
工作区仍为 `[M,H,43,258]`，M1 使用 33 个 live partial，
每个 partial 覆盖 64 个选中 slot。未对齐指针和 M>1 保持原读取顺序。
该实现适用于 TP4 INT4 的 6 heads 和 TP8 INT4/FP8 的 3 heads。

## Qwen3.8 TP8 INT4 attention 输出投影

M1 `[1,768]` 到 `[1,2560]` 的 Q4_0 输出投影使用每个 SG16 两行、
每个工作组一个 subgroup，保留原通用 kernel 的逐 lane FP32 FMA 链、
signed scale、SG16 归约及未对齐 storage offset 契约。
其余形状与 M>1 的工作组打包保持原配置。

## Qwen3.8 M1 QSA 元数据及 partial 读取

对齐的 M1 SG32 worker 在寄存器中批量保存其索引和物理页号，
按原 token 顺序取用，并跳过最后一个有效物理 slot 后的无效尾部。
任意位置的负索引、缺页、越界 request 仍按原契约屏蔽。
默认 M1 合并使用 64 列 tile，提前读取四个 partial，保留逐列 FP32 累加顺序；
显式 `VLLM_XPU_QWEN38_QSA_M1_MERGE_TILE` 覆盖和 M>1 保持原选择。
没有增加 workspace、图节点或改变公共 ABI。
