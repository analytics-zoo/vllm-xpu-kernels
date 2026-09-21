# Independent partial-build target, using the upstream TLA arithmetic helpers.
define_gpu_extension_target(
  _moe_decode_C
  DESTINATION vllm_xpu_kernels
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES "csrc/xpu/moe_decode/moe_decode.cpp"
  COMPILE_FLAGS ${SYCL_TLA_KERNELS_COMPILE_FLAGS}
    -fsycl-targets=spir64_gen -fno-fast-math -ffp-contract=off
    -Wno-deprecated-declarations -Wno-pass-failed
  LINK_FLAGS -fsycl -fsycl-targets=spir64_gen
    "SHELL:-Xs '-device bmg-g31-a0 -internal_options -cl-intel-256-GRF-per-thread'"
    "SHELL:-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate"
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/csrc
    ${PROJECT_SOURCE_DIR}/csrc/xpu/grouped_gemm/xe_2 ${SYCL_TLA_INCLUDE_DIRS}
  LIBRARIES torch_xpu c10_xpu
  USE_SABI 3
  WITH_SOABI)
