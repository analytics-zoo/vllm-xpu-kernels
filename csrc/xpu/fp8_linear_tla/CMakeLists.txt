# BMG-G31 FP8 NT small-M linear. Other hardware uses the original op.
define_gpu_extension_target(
  _linear_C
  DESTINATION vllm_xpu_kernels
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/fp8_linear_tla.cpp"
  COMPILE_FLAGS ${SYCL_TLA_KERNELS_COMPILE_FLAGS}
    -fsycl-targets=spir64_gen -fno-fast-math
    -Wno-deprecated-declarations -Wno-pass-failed
  LINK_FLAGS -fsycl -fsycl-targets=spir64_gen
    "SHELL:-Xs '-device bmg-g31-a0 -options -doubleGRF'"
    "SHELL:-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate"
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/csrc ${SYCL_TLA_INCLUDE_DIRS}
  LIBRARIES torch_xpu c10_xpu
  USE_SABI 3
  WITH_SOABI)
