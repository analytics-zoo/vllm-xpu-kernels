# Included from repository root: keep install rules in root scope.
define_gpu_extension_target(
  _fp16_C
  DESTINATION vllm_xpu_kernels
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES "csrc/xpu/fp16_linear/fp16_linear_tla.cpp"
  COMPILE_FLAGS ${SYCL_TLA_KERNELS_COMPILE_FLAGS}
    -fsycl-targets=spir64_gen -fno-fast-math -Wno-deprecated-declarations
  LINK_FLAGS -fsycl -fsycl-targets=spir64_gen
    "SHELL:-Xs '-device bmg-g31-a0 -options -doubleGRF'"
    "SHELL:-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate"
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${VLLM_XPU_INCLUDE_DIR} ${SYCL_TLA_INCLUDE_DIRS}
  USE_SABI 3
  WITH_SOABI)
