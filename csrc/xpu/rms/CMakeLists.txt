# Included from repository root: keep install rules in root scope.
define_gpu_extension_target(
  _rms_C
  DESTINATION vllm_xpu_kernels
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES "csrc/xpu/rms/weightless_rms.cpp" "csrc/xpu/rms/small_m_rms.cpp"
  COMPILE_FLAGS ${VLLM_GPU_COMPILE_FLAGS} -ffp-contract=off
  LINK_FLAGS ${VLLM_GPU_LINK_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/csrc ${VLLM_XPU_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)
