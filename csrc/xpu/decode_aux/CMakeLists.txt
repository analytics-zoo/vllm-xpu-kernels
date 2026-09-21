define_gpu_extension_target(
  _decode_aux_C
  DESTINATION vllm_xpu_kernels
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES "csrc/xpu/decode_aux/small_m_aux.cpp"
  COMPILE_FLAGS ${VLLM_GPU_COMPILE_FLAGS} -fno-fast-math -ffp-contract=off
  LINK_FLAGS ${VLLM_GPU_LINK_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/csrc ${VLLM_XPU_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)
