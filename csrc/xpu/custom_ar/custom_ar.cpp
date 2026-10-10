// Low-latency one-shot custom all-reduce for 2 Intel dGPUs (TP=2) over
// PCIe P2P, using Level Zero IPC memory. Registered in _xpu_C; the Python side
// is vllm_xpu_kernels/custom_all_reduce.py.
//
// Setup (once per communicator / device):
//   h = custom_ar_create(max_bytes)          allocate own IPC buffer
//   info = custom_ar_export(h)               [pid, dma-buf fd, cookie, raw handle]
//   (exchange info with the peer over a CPU process group)
//   custom_ar_open(h, rank, peer_info, 7)    import the peer buffer, see there
//     (pidfd_open(peer_pid) + pidfd_getfd(peer_fd) or the driver's own
//     pidfd import of the raw handle), verified via the peer's cookie.
// Per call: custom_ar_all_reduce_(x, h)  in place, one kernel launch on the
//   current torch XPU queue, no host sync, no allocation.
//
// Buffer layout (both ranks identical):
//   [0, kCtrlBytes)  control: flag[w] one per 64-byte line (written by the
//                    PEER), epoch[w] (written only by the owner), err, cookie
//   [kCtrlBytes, +2*max_bytes)  data, 2 halves (parity of the slot epoch).
// Work-group w (slot w) owns bytes [w*kChunk, (w+1)*kChunk) of x.
// Protocol of slot w, epoch e = epoch[w] + 1 (device-side counter, so the
// host passes nothing and the launch is graph-capture friendly):
//   1. load own chunk into registers, store it to peer.data[e&1][chunk]
//   2. release fence (system scope), barrier, leader stores peer.flag[w] = e
//      (plain store, see launch())
//   3. leader spins (acquire, system scope) until own.flag[w] >= e
//   4. barrier, acquire fence, read own.data[e&1][chunk] (= peer's chunk),
//      out = rank0 + rank1 (fp32 add, rounded; same order on both ranks ->
//      bit-identical), store into x; leader stores epoch[w] = e.
// Safety without resets: a rank writes parity (e+1)&1 of slot w only after it
// saw the peer's flag e, i.e. the peer's previous kernel (last reader of that
// parity) has finished (in-order queue). ">=" in step 3 because the peer may
// already have advanced to e+1 before we observed e.
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sys/syscall.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace vllm {
namespace custom_ar {

constexpr int kWgSize = 256;          // work-items per work-group
constexpr int kVecBytes = 16;         // bytes per work-item
constexpr int64_t kChunk = kWgSize * kVecBytes;  // 4 KB per work-group
constexpr int kMaxSlots = 256;        // -> max 1 MB per call
// control region, in uint32 words: flag of slot w at w * kFlagStride (one
// flag per 64-byte line, written by the peer), epoch[w] at kEpochWord + w
// (owner only), error word at kErrWord.
constexpr int kFlagStride = 16;
constexpr int kEpochWord = kMaxSlots * kFlagStride;
constexpr int kErrWord = kEpochWord + kMaxSlots;
constexpr int64_t kCtrlBytes = 32768;
constexpr int64_t kCookieOff = kCtrlBytes - 8;
constexpr uint32_t kSpinLimit = 1u << 30;  // ~ seconds; avoids a hard GPU hang

#define ZE_CHECK(call)                                                   \
  do {                                                                   \
    ze_result_t _r = (call);                                             \
    TORCH_CHECK(_r == ZE_RESULT_SUCCESS, #call " failed: ze_result 0x", \
                std::hex, static_cast<uint32_t>(_r));                    \
  } while (0)

struct State {
  int device = -1;
  int rank = -1;
  int64_t max_bytes = 0;
  ze_context_handle_t ctx = nullptr;
  ze_device_handle_t dev = nullptr;
  char* own = nullptr;   // own buffer (device memory)
  char* peer = nullptr;  // peer buffer opened via IPC
  ze_ipc_mem_handle_t ipc{};
  uint64_t cookie = 0;
  bool exported = false;
  int imported_fd = -1;
};

static std::mutex g_mu;
static std::vector<std::unique_ptr<State>> g_states;

static State& get_state(int64_t h) {
  std::lock_guard<std::mutex> lk(g_mu);
  TORCH_CHECK(h >= 0 && h < (int64_t)g_states.size() && g_states[h],
              "custom_ar: invalid handle ", h);
  return *g_states[h];
}

// ---------------------------------------------------------------- kernel
template <typename T>
struct alignas(16) Vec {
  T v[kVecBytes / sizeof(T)];
};

template <typename T>
class ArKernel;

// Memory ordering (measured alternatives in STATUS.md):
//   writer: every work-item issues a system-scope release fence after its
//   data stores to peer memory; work-group barrier; the leader issues another
//   system-scope release fence and a PLAIN (volatile) store of the flag.
//   reader: the leader spins with system-scope acquire loads on its own flag
//   (in its own device memory); barrier; every work-item issues a
//   system-scope acquire fence before reading the peer's data.
// Measured on B70 x2 (P2P across sockets): an atomic_ref::store to peer
// memory costs ~2 us per work-group (serialized), and flags of different
// work-groups sharing a 64-byte line serialize too (~0.5 us per work-group)
// -- hence plain stores and one flag per 64-byte line. The fences are free.
template <typename T>
static void launch(sycl::queue& q, T* x, int64_t n, char* own, char* peer,
                   int64_t max_bytes, int rank) {
  constexpr int kPer = kVecBytes / sizeof(T);
  const int64_t bytes = n * (int64_t)sizeof(T);
  const int nwg = (int)((bytes + kChunk - 1) / kChunk);
  const bool aligned16 = (reinterpret_cast<uintptr_t>(x) % 16) == 0;
  uint32_t* own_flag = reinterpret_cast<uint32_t*>(own);
  uint32_t* own_epoch = own_flag + kEpochWord;
  uint32_t* own_err = own_flag + kErrWord;
  uint32_t* peer_flag = reinterpret_cast<uint32_t*>(peer);
  char* own_data = own + kCtrlBytes;
  char* peer_data = peer + kCtrlBytes;
  using sycl::memory_order;
  using sycl::memory_scope;
  using sys_ref = sycl::atomic_ref<uint32_t, memory_order::relaxed,
                                   memory_scope::system,
                                   sycl::access::address_space::global_space>;

  auto kern = [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
    const int w = it.get_group(0);
    const int lid = it.get_local_id(0);
    const uint32_t e = own_epoch[w] + 1;
    const int64_t half_off = (e & 1) ? max_bytes : 0;
    const int64_t off = w * kChunk + lid * kVecBytes;  // byte offset
    const int64_t i0 = off / sizeof(T);                // element index
    const bool full = aligned16 && (i0 + kPer <= n);

    // 1. own chunk -> registers -> peer buffer (PCIe P2P writes)
    Vec<T> mine;
    if (full) {
      mine = *reinterpret_cast<const Vec<T>*>(x + i0);
    } else {
#pragma unroll
      for (int k = 0; k < kPer; ++k)
        mine.v[k] = (i0 + k < n) ? x[i0 + k] : T(0);
    }
    if (i0 < n) *reinterpret_cast<Vec<T>*>(peer_data + half_off + off) = mine;
    sycl::atomic_fence(memory_order::release, memory_scope::system);
    sycl::group_barrier(it.get_group());
    if (lid == 0) {
      // 2. publish: peer.flag[w] = e
      sycl::atomic_fence(memory_order::release, memory_scope::system);
      *reinterpret_cast<volatile uint32_t*>(peer_flag + w * kFlagStride) = e;
      // 3. wait for the peer's chunk: own.flag[w] >= e
      sys_ref of(own_flag[w * kFlagStride]);
      uint32_t spins = 0;
      while ((int32_t)(of.load(memory_order::acquire) - e) < 0) {
        if (++spins > kSpinLimit) {
          own_err[0] = 1;  // peer never arrived; the result is garbage
          break;
        }
      }
      sycl::atomic_fence(memory_order::acquire, memory_scope::system);
    }
    sycl::group_barrier(it.get_group());
    sycl::atomic_fence(memory_order::acquire, memory_scope::system);
    // 4. out = rank0 + rank1 (same order on both ranks -> bit-identical)
    if (i0 < n) {
      Vec<T> theirs =
          *reinterpret_cast<const Vec<T>*>(own_data + half_off + off);
      Vec<T> out;
#pragma unroll
      for (int k = 0; k < kPer; ++k) {
        float a = (float)mine.v[k], b = (float)theirs.v[k];
        out.v[k] = rank == 0 ? T(a + b) : T(b + a);
      }
      if (full) {
        *reinterpret_cast<Vec<T>*>(x + i0) = out;
      } else {
#pragma unroll
        for (int k = 0; k < kPer; ++k)
          if (i0 + k < n) x[i0 + k] = out.v[k];
      }
    }
    if (lid == 0) own_epoch[w] = e;
  };
  // nd_launch (sycl_ext_oneapi_enqueue_functions): no sycl::event created.
  sycl::ext::oneapi::experimental::nd_launch<ArKernel<T>>(
      q, sycl::nd_range<1>(nwg * kWgSize, kWgSize), kern);
}

// ---------------------------------------------------------------- ops
int64_t custom_ar_create(int64_t max_bytes) {
  TORCH_CHECK(max_bytes > 0 && max_bytes % kChunk == 0,
              "custom_ar: max_bytes must be a positive multiple of ", kChunk);
  TORCH_CHECK(max_bytes <= kMaxSlots * kChunk, "custom_ar: max_bytes > ",
              kMaxSlots * kChunk);
  auto st = std::make_unique<State>();
  st->device = c10::xpu::current_device();
  st->max_bytes = max_bytes;
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  st->ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      q.get_context());
  st->dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      q.get_device());
  const size_t total = kCtrlBytes + 2 * max_bytes;
  ze_device_mem_alloc_desc_t desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                  nullptr, 0, 0};
  void* p = nullptr;
  ZE_CHECK(zeMemAllocDevice(st->ctx, &desc, total, 4096, st->dev, &p));
  st->own = static_cast<char*>(p);
  q.memset(st->own, 0, total).wait();
  // cookie identifying this buffer, checked by the peer after the IPC open
  st->cookie = ((uint64_t)getpid() << 32) ^ (uint64_t)(uintptr_t)st->own ^
               0x9e3779b97f4a7c15ull;
  q.memcpy(st->own + kCookieOff, &st->cookie, sizeof(uint64_t)).wait();
  std::lock_guard<std::mutex> lk(g_mu);
  g_states.push_back(std::move(st));
  return (int64_t)g_states.size() - 1;
}

// Returns [pid, fd, w0..w15]: the exporting pid, the dma-buf fd (valid only in
// THIS process) and the raw ze_ipc_mem_handle_t as 16 uint32 words.
std::vector<int64_t> custom_ar_export(int64_t h) {
  State& st = get_state(h);
  if (!st.exported) {
    ZE_CHECK(zeMemGetIpcHandle(st.ctx, st.own, &st.ipc));
    st.exported = true;
  }
  uint64_t fd = 0;
  if (zeMemGetFileDescriptorFromIpcHandleExp(st.ctx, st.ipc, &fd) !=
      ZE_RESULT_SUCCESS) {
    // Extension not supported by this driver: on Linux the Intel handle
    // starts with the dma-buf fd (int).
    int fd32 = 0;
    std::memcpy(&fd32, st.ipc.data, sizeof(fd32));
    fd = (uint64_t)fd32;
  }
  std::vector<int64_t> out{(int64_t)getpid(), (int64_t)fd,
                           (int64_t)(st.cookie & 0xffffffffu),
                           (int64_t)(st.cookie >> 32)};
  uint32_t w[ZE_MAX_IPC_HANDLE_SIZE / 4];
  std::memcpy(w, st.ipc.data, sizeof(w));
  for (uint32_t v : w) out.push_back(v);
  return out;
}

// Opens the peer buffer from the peer's custom_ar_export() result. Tries, in
// order: (0) the raw peer handle (drivers with opaque / pidfd IPC handles
// import the fd themselves), (1) pidfd_getfd + zeMemGetIpcHandleFromFileDescriptorExp,
// (2) pidfd_getfd + the peer handle with its fd field replaced by the local fd.
// Returns the mode that succeeded. mode_mask selects allowed modes (bit i).
int64_t custom_ar_open(int64_t h, int64_t rank, std::vector<int64_t> peer,
                       int64_t mode_mask) {
  State& st = get_state(h);
  TORCH_CHECK(rank == 0 || rank == 1, "custom_ar: world size 2 only, rank ",
              rank);
  TORCH_CHECK(st.peer == nullptr, "custom_ar: peer already opened");
  constexpr size_t kHdr = 4;  // pid, fd, cookie lo, cookie hi
  TORCH_CHECK(peer.size() == kHdr + ZE_MAX_IPC_HANDLE_SIZE / 4,
              "custom_ar: bad peer handle info");
  const uint64_t peer_cookie =
      (uint64_t)(uint32_t)peer[2] | ((uint64_t)(uint32_t)peer[3] << 32);
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  // true if p maps the peer's buffer (its cookie is readable through p)
  auto verify = [&](void* p) {
    uint64_t c = 0;
    q.memcpy(&c, static_cast<char*>(p) + kCookieOff, sizeof(c)).wait();
    if (c == peer_cookie) return true;
    zeMemCloseIpcHandle(st.ctx, p);
    return false;
  };
  ze_ipc_mem_handle_t raw{};
  uint32_t w[ZE_MAX_IPC_HANDLE_SIZE / 4];
  for (size_t i = 0; i < ZE_MAX_IPC_HANDLE_SIZE / 4; ++i)
    w[i] = (uint32_t)peer[kHdr + i];
  std::memcpy(raw.data, w, sizeof(w));
  void* p = nullptr;
  int64_t mode = -1;
  ze_result_t r0 = ZE_RESULT_ERROR_UNKNOWN, r1 = r0, r2 = r0;
  if (mode_mask & 1) {
    r0 = zeMemOpenIpcHandle(st.ctx, st.dev, raw, 0, &p);
    if (r0 == ZE_RESULT_SUCCESS) {
      if (verify(p)) mode = 0;
      else r0 = ZE_RESULT_ERROR_INVALID_ARGUMENT;
    }
  }
  if (mode < 0 && (mode_mask & 6)) {
    int pidfd = (int)syscall(SYS_pidfd_open, (pid_t)peer[0], 0);
    TORCH_CHECK(pidfd >= 0, "custom_ar: pidfd_open(", peer[0],
                ") failed, errno ", errno);
    int fd = (int)syscall(SYS_pidfd_getfd, pidfd, (int)peer[1], 0);
    int err = errno;
    close(pidfd);
    TORCH_CHECK(fd >= 0, "custom_ar: pidfd_getfd(", peer[1],
                ") failed, errno ", err,
                " (needs same user / ptrace permission)");
    if (mode_mask & 2) {
      ze_ipc_mem_handle_t ph{};
      r1 = zeMemGetIpcHandleFromFileDescriptorExp(st.ctx, (uint64_t)fd, &ph);
      if (r1 == ZE_RESULT_SUCCESS) {
        r1 = zeMemOpenIpcHandle(st.ctx, st.dev, ph, 0, &p);
        if (r1 == ZE_RESULT_SUCCESS) {
          if (verify(p)) mode = 1;
          else r1 = ZE_RESULT_ERROR_INVALID_ARGUMENT;
        }
      }
    }
    if (mode < 0 && (mode_mask & 4)) {
      ze_ipc_mem_handle_t ph = raw;
      int fd32 = fd;
      std::memcpy(ph.data, &fd32, sizeof(fd32));
      // the fd is now local: also make an embedded exporter pid (opaque
      // handle layout) point at this process
      uint32_t pw[ZE_MAX_IPC_HANDLE_SIZE / 4];
      std::memcpy(pw, ph.data, sizeof(pw));
      for (int i = 1; i < ZE_MAX_IPC_HANDLE_SIZE / 4; ++i)
        if (pw[i] == (uint32_t)peer[0]) pw[i] = (uint32_t)getpid();
      std::memcpy(ph.data + 4, pw + 1, sizeof(pw) - 4);
      r2 = zeMemOpenIpcHandle(st.ctx, st.dev, ph, 0, &p);
      if (r2 == ZE_RESULT_SUCCESS) {
        if (verify(p)) mode = 2;
        else r2 = ZE_RESULT_ERROR_INVALID_ARGUMENT;
      }
    }
    if (mode < 0) close(fd);
    else st.imported_fd = fd;
  }
  TORCH_CHECK(mode >= 0, "custom_ar: could not open peer IPC handle: raw 0x",
              std::hex, (uint32_t)r0, " fdExp 0x", (uint32_t)r1,
              " patched 0x", (uint32_t)r2);
  st.peer = static_cast<char*>(p);
  st.rank = (int)rank;
  return mode;
}

void custom_ar_all_reduce_(at::Tensor& x, int64_t h) {
  State& st = get_state(h);
  TORCH_CHECK(st.peer != nullptr, "custom_ar: peer not opened");
  TORCH_CHECK(x.is_xpu() && x.get_device() == st.device,
              "custom_ar: x must be on xpu:", st.device);
  TORCH_CHECK(x.is_contiguous(), "custom_ar: x must be contiguous");
  const int64_t bytes = x.numel() * x.element_size();
  TORCH_CHECK(bytes <= st.max_bytes, "custom_ar: ", bytes,
              " bytes > max_bytes ", st.max_bytes);
  if (x.numel() == 0) return;
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  switch (x.scalar_type()) {
    case at::kHalf:
      launch(q, reinterpret_cast<sycl::half*>(x.data_ptr()), x.numel(),
             st.own, st.peer, st.max_bytes, st.rank);
      break;
    case at::kBFloat16:
      launch(q, reinterpret_cast<sycl::ext::oneapi::bfloat16*>(x.data_ptr()),
             x.numel(), st.own, st.peer, st.max_bytes, st.rank);
      break;
    case at::kFloat:
      launch(q, x.data_ptr<float>(), x.numel(), st.own, st.peer,
             st.max_bytes, st.rank);
      break;
    default:
      TORCH_CHECK(false, "custom_ar: unsupported dtype ", x.scalar_type(),
                  " (half, bfloat16, float)");
  }
}

// Debug: nonzero if a spin wait timed out (peer never arrived).
int64_t custom_ar_error(int64_t h) {
  State& st = get_state(h);
  uint32_t v = 0;
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  q.memcpy(&v, reinterpret_cast<uint32_t*>(st.own) + kErrWord,
           sizeof(v))
      .wait();
  return v;
}

void custom_ar_destroy(int64_t h) {
  std::unique_ptr<State> st;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    TORCH_CHECK(h >= 0 && h < (int64_t)g_states.size() && g_states[h],
                "custom_ar: invalid handle ", h);
    st = std::move(g_states[h]);
  }
  c10::xpu::getCurrentXPUStream().queue().wait();
  if (st->peer) zeMemCloseIpcHandle(st->ctx, st->peer);
  if (st->imported_fd >= 0) close(st->imported_fd);
  if (st->exported) zeMemPutIpcHandle(st->ctx, st->ipc);
  if (st->own) zeMemFree(st->ctx, st->own);
}

int64_t custom_ar_max_bytes_limit() { return kMaxSlots * kChunk; }

}  // namespace custom_ar
}  // namespace vllm

// _xpu_C entry points (declared in xpu/ops.h).
int64_t custom_ar_create(int64_t max_bytes) {
  return vllm::custom_ar::custom_ar_create(max_bytes);
}
std::vector<int64_t> custom_ar_export(int64_t h) {
  return vllm::custom_ar::custom_ar_export(h);
}
int64_t custom_ar_open(
    int64_t h, int64_t rank, std::vector<int64_t> peer, int64_t mode_mask) {
  return vllm::custom_ar::custom_ar_open(h, rank, std::move(peer), mode_mask);
}
void custom_ar_all_reduce_(at::Tensor& x, int64_t h) {
  vllm::custom_ar::custom_ar_all_reduce_(x, h);
}
int64_t custom_ar_error(int64_t h) { return vllm::custom_ar::custom_ar_error(h); }
void custom_ar_destroy(int64_t h) { vllm::custom_ar::custom_ar_destroy(h); }
int64_t custom_ar_max_bytes_limit() {
  return vllm::custom_ar::custom_ar_max_bytes_limit();
}
