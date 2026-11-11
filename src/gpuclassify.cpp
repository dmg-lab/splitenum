#include "gpuclassify.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "isotopy_graph.h"
#include "isotopy_gpu.cuh"

#include "gpudevice.h"
#include "gpurefine.h"
#include "walk.h"

namespace SplitEnum {
namespace {
std::atomic<uint64_t> g_nodes{0}, g_batches{0}, g_fallbacks{0}, g_new_codes{0},
    g_checked{0}, g_mismatches{0}, g_run_ns{0}, g_resolve_ns{0};

uint64_t ns_since(std::chrono::steady_clock::time_point t0) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - t0)
                                   .count());
}

struct Reference {
  ClassifyResult res;
  uint64_t code = 0;
};

Reference cpu_reference(const SplitTable& table, SignMask sig,
                        const uint8_t* bytes, int nedges) {
  Reference out;
  std::vector<Isotopy::Edge> edges(nedges);
  for (int e = 0; e < nedges; ++e) edges[e] = {bytes[2 * e], bytes[2 * e + 1]};
  try {
    Isotopy::Graph g(table.delta, unpack_signs(sig, table.npoints), edges);
    out.res.viro = g.viro_notation();
    out.res.p = g.p_regions;
    out.res.n = g.n_regions;
    out.res.mcurve = g.is_mcurve();
    out.code =
        IsotopyGPU::tree_code_from_adjacency(g.root_region, g.region_adjacency);
  } catch (const std::exception&) {
    out.res = ClassifyResult();
    out.res.viro = kErrType;
    out.res.err = true;
  }
  return out;
}

#ifndef SPLITENUM_CUDA
}

int gpu_device_count() { return -1; }
GpuDevice* gpu_device_open(int, int, int, const RefineTables*, size_t) {
  return nullptr;
}
void gpu_device_launch(GpuDevice*, const uint8_t*, const uint64_t*, int,
                       void*) {}
void gpu_device_launch_ids(GpuDevice*, const uint64_t*, const uint32_t*,
                           const uint16_t*, int, size_t, void*, uint8_t*) {}
void gpu_device_wait(GpuDevice*) {}
void gpu_device_close(GpuDevice*) {}
void* gpu_host_alloc(size_t) { return nullptr; }
void gpu_host_free(void*) {}

namespace {
#endif

constexpr size_t kIdsPerNode = 32;
constexpr size_t kIdsSlack = 256;

struct NodeRec {
  SignMask sig;
  Profile profile;
  uint32_t weight;
  uint32_t stack_at;
  uint16_t stack_len;
  uint16_t depth;
};

template <typename T>
class HostArray {
 public:
  HostArray() = default;
  HostArray(const HostArray&) = delete;
  HostArray& operator=(const HostArray&) = delete;
  ~HostArray() { release(); }

  void allocate(size_t n, bool pinned) {
    release();
    pinned_ = pinned;
    if (pinned) {
      p_ = static_cast<T*>(gpu_host_alloc(n * sizeof(T)));
      std::fill(p_, p_ + n, T{});
    } else {
      p_ = new T[n]();
    }
  }

  T* data() { return p_; }
  const T* data() const { return p_; }
  T& operator[](size_t i) { return p_[i]; }

 private:
  void release() {
    if (pinned_)
      gpu_host_free(p_);
    else
      delete[] p_;
    p_ = nullptr;
  }

  T* p_ = nullptr;
  bool pinned_ = false;
};

struct Buffer {
  HostArray<uint8_t> edges;
  HostArray<uint64_t> signs;
  HostArray<IsotopyGPU::Result> results;
  HostArray<uint32_t> offsets;
  HostArray<uint16_t> ids;
  std::vector<NodeRec> recs;
  std::vector<uint16_t> stacks;
  GpuDevice* device = nullptr;
  int n = 0;
  size_t nids = 0;
  bool inflight = false;
  bool check = false;
  bool has_edges = false;
};
}

RefineTables refine_tables(const LeanRefine& lean) {
  RefineTables rt{};
  rt.nedges = lean.nedges();
  rt.pool_size = lean.pool_size();
  rt.nsplits = lean.nsplits();
  std::copy(lean.valid(), lean.valid() + kRefineWords, rt.valid);
  rt.forced = lean.forced_rows();
  rt.blocked = lean.blocked_rows();
  rt.cross = lean.cross_rows();
  rt.pool = lean.pool();
  return rt;
}

BatchStats batch_stats() {
  BatchStats s;
  s.nodes = g_nodes.load();
  s.batches = g_batches.load();
  s.fallbacks = g_fallbacks.load();
  s.new_codes = g_new_codes.load();
  s.checked = g_checked.load();
  s.mismatches = g_mismatches.load();
  s.run_ns = g_run_ns.load();
  s.resolve_ns = g_resolve_ns.load();
  return s;
}

void print_batch_stats(std::FILE* fp) {
  const BatchStats s = batch_stats();
  if (s.batches == 0) return;
  std::fprintf(fp,
               "batch classify: %llu nodes in %llu batches, %llu cpu "
               "fallbacks, %llu codes resolved, %llu cross-checked, %llu "
               "mismatches, %.1f thread-s waiting on the kernel, %.1f "
               "thread-s resolving\n",
               static_cast<unsigned long long>(s.nodes),
               static_cast<unsigned long long>(s.batches),
               static_cast<unsigned long long>(s.fallbacks),
               static_cast<unsigned long long>(s.new_codes),
               static_cast<unsigned long long>(s.checked),
               static_cast<unsigned long long>(s.mismatches),
               static_cast<double>(s.run_ns) * 1e-9,
               static_cast<double>(s.resolve_ns) * 1e-9);
}

struct BatchClassifier::Impl {
  const SplitTable& table;
  const LeanRefine& lean;
  Classify backend;
  int capacity;
  uint64_t check_every;
  bool refine;
  IsotopyGPU::Tables tables;
  RefineTables host_rt{};
  int stride = 0;
  size_t ids_capacity = 0;
  Buffer bufs[2];
  int cur = 0;
  uint64_t batch_no = 0;
  std::unordered_map<uint64_t, uint32_t> by_code;
  std::vector<ClassifyResult> known;
  IsotopyGPU::Result feed_result{};
  std::unique_ptr<IsotopyGPU::Scratch> host_scratch;
  std::vector<uint8_t> cpu_edges;

  Impl(const SplitTable& t, const LeanRefine& l, Classify b, int cap,
       uint64_t check, bool gpu_refine)
      : table(t), lean(l), backend(b), capacity(cap), check_every(check),
        refine(gpu_refine), tables(IsotopyGPU::build_tables(t.delta)),
        host_rt(refine_tables(l)),
        ids_capacity(static_cast<size_t>(cap) * kIdsPerNode + kIdsSlack),
        host_scratch(std::make_unique<IsotopyGPU::Scratch>()),
        cpu_edges(2 * static_cast<size_t>(l.nedges()) +
                  2 * 64 * LeanRefine::kWords) {}

  bool async() const { return backend == Classify::Gpu; }

  bool full(const Buffer& b) const {
    return b.n == capacity ||
           (refine && b.nids > static_cast<size_t>(capacity) * kIdsPerNode);
  }

  const uint16_t* stack_of(const Buffer& b, const NodeRec& rec) const {
    return (refine ? b.ids.data() : b.stacks.data()) + rec.stack_at;
  }

  uint8_t* edge_row(Buffer& b, int i) {
    return b.edges.data() + static_cast<size_t>(i) * stride;
  }

  void start(Buffer& b) {
    b.has_edges = !refine;
    if (backend == Classify::Gpu) {
      if (refine) {
        gpu_device_launch_ids(b.device, b.signs.data(), b.offsets.data(),
                              b.ids.data(), b.n, b.nids, b.results.data(),
                              b.check ? b.edges.data() : nullptr);
        b.has_edges = b.check;
      } else {
        gpu_device_launch(b.device, b.edges.data(), b.signs.data(), b.n,
                          b.results.data());
      }
    } else if (backend == Classify::Feed) {
      std::fill(b.results.data(), b.results.data() + b.n, feed_result);
    } else if (refine) {
      for (int i = 0; i < b.n; ++i) {
        RefineScratch rs;
        const uint32_t at = b.offsets[i];
        refine_classify_one(
            tables, host_rt, b.ids.data() + at,
            static_cast<int>(b.offsets[i + 1] - at),
            b.signs.data() + static_cast<size_t>(i) * IsotopyGPU::kQ1Words,
            rs, edge_row(b, i), *host_scratch, IsotopyGPU::Warp<1>{},
            b.results[i]);
      }
      b.has_edges = true;
    } else {
      for (int i = 0; i < b.n; ++i)
        b.results[i] = IsotopyGPU::classify_host(
            tables, edge_row(b, i),
            b.signs.data() + static_cast<size_t>(i) * IsotopyGPU::kQ1Words);
    }
    b.inflight = true;
  }

  void await(Buffer& b) {
    if (backend == Classify::Gpu) gpu_device_wait(b.device);
  }

  void put_sig(Buffer& b, SignMask sig) {
    uint64_t* s =
        b.signs.data() + static_cast<size_t>(b.n) * IsotopyGPU::kQ1Words;
    s[0] = sig;
    for (int w = 1; w < IsotopyGPU::kQ1Words; ++w) s[w] = 0;
  }

  bool append(Buffer& b, SignMask sig, const uint64_t* forced,
              const uint64_t* blocked) {
    uint64_t acc[LeanRefine::kWords];
    std::copy(forced, forced + LeanRefine::kWords, acc);
    if (!lean.sweep(acc, blocked)) return false;
    lean.edges(acc, edge_row(b, b.n));
    put_sig(b, sig);
    return true;
  }

  void append_ids(Buffer& b, SignMask sig, const uint16_t* stack,
                  size_t len) {
    put_sig(b, sig);
    if (b.n == 0) b.offsets[0] = 0;
    std::copy(stack, stack + len, b.ids.data() + b.nids);
    b.nids += len;
    b.offsets[b.n + 1] = static_cast<uint32_t>(b.nids);
  }

  bool cpu_refine(const uint16_t* ids, int n) {
    uint64_t acc[LeanRefine::kWords];
    if (!lean.refine(ids, n, acc)) return false;
    lean.edges(acc, cpu_edges.data());
    return true;
  }

  void launch(Buffer& b) {
    b.check = check_every != 0 && batch_no % check_every == 0;
    ++batch_no;
    const auto t0 = std::chrono::steady_clock::now();
    start(b);
    g_run_ns.fetch_add(ns_since(t0));
  }

  void finish(Buffer& b, TypeTable& types, uint64_t& errs) {
    const auto t0 = std::chrono::steady_clock::now();
    await(b);
    g_run_ns.fetch_add(ns_since(t0));
    const auto t1 = std::chrono::steady_clock::now();
    resolve(b, types, errs);
    g_resolve_ns.fetch_add(ns_since(t1));
    b.n = 0;
    b.nids = 0;
    b.inflight = false;
    b.recs.clear();
    b.stacks.clear();
  }

  void resolve(Buffer& b, TypeTable& types, uint64_t& errs) {
    const int nedges = tables.nedges;
    const size_t nbytes = 2 * static_cast<size_t>(nedges);
    uint64_t fallbacks = 0, fresh = 0, checked = 0;
    for (int i = 0; i < b.n; ++i) {
      const IsotopyGPU::Result& r = b.results[i];
      const NodeRec& rec = b.recs[i];
      const uint16_t* stack = stack_of(b, rec);
      Reference ref;
      const ClassifyResult* res = nullptr;
      const uint8_t* bytes = b.has_edges ? edge_row(b, i) : nullptr;
      bool refined = true;
      if (refine && (b.check || r.flags != IsotopyGPU::kOk)) {
        refined = cpu_refine(stack, rec.stack_len);
        const bool kernel_ok = r.flags != kRefineFailed;
        const bool same =
            refined == kernel_ok &&
            (!refined || !bytes ||
             std::memcmp(bytes, cpu_edges.data(), nbytes) == 0);
        if (!same) {
          report_refine_mismatch(rec, refined, kernel_ok);
          ++errs;
        }
        if (refined) bytes = cpu_edges.data();
      }
      const auto reference = [&]() {
        if (!bytes) {
          if (!cpu_refine(stack, rec.stack_len)) {
            report_refine_mismatch(rec, false, true);
            ++errs;
            Reference failed;
            failed.res.viro = kErrType;
            failed.res.err = true;
            return failed;
          }
          bytes = cpu_edges.data();
        }
        return cpu_reference(table, rec.sig, bytes, nedges);
      };
      if (!refined) {
        ref.res.viro = kErrType;
        ref.res.err = true;
        res = &ref.res;
      } else if (r.flags != IsotopyGPU::kOk) {
        ref = reference();
        ++fallbacks;
        res = &ref.res;
      } else {
        auto it = by_code.find(r.code);
        if (it == by_code.end()) {
          ref = reference();
          ++fresh;
          if (ref.res.err || ref.code != r.code || ref.res.p != r.p ||
              ref.res.n != r.n) {
            report_mismatch(r, rec, ref);
            ++errs;
            res = &ref.res;
          } else {
            it = by_code
                     .emplace(r.code, static_cast<uint32_t>(known.size()))
                     .first;
            known.push_back(ref.res);
            res = &known[it->second];
          }
        } else {
          const ClassifyResult& k = known[it->second];
          bool ok = k.p == r.p && k.n == r.n;
          if (b.check || !ok) {
            ref = reference();
            ++checked;
            ok = ok && !ref.res.err && ref.code == r.code &&
                 ref.res.viro == k.viro && ref.res.p == k.p && ref.res.n == k.n;
          }
          if (ok) {
            res = &k;
          } else {
            report_mismatch(r, rec, ref);
            ++errs;
            res = &ref.res;
          }
        }
      }
      if (res->err) ++errs;
      types.add(res->viro, res->p, res->n, res->mcurve, rec.profile,
                rec.weight, rec.depth, stack, rec.stack_len);
    }
    g_nodes.fetch_add(static_cast<uint64_t>(b.n));
    g_batches.fetch_add(1);
    g_fallbacks.fetch_add(fallbacks);
    g_new_codes.fetch_add(fresh);
    g_checked.fetch_add(checked);
  }

  void report_mismatch(const IsotopyGPU::Result& r, const NodeRec& rec,
                       const Reference& ref) {
    if (g_mismatches.fetch_add(1) >= 10) return;
    std::fprintf(stderr,
                 "splitenum: kernel disagrees with Isotopy::Graph: sig %016llx "
                 "kernel code %016llx p %d n %d flags %u, cpu code %016llx "
                 "p %d n %d %s\n",
                 static_cast<unsigned long long>(rec.sig),
                 static_cast<unsigned long long>(r.code), r.p, r.n, r.flags,
                 static_cast<unsigned long long>(ref.code), ref.res.p,
                 ref.res.n, ref.res.viro.c_str());
  }

  void report_refine_mismatch(const NodeRec& rec, bool cpu_ok,
                              bool kernel_ok) {
    if (g_mismatches.fetch_add(1) >= 10) return;
    std::fprintf(stderr,
                 "splitenum: kernel refine disagrees with LeanRefine: sig "
                 "%016llx depth %u, cpu %s, kernel %s\n",
                 static_cast<unsigned long long>(rec.sig), rec.depth,
                 cpu_ok ? "refines" : "fails",
                 kernel_ok ? "refines" : "fails");
  }
};

BatchClassifier::BatchClassifier(const SplitTable& table,
                                 const LeanRefine& lean, Classify backend,
                                 int slot, int capacity, uint64_t check_every,
                                 bool refine_gpu)
    : impl_(std::make_unique<Impl>(table, lean, backend, std::max(capacity, 1),
                                   check_every, refine_gpu)) {
  Impl& m = *impl_;
  if (table.delta > IsotopyGPU::kMaxDelta)
    throw std::runtime_error("the batch kernel is built for delta <= " +
                             std::to_string(IsotopyGPU::kMaxDelta));
  if (m.tables.nedges != lean.nedges() || m.tables.nverts != table.npoints)
    throw std::runtime_error("the batch kernel and the lean refine disagree "
                             "on the triangulation size");
  if (lean.pool_size() > 64 * kRefineWords)
    throw std::runtime_error("the refine pool exceeds the kernel bitset");
  m.stride = 2 * m.tables.nedges;
  int ndev = 0;
  if (backend == Classify::Gpu) {
    ndev = gpu_device_count();
    if (ndev < 0)
      throw std::runtime_error("--classify gpu: this binary has no CUDA "
                               "backend, build splitenum-gpu with make gpu");
    if (ndev == 0)
      throw std::runtime_error("--classify gpu: no CUDA device visible");
  }
  const int nbufs = m.async() ? 2 : 1;
  for (int k = 0; k < nbufs; ++k) {
    Buffer& b = m.bufs[k];
    b.edges.allocate(static_cast<size_t>(m.capacity) * m.stride +
                         2 * 64 * LeanRefine::kWords,
                     m.async());
    b.signs.allocate(static_cast<size_t>(m.capacity) * IsotopyGPU::kQ1Words,
                     m.async());
    b.results.allocate(static_cast<size_t>(m.capacity), m.async());
    if (m.refine) {
      b.offsets.allocate(static_cast<size_t>(m.capacity) + 1, m.async());
      b.ids.allocate(m.ids_capacity, m.async());
    }
    b.recs.reserve(m.capacity);
    if (m.async())
      b.device = gpu_device_open(slot % ndev, table.delta, m.capacity,
                                 m.refine ? &m.host_rt : nullptr,
                                 m.ids_capacity);
  }
  if (backend == Classify::Feed) {
    const uint64_t zero[LeanRefine::kWords] = {};
    Buffer& b = m.bufs[0];
    m.append(b, table.harnack, zero, zero);
    m.feed_result = IsotopyGPU::classify_host(m.tables, b.edges.data(),
                                              b.signs.data());
    m.by_code.emplace(m.feed_result.code, 0);
    m.known.push_back(
        cpu_reference(table, table.harnack, b.edges.data(), m.tables.nedges)
            .res);
  }
}

BatchClassifier::~BatchClassifier() {
  if (!impl_) return;
  for (Buffer& b : impl_->bufs) gpu_device_close(b.device);
}

bool BatchClassifier::refines() const { return impl_->refine; }

bool BatchClassifier::full() const {
  return impl_->full(impl_->bufs[impl_->cur]);
}

bool BatchClassifier::add(SignMask sig, const uint64_t* forced,
                          const uint64_t* blocked, Profile profile,
                          uint32_t weight, int depth, const uint16_t* stack,
                          size_t stack_len) {
  Impl& m = *impl_;
  Buffer& b = m.bufs[m.cur];
  if (!m.append(b, sig, forced, blocked)) return false;
  NodeRec rec;
  rec.sig = sig;
  rec.profile = profile;
  rec.weight = weight;
  rec.stack_at = static_cast<uint32_t>(b.stacks.size());
  rec.stack_len = static_cast<uint16_t>(stack_len);
  rec.depth = static_cast<uint16_t>(depth);
  b.recs.push_back(rec);
  b.stacks.insert(b.stacks.end(), stack, stack + stack_len);
  ++b.n;
  return true;
}

void BatchClassifier::add_ids(SignMask sig, Profile profile, uint32_t weight,
                              int depth, const uint16_t* stack,
                              size_t stack_len) {
  Impl& m = *impl_;
  Buffer& b = m.bufs[m.cur];
  NodeRec rec;
  rec.sig = sig;
  rec.profile = profile;
  rec.weight = weight;
  rec.stack_at = static_cast<uint32_t>(b.nids);
  rec.stack_len = static_cast<uint16_t>(stack_len);
  rec.depth = static_cast<uint16_t>(depth);
  m.append_ids(b, sig, stack, stack_len);
  b.recs.push_back(rec);
  ++b.n;
}

void BatchClassifier::rotate(TypeTable& types, uint64_t& errs) {
  Impl& m = *impl_;
  if (!m.async()) {
    flush(types, errs);
    return;
  }
  Buffer& filled = m.bufs[m.cur];
  if (filled.n > 0) m.launch(filled);
  m.cur ^= 1;
  Buffer& next = m.bufs[m.cur];
  if (next.inflight) m.finish(next, types, errs);
}

void BatchClassifier::flush(TypeTable& types, uint64_t& errs) {
  Impl& m = *impl_;
  Buffer& open = m.bufs[m.cur];
  if (open.n > 0) m.launch(open);
  for (int k = 1; k <= 2; ++k) {
    Buffer& b = m.bufs[(m.cur + k) & 1];
    if (b.inflight) m.finish(b, types, errs);
  }
}

void BatchClassifier::pin() {
  Impl& m = *impl_;
  if (m.backend == Classify::Feed) return;
  Buffer& b = m.bufs[m.cur];
  if (b.n != 0 || b.inflight)
    throw std::logic_error("pin on a non-empty batch");
  constexpr int kW = LeanRefine::kWords;
  const uint64_t zero[kW] = {};
  const int nsplits = static_cast<int>(m.table.size());
  const size_t nbytes = static_cast<size_t>(m.stride);
  std::vector<std::string> labels;
  const auto verify = [&]() {
    b.check = true;
    m.start(b);
    m.await(b);
    for (int i = 0; i < b.n; ++i) {
      const IsotopyGPU::Result& r = b.results[i];
      const uint64_t sig =
          b.signs[static_cast<size_t>(i) * IsotopyGPU::kQ1Words];
      const uint8_t* bytes = m.edge_row(b, i);
      if (m.refine) {
        const uint32_t at = b.offsets[i];
        const bool cpu_ok = m.cpu_refine(
            b.ids.data() + at, static_cast<int>(b.offsets[i + 1] - at));
        const bool kernel_ok = r.flags != kRefineFailed;
        if (cpu_ok != kernel_ok ||
            (cpu_ok && std::memcmp(bytes, m.cpu_edges.data(), nbytes) != 0))
          throw std::runtime_error(
              "the batch kernel refine disagrees with LeanRefine on " +
              labels[i]);
        if (!cpu_ok) continue;
      }
      const Reference ref = cpu_reference(m.table, sig, bytes, m.tables.nedges);
      if (r.flags != IsotopyGPU::kOk || ref.res.err || r.code != ref.code ||
          r.p != ref.res.p || r.n != ref.res.n)
        throw std::runtime_error(
            "the batch kernel disagrees with Isotopy::Graph on " + labels[i]);
    }
    b.n = 0;
    b.nids = 0;
    b.inflight = false;
    b.check = false;
    labels.clear();
  };
  const auto push = [&](const std::vector<uint16_t>& ids, SignMask sig,
                        std::string label) {
    if (m.refine) {
      m.append_ids(b, sig, ids.data(), ids.size());
    } else {
      const uint64_t* f = ids.empty() ? zero : m.lean.forced_row(ids[0]);
      const uint64_t* bl = ids.empty() ? zero : m.lean.blocked_row(ids[0]);
      if (!m.append(b, sig, f, bl))
        throw std::runtime_error("lean refine failed while pinning the "
                                 "batch kernel");
    }
    labels.push_back(std::move(label));
    ++b.n;
    if (m.full(b)) verify();
  };
  push({}, m.table.harnack, "the empty collection");
  for (int s = 0; s < nsplits; ++s)
    push({static_cast<uint16_t>(s)}, m.table.harnack ^ m.table.recs[s].twist,
         "split " + std::to_string(s));
  if (m.refine) {
    std::mt19937_64 rng(0x5eed5eedULL);
    for (int k = 0; k < 4096; ++k) {
      const int len = 1 + static_cast<int>(rng() % 8);
      std::vector<uint16_t> ids(len);
      SignMask sig = m.table.harnack;
      for (uint16_t& id : ids) {
        id = static_cast<uint16_t>(rng() % nsplits);
        sig ^= m.table.recs[id].twist;
      }
      std::string label = "random collection";
      for (uint16_t id : ids) label += " " + std::to_string(id);
      push(ids, sig, std::move(label));
    }
  }
  if (b.n > 0) verify();
}
}
