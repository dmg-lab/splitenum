#include "jobs.h"

#include "gpuclassify.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace SplitEnum {
namespace {
constexpr uint32_t kJobMagic = 0x424a4553;
constexpr uint32_t kResultMagic = 0x524a4553;
constexpr uint32_t kBatchMagic = 0x514a4553;
constexpr uint32_t kStateMagic = 0x534a4553;
constexpr uint16_t kWireVersion = 3;

class Writer {
 public:
  explicit Writer(std::vector<uint8_t>& out) : out_(out) {}

  template <typename T>
  void put(T v) {
    const size_t at = out_.size();
    out_.resize(at + sizeof v);
    std::memcpy(out_.data() + at, &v, sizeof v);
  }

  template <typename T>
  void put_array(const std::vector<T>& v) {
    put(static_cast<uint32_t>(v.size()));
    const size_t at = out_.size();
    out_.resize(at + v.size() * sizeof(T));
    if (!v.empty()) std::memcpy(out_.data() + at, v.data(), v.size() * sizeof(T));
  }

  void put_string(const std::string& s) {
    put(static_cast<uint32_t>(s.size()));
    out_.insert(out_.end(), s.begin(), s.end());
  }

 private:
  std::vector<uint8_t>& out_;
};

class Reader {
 public:
  explicit Reader(const std::vector<uint8_t>& in)
      : p_(in.data()), end_(in.data() + in.size()) {}

  template <typename T>
  T get() {
    need(sizeof(T));
    T v;
    std::memcpy(&v, p_, sizeof v);
    p_ += sizeof v;
    return v;
  }

  template <typename T>
  std::vector<T> get_array() {
    const uint32_t n = get<uint32_t>();
    need(static_cast<size_t>(n) * sizeof(T));
    std::vector<T> v(n);
    if (n) std::memcpy(v.data(), p_, n * sizeof(T));
    p_ += n * sizeof(T);
    return v;
  }

  std::string get_string() {
    const uint32_t n = get<uint32_t>();
    need(n);
    std::string s(reinterpret_cast<const char*>(p_), n);
    p_ += n;
    return s;
  }

  bool done() const { return p_ == end_; }

 private:
  void need(size_t n) const {
    if (static_cast<size_t>(end_ - p_) < n)
      throw std::runtime_error("job message truncated");
  }

  const uint8_t* p_;
  const uint8_t* end_;
};

void put_header(Writer& w, uint32_t magic, uint64_t config_hash) {
  w.put(magic);
  w.put(kWireVersion);
  w.put(config_hash);
}

void check_header(Reader& r, uint32_t magic, uint64_t config_hash) {
  if (r.get<uint32_t>() != magic)
    throw std::runtime_error("job message has the wrong magic");
  if (r.get<uint16_t>() != kWireVersion)
    throw std::runtime_error("job message has an unknown version");
  if (r.get<uint64_t>() != config_hash)
    throw std::runtime_error(
        "job message belongs to a different walk configuration");
}

void put_job(Writer& w, const Job& j) {
  w.put(j.lo);
  w.put(j.hi);
  w.put(static_cast<uint8_t>(j.root));
  w.put_array(j.prefix);
}

Job get_job(Reader& r) {
  Job j;
  j.lo = r.get<uint16_t>();
  j.hi = r.get<uint16_t>();
  j.root = r.get<uint8_t>() != 0;
  j.prefix = r.get_array<uint16_t>();
  return j;
}

void finish(Reader& r) {
  if (!r.done()) throw std::runtime_error("job message has trailing bytes");
}
}

uint64_t JobResult::nodes() const {
  uint64_t s = 0;
  for (uint64_t v : nodes_per_depth) s += v;
  return s;
}

Job root_job(const Walk& w) {
  Job j;
  j.lo = 0;
  j.hi = static_cast<uint16_t>(w.table.size());
  j.root = true;
  return j;
}

JobResult execute_job(Ctx& c, const Walk& w, const Job& job, uint64_t budget) {
  if (budget == 0) throw std::invalid_argument("job budget must be positive");
  c.reset();
  const uint64_t calls0 = c.classify_calls;
  JobResult r;
  run_job(c, w, job, budget, r.rest);
  if (!c.count_only) {
    if (c.batch && !c.batch_defer)
      c.batch->flush(c.chunk_types, c.chunk_errs);
    r.types = c.chunk_types.types();
    r.classified = c.classified;
    r.errs = c.chunk_errs;
  }
  r.nodes_per_depth = c.per_depth;
  r.classify_calls = c.classify_calls - calls0;
  r.jobs = 1;
  return r;
}

std::vector<Job> split_job(const Walk& w, const Job& job, int parts) {
  const std::vector<uint16_t> cand = job_candidates(w, job);
  const size_t m = cand.size();
  const size_t k = std::min<size_t>(std::max(parts, 1), std::max<size_t>(m, 1));
  if (k <= 1) return {job};
  std::vector<Job> out(k);
  for (size_t i = 0; i < k; ++i) {
    out[i].prefix = job.prefix;
    out[i].lo = i == 0 ? job.lo : cand[i * m / k];
    out[i].hi = i + 1 == k ? job.hi : cand[(i + 1) * m / k];
    out[i].root = i == 0 && job.root;
  }
  return out;
}

void JobFold::fold(const JobResult& r) {
  if (r.nodes_per_depth.size() > nodes_per_depth.size())
    nodes_per_depth.resize(r.nodes_per_depth.size(), 0);
  for (size_t d = 0; d < r.nodes_per_depth.size(); ++d)
    nodes_per_depth[d] += r.nodes_per_depth[d];
  for (const TypeInfo& t : r.types)
    types.add(t.viro, t.p, t.n, t.mcurve, t.profile, t.count, t.first_depth,
              t.witness.data(), t.witness.size());
  classified += r.classified;
  errs += r.errs;
  classify_calls += r.classify_calls;
  jobs += r.jobs;
}

JobResult JobFold::snapshot() const {
  JobResult r;
  r.nodes_per_depth = nodes_per_depth;
  r.types = types.types();
  r.classified = classified;
  r.errs = errs;
  r.classify_calls = classify_calls;
  r.jobs = jobs;
  return r;
}

uint64_t JobFold::total_nodes() const {
  uint64_t s = 0;
  for (uint64_t v : nodes_per_depth) s += v;
  return s;
}

uint64_t job_config_hash(const Walk& w) {
  uint64_t h = w.base.table_hash;
  const auto mix = [&h](uint64_t v) {
    h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  };
  mix(static_cast<uint64_t>(w.base.nsplits));
  mix(w.base.symmetry);
  mix(w.base.count_only);
  mix(static_cast<uint64_t>(w.base.max_depth));
  mix(uint64_t{7});
  return h;
}

std::vector<uint8_t> encode_job(const Job& job, uint64_t config_hash) {
  std::vector<uint8_t> out;
  Writer w(out);
  put_header(w, kJobMagic, config_hash);
  put_job(w, job);
  return out;
}

Job decode_job(const std::vector<uint8_t>& bytes, uint64_t config_hash) {
  Reader r(bytes);
  check_header(r, kJobMagic, config_hash);
  Job j = get_job(r);
  finish(r);
  return j;
}

namespace {
void put_result(Writer& w, const JobResult& res) {
  w.put_array(res.nodes_per_depth);
  w.put(res.classified);
  w.put(res.errs);
  w.put(res.classify_calls);
  w.put(res.jobs);
  w.put(static_cast<uint8_t>(res.stopped));
  w.put(static_cast<uint32_t>(res.types.size()));
  for (const TypeInfo& t : res.types) {
    w.put_string(t.viro);
    w.put(static_cast<int32_t>(t.p));
    w.put(static_cast<int32_t>(t.n));
    w.put(static_cast<uint8_t>(t.mcurve));
    w.put(t.profile);
    w.put(t.count);
    w.put(static_cast<int32_t>(t.first_depth));
    w.put_array(t.witness);
  }
  w.put(static_cast<uint32_t>(res.rest.size()));
  for (const Job& j : res.rest) put_job(w, j);
}

JobResult get_result(Reader& r) {
  JobResult res;
  res.nodes_per_depth = r.get_array<uint64_t>();
  res.classified = r.get<uint64_t>();
  res.errs = r.get<uint64_t>();
  res.classify_calls = r.get<uint64_t>();
  res.jobs = r.get<uint64_t>();
  res.stopped = r.get<uint8_t>() != 0;
  res.types.resize(r.get<uint32_t>());
  for (TypeInfo& t : res.types) {
    t.viro = r.get_string();
    t.p = r.get<int32_t>();
    t.n = r.get<int32_t>();
    t.mcurve = r.get<uint8_t>() != 0;
    t.profile = r.get<Profile>();
    t.count = r.get<uint64_t>();
    t.first_depth = r.get<int32_t>();
    t.witness = r.get_array<uint16_t>();
  }
  res.rest.resize(r.get<uint32_t>());
  for (Job& j : res.rest) j = get_job(r);
  return res;
}
}

std::vector<uint8_t> encode_result(const JobResult& res, uint64_t config_hash) {
  std::vector<uint8_t> out;
  Writer w(out);
  put_header(w, kResultMagic, config_hash);
  put_result(w, res);
  return out;
}

JobResult decode_result(const std::vector<uint8_t>& bytes,
                        uint64_t config_hash) {
  Reader r(bytes);
  check_header(r, kResultMagic, config_hash);
  JobResult res = get_result(r);
  finish(r);
  return res;
}

std::vector<uint8_t> encode_jobs(const JobBatch& batch, uint64_t config_hash) {
  std::vector<uint8_t> out;
  Writer w(out);
  put_header(w, kBatchMagic, config_hash);
  w.put(batch.seconds);
  w.put(static_cast<uint32_t>(batch.jobs.size()));
  for (const Job& j : batch.jobs) put_job(w, j);
  return out;
}

JobBatch decode_jobs(const std::vector<uint8_t>& bytes, uint64_t config_hash) {
  Reader r(bytes);
  check_header(r, kBatchMagic, config_hash);
  JobBatch batch;
  batch.seconds = r.get<double>();
  batch.jobs.resize(r.get<uint32_t>());
  for (Job& j : batch.jobs) j = get_job(r);
  finish(r);
  return batch;
}

std::vector<uint8_t> encode_state(const JobState& state, uint64_t config_hash) {
  std::vector<uint8_t> out;
  Writer w(out);
  put_header(w, kStateMagic, config_hash);
  w.put(state.seconds);
  put_result(w, state.done);
  return out;
}

JobState decode_state(const std::vector<uint8_t>& bytes, uint64_t config_hash) {
  Reader r(bytes);
  check_header(r, kStateMagic, config_hash);
  JobState state;
  state.seconds = r.get<double>();
  state.done = get_result(r);
  finish(r);
  return state;
}
}
