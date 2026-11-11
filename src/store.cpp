#include "store.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace SplitEnum {
namespace {
std::string join_ids(const std::vector<uint16_t>& v) {
  std::string s;
  for (size_t k = 0; k < v.size(); ++k) {
    if (k) s += ';';
    s += std::to_string(v[k]);
  }
  return s;
}

std::string quote(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"') out += "\"\"";
    else out += c;
  }
  out += '"';
  return out;
}
}

void TypeTable::add(const std::string& viro, int p, int n, bool mcurve,
                    Profile profile, uint64_t count, int depth,
                    const uint16_t* stack, size_t stack_len) {
  std::string id = viro;
  id += '\0';
  id.append(reinterpret_cast<const char*>(&profile), sizeof profile);
  const auto [it, fresh] =
      id_of_.try_emplace(std::move(id), static_cast<uint32_t>(types_.size()));
  if (fresh) {
    types_.push_back({viro, p, n, mcurve, profile, count, depth,
                      std::vector<uint16_t>(stack, stack + stack_len)});
    return;
  }
  TypeInfo& t = types_[it->second];
  t.count += count;
  if (depth < t.first_depth ||
      (depth == t.first_depth &&
       std::lexicographical_compare(stack, stack + stack_len,
                                    t.witness.begin(), t.witness.end()))) {
    t.first_depth = depth;
    t.witness.assign(stack, stack + stack_len);
  }
}

void TypeTable::write_csv(const std::string& path, int delta) const {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << "viro,p,n,is_mcurve,slack,support_degree,count,first_depth,"
       "first_witness\n";
  for (const TypeInfo& t : types_)
    f << quote(t.viro) << ',' << t.p << ',' << t.n << ',' << (t.mcurve ? 1 : 0)
      << ',' << quote(profile_text(t.profile)) << ','
      << support_degree(delta, t.profile) << ',' << t.count << ',' << t.first_depth << ','
      << quote(join_ids(t.witness)) << '\n';
}

void RunStats::add_depth(int d, uint64_t k) {
  if (static_cast<size_t>(d) >= nodes_per_depth.size())
    nodes_per_depth.resize(d + 1, 0);
  nodes_per_depth[d] += k;
}

void write_run_json(const std::string& path, const RunStats& s) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  const auto num = [&](const char* k, const auto& v) {
    f << "  \"" << k << "\": " << v << ",\n";
  };
  const auto str = [&](const char* k, const std::string& v) {
    f << "  \"" << k << "\": " << quote(v) << ",\n";
  };
  const auto flag = [&](const char* k, bool v) {
    f << "  \"" << k << "\": " << (v ? "true" : "false") << ",\n";
  };
  f << "{\n";
  str("cli", s.cli);
  num("delta", s.delta);
  str("family", s.family);
  flag("drop_same_side", s.drop_same_side);
  flag("drop_trivial", s.drop_trivial);
  flag("drop_collinear", s.drop_collinear);
  num("max_depth", s.max_depth);
  num("threads", s.threads);
  flag("count_only", s.count_only);
  flag("symmetry", s.symmetry);
  num("symmetry_group_order", s.sym_order);
  num("nsplits", s.nsplits);
  num("compat_pairs", s.compat_pairs);
  num("total_nodes", s.total_nodes);
  num("max_depth_reached", s.max_depth_reached);
  num("classified_nodes", s.classified);
  num("errors", s.errors);
  num("seconds", s.seconds);
  str("scheduler", s.scheduler_jobs ? "jobs" : "chunks");
  if (s.scheduler_jobs) {
    num("jobs_done", s.jobs_done);
  } else {
    num("chunks_done", s.chunks_done);
    num("chunks_total", s.chunks_total);
  }
  f << "  \"nodes_per_depth\": [";
  for (size_t d = 0; d < s.nodes_per_depth.size(); ++d) {
    if (d) f << ", ";
    f << s.nodes_per_depth[d];
  }
  f << "]\n}\n";
}

void write_splits_csv(const std::string& path, const SplitTable& table) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << "split_id,kind,x0,y0,x1,y1,x2,y2,twist_hex\n";
  for (int i = 0; i < table.size(); ++i) {
    const Split& s = table.splits[i];
    f << i << ',' << (s.size() == 2 ? "simple" : "double");
    for (size_t k = 0; k < 3; ++k) {
      if (k < s.size()) f << ',' << s[k].first << ',' << s[k].second;
      else f << ",,";
    }
    std::ostringstream hex;
    hex << std::hex << table.recs[i].twist;
    f << ",0x" << hex.str() << '\n';
  }
}
}
