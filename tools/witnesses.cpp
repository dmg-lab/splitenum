#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "isotopy_graph.h"
#include "splits.h"

using namespace SplitEnum;

static std::string slurp(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream b;
  b << f.rdbuf();
  return b.str();
}

static long long json_int(const std::string& j, const std::string& key) {
  std::smatch m;
  if (!std::regex_search(j, m, std::regex("\"" + key + "\": *(-?[0-9]+)")))
    throw std::runtime_error("run.json lacks " + key);
  return std::stoll(m[1]);
}

static bool json_bool(const std::string& j, const std::string& key, bool absent) {
  std::smatch m;
  if (!std::regex_search(j, m, std::regex("\"" + key + "\": *(true|false)"))) return absent;
  return m[1] == "true";
}

static std::string table_csv(const SplitTable& t) {
  std::ostringstream f;
  f << "split_id,kind,x0,y0,x1,y1,x2,y2,twist_hex\n";
  for (int i = 0; i < t.size(); ++i) {
    const Split& s = t.splits[i];
    f << i << ',' << (s.size() == 2 ? "simple" : "double");
    for (size_t k = 0; k < 3; ++k) {
      if (k < s.size()) f << ',' << s[k].first << ',' << s[k].second;
      else f << ",,";
    }
    f << ",0x" << std::hex << t.recs[i].twist << std::dec << '\n';
  }
  return f.str();
}

static std::string split_text(const SplitTable& t, int id) {
  std::string out = std::to_string(id) + ":";
  const Split& s = t.splits[id];
  for (size_t j = 0; j < s.size(); ++j)
    out += (j ? "-(" : "(") + std::to_string(s[j].first) + "," + std::to_string(s[j].second) + ")";
  return out;
}

static std::string grouped(long long v) {
  std::string s = std::to_string(v), out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (i && (s.size() - i) % 3 == 0) out += ',';
    out += s[i];
  }
  return out;
}

static std::vector<std::string> csv_fields(const std::string& line) {
  std::vector<std::string> out(1);
  bool quoted = false;
  for (char c : line) {
    if (c == '"') quoted = !quoted;
    else if (c == ',' && !quoted) out.emplace_back();
    else out.back() += c;
  }
  return out;
}

struct Row {
  std::string viro, slack;
  int p, n, support;
  long long count;
  std::vector<int> ids;
};

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fputs("usage: witnesses RUN_DIR OUT_PREFIX\n", stderr);
    return 2;
  }
  const std::string dir = argv[1], prefix = argv[2];
  const std::string j = slurp(dir + "/run.json");
  const int delta = static_cast<int>(json_int(j, "delta"));
  if (j.find("\"family\": \"all\"") == std::string::npos)
    throw std::runtime_error("only --family all is supported");
  const bool same_side = json_bool(j, "drop_same_side", true);
  const bool trivial = json_bool(j, "drop_trivial", true);
  const bool collinear = json_bool(j, "drop_collinear", false);
  if (j.find("\"drop_dominated\"") != std::string::npos)
    throw std::runtime_error(dir + " predates 2026-10-22: its table held boundary-apex "
                             "doubles and the --dominated filter, which are gone; "
                             "verify it with a build from before that date");
  const SplitTable t = build_splits(delta, Family::All, same_side, trivial, collinear);
  if (t.size() != json_int(j, "nsplits") || table_csv(t) != slurp(dir + "/splits.csv"))
    throw std::runtime_error("rebuilt split table differs from " + dir + "/splits.csv");

  std::vector<Row> rows;
  std::istringstream types(slurp(dir + "/types.csv"));
  std::string line;
  std::getline(types, line);
  const std::vector<std::string> header = csv_fields(line);
  const auto col = [&](const std::string& name) {
    const auto it = std::find(header.begin(), header.end(), name);
    if (it == header.end()) throw std::runtime_error("types.csv lacks column " + name);
    return static_cast<size_t>(it - header.begin());
  };
  const size_t c_viro = col("viro"), c_p = col("p"), c_n = col("n"), c_slack = col("slack"),
               c_support = col("support_degree"), c_count = col("count"), c_wit = col("first_witness");
  while (std::getline(types, line)) {
    if (line.empty()) continue;
    const std::vector<std::string> f = csv_fields(line);
    Row r{f[c_viro], f[c_slack], std::stoi(f[c_p]), std::stoi(f[c_n]), std::stoi(f[c_support]),
          std::stoll(f[c_count]), {}};
    std::stringstream ws(f[c_wit]);
    for (std::string tok; std::getline(ws, tok, ';');) r.ids.push_back(std::stoi(tok));
    rows.push_back(r);
  }
  std::map<std::string, long long> total;
  for (const Row& r : rows) total[r.viro] += r.count;
  std::stable_sort(rows.begin(), rows.end(), [&](const Row& a, const Row& b) {
    if (a.viro != b.viro)
      return total[a.viro] != total[b.viro] ? total[a.viro] > total[b.viro] : a.viro < b.viro;
    return a.support != b.support ? a.support < b.support : a.slack < b.slack;
  });

  const std::string pcomdir = prefix + "-pcom";
  std::filesystem::create_directories(pcomdir);
  std::ofstream txt(prefix + "-splits.txt");
  const char* onoff[] = {"keep", "drop"};
  txt << "delta-" << delta << " M-schemes realized by Harnack splits: witness splits as lattice coordinates\n"
      << "configuration: --family all --same-side " << onoff[same_side] << " --trivial " << onoff[trivial]
      << " --collinear " << onoff[collinear] << ", " << t.size()
      << " splits, " << grouped(json_int(j, "total_nodes")) << " nodes\n"
      << "simple split: A-B; double split: A-apex-B; id = row of splits.csv of that run\n"
      << "slack: sorted distances to the three edges; support: smallest degree the collection fits, 0 for none\n"
      << "every witness re-verified: harnack XOR twists, refine, Isotopy::Graph reproduces the scheme, M-curve, pcom round trip, slack recomputed\n\n";
  char head[256];
  std::snprintf(head, sizeof head, "%-22s%-10s%-9s%-8s%18s  %s\n", "scheme", "(p,n)", "slack", "support", "count",
                "splits");
  txt << head << std::string(110, '-') << '\n';

  const int ntri = delta * delta;
  int bad = 0;
  std::string previous;
  for (const Row& r : rows) {
    std::vector<Isotopy::Edge> edges;
    SignMask sig = t.harnack;
    std::string desc;
    for (int id : r.ids) {
      const auto& rec = t.recs[id];
      edges.push_back(rec.idx_a);
      if (rec.nsegs == 2) edges.push_back(rec.idx_b);
      sig ^= rec.twist;
      const std::string text = split_text(t, id);
      desc += (desc.empty() ? "" : " ") + text.substr(text.find(':') + 1);
    }
    const auto tris = Isotopy::refine(delta, edges);
    const auto signs = unpack_signs(sig, t.npoints);
    Isotopy::Graph g(delta, signs, *tris);
    const Profile prof = profile_of(t, std::vector<uint16_t>(r.ids.begin(), r.ids.end()).data(),
                                    static_cast<int>(r.ids.size()));
    const std::string origin = r.viro + " splitenum degree-" + std::to_string(delta) + " Harnack splits, support degree " +
                               std::to_string(r.support) + ": " + (desc.empty() ? "none" : desc);
    const std::string pcom = Utils::signs_and_triangles_to_pcom(signs, *tris, origin);
    auto [s2, t2] = Utils::pcom_to_signs_and_triangles_vec(pcom);
    Isotopy::Graph g2(delta, s2, t2);
    const bool ok = g.viro_notation() == r.viro && g2.viro_notation() == r.viro && g2.is_mcurve() &&
                    g.p_regions == r.p && g.n_regions == r.n && static_cast<int>(tris->size()) == ntri &&
                    profile_text(prof) == r.slack && support_degree(delta, prof) == r.support;
    bad += !ok;

    std::string name;
    for (char c : r.viro) name += (c == '<' || c == '>') ? '_' : c;
    std::string suffix;
    for (char c : r.slack) if (c != ';') suffix += c;
    name = std::to_string(r.p) + "_" + std::to_string(r.n) + "_" + name + "_s" + suffix + ".pcom";
    std::ofstream(pcomdir + "/" + name) << pcom;

    const bool first = r.viro != previous;
    previous = r.viro;
    const std::string pn = "(" + std::to_string(r.p) + "," + std::to_string(r.n) + ")";
    std::snprintf(head, sizeof head, "%-22s%-10s%-9s%-8d%18s  ", first ? r.viro.c_str() : "",
                  first ? pn.c_str() : "", r.slack.c_str(), r.support, grouped(r.count).c_str());
    txt << head;
    if (r.ids.empty()) txt << "none: Harnack curve\n";
    for (size_t k = 0; k < r.ids.size(); ++k)
      txt << (k ? std::string(69, ' ') : "") << split_text(t, r.ids[k]) << '\n';
    std::printf("%s %s %s %s\n", ok ? "ok " : "BAD", name.c_str(), g2.viro_notation().c_str(), r.slack.c_str());
  }
  return bad;
}
