#include "chunkio.h"

#include <bit>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace SplitEnum {
namespace {
static_assert(std::endian::native == std::endian::little,
              "chunk files are written in host byte order and shared between "
              "shard hosts; only little-endian hosts are supported");

constexpr uint32_t kMagic = 0x4b574553;
constexpr uint32_t kVersion = 7;
constexpr uint32_t kMinVersion = 7;

template <typename T>
void put(std::ofstream& f, T v) {
  f.write(reinterpret_cast<const char*>(&v), sizeof v);
}

template <typename T>
void get(std::istream& f, T& v) {
  f.read(reinterpret_cast<char*>(&v), sizeof v);
}

void write_header(std::ofstream& f, const ChunkHeader& h) {
  put(f, kMagic);
  put(f, kVersion);
  put(f, static_cast<int32_t>(h.delta));
  put(f, static_cast<int32_t>(h.family));
  put(f, static_cast<uint8_t>(h.drop_same_side));
  put(f, static_cast<uint8_t>(h.count_only));
  put(f, static_cast<uint8_t>(h.symmetry));
  put(f, static_cast<int32_t>(h.max_depth));
  put(f, static_cast<int32_t>(h.nsplits));
  put(f, h.table_hash);
  put(f, static_cast<int32_t>(h.chunk_index));
  put(f, h.pair_lo);
  put(f, h.pair_hi);
  put(f, h.chunk_count);
  put(f, h.npairs);
  put(f, h.seconds);
  put(f, static_cast<uint32_t>(h.nodes_per_depth.size()));
  for (uint64_t v : h.nodes_per_depth) put(f, v);
  put(f, h.nrecords);
  put(f, h.nclassified);
  put(f, h.nerr);
}

ChunkHeader read_header(std::istream& f, const std::string& path) {
  uint32_t magic = 0, version = 0;
  get(f, magic);
  get(f, version);
  if (!f || magic != kMagic)
    throw std::runtime_error(path + ": not a splitenum chunk file");
  if (version < kMinVersion || version > kVersion)
    throw std::runtime_error(path + ": chunk format version " +
                             std::to_string(version) + ", expected " +
                             std::to_string(kMinVersion) + ".." +
                             std::to_string(kVersion) +
                             "; re-walk into a fresh --out directory");
  ChunkHeader h;
  int32_t i32;
  uint8_t u8;
  get(f, i32); h.delta = i32;
  get(f, i32); h.family = i32;
  get(f, u8); h.drop_same_side = u8 != 0;
  get(f, u8); h.count_only = u8 != 0;
  get(f, u8); h.symmetry = u8 != 0;
  get(f, i32); h.max_depth = i32;
  get(f, i32); h.nsplits = i32;
  get(f, h.table_hash);
  get(f, i32); h.chunk_index = i32;
  get(f, h.pair_lo);
  get(f, h.pair_hi);
  get(f, h.chunk_count);
  get(f, h.npairs);
  get(f, h.seconds);
  uint32_t depths = 0;
  get(f, depths);
  h.nodes_per_depth.resize(depths);
  for (uint32_t d = 0; d < depths; ++d) get(f, h.nodes_per_depth[d]);
  get(f, h.nrecords);
  get(f, h.nclassified);
  get(f, h.nerr);
  if (!f) throw std::runtime_error(path + ": truncated chunk header");
  return h;
}
}

uint64_t table_hash(const SplitTable& table, Family family,
                    bool drop_same_side) {
  uint64_t h = 1469598103934665603ull;
  const auto mix = [&h](uint64_t v) {
    for (int k = 0; k < 8; ++k) {
      h ^= (v >> (8 * k)) & 0xff;
      h *= 1099511628211ull;
    }
  };
  mix(static_cast<uint64_t>(table.delta));
  mix(static_cast<uint64_t>(static_cast<int>(family)));
  mix(static_cast<uint64_t>(drop_same_side));
  mix(static_cast<uint64_t>(table.size()));
  for (const Split& s : table.splits) {
    mix(s.size());
    for (const Point& p : s) {
      mix(static_cast<uint64_t>(p.first));
      mix(static_cast<uint64_t>(p.second));
    }
  }
  return h;
}

ChunkHeader read_chunk_header(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  return read_header(f, path);
}

void write_chunk(const std::string& path, const ChunkHeader& hdr,
                 const std::vector<TypeInfo>& types) {
  if (hdr.nrecords != types.size())
    throw std::logic_error("chunk header type count mismatch");
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot write " + tmp);
    write_header(f, hdr);
    for (const TypeInfo& t : types) {
      put(f, static_cast<uint32_t>(t.viro.size()));
      f.write(t.viro.data(), static_cast<std::streamsize>(t.viro.size()));
      put(f, static_cast<int32_t>(t.p));
      put(f, static_cast<int32_t>(t.n));
      put(f, static_cast<uint8_t>(t.mcurve));
      put(f, t.profile);
      put(f, t.count);
      put(f, static_cast<uint8_t>(t.first_depth));
      put(f, static_cast<uint8_t>(t.witness.size()));
      f.write(reinterpret_cast<const char*>(t.witness.data()),
              static_cast<std::streamsize>(t.witness.size() * 2));
    }
    if (!f) throw std::runtime_error("write failed on " + tmp);
  }
  fs::rename(tmp, path);
}

std::vector<TypeInfo> read_chunk(const std::string& path,
                                 ChunkHeader* hdr_out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  const ChunkHeader h = read_header(f, path);
  std::vector<TypeInfo> types(h.nrecords);
  for (TypeInfo& t : types) {
    uint32_t vlen = 0;
    get(f, vlen);
    t.viro.resize(vlen);
    f.read(t.viro.data(), vlen);
    int32_t i32;
    uint8_t u8;
    get(f, i32); t.p = i32;
    get(f, i32); t.n = i32;
    get(f, u8); t.mcurve = u8 != 0;
    get(f, t.profile);
    get(f, t.count);
    get(f, u8); t.first_depth = u8;
    uint8_t wlen = 0;
    get(f, wlen);
    t.witness.resize(wlen);
    f.read(reinterpret_cast<char*>(t.witness.data()), 2 * wlen);
  }
  if (!f) throw std::runtime_error(path + ": truncated type table");
  if (hdr_out) *hdr_out = h;
  return types;
}

void clean_tmp_files(const std::string& dir) {
  if (!fs::exists(dir)) return;
  for (const auto& e : fs::directory_iterator(dir))
    if (e.path().extension() == ".tmp") fs::remove(e.path());
}
}
