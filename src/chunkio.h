#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "splits.h"
#include "store.h"

namespace SplitEnum {
constexpr int kShallowChunk = -1;

struct ChunkHeader {
  int delta = 0;
  int family = 0;
  bool drop_same_side = true;
  bool count_only = false;
  bool symmetry = false;
  int max_depth = 0;
  int nsplits = 0;
  uint64_t table_hash = 0;
  int chunk_index = kShallowChunk;
  uint64_t pair_lo = 0;
  uint64_t pair_hi = 0;
  uint64_t chunk_count = 0;
  uint64_t npairs = 0;
  double seconds = 0.0;
  std::vector<uint64_t> nodes_per_depth;
  uint64_t nrecords = 0;
  uint64_t nclassified = 0;
  uint64_t nerr = 0;

  bool config_matches(const ChunkHeader& o) const {
    return delta == o.delta && family == o.family &&
           drop_same_side == o.drop_same_side && count_only == o.count_only &&
           symmetry == o.symmetry && max_depth == o.max_depth &&
           nsplits == o.nsplits && table_hash == o.table_hash &&
           chunk_count == o.chunk_count && npairs == o.npairs;
  }
};

uint64_t table_hash(const SplitTable& table, Family family,
                    bool drop_same_side);

ChunkHeader read_chunk_header(const std::string& path);

void write_chunk(const std::string& path, const ChunkHeader& hdr,
                 const std::vector<TypeInfo>& types);

std::vector<TypeInfo> read_chunk(const std::string& path,
                                 ChunkHeader* hdr_out = nullptr);

void clean_tmp_files(const std::string& dir);
}
