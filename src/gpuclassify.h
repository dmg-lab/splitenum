#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>

#include "enumerate.h"
#include "gpurefine.h"
#include "refine.h"
#include "splits.h"
#include "store.h"

namespace SplitEnum {
struct BatchStats {
  uint64_t nodes = 0;
  uint64_t batches = 0;
  uint64_t fallbacks = 0;
  uint64_t new_codes = 0;
  uint64_t checked = 0;
  uint64_t mismatches = 0;
  uint64_t run_ns = 0;
  uint64_t resolve_ns = 0;
};

RefineTables refine_tables(const LeanRefine& lean);

BatchStats batch_stats();
void print_batch_stats(std::FILE* fp);

class BatchClassifier {
 public:
  BatchClassifier(const SplitTable& table, const LeanRefine& lean,
                  Classify backend, int slot, int capacity,
                  uint64_t check_every, bool refine_gpu = true);
  ~BatchClassifier();
  BatchClassifier(const BatchClassifier&) = delete;
  BatchClassifier& operator=(const BatchClassifier&) = delete;

  bool refines() const;
  bool full() const;
  bool add(SignMask sig, const uint64_t* forced, const uint64_t* blocked,
           Profile profile, uint32_t weight, int depth, const uint16_t* stack,
           size_t stack_len);
  void add_ids(SignMask sig, Profile profile, uint32_t weight, int depth,
               const uint16_t* stack, size_t stack_len);
  void rotate(TypeTable& types, uint64_t& errs);
  void flush(TypeTable& types, uint64_t& errs);
  void pin();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
