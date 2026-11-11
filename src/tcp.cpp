#include "tcp.h"

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <thread>

#include <poll.h>

#include "gpuclassify.h"
#include "pool.h"

namespace fs = std::filesystem;

namespace SplitEnum {
namespace {
std::vector<uint8_t> hash_bytes(uint64_t h) {
  std::vector<uint8_t> b(8);
  for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>(h >> (8 * i));
  return b;
}

uint64_t bytes_hash(const std::vector<uint8_t>& b) {
  uint64_t h = 0;
  for (size_t i = 0; i < 8 && i < b.size(); ++i) h |= uint64_t{b[i]} << (8 * i);
  return h;
}

std::vector<uint8_t> text_bytes(const std::string& s) {
  return std::vector<uint8_t>(s.begin(), s.end());
}
}

TcpRunner::TcpRunner(const std::string& dir, int port)
    : dir_(dir), listener_(listen_socket(port)), slots_(kSlots) {
  port_ = bound_port(listener_);
}

void TcpRunner::start(const Walk& w) {
  hash_ = job_config_hash(w);
  write_addr(dir_, local_host(), port_);
  std::fprintf(stderr, "splitenum: master listening on %s:%d\n",
               local_host().c_str(), port_);
}

bool TcpRunner::available(int slot) const {
  return slots_[slot].sock.is_open() && !slots_[slot].busy;
}

int TcpRunner::connected() const {
  int n = 0;
  for (const Slot& s : slots_) n += s.sock.is_open();
  return n;
}

void TcpRunner::submit(int slot, std::vector<Job> jobs, double seconds) {
  Slot& s = slots_[slot];
  JobBatch batch;
  batch.jobs = jobs;
  batch.seconds = seconds;
  s.inflight = std::move(jobs);
  s.busy = true;
  s.deadline = Clock::now() + std::chrono::seconds(600) +
               std::chrono::duration_cast<Clock::duration>(
                   std::chrono::duration<double>(seconds));
  if (!s.sock.send_frame(kFrameJobs, encode_jobs(batch, hash_))) drop(slot);
}

void TcpRunner::drop(int slot) {
  Slot& s = slots_[slot];
  if (s.sock.is_open())
    std::fprintf(stderr, "\nsplitenum: worker on slot %d lost%s\n", slot,
                 s.busy ? ", its jobs are requeued" : "");
  s.sock.close();
  if (s.busy) lost_.push_back(slot);
}

void TcpRunner::accept_one() {
  Socket sock = accept_socket(listener_);
  if (!sock.is_open()) return;
  sock.set_recv_timeout(10);
  uint8_t tag = 0;
  std::vector<uint8_t> bytes;
  if (!sock.recv_frame(tag, bytes) || tag != kFrameHello) return;
  if (bytes_hash(bytes) != hash_) {
    sock.send_frame(kFrameRefuse,
                    text_bytes("worker runs a different walk configuration"));
    std::fprintf(stderr, "\nsplitenum: refused a worker with another "
                         "configuration\n");
    return;
  }
  sock.set_recv_timeout(0);
  for (int i = 0; i < kSlots; ++i) {
    Slot& s = slots_[i];
    if (s.sock.is_open() || s.busy) continue;
    if (!sock.send_frame(kFrameHello, {})) return;
    s.sock = std::move(sock);
    std::fprintf(stderr, "\nsplitenum: worker joined on slot %d, %d connected\n",
                 i, connected());
    return;
  }
  sock.send_frame(kFrameRefuse, text_bytes("master has no free slot"));
}

bool TcpRunner::poll_once(int timeout_ms, int& slot, JobResult& out) {
  if (!lost_.empty()) {
    slot = lost_.front();
    lost_.pop_front();
    Slot& s = slots_[slot];
    out = JobResult();
    out.rest = std::move(s.inflight);
    s.inflight.clear();
    s.busy = false;
    return true;
  }
  std::vector<pollfd> fds;
  std::vector<int> owner;
  fds.push_back({listener_.fd(), POLLIN, 0});
  owner.push_back(-1);
  for (int i = 0; i < kSlots; ++i)
    if (slots_[i].sock.is_open()) {
      fds.push_back({slots_[i].sock.fd(), POLLIN, 0});
      owner.push_back(i);
    }
  ::poll(fds.data(), fds.size(), timeout_ms);
  const auto now = Clock::now();
  for (size_t k = 0; k < fds.size(); ++k) {
    if (owner[k] < 0) {
      if (fds[k].revents & POLLIN) accept_one();
      continue;
    }
    const int i = owner[k];
    Slot& s = slots_[i];
    if (!(fds[k].revents & (POLLIN | POLLHUP | POLLERR))) {
      if (s.busy && now > s.deadline) drop(i);
      continue;
    }
    uint8_t tag = 0;
    std::vector<uint8_t> bytes;
    if (!s.sock.recv_frame(tag, bytes) || tag != kFrameResult || !s.busy) {
      drop(i);
      continue;
    }
    try {
      out = decode_result(bytes, hash_);
    } catch (const std::exception&) {
      drop(i);
      continue;
    }
    s.busy = false;
    s.inflight.clear();
    slot = i;
    return true;
  }
  return false;
}

int TcpRunner::wait(JobResult& out) {
  int slot = -1;
  while (!poll_once(500, slot, out)) {
  }
  return slot;
}

void TcpRunner::idle_wait(const std::atomic<bool>* stop) {
  const auto end = Clock::now() + std::chrono::seconds(1);
  while (Clock::now() < end) {
    if (stop && stop->load()) return;
    int slot = -1;
    JobResult r;
    if (poll_once(200, slot, r))
      throw std::logic_error("idle_wait received a result");
    for (int i = 0; i < kSlots; ++i)
      if (available(i)) return;
  }
}

void TcpRunner::finish() {
  for (Slot& s : slots_)
    if (s.sock.is_open()) {
      s.sock.send_frame(kFrameStop, {});
      s.sock.close();
    }
  remove_addr(dir_);
}

void run_worker(const Options& opt) {
  using Clock = std::chrono::steady_clock;
  const StopSignals signals(true);
  const Walk w = prepare(opt);
  const uint64_t hash = job_config_hash(w);
  int nthreads = opt.threads;
  if (nthreads <= 0)
    nthreads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  Pool pool(w, nthreads, opt.budget);
  const std::string& dir = opt.connect;
  const auto patience = std::chrono::duration_cast<Clock::duration>(
      std::chrono::duration<double>(opt.connect_patience));
  auto last_contact = Clock::now();
  bool announced_wait = false;
  while (!stop_flag().load()) {
    std::string host;
    int port = 0;
    Socket sock;
    if (read_addr(dir, host, port)) sock = connect_socket(host, port);
    if (!sock.is_open()) {
      if (!fs::exists(addr_path(dir)) && fs::exists(dir + "/run.json")) {
        std::fprintf(stderr, "splitenum: the run in %s is finished\n",
                     dir.c_str());
        break;
      }
      if (Clock::now() - last_contact > patience)
        throw std::runtime_error("no master reachable through " +
                                 addr_path(dir));
      if (!announced_wait)
        std::fprintf(stderr, "splitenum: waiting for a master at %s\n",
                     addr_path(dir).c_str());
      announced_wait = true;
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }
    uint8_t tag = 0;
    std::vector<uint8_t> bytes;
    if (!sock.send_frame(kFrameHello, hash_bytes(hash)) ||
        !sock.recv_frame(tag, bytes)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }
    if (tag == kFrameRefuse)
      throw std::runtime_error("master refused this worker: " +
                               std::string(bytes.begin(), bytes.end()));
    std::fprintf(stderr, "splitenum: worker connected to %s:%d with %d "
                         "threads\n",
                 host.c_str(), port, nthreads);
    announced_wait = false;
    last_contact = Clock::now();
    while (true) {
      while (!sock.readable(500))
        if (stop_flag().load()) {
          print_batch_stats(stderr);
          return;
        }
      if (!sock.recv_frame(tag, bytes)) break;
      last_contact = Clock::now();
      if (tag == kFrameStop) {
        std::fprintf(stderr, "splitenum: master finished the run\n");
        print_batch_stats(stderr);
        return;
      }
      if (tag != kFrameJobs) break;
      JobBatch batch = decode_jobs(bytes, hash);
      JobResult r = pool.run(std::move(batch.jobs), batch.seconds, &stop_flag());
      const bool leaving = r.stopped;
      r.stopped = false;
      if (!sock.send_frame(kFrameResult, encode_result(r, hash))) break;
      last_contact = Clock::now();
      if (leaving) {
        std::fprintf(stderr, "splitenum: worker stopping, its slice went "
                             "back to the master\n");
        print_batch_stats(stderr);
        return;
      }
    }
    std::fprintf(stderr, "splitenum: lost the master, reconnecting\n");
  }
  print_batch_stats(stderr);
}
}
