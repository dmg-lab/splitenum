#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace SplitEnum {
enum : uint8_t {
  kFrameHello = 1,
  kFrameRefuse = 2,
  kFrameJobs = 3,
  kFrameResult = 4,
  kFrameStop = 5,
};

class Socket {
 public:
  Socket() = default;
  explicit Socket(int fd) : fd_(fd) {}
  ~Socket();
  Socket(Socket&& o) noexcept;
  Socket& operator=(Socket&& o) noexcept;
  Socket(const Socket&) = delete;
  Socket& operator=(const Socket&) = delete;

  int fd() const { return fd_; }
  bool is_open() const { return fd_ >= 0; }
  void close();

  bool send_frame(uint8_t tag, const std::vector<uint8_t>& bytes);
  bool recv_frame(uint8_t& tag, std::vector<uint8_t>& bytes);
  bool readable(int timeout_ms) const;
  void set_recv_timeout(int seconds);

 private:
  int fd_ = -1;
};

Socket listen_socket(int port);
int bound_port(const Socket& s);
Socket accept_socket(const Socket& listener);
Socket connect_socket(const std::string& host, int port);
std::string local_host();

std::string addr_path(const std::string& dir);
void write_addr(const std::string& dir, const std::string& host, int port);
bool read_addr(const std::string& dir, std::string& host, int& port);
void remove_addr(const std::string& dir);
}
