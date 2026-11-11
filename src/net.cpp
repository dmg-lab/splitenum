#include "net.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace SplitEnum {
namespace {
constexpr uint64_t kMaxFrame = uint64_t{1} << 31;

bool write_all(int fd, const uint8_t* p, size_t n) {
  while (n > 0) {
    const ssize_t k = ::send(fd, p, n, MSG_NOSIGNAL);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) return false;
    p += k;
    n -= static_cast<size_t>(k);
  }
  return true;
}

bool read_all(int fd, uint8_t* p, size_t n) {
  while (n > 0) {
    const ssize_t k = ::recv(fd, p, n, 0);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) return false;
    p += k;
    n -= static_cast<size_t>(k);
  }
  return true;
}

void keepalive(int fd) {
  int on = 1;
  setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &on, sizeof on);
  int idle = 60, intvl = 10, cnt = 6;
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof idle);
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof intvl);
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof cnt);
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof on);
}
}

Socket::~Socket() { close(); }

Socket::Socket(Socket&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }

Socket& Socket::operator=(Socket&& o) noexcept {
  if (this != &o) {
    close();
    fd_ = o.fd_;
    o.fd_ = -1;
  }
  return *this;
}

void Socket::close() {
  if (fd_ >= 0) ::close(fd_);
  fd_ = -1;
}

bool Socket::send_frame(uint8_t tag, const std::vector<uint8_t>& bytes) {
  if (fd_ < 0) return false;
  uint8_t head[9];
  const uint64_t n = bytes.size();
  for (int i = 0; i < 8; ++i) head[i] = static_cast<uint8_t>(n >> (8 * i));
  head[8] = tag;
  return write_all(fd_, head, sizeof head) &&
         write_all(fd_, bytes.data(), bytes.size());
}

bool Socket::recv_frame(uint8_t& tag, std::vector<uint8_t>& bytes) {
  if (fd_ < 0) return false;
  uint8_t head[9];
  if (!read_all(fd_, head, sizeof head)) return false;
  uint64_t n = 0;
  for (int i = 0; i < 8; ++i) n |= uint64_t{head[i]} << (8 * i);
  if (n > kMaxFrame) return false;
  tag = head[8];
  bytes.resize(n);
  return read_all(fd_, bytes.data(), n);
}

bool Socket::readable(int timeout_ms) const {
  if (fd_ < 0) return false;
  pollfd p{fd_, POLLIN, 0};
  const int r = ::poll(&p, 1, timeout_ms);
  return r > 0;
}

void Socket::set_recv_timeout(int seconds) {
  timeval tv{seconds, 0};
  setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
}

Socket listen_socket(int port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) throw std::runtime_error("socket: " + std::string(strerror(errno)));
  Socket s(fd);
  int on = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = htonl(INADDR_ANY);
  a.sin_port = htons(static_cast<uint16_t>(port));
  if (::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0)
    throw std::runtime_error("bind port " + std::to_string(port) + ": " +
                             strerror(errno));
  if (::listen(fd, 64) < 0)
    throw std::runtime_error("listen: " + std::string(strerror(errno)));
  return s;
}

int bound_port(const Socket& s) {
  sockaddr_in a{};
  socklen_t len = sizeof a;
  if (getsockname(s.fd(), reinterpret_cast<sockaddr*>(&a), &len) < 0) return -1;
  return ntohs(a.sin_port);
}

Socket accept_socket(const Socket& listener) {
  const int fd = ::accept(listener.fd(), nullptr, nullptr);
  if (fd < 0) return Socket();
  keepalive(fd);
  return Socket(fd);
}

Socket connect_socket(const std::string& host, int port) {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0)
    return Socket();
  Socket out;
  for (addrinfo* p = res; p; p = p->ai_next) {
    const int fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd < 0) continue;
    if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) {
      keepalive(fd);
      out = Socket(fd);
      break;
    }
    ::close(fd);
  }
  freeaddrinfo(res);
  return out;
}

std::string local_host() {
  char buf[256] = {};
  if (gethostname(buf, sizeof buf - 1) != 0) return "localhost";
  return buf;
}

std::string addr_path(const std::string& dir) {
  return dir + "/jobs/master.addr";
}

void write_addr(const std::string& dir, const std::string& host, int port) {
  const std::string path = addr_path(dir);
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::trunc);
    f << host << ' ' << port << '\n';
    f.flush();
    if (!f) throw std::runtime_error("cannot write " + tmp);
  }
  fs::rename(tmp, path);
}

bool read_addr(const std::string& dir, std::string& host, int& port) {
  std::ifstream f(addr_path(dir));
  return static_cast<bool>(f >> host >> port);
}

void remove_addr(const std::string& dir) {
  std::error_code ec;
  fs::remove(addr_path(dir), ec);
}
}
