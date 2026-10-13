// Thin MPI communicator wrapper. When MPI is not initialised (plain `python
// script.py`), a Comm behaves as a serial communicator of size 1, so every
// kernel has a single code path.
#pragma once

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cfdpost/common.hpp"

namespace cfdpost {

bool mpi_is_initialized();
bool mpi_is_finalized();
// Initialise MPI (MPI_THREAD_FUNNELED). Returns false if already initialised.
bool mpi_init();
void mpi_finalize();
std::string mpi_library_version();

class Comm {
 public:
  // World communicator if MPI is running, otherwise serial.
  static Comm world();
  static Comm self();

  int rank() const { return rank_; }
  int size() const { return size_; }
  bool parallel() const { return size_ > 1; }
  bool is_mpi() const { return mpi_; }
  MPI_Comm raw() const { return comm_ ? *comm_ : MPI_COMM_SELF; }

  void barrier() const;
  Comm split(int color, int key) const;

  // Element-wise reductions in place.
  void allreduce_sum(double* data, std::size_t n) const;
  void allreduce_sum(std::int64_t* data, std::size_t n) const;
  void allreduce_min(double* data, std::size_t n) const;
  void allreduce_max(double* data, std::size_t n) const;
  std::int64_t allreduce_sum(std::int64_t v) const;

  // Byte messages (used for pickled Python objects).
  void bcast_bytes(std::string& buf, int root) const;
  std::vector<std::string> gather_bytes(const std::string& buf, int root) const;
  std::vector<std::string> allgather_bytes(const std::string& buf) const;
  void send_bytes(const std::string& buf, int dest, int tag) const;
  // Returns (payload, source, tag). source/tag may be MPI_ANY_*.
  std::string recv_bytes(int source, int tag, int* out_source, int* out_tag) const;

  // Variable all-to-all of plain-old-data. sendcounts are element counts per rank;
  // returns received data and fills recvcounts.
  template <class T>
  std::vector<T> alltoallv(const std::vector<T>& send, const std::vector<std::int64_t>& sendcounts,
                           std::vector<std::int64_t>& recvcounts) const;
  template <class T>
  std::vector<T> gatherv(const std::vector<T>& send, int root, std::vector<std::int64_t>& counts) const;

 private:
  void alltoallv_bytes(const char* send, const std::vector<std::int64_t>& sendbytes, char* recv,
                       const std::vector<std::int64_t>& recvbytes) const;
  std::vector<std::int64_t> exchange_counts(const std::vector<std::int64_t>& sendcounts) const;
  void gatherv_bytes(const char* send, std::int64_t nbytes, char* recv,
                     const std::vector<std::int64_t>& recvbytes, int root) const;

  std::shared_ptr<MPI_Comm> comm_;  // null => serial
  int rank_ = 0;
  int size_ = 1;
  bool mpi_ = false;
};

template <class T>
std::vector<T> Comm::alltoallv(const std::vector<T>& send, const std::vector<std::int64_t>& sendcounts,
                               std::vector<std::int64_t>& recvcounts) const {
  if (size_ == 1) {
    recvcounts = sendcounts;
    return send;
  }
  recvcounts = exchange_counts(sendcounts);
  std::vector<std::int64_t> sb(size_), rb(size_);
  std::int64_t ntot = 0;
  for (int r = 0; r < size_; ++r) {
    sb[r] = sendcounts[r] * static_cast<std::int64_t>(sizeof(T));
    rb[r] = recvcounts[r] * static_cast<std::int64_t>(sizeof(T));
    ntot += recvcounts[r];
  }
  std::vector<T> recv(static_cast<std::size_t>(ntot));
  alltoallv_bytes(reinterpret_cast<const char*>(send.data()), sb, reinterpret_cast<char*>(recv.data()), rb);
  return recv;
}

template <class T>
std::vector<T> Comm::gatherv(const std::vector<T>& send, int root, std::vector<std::int64_t>& counts) const {
  if (size_ == 1) {
    counts.assign(1, static_cast<std::int64_t>(send.size()));
    return send;
  }
  std::int64_t n = static_cast<std::int64_t>(send.size());
  counts.assign(size_, 0);
  MPI_Gather(&n, 1, MPI_INT64_T, counts.data(), 1, MPI_INT64_T, root, raw());
  std::vector<std::int64_t> bytes(size_);
  std::int64_t ntot = 0;
  for (int r = 0; r < size_; ++r) {
    bytes[r] = counts[r] * static_cast<std::int64_t>(sizeof(T));
    ntot += counts[r];
  }
  std::vector<T> recv(rank_ == root ? static_cast<std::size_t>(ntot) : 0);
  gatherv_bytes(reinterpret_cast<const char*>(send.data()), n * static_cast<std::int64_t>(sizeof(T)),
                reinterpret_cast<char*>(recv.data()), bytes, root);
  return recv;
}

// Balanced split of [0, n) into `parts` contiguous blocks.
inline std::int64_t block_begin(std::int64_t n, int parts, int i) {
  return (n / parts) * i + std::min<std::int64_t>(i, n % parts);
}

}  // namespace cfdpost
