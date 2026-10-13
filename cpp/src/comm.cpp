#include "cfdpost/comm.hpp"

#include <climits>
#include <cstring>

namespace cfdpost {

namespace {
constexpr std::int64_t kChunk = std::int64_t(1) << 30;  // MPI counts are int: send in <=1 GiB pieces

void check(int err, const char* what) {
  if (err != MPI_SUCCESS) fail(std::string("MPI error in ") + what);
}

void free_comm(MPI_Comm* c) {
  if (c && *c != MPI_COMM_NULL && *c != MPI_COMM_WORLD && *c != MPI_COMM_SELF && !mpi_is_finalized())
    MPI_Comm_free(c);
  delete c;
}
}  // namespace

bool mpi_is_initialized() {
  int f = 0;
  MPI_Initialized(&f);
  return f != 0;
}

bool mpi_is_finalized() {
  int f = 0;
  MPI_Finalized(&f);
  return f != 0;
}

bool mpi_init() {
  if (mpi_is_initialized()) return false;
  int provided = 0;
  check(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided), "MPI_Init_thread");
  return true;
}

void mpi_finalize() {
  if (mpi_is_initialized() && !mpi_is_finalized()) MPI_Finalize();
}

std::string mpi_library_version() {
  char buf[MPI_MAX_LIBRARY_VERSION_STRING];
  int len = 0;
  MPI_Get_library_version(buf, &len);
  return std::string(buf, static_cast<std::size_t>(len));
}

Comm Comm::world() {
  Comm c;
  if (mpi_is_initialized() && !mpi_is_finalized()) {
    c.comm_ = std::shared_ptr<MPI_Comm>(new MPI_Comm(MPI_COMM_WORLD), free_comm);
    MPI_Comm_rank(MPI_COMM_WORLD, &c.rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &c.size_);
    c.mpi_ = true;
  }
  return c;
}

Comm Comm::self() { return Comm(); }

void Comm::barrier() const {
  if (mpi_) MPI_Barrier(raw());
}

Comm Comm::split(int color, int key) const {
  if (!mpi_) return Comm();
  MPI_Comm nc;
  check(MPI_Comm_split(raw(), color, key, &nc), "MPI_Comm_split");
  Comm c;
  c.comm_ = std::shared_ptr<MPI_Comm>(new MPI_Comm(nc), free_comm);
  MPI_Comm_rank(nc, &c.rank_);
  MPI_Comm_size(nc, &c.size_);
  c.mpi_ = true;
  return c;
}

namespace {
template <class T>
void allreduce_chunked(MPI_Comm comm, T* data, std::size_t n, MPI_Datatype dt, MPI_Op op) {
  const std::size_t chunk = static_cast<std::size_t>(INT_MAX / 2);
  for (std::size_t i = 0; i < n; i += chunk) {
    int cnt = static_cast<int>(std::min(chunk, n - i));
    check(MPI_Allreduce(MPI_IN_PLACE, data + i, cnt, dt, op, comm), "MPI_Allreduce");
  }
}
}  // namespace

void Comm::allreduce_sum(double* d, std::size_t n) const {
  if (size_ > 1) allreduce_chunked(raw(), d, n, MPI_DOUBLE, MPI_SUM);
}
void Comm::allreduce_sum(std::int64_t* d, std::size_t n) const {
  if (size_ > 1) allreduce_chunked(raw(), d, n, MPI_INT64_T, MPI_SUM);
}
void Comm::allreduce_min(double* d, std::size_t n) const {
  if (size_ > 1) allreduce_chunked(raw(), d, n, MPI_DOUBLE, MPI_MIN);
}
void Comm::allreduce_max(double* d, std::size_t n) const {
  if (size_ > 1) allreduce_chunked(raw(), d, n, MPI_DOUBLE, MPI_MAX);
}
std::int64_t Comm::allreduce_sum(std::int64_t v) const {
  allreduce_sum(&v, 1);
  return v;
}

void Comm::bcast_bytes(std::string& buf, int root) const {
  if (size_ == 1) return;
  std::int64_t n = static_cast<std::int64_t>(buf.size());
  check(MPI_Bcast(&n, 1, MPI_INT64_T, root, raw()), "MPI_Bcast");
  buf.resize(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; i += kChunk) {
    int cnt = static_cast<int>(std::min(kChunk, n - i));
    check(MPI_Bcast(buf.data() + i, cnt, MPI_BYTE, root, raw()), "MPI_Bcast");
  }
}

void Comm::gatherv_bytes(const char* send, std::int64_t nbytes, char* recv,
                         const std::vector<std::int64_t>& recvbytes, int root) const {
  // Point-to-point in chunks: robust for >2 GiB totals.
  std::vector<MPI_Request> reqs;
  if (rank_ == root) {
    std::int64_t off = 0;
    for (int r = 0; r < size_; ++r) {
      if (r == root) {
        std::memcpy(recv + off, send, static_cast<std::size_t>(nbytes));
      } else {
        for (std::int64_t i = 0, t = 0; i < recvbytes[r]; i += kChunk, ++t) {
          reqs.emplace_back();
          int cnt = static_cast<int>(std::min(kChunk, recvbytes[r] - i));
          MPI_Irecv(recv + off + i, cnt, MPI_BYTE, r, 7000 + static_cast<int>(t), raw(), &reqs.back());
        }
      }
      off += recvbytes[r];
    }
  } else {
    for (std::int64_t i = 0, t = 0; i < nbytes; i += kChunk, ++t) {
      reqs.emplace_back();
      int cnt = static_cast<int>(std::min(kChunk, nbytes - i));
      MPI_Isend(send + i, cnt, MPI_BYTE, root, 7000 + static_cast<int>(t), raw(), &reqs.back());
    }
  }
  if (!reqs.empty()) check(MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE), "Waitall");
}

std::vector<std::string> Comm::gather_bytes(const std::string& buf, int root) const {
  if (size_ == 1) return {buf};
  std::vector<std::int64_t> counts;
  std::vector<char> v(buf.begin(), buf.end());
  std::vector<char> all = gatherv(v, root, counts);
  std::vector<std::string> out;
  if (rank_ == root) {
    std::int64_t off = 0;
    for (int r = 0; r < size_; ++r) {
      out.emplace_back(all.data() + off, static_cast<std::size_t>(counts[r]));
      off += counts[r];
    }
  }
  return out;
}

std::vector<std::string> Comm::allgather_bytes(const std::string& buf) const {
  if (size_ == 1) return {buf};
  std::vector<std::string> g = gather_bytes(buf, 0);
  std::string packed;
  if (rank_ == 0) {
    for (auto& s : g) {
      std::int64_t n = static_cast<std::int64_t>(s.size());
      packed.append(reinterpret_cast<const char*>(&n), sizeof n);
      packed += s;
    }
  }
  bcast_bytes(packed, 0);
  std::vector<std::string> out;
  std::size_t p = 0;
  for (int r = 0; r < size_; ++r) {
    std::int64_t n;
    std::memcpy(&n, packed.data() + p, sizeof n);
    p += sizeof n;
    out.emplace_back(packed.data() + p, static_cast<std::size_t>(n));
    p += static_cast<std::size_t>(n);
  }
  return out;
}

void Comm::send_bytes(const std::string& buf, int dest, int tag) const {
  if (!mpi_) fail("send_bytes needs MPI");
  std::int64_t n = static_cast<std::int64_t>(buf.size());
  check(MPI_Send(&n, 1, MPI_INT64_T, dest, tag, raw()), "MPI_Send");
  for (std::int64_t i = 0; i < n; i += kChunk) {
    int cnt = static_cast<int>(std::min(kChunk, n - i));
    check(MPI_Send(buf.data() + i, cnt, MPI_BYTE, dest, tag, raw()), "MPI_Send");
  }
}

std::string Comm::recv_bytes(int source, int tag, int* out_source, int* out_tag) const {
  if (!mpi_) fail("recv_bytes needs MPI");
  std::int64_t n = 0;
  MPI_Status st;
  check(MPI_Recv(&n, 1, MPI_INT64_T, source, tag, raw(), &st), "MPI_Recv");
  // the payload must come from the same sender / tag as the header
  int src = st.MPI_SOURCE, tg = st.MPI_TAG;
  std::string buf(static_cast<std::size_t>(n), '\0');
  for (std::int64_t i = 0; i < n; i += kChunk) {
    int cnt = static_cast<int>(std::min(kChunk, n - i));
    check(MPI_Recv(buf.data() + i, cnt, MPI_BYTE, src, tg, raw(), MPI_STATUS_IGNORE), "MPI_Recv");
  }
  if (out_source) *out_source = src;
  if (out_tag) *out_tag = tg;
  return buf;
}

std::vector<std::int64_t> Comm::exchange_counts(const std::vector<std::int64_t>& sendcounts) const {
  std::vector<std::int64_t> recv(size_);
  check(MPI_Alltoall(sendcounts.data(), 1, MPI_INT64_T, recv.data(), 1, MPI_INT64_T, raw()), "MPI_Alltoall");
  return recv;
}

void Comm::alltoallv_bytes(const char* send, const std::vector<std::int64_t>& sendbytes, char* recv,
                           const std::vector<std::int64_t>& recvbytes) const {
  std::vector<std::int64_t> soff(size_ + 1, 0), roff(size_ + 1, 0);
  for (int r = 0; r < size_; ++r) {
    soff[r + 1] = soff[r] + sendbytes[r];
    roff[r + 1] = roff[r] + recvbytes[r];
  }
  std::vector<MPI_Request> reqs;
  for (int r = 0; r < size_; ++r) {
    if (r == rank_) continue;
    for (std::int64_t i = 0, t = 0; i < recvbytes[r]; i += kChunk, ++t) {
      reqs.emplace_back();
      int cnt = static_cast<int>(std::min(kChunk, recvbytes[r] - i));
      MPI_Irecv(recv + roff[r] + i, cnt, MPI_BYTE, r, 8000 + static_cast<int>(t), raw(), &reqs.back());
    }
  }
  for (int r = 0; r < size_; ++r) {
    if (r == rank_) continue;
    for (std::int64_t i = 0, t = 0; i < sendbytes[r]; i += kChunk, ++t) {
      reqs.emplace_back();
      int cnt = static_cast<int>(std::min(kChunk, sendbytes[r] - i));
      MPI_Isend(send + soff[r] + i, cnt, MPI_BYTE, r, 8000 + static_cast<int>(t), raw(), &reqs.back());
    }
  }
  std::memcpy(recv + roff[rank_], send + soff[rank_], static_cast<std::size_t>(sendbytes[rank_]));
  if (!reqs.empty()) check(MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE), "Waitall");
}

}  // namespace cfdpost
