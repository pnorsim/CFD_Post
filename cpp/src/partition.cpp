#include "cfdpost/partition.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "cfdpost/cells.hpp"
#include "cfdpost/contour.hpp"

namespace cfdpost {

VolumePartition::VolumePartition(const std::string& vtu_path, Comm comm) : comm_(std::move(comm)) {
  file_ = std::make_shared<VtkXmlFile>(vtu_path);
  const VtkXmlFile& f = *file_;
  if (f.dataset_type() != "UnstructuredGrid") fail(vtu_path + " is not an UnstructuredGrid");
  n_points_ = f.num_points();
  n_cells_ = f.num_cells();
  int r = comm_.rank(), p = comm_.size();
  cell_begin_ = block_begin(n_cells_, p, r);
  cell_end_ = block_begin(n_cells_, p, r + 1);
  own_begin_ = block_begin(n_points_, p, r);
  own_end_ = block_begin(n_points_, p, r + 1);

  index_t nc = cell_end_ - cell_begin_;
  const VtkArrayInfo& off = f.get("Cells", "offsets");
  const VtkArrayInfo& typ = f.get("Cells", "types");
  const VtkArrayInfo& con = f.get("Cells", "connectivity");
  std::vector<index_t> ends(static_cast<std::size_t>(nc) + 1, 0);
  if (cell_begin_ > 0)
    f.read_values(off, cell_begin_ - 1, nc + 1, ends.data());
  else if (nc > 0)
    f.read_values(off, 0, nc, ends.data() + 1);
  cells_.types.resize(static_cast<std::size_t>(nc));
  if (nc > 0) f.read_values(typ, cell_begin_, nc, cells_.types.data());
  index_t c0 = ends[0], c1 = ends[static_cast<std::size_t>(nc)];
  std::vector<index_t> conn(static_cast<std::size_t>(c1 - c0));
  if (c1 > c0) f.read_values(con, c0, c1 - c0, conn.data());
  cells_.offsets.resize(static_cast<std::size_t>(nc) + 1);
  for (index_t i = 0; i <= nc; ++i) cells_.offsets[i] = ends[i] - c0;
  build_local(conn.data(), static_cast<index_t>(conn.size()));
  conn.clear();
  conn.shrink_to_fit();

  const VtkArrayInfo& pts = f.get("Points", "Points");
  if (pts.ncomp != 3) fail("Points must have 3 components");
  setup_exchange();
  // each rank reads its contiguous owned block and the halo comes over MPI:
  // the file is read once in total (sequential I/O, parallel-filesystem friendly)
  std::vector<double> own(static_cast<std::size_t>(n_owned() * 3));
  f.read_values(pts, own_begin_ * 3, n_owned() * 3, own.data());
  xyz_ = to_touched(own.data(), 3);
}

VolumePartition::VolumePartition(const double* xyz, index_t npts, const std::uint8_t* types, const index_t* offsets,
                                 const index_t* conn, index_t ncells, Comm comm)
    : comm_(std::move(comm)) {
  n_points_ = npts;
  n_cells_ = ncells;
  int r = comm_.rank(), p = comm_.size();
  cell_begin_ = block_begin(n_cells_, p, r);
  cell_end_ = block_begin(n_cells_, p, r + 1);
  own_begin_ = block_begin(n_points_, p, r);
  own_end_ = block_begin(n_points_, p, r + 1);
  index_t nc = cell_end_ - cell_begin_;
  cells_.types.assign(types + cell_begin_, types + cell_end_);
  index_t c0 = offsets[cell_begin_];
  cells_.offsets.resize(static_cast<std::size_t>(nc) + 1);
  for (index_t i = 0; i <= nc; ++i) cells_.offsets[i] = offsets[cell_begin_ + i] - c0;
  build_local(conn + c0, offsets[cell_end_] - c0);
  xyz_.resize(touched_.size() * 3);
  for (std::size_t i = 0; i < touched_.size(); ++i)
    for (int d = 0; d < 3; ++d) xyz_[3 * i + d] = xyz[3 * touched_[i] + d];
  setup_exchange();
}

void VolumePartition::build_local(const index_t* conn, index_t n) {
  index_t mn = n_points_, mx = -1;
#pragma omp parallel for reduction(min : mn) reduction(max : mx)
  for (index_t i = 0; i < n; ++i) {
    mn = std::min(mn, conn[i]);
    mx = std::max(mx, conn[i]);
  }
  if (n > 0 && (mn < 0 || mx >= n_points_)) fail("cell connectivity references a point out of range");
  dim_ = 0;
  for (auto t : cells_.types) dim_ = std::max(dim_, cell_dim(t));
  int dmax = dim_;
  if (comm_.size() > 1) {
    std::int64_t d = dim_;
    std::vector<double> v{static_cast<double>(d)};
    comm_.allreduce_max(v.data(), 1);
    dmax = static_cast<int>(v[0]);
  }
  dim_ = dmax;
  touched_.clear();
  if (n == 0) {
    cells_.conn.clear();
    return;
  }
  index_t range = mx - mn + 1;
  cells_.conn.resize(static_cast<std::size_t>(n));
  if (range <= 2 * n) {
    // compact id range: dense lookup table
    std::vector<lindex_t> map(static_cast<std::size_t>(range), 0);
#pragma omp parallel for
    for (index_t i = 0; i < n; ++i) __atomic_store_n(&map[conn[i] - mn], 1, __ATOMIC_RELAXED);
    index_t cnt = 0;
    for (index_t i = 0; i < range; ++i) {
      if (map[i]) {
        map[i] = static_cast<lindex_t>(cnt++);
        touched_.push_back(mn + i);
      } else {
        map[i] = -1;
      }
    }
#pragma omp parallel for
    for (index_t i = 0; i < n; ++i) cells_.conn[i] = map[conn[i] - mn];
  } else {
    // scattered ids (typical: a block of cells references nodes all over the mesh):
    // memory O(local connectivity) instead of O(global points)
    touched_.assign(conn, conn + n);
    std::sort(touched_.begin(), touched_.end());
    touched_.erase(std::unique(touched_.begin(), touched_.end()), touched_.end());
#pragma omp parallel for
    for (index_t i = 0; i < n; ++i)
      cells_.conn[i] = static_cast<lindex_t>(std::lower_bound(touched_.begin(), touched_.end(), conn[i]) -
                                             touched_.begin());
  }
  touched_.shrink_to_fit();
  if (static_cast<index_t>(touched_.size()) > INT32_MAX)
    fail("too many points on one rank for 32-bit local ids; use more ranks");
}

void VolumePartition::setup_exchange() {
  int p = comm_.size();
  identity_ = p == 1 && n_touched() == n_points_;
  send_counts_.assign(p, 0);
  for (int r = 0; r < p; ++r) {
    index_t b = block_begin(n_points_, p, r), e = block_begin(n_points_, p, r + 1);
    send_counts_[r] = std::lower_bound(touched_.begin(), touched_.end(), e) -
                      std::lower_bound(touched_.begin(), touched_.end(), b);
  }
  std::vector<index_t> req = comm_.alltoallv(touched_, send_counts_, recv_counts_);
  recv_local_.resize(req.size());
  for (std::size_t i = 0; i < req.size(); ++i) recv_local_[i] = req[i] - own_begin_;
}

std::vector<double> VolumePartition::to_touched(const double* owned, int k) const {
  if (identity_) return std::vector<double>(owned, owned + n_owned() * k);
  std::vector<double> send(recv_local_.size() * k);
#pragma omp parallel for if (recv_local_.size() > 100000)
  for (std::size_t i = 0; i < recv_local_.size(); ++i)
    std::memcpy(&send[i * k], owned + recv_local_[i] * k, sizeof(double) * k);
  std::vector<std::int64_t> sc(recv_counts_), rc;
  for (auto& c : sc) c *= k;
  return comm_.alltoallv(send, sc, rc);
}

std::vector<double> VolumePartition::sum_to_owned(const double* touched, int k) const {
  if (identity_) return std::vector<double>(touched, touched + n_touched() * k);
  std::vector<double> send(touched, touched + n_touched() * k);
  std::vector<std::int64_t> sc(send_counts_), rc;
  for (auto& c : sc) c *= k;
  std::vector<double> recv = comm_.alltoallv(send, sc, rc);
  std::vector<double> owned(static_cast<std::size_t>(n_owned() * k), 0.0);
  // contributions of different requesters may hit the same point: one block at a time
  std::size_t pos = 0;
  for (int r = 0; r < comm_.size(); ++r) {
    std::size_t n = static_cast<std::size_t>(recv_counts_[r]);
#pragma omp parallel for if (n > 100000)
    for (std::size_t i = 0; i < n; ++i)
      for (int q = 0; q < k; ++q) owned[recv_local_[pos + i] * k + q] += recv[(pos + i) * k + q];
    pos += n;
  }
  return owned;
}

std::vector<double> VolumePartition::xyz_owned() const {
  if (!xyz_owned_override_.empty()) return xyz_owned_override_;
  std::vector<double> out(static_cast<std::size_t>(n_owned() * 3));
  if (file_) {
    file_->read_values(file_->get("Points", "Points"), own_begin_ * 3, n_owned() * 3, out.data());
  } else {
    // gather from touched copies through the owners (every point is touched by someone
    // unless orphaned; orphans keep 0)
    std::vector<double> one(touched_.size(), 1.0);
    std::vector<double> cnt = sum_to_owned(one.data(), 1);
    std::vector<double> s = sum_to_owned(xyz_.data(), 3);
    for (index_t i = 0; i < n_owned(); ++i)
      for (int d = 0; d < 3; ++d) out[3 * i + d] = cnt[i] > 0 ? s[3 * i + d] / cnt[i] : 0.0;
  }
  return out;
}

void VolumePartition::set_coordinates(const double* owned_xyz) {
  xyz_owned_override_.assign(owned_xyz, owned_xyz + n_owned() * 3);
  xyz_ = to_touched(owned_xyz, 3);
}

std::vector<double> VolumePartition::read_point_data(const std::string& name, bool owned, int& ncomp) const {
  if (!file_) fail("partition was not built from a file");
  const VtkArrayInfo& a = file_->get("PointData", name);
  ncomp = a.ncomp;
  if (owned) {
    std::vector<double> out(static_cast<std::size_t>(n_owned() * ncomp));
    file_->read_values(a, own_begin_ * ncomp, n_owned() * ncomp, out.data());
    return out;
  }
  // owned block from the file, halo through MPI (see constructor)
  std::vector<double> own(static_cast<std::size_t>(n_owned() * ncomp));
  file_->read_values(a, own_begin_ * ncomp, n_owned() * ncomp, own.data());
  return to_touched(own.data(), ncomp);
}

std::vector<double> VolumePartition::read_cell_data(const std::string& name, int& ncomp) const {
  if (!file_) fail("partition was not built from a file");
  const VtkArrayInfo& a = file_->get("CellData", name);
  ncomp = a.ncomp;
  std::vector<double> out(static_cast<std::size_t>((cell_end_ - cell_begin_) * ncomp));
  file_->read_values(a, cell_begin_ * ncomp, (cell_end_ - cell_begin_) * ncomp, out.data());
  return out;
}

void VolumePartition::build_inverse() const {
  if (!n2c_off_.empty()) return;
  index_t nt = n_touched(), nc = cells_.size();
  std::vector<index_t> cnt(static_cast<std::size_t>(nt) + 1, 0);
#pragma omp parallel for
  for (index_t c = 0; c < nc; ++c) {
    if (cell_dim(cells_.types[c]) != dim_) continue;
    for (index_t j = cells_.offsets[c]; j < cells_.offsets[c + 1]; ++j)
      __atomic_fetch_add(&cnt[cells_.conn[j] + 1], 1, __ATOMIC_RELAXED);
  }
  for (index_t i = 0; i < nt; ++i) cnt[i + 1] += cnt[i];
  n2c_.resize(static_cast<std::size_t>(cnt[nt]));
  std::vector<index_t> pos(cnt.begin(), cnt.end() - 1);
#pragma omp parallel for
  for (index_t c = 0; c < nc; ++c) {
    if (cell_dim(cells_.types[c]) != dim_) continue;
    for (index_t j = cells_.offsets[c]; j < cells_.offsets[c + 1]; ++j) {
      index_t at = __atomic_fetch_add(&pos[cells_.conn[j]], 1, __ATOMIC_RELAXED);
      n2c_[at] = static_cast<lindex_t>(c);
    }
  }
#pragma omp parallel for schedule(dynamic, 4096)
  for (index_t i = 0; i < nt; ++i) std::sort(n2c_.begin() + cnt[i], n2c_.begin() + cnt[i + 1]);
  n2c_off_.swap(cnt);
}

namespace {
constexpr int kMaxComp = 16;  // components per Green-Gauss pass
}

std::vector<double> VolumePartition::gradient(const double* owned, int k) const {
  std::vector<double> out(static_cast<std::size_t>(n_owned() * k * 3), 0.0);
  build_inverse();
  index_t nc = cells_.size(), nt = n_touched();
  for (int k0 = 0; k0 < k; k0 += kMaxComp) {
    int kk = std::min(kMaxComp, k - k0);
    std::vector<double> sub(static_cast<std::size_t>(n_owned() * kk));
    for (index_t i = 0; i < n_owned(); ++i)
      for (int q = 0; q < kk; ++q) sub[i * kk + q] = owned[i * k + k0 + q];
    std::vector<double> ft = to_touched(sub.data(), kk);
    const int w = 3 * kk + 1;
    std::vector<double> cg(static_cast<std::size_t>(nc * w), 0.0);
#pragma omp parallel
    {
      Vec3 x[8];
      double f[8 * kMaxComp];
      double g[3 * kMaxComp];
#pragma omp for schedule(static)
      for (index_t c = 0; c < nc; ++c) {
        int type = cells_.types[c];
        if (cell_dim(type) != dim_) continue;
        index_t b = cells_.offsets[c];
        int nn = static_cast<int>(cells_.offsets[c + 1] - b);
        if (nn > 8) continue;  // polygons with many nodes are not volume cells
        for (int i = 0; i < nn; ++i) {
          lindex_t li = cells_.conn[b + i];
          x[i] = load3(xyz_.data(), li);
          for (int q = 0; q < kk; ++q) f[i * kk + q] = ft[static_cast<std::size_t>(li) * kk + q];
        }
        double v = dim_ == 3 ? cell_gradient_3d(type, x, f, kk, g) : cell_gradient_2d(nn, x, f, kk, g);
        double av = std::fabs(v);
        double* o = &cg[c * w];
        o[0] = av;
        for (int q = 0; q < 3 * kk; ++q) o[1 + q] = av * g[q];
      }
    }
    std::vector<double> acc(static_cast<std::size_t>(nt * w), 0.0);
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < nt; ++i) {
      double* a = &acc[i * w];
      for (index_t j = n2c_off_[i]; j < n2c_off_[i + 1]; ++j) {
        const double* o = &cg[static_cast<std::size_t>(n2c_[j]) * w];
        for (int q = 0; q < w; ++q) a[q] += o[q];
      }
    }
    cg.clear();
    cg.shrink_to_fit();
    std::vector<double> own = sum_to_owned(acc.data(), w);
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < n_owned(); ++i) {
      double wsum = own[i * w];
      for (int q = 0; q < kk; ++q)
        for (int d = 0; d < 3; ++d)
          out[(i * k + k0 + q) * 3 + d] = wsum > 0 ? own[i * w + 1 + 3 * q + d] / wsum : std::nan("");
    }
  }
  return out;
}

std::vector<double> VolumePartition::cell_volumes() const {
  index_t nc = cells_.size();
  std::vector<double> v(static_cast<std::size_t>(nc), 0.0);
#pragma omp parallel
  {
    Vec3 x[8];
    double f[8], g[3];
#pragma omp for schedule(static)
    for (index_t c = 0; c < nc; ++c) {
      int type = cells_.types[c];
      index_t b = cells_.offsets[c];
      int nn = static_cast<int>(cells_.offsets[c + 1] - b);
      if (cell_dim(type) != dim_ || nn > 8) continue;
      for (int i = 0; i < nn; ++i) {
        x[i] = load3(xyz_.data(), cells_.conn[b + i]);
        f[i] = 0.0;
      }
      v[c] = dim_ == 3 ? cell_volume_3d(type, x) : cell_gradient_2d(nn, x, f, 1, g);
    }
  }
  return v;
}

std::vector<double> VolumePartition::nodal_volume() const {
  std::vector<double> cv = cell_volumes();
  build_inverse();
  index_t nt = n_touched();
  std::vector<double> acc(static_cast<std::size_t>(nt), 0.0);
#pragma omp parallel for schedule(static)
  for (index_t i = 0; i < nt; ++i) {
    double s = 0;
    for (index_t j = n2c_off_[i]; j < n2c_off_[i + 1]; ++j) {
      lindex_t c = n2c_[j];
      s += std::fabs(cv[c]) / static_cast<double>(cells_.offsets[c + 1] - cells_.offsets[c]);
    }
    acc[i] = s;
  }
  return sum_to_owned(acc.data(), 1);
}

ContourResult VolumePartition::contour_touched(const double* scalar, double iso, const double* fields, int nf,
                                               bool gather) const {
  ContourResult local = contour_cells(cells_, touched_.data(), xyz_.data(), scalar, iso, fields, nf, cell_begin_);
  if (!gather || comm_.size() == 1) return local;
  return gather_contour(local, comm_, 0);
}

}  // namespace cfdpost
