// Distributed unstructured volume.
//
// Cells are split in contiguous blocks over the ranks of a communicator; each
// rank reads only its cells (and the points they touch) from the file.
// Point data live in two layouts:
//   owned   : the contiguous block of global points [own_begin, own_end) of this
//             rank. Every point is owned by exactly one rank, so element-wise
//             numpy code and global reductions work on owned arrays.
//   touched : the sorted global ids of the points referenced by the local cells
//             (what cell kernels need). to_touched / sum_to_owned move data
//             between the two layouts with one all-to-all each.
// With one rank and every point referenced, both layouts are the identity.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cfdpost/comm.hpp"
#include "cfdpost/common.hpp"
#include "cfdpost/vtu.hpp"

namespace cfdpost {

struct ContourResult;

class VolumePartition {
 public:
  // Read the cells of this rank from a .vtu file (UnstructuredGrid).
  VolumePartition(const std::string& vtu_path, Comm comm);
  // Build from global in-memory arrays (every rank passes the full arrays).
  VolumePartition(const double* xyz, index_t npts, const std::uint8_t* types, const index_t* offsets,
                  const index_t* conn, index_t ncells, Comm comm);

  const Comm& comm() const { return comm_; }
  index_t n_points() const { return n_points_; }
  index_t n_cells() const { return n_cells_; }
  index_t cell_begin() const { return cell_begin_; }
  index_t cell_end() const { return cell_end_; }
  index_t own_begin() const { return own_begin_; }
  index_t own_end() const { return own_end_; }
  index_t n_owned() const { return own_end_ - own_begin_; }
  index_t n_touched() const { return static_cast<index_t>(touched_.size()); }
  int dim() const { return dim_; }
  const std::vector<index_t>& touched() const { return touched_; }
  const CellArray<lindex_t>& cells() const { return cells_; }
  const std::vector<double>& xyz_touched() const { return xyz_; }
  std::vector<double> xyz_owned() const;
  const VtkXmlFile* file() const { return file_.get(); }

  // Replace coordinates (e.g. deformed mesh with the same topology); owned layout.
  void set_coordinates(const double* owned_xyz);

  // Point / cell data from the file.
  std::vector<double> read_point_data(const std::string& name, bool owned, int& ncomp) const;
  std::vector<double> read_cell_data(const std::string& name, int& ncomp) const;  // local cells

  // Layout changes (k components per point).
  std::vector<double> to_touched(const double* owned, int k) const;
  std::vector<double> sum_to_owned(const double* touched, int k) const;

  // Nodal gradient of an owned field with k components -> owned (n_owned, k, 3).
  // Cell-wise Green-Gauss gradients averaged to the nodes with volume weights
  // (exact for linear fields, also at boundary nodes).
  std::vector<double> gradient(const double* owned, int k) const;
  // Median-dual-like nodal volume: sum over cells of |V_c| / nodes(c), owned.
  std::vector<double> nodal_volume() const;
  // Signed volumes of the local cells.
  std::vector<double> cell_volumes() const;

  // Iso-surface (3D) / iso-line (2D) of a touched-layout scalar; fields (touched
  // layout, nf values per point) are interpolated onto it. With gather=true the
  // pieces are merged on rank 0 (other ranks get an empty result).
  ContourResult contour_touched(const double* scalar, double iso, const double* fields, int nf, bool gather) const;

 private:
  void build_local(const index_t* conn_global, index_t nconn);
  void setup_exchange();
  void build_inverse() const;

  Comm comm_;
  std::shared_ptr<VtkXmlFile> file_;
  index_t n_points_ = 0, n_cells_ = 0;
  index_t cell_begin_ = 0, cell_end_ = 0;
  index_t own_begin_ = 0, own_end_ = 0;
  int dim_ = 3;
  std::vector<index_t> touched_;
  CellArray<lindex_t> cells_;
  std::vector<double> xyz_;
  std::vector<double> xyz_owned_override_;

  bool identity_ = false;
  std::vector<std::int64_t> send_counts_;  // touched points per owner rank
  std::vector<std::int64_t> recv_counts_;  // requested owned points per requester rank
  std::vector<index_t> recv_local_;        // owned-local ids, grouped by requester rank

  mutable std::vector<index_t> n2c_off_;  // touched point -> local cells (lazy)
  mutable std::vector<lindex_t> n2c_;
};

}  // namespace cfdpost
