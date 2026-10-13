// Iso-surfaces / iso-lines of point scalars on mixed unstructured cells.
//
// "Marching polyhedra": every cell face that changes sign contributes an
// oriented segment between its two edge crossings; the segments of a cell
// chain into closed loops, which are the output polygons. No lookup tables, any
// VTK cell type, and crack-free across cells because each crossing is keyed by
// the global ids of its edge. Ambiguous quad faces are resolved with the face
// centre value, identically from both sides. Polygon normals point towards
// increasing scalar.
#pragma once

#include <vector>

#include "cfdpost/comm.hpp"
#include "cfdpost/common.hpp"

namespace cfdpost {

struct ContourResult {
  std::vector<index_t> edge_a, edge_b;  // global point ids of the cut edge (a < b)
  std::vector<double> t;                // position along the edge from a to b
  std::vector<double> xyz;              // n*3
  std::vector<double> fields;           // n*nf interpolated point fields
  int nf = 0;
  CellArray<index_t> polys;             // from 3D cells
  CellArray<index_t> lines;             // from 2D cells (segments)
  std::vector<index_t> poly_cell, line_cell;  // source (global) cell ids
  index_t num_points() const { return static_cast<index_t>(t.size()); }
};

// Contour local cells. conn uses local ids; gids maps local -> global ids;
// cell_gid0 is the global id of local cell 0.
ContourResult contour_cells(const CellArray<lindex_t>& cells, const index_t* gids, const double* xyz,
                            const double* scalar, double iso, const double* fields, int nf, index_t cell_gid0);

// Merge the per-rank pieces on `root` (deduplicating shared edge points).
ContourResult gather_contour(const ContourResult& local, const Comm& comm, int root);

}  // namespace cfdpost
