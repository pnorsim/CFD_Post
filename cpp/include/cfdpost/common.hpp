// Common types and helpers for the cfdpost C++ core.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cfdpost {

using index_t = std::int64_t;   // global ids (points, cells)
using lindex_t = std::int32_t;  // rank-local ids

struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

[[noreturn]] inline void fail(const std::string& msg) { throw Error(msg); }

// VTK cell type codes (SU2 uses the same codes in its mesh files).
enum CellType : std::uint8_t {
  VTK_VERTEX = 1,
  VTK_LINE = 3,
  VTK_TRIANGLE = 5,
  VTK_POLYGON = 7,
  VTK_QUAD = 9,
  VTK_TETRA = 10,
  VTK_HEXAHEDRON = 12,
  VTK_WEDGE = 13,
  VTK_PYRAMID = 14,
};

// Number of nodes of a fixed-size cell type, 0 if variable / unknown.
inline int cell_num_nodes(int type) {
  switch (type) {
    case VTK_VERTEX: return 1;
    case VTK_LINE: return 2;
    case VTK_TRIANGLE: return 3;
    case VTK_QUAD: return 4;
    case VTK_TETRA: return 4;
    case VTK_HEXAHEDRON: return 8;
    case VTK_WEDGE: return 6;
    case VTK_PYRAMID: return 5;
    default: return 0;
  }
}

// Topological dimension of a cell type.
inline int cell_dim(int type) {
  switch (type) {
    case VTK_VERTEX: return 0;
    case VTK_LINE: return 1;
    case VTK_TRIANGLE:
    case VTK_QUAD:
    case VTK_POLYGON: return 2;
    case VTK_TETRA:
    case VTK_HEXAHEDRON:
    case VTK_WEDGE:
    case VTK_PYRAMID: return 3;
    default: return -1;
  }
}

inline int num_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

// Cells in compressed-row form: cell c has nodes conn[offsets[c] .. offsets[c+1]).
template <class I>
struct CellArray {
  std::vector<std::uint8_t> types;
  std::vector<index_t> offsets{0};
  std::vector<I> conn;
  index_t size() const { return static_cast<index_t>(types.size()); }
};

}  // namespace cfdpost
