// Multithreaded reader for native SU2 ASCII meshes (.su2).
#pragma once

#include <string>
#include <vector>

#include "cfdpost/common.hpp"

namespace cfdpost {

struct Su2Marker {
  std::string tag;
  CellArray<index_t> elems;  // boundary elements, global point ids
};

struct Su2Mesh {
  int ndim = 0;
  index_t npoin = 0;
  index_t nelem = 0;           // from the header, also when elements are skipped
  std::vector<double> xyz;     // npoin*3 (z = 0 in 2D)
  CellArray<index_t> elems;    // volume elements (empty when skipped)
  std::vector<Su2Marker> markers;
};

struct Su2ReadOptions {
  bool points = true;
  bool elements = true;
  bool markers = true;
  std::vector<std::string> only_markers;  // empty = all
  int zone = 1;                           // for multi-zone files (IZONE=)
};

Su2Mesh read_su2(const std::string& path, const Su2ReadOptions& opt);

// Marker names and element counts only (fast: skips points and elements).
std::vector<std::pair<std::string, index_t>> su2_marker_list(const std::string& path, int zone = 1);

}  // namespace cfdpost
