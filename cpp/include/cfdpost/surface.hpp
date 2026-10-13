// Surface integration of pressure / friction loads.
//
// Faces (tri, quad, polygon) are split into triangles around their centroid,
// nodal Cp and Cf are linear on each triangle, and every integral (force,
// moment, strip loads) is evaluated exactly for that piecewise-linear field.
// All quantities are per unit dynamic pressure: dF/q = (-Cp n + Cf) dA with n
// the unit normal pointing into the fluid.
#pragma once

#include <vector>

#include "cfdpost/common.hpp"

namespace cfdpost {

struct SurfaceInput {
  const double* xyz = nullptr;  // points (n*3)
  index_t npts = 0;
  const std::uint8_t* types = nullptr;
  const index_t* offsets = nullptr;  // nfaces+1
  const index_t* conn = nullptr;
  index_t nfaces = 0;
  const double* cp = nullptr;      // n (required)
  const double* cf = nullptr;      // n*3 (optional)
  const double* sign = nullptr;    // per face, +1 if the right-hand normal points into the fluid, -1 otherwise
  double ref[3] = {0, 0, 0};       // moment reference point
};

// Per face, 15 values: area vector into the fluid S (3), pressure force (3),
// viscous force (3), total moment about ref (3), centroid (3).
std::vector<double> face_loads(const SurfaceInput& in);

// Loads in slabs edges[k] <= axis.x < edges[k+1]; faces are clipped exactly.
// Per strip, 20 values: pressure force (3), viscous force (3), moment (3),
// wetted area (1), area vector sum (3), sum |S_i| per component (3),
// min / max of chord_dir.x over the clipped surface (2), strip-wise integral of
// Cp dA (1), clipped area count flag (1 if any surface in strip).
std::vector<double> strip_loads(const SurfaceInput& in, const double axis[3], const std::vector<double>& edges,
                                const double chord_dir[3]);

// Orientation of boundary faces from the volume mesh: for each face, +1 if its
// right-hand normal points into the adjacent volume cell (into the fluid), -1
// if it points out of it, 0 if no adjacent cell was found.
std::vector<std::int8_t> orient_faces(const double* xyz, index_t npts, const std::uint8_t* vtypes,
                                      const index_t* voffsets, const index_t* vconn, index_t ncells,
                                      const std::uint8_t* ftypes, const index_t* foffsets, const index_t* fconn,
                                      index_t nfaces);

}  // namespace cfdpost
