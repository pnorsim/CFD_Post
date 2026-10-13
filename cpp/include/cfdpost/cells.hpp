// Cell topology (faces in VTK node ordering) and per-cell geometry kernels.
#pragma once

#include <array>
#include <cmath>

#include "cfdpost/common.hpp"

namespace cfdpost {

struct Vec3 {
  double x = 0, y = 0, z = 0;
  Vec3() = default;
  Vec3(double a, double b, double c) : x(a), y(b), z(c) {}
  Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
  Vec3& operator+=(const Vec3& o) {
    x += o.x;
    y += o.y;
    z += o.z;
    return *this;
  }
  double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
  Vec3 cross(const Vec3& o) const { return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x}; }
  double norm() const { return std::sqrt(dot(*this)); }
};

inline Vec3 load3(const double* xyz, index_t i) { return {xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]}; }

// Faces of 3D cells, outward for positively oriented VTK cells.
// Face k of a cell: kFaceSize[k] local node indices in kFaceNodes[k].
struct CellFaces {
  int nfaces;
  int size[6];
  int nodes[6][4];
};

const CellFaces& faces_of(int type);  // 3D types only

// Edges (as local node index pairs) of a 2D cell given its node count: (i, i+1 mod n).

// Green-Gauss gradient of k fields over one 3D cell.
//   x: node coordinates (nn), f: values f[node*k + comp]
//   grad: out (k*3), returns the signed volume.
// Quadrilateral faces are split into 4 triangles around the face centroid, so
// the result is exact for linear fields on any (also warped) cell.
double cell_gradient_3d(int type, const Vec3* x, const double* f, int k, double* grad);

// Same for 2D cells (tri, quad, polygon) lying in the xy plane.
double cell_gradient_2d(int nn, const Vec3* x, const double* f, int k, double* grad);

// Signed volume of a 3D cell (same face splitting as the gradient).
double cell_volume_3d(int type, const Vec3* x);

}  // namespace cfdpost
