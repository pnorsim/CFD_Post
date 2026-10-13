#include "cfdpost/cells.hpp"

namespace cfdpost {

namespace {
// VTK local face definitions; normals (right-hand rule) point out of the cell.
const CellFaces kTet{4, {3, 3, 3, 3}, {{0, 1, 3}, {1, 2, 3}, {2, 0, 3}, {0, 2, 1}}};
const CellFaces kHex{6,
                     {4, 4, 4, 4, 4, 4},
                     {{0, 4, 7, 3}, {1, 2, 6, 5}, {0, 1, 5, 4}, {3, 7, 6, 2}, {0, 3, 2, 1}, {4, 5, 6, 7}}};
const CellFaces kWedge{5, {3, 3, 4, 4, 4}, {{0, 1, 2}, {3, 5, 4}, {0, 3, 4, 1}, {1, 4, 5, 2}, {2, 5, 3, 0}}};
const CellFaces kPyramid{5, {4, 3, 3, 3, 3}, {{0, 3, 2, 1}, {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}}};
}  // namespace

const CellFaces& faces_of(int type) {
  switch (type) {
    case VTK_TETRA: return kTet;
    case VTK_HEXAHEDRON: return kHex;
    case VTK_WEDGE: return kWedge;
    case VTK_PYRAMID: return kPyramid;
    default: fail("faces_of: not a 3D cell type: " + std::to_string(type));
  }
}

namespace {

// Visit the triangles of the cell boundary: fn(a, b, c, fa_index_list...) with
// coordinates relative to `o`. For quads a centroid point is synthesised.
template <class Fn>
inline void for_each_boundary_triangle(int type, const Vec3* x, const Vec3& o, Fn&& fn) {
  const CellFaces& cf = faces_of(type);
  for (int fi = 0; fi < cf.nfaces; ++fi) {
    const int* n = cf.nodes[fi];
    if (cf.size[fi] == 3) {
      fn(x[n[0]] - o, x[n[1]] - o, x[n[2]] - o, n[0], n[1], n[2], -1, fi);
    } else {
      Vec3 m = (x[n[0]] + x[n[1]] + x[n[2]] + x[n[3]]) * 0.25 - o;
      for (int e = 0; e < 4; ++e) {
        int a = n[e], b = n[(e + 1) % 4];
        fn(x[a] - o, x[b] - o, m, a, b, -1, fi, fi);
      }
    }
  }
}

}  // namespace

double cell_gradient_3d(int type, const Vec3* x, const double* f, int k, double* grad) {
  const CellFaces& cf = faces_of(type);
  int nn = cell_num_nodes(type);
  Vec3 o;
  for (int i = 0; i < nn; ++i) o += x[i];
  o = o * (1.0 / nn);
  for (int c = 0; c < 3 * k; ++c) grad[c] = 0.0;
  double vol = 0.0;
  double fm[16];  // face-centroid values for the current quad face (k <= 16)
  int cur_face = -1;
  for_each_boundary_triangle(type, x, o, [&](const Vec3& a, const Vec3& b, const Vec3& c, int ia, int ib, int ic,
                                             int quad_face, int fi) {
    Vec3 S = (b - a).cross(c - a) * 0.5;
    vol += (a + b + c).dot(S) / 9.0;  // (1/3) * centroid . S
    if (quad_face >= 0 && cur_face != fi) {
      const int* n = cf.nodes[fi];
      for (int q = 0; q < k; ++q)
        fm[q] = 0.25 * (f[n[0] * k + q] + f[n[1] * k + q] + f[n[2] * k + q] + f[n[3] * k + q]);
      cur_face = fi;
    }
    for (int q = 0; q < k; ++q) {
      double fc = ic >= 0 ? f[ic * k + q] : fm[q];
      double fa = (f[ia * k + q] + f[ib * k + q] + fc) / 3.0;
      grad[3 * q] += fa * S.x;
      grad[3 * q + 1] += fa * S.y;
      grad[3 * q + 2] += fa * S.z;
    }
  });
  if (vol != 0.0)
    for (int c = 0; c < 3 * k; ++c) grad[c] /= vol;
  return vol;
}

double cell_volume_3d(int type, const Vec3* x) {
  int nn = cell_num_nodes(type);
  Vec3 o;
  for (int i = 0; i < nn; ++i) o += x[i];
  o = o * (1.0 / nn);
  double vol = 0.0;
  for_each_boundary_triangle(type, x, o, [&](const Vec3& a, const Vec3& b, const Vec3& c, int, int, int, int, int) {
    vol += (a + b + c).dot((b - a).cross(c - a)) / 18.0;
  });
  return vol;
}

double cell_gradient_2d(int nn, const Vec3* x, const double* f, int k, double* grad) {
  // line integral: grad f = (1/A) sum_edges f_e n_e L_e, n_e = (dy, -dx) for CCW polygons
  for (int c = 0; c < 3 * k; ++c) grad[c] = 0.0;
  double area = 0.0;
  for (int e = 0; e < nn; ++e) {
    const Vec3& a = x[e];
    const Vec3& b = x[(e + 1) % nn];
    area += 0.5 * (a.x * b.y - b.x * a.y);
    double nx = b.y - a.y, ny = -(b.x - a.x);
    for (int q = 0; q < k; ++q) {
      double fe = 0.5 * (f[e * k + q] + f[((e + 1) % nn) * k + q]);
      grad[3 * q] += fe * nx;
      grad[3 * q + 1] += fe * ny;
    }
  }
  if (area != 0.0)
    for (int c = 0; c < 3 * k; ++c) grad[c] /= area;
  return area;
}

}  // namespace cfdpost
