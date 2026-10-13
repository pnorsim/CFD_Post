#include "cfdpost/surface.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>

#include "cfdpost/cells.hpp"

namespace cfdpost {

namespace {

// A vertex carrying the linear fields.
struct V {
  Vec3 r;
  double cp;
  Vec3 cf;
};

inline V lerp(const V& a, const V& b, double t) {
  return {a.r + (b.r - a.r) * t, a.cp + (b.cp - a.cp) * t, a.cf + (b.cf - a.cf) * t};
}

struct Acc {
  Vec3 fp, fv, m, S, absS, cen;
  double area = 0, cpA = 0;
  double cmin = std::numeric_limits<double>::infinity();
  double cmax = -std::numeric_limits<double>::infinity();
};

// Exact integrals over one triangle with linear cp, cf; sgn orients the normal.
inline void integrate_tri(const V& a, const V& b, const V& c, double sgn, const Vec3& ref, Acc& acc) {
  Vec3 S = (b.r - a.r).cross(c.r - a.r) * (0.5 * sgn);
  double A = S.norm();
  if (A == 0) return;
  Vec3 n = S * (1.0 / A);
  const V* v[3] = {&a, &b, &c};
  Vec3 g[3], r[3];
  for (int i = 0; i < 3; ++i) {
    g[i] = v[i]->cf + n * (-v[i]->cp);
    r[i] = v[i]->r - ref;
  }
  double cpm = (a.cp + b.cp + c.cp) / 3.0;
  acc.fp += S * (-cpm);
  acc.fv += (a.cf + b.cf + c.cf) * (A / 3.0);
  // int (r x g) dA = A/12 * sum_ij (1 + d_ij) r_i x g_j
  Vec3 m;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) m += r[i].cross(g[j]) * (i == j ? 2.0 : 1.0);
  acc.m += m * (A / 12.0);
  acc.S += S;
  acc.absS += Vec3(std::fabs(S.x), std::fabs(S.y), std::fabs(S.z));
  acc.area += A;
  acc.cpA += cpm * A;
  acc.cen += (a.r + b.r + c.r) * (A / 3.0);
}

inline V vertex(const SurfaceInput& in, index_t i) {
  V v;
  v.r = load3(in.xyz, i);
  v.cp = in.cp[i];
  if (in.cf) v.cf = load3(in.cf, i);
  return v;
}

// Triangles of face f (centroid fan for quads/polygons).
template <class Fn>
inline void face_triangles(const SurfaceInput& in, index_t f, Fn&& fn) {
  index_t b = in.offsets[f];
  int nn = static_cast<int>(in.offsets[f + 1] - b);
  if (nn < 3) return;
  if (nn == 3) {
    fn(vertex(in, in.conn[b]), vertex(in, in.conn[b + 1]), vertex(in, in.conn[b + 2]));
    return;
  }
  V c{Vec3(), 0.0, Vec3()};
  V vs[64];
  nn = std::min(nn, 64);
  for (int i = 0; i < nn; ++i) {
    vs[i] = vertex(in, in.conn[b + i]);
    c.r += vs[i].r;
    c.cp += vs[i].cp;
    c.cf += vs[i].cf;
  }
  double s = 1.0 / nn;
  c.r = c.r * s;
  c.cp *= s;
  c.cf = c.cf * s;
  for (int i = 0; i < nn; ++i) fn(vs[i], vs[(i + 1) % nn], c);
}

// Clip polygon p (n vertices) to axis.r >= lo (keep_above) or <= hi.
int clip(const V* p, int n, const Vec3& ax, double lim, bool keep_above, V* out) {
  int m = 0;
  for (int i = 0; i < n; ++i) {
    const V& a = p[i];
    const V& b = p[(i + 1) % n];
    double da = ax.dot(a.r) - lim, db = ax.dot(b.r) - lim;
    if (!keep_above) {
      da = -da;
      db = -db;
    }
    bool ina = da >= 0, inb = db >= 0;
    if (ina) out[m++] = a;
    if (ina != inb) out[m++] = lerp(a, b, da / (da - db));
  }
  return m;
}

}  // namespace

std::vector<double> face_loads(const SurfaceInput& in) {
  std::vector<double> out(static_cast<std::size_t>(in.nfaces) * 15, 0.0);
  Vec3 ref(in.ref[0], in.ref[1], in.ref[2]);
#pragma omp parallel for schedule(static)
  for (index_t f = 0; f < in.nfaces; ++f) {
    Acc acc;
    double sg = in.sign ? in.sign[f] : 1.0;
    face_triangles(in, f, [&](const V& a, const V& b, const V& c) { integrate_tri(a, b, c, sg, ref, acc); });
    double* o = &out[f * 15];
    Vec3 cen = acc.area > 0 ? acc.cen * (1.0 / acc.area) : Vec3();
    Vec3 vals[5] = {acc.S, acc.fp, acc.fv, acc.m, cen};
    for (int k = 0; k < 5; ++k) {
      o[3 * k] = vals[k].x;
      o[3 * k + 1] = vals[k].y;
      o[3 * k + 2] = vals[k].z;
    }
  }
  return out;
}

std::vector<double> strip_loads(const SurfaceInput& in, const double axis_[3], const std::vector<double>& edges,
                                const double chord_[3]) {
  Vec3 ax(axis_[0], axis_[1], axis_[2]);
  ax = ax * (1.0 / ax.norm());
  Vec3 ch(chord_[0], chord_[1], chord_[2]);
  Vec3 ref(in.ref[0], in.ref[1], in.ref[2]);
  int nb = static_cast<int>(edges.size()) - 1;
  if (nb < 1) fail("strip_loads: need at least 2 edges");
  for (int k = 0; k < nb; ++k)
    if (!(edges[k] < edges[k + 1])) fail("strip_loads: edges must be increasing");
  int nt = num_threads();
  std::vector<std::vector<Acc>> accs(nt, std::vector<Acc>(nb));
#pragma omp parallel
  {
    int t = 0;
#ifdef _OPENMP
    t = omp_get_thread_num();
#endif
    std::vector<Acc>& acc = accs[t];
    V p0[3], p1[8], p2[8];
#pragma omp for schedule(dynamic, 1024)
    for (index_t f = 0; f < in.nfaces; ++f) {
      double sg = in.sign ? in.sign[f] : 1.0;
      face_triangles(in, f, [&](const V& a, const V& b, const V& c) {
        double sa = ax.dot(a.r), sb = ax.dot(b.r), sc = ax.dot(c.r);
        double smin = std::min({sa, sb, sc}), smax = std::max({sa, sb, sc});
        // strips are [e_k, e_k+1), the last one closed: faces lying on the last plane count
        if (smax < edges.front() || smin > edges.back()) return;
        int k0 = static_cast<int>(std::upper_bound(edges.begin(), edges.end(), smin) - edges.begin()) - 1;
        int k1 = static_cast<int>(std::upper_bound(edges.begin(), edges.end(), smax) - edges.begin()) - 1;
        k0 = std::min(std::max(k0, 0), nb - 1);
        k1 = std::min(std::max(k1, 0), nb - 1);
        p0[0] = a;
        p0[1] = b;
        p0[2] = c;
        for (int k = k0; k <= k1; ++k) {
          Acc& A = acc[k];
          int n1 = 3;
          const V* poly = p0;
          if (k == k0 && k == k1 && smin >= edges[k] && smax <= edges[k + 1]) {
            // fully inside
          } else {
            n1 = clip(p0, 3, ax, edges[k], true, p1);
            if (n1 < 3) continue;
            n1 = clip(p1, n1, ax, edges[k + 1], false, p2);
            if (n1 < 3) continue;
            poly = p2;
          }
          for (int i = 0; i < n1; ++i) {
            double cv = ch.dot(poly[i].r);
            A.cmin = std::min(A.cmin, cv);
            A.cmax = std::max(A.cmax, cv);
          }
          for (int i = 1; i + 1 < n1; ++i) integrate_tri(poly[0], poly[i], poly[i + 1], sg, ref, A);
        }
      });
    }
  }
  std::vector<double> out(static_cast<std::size_t>(nb) * 20, 0.0);
  for (int k = 0; k < nb; ++k) {
    Acc s;
    for (int t = 0; t < nt; ++t) {
      const Acc& a = accs[t][k];
      s.fp += a.fp;
      s.fv += a.fv;
      s.m += a.m;
      s.S += a.S;
      s.absS += a.absS;
      s.area += a.area;
      s.cpA += a.cpA;
      s.cmin = std::min(s.cmin, a.cmin);
      s.cmax = std::max(s.cmax, a.cmax);
    }
    double* o = &out[k * 20];
    Vec3 v3[3] = {s.fp, s.fv, s.m};
    for (int j = 0; j < 3; ++j) {
      o[3 * j] = v3[j].x;
      o[3 * j + 1] = v3[j].y;
      o[3 * j + 2] = v3[j].z;
    }
    o[9] = s.area;
    o[10] = s.S.x;
    o[11] = s.S.y;
    o[12] = s.S.z;
    o[13] = s.absS.x;
    o[14] = s.absS.y;
    o[15] = s.absS.z;
    bool any = s.area > 0;
    o[16] = any ? s.cmin : std::nan("");
    o[17] = any ? s.cmax : std::nan("");
    o[18] = s.cpA;
    o[19] = any ? 1.0 : 0.0;
  }
  return out;
}

namespace {
struct FaceKey {
  std::array<index_t, 4> n;  // sorted, -1 padded
  bool operator==(const FaceKey& o) const { return n == o.n; }
};
struct FaceKeyHash {
  std::size_t operator()(const FaceKey& k) const {
    std::size_t h = 1469598103934665603ull;
    for (auto v : k.n) h = (h ^ static_cast<std::size_t>(v)) * 1099511628211ull;
    return h;
  }
};
inline FaceKey make_key(const index_t* nodes, int n) {
  FaceKey k;
  k.n = {-1, -1, -1, -1};
  for (int i = 0; i < n && i < 4; ++i) k.n[i] = nodes[i];
  std::sort(k.n.begin(), k.n.begin() + std::min(n, 4));
  return k;
}
}  // namespace

std::vector<std::int8_t> orient_faces(const double* xyz, index_t npts, const std::uint8_t* vtypes,
                                      const index_t* voffsets, const index_t* vconn, index_t ncells,
                                      const std::uint8_t* ftypes, const index_t* foffsets, const index_t* fconn,
                                      index_t nfaces) {
  (void)ftypes;
  std::unordered_map<FaceKey, index_t, FaceKeyHash> map;
  map.reserve(static_cast<std::size_t>(nfaces) * 2);
  std::vector<char> on_bnd(static_cast<std::size_t>(npts), 0);
  for (index_t f = 0; f < nfaces; ++f) {
    int nn = static_cast<int>(foffsets[f + 1] - foffsets[f]);
    map.emplace(make_key(fconn + foffsets[f], nn), f);
    for (int i = 0; i < nn; ++i) on_bnd[fconn[foffsets[f] + i]] = 1;
  }
  std::vector<std::int8_t> sign(static_cast<std::size_t>(nfaces), 0);
#pragma omp parallel for schedule(dynamic, 8192)
  for (index_t c = 0; c < ncells; ++c) {
    int type = vtypes[c];
    if (cell_dim(type) != 3) continue;
    const index_t* cn = vconn + voffsets[c];
    int nn = cell_num_nodes(type), nb = 0;
    for (int i = 0; i < nn; ++i) nb += on_bnd[cn[i]];
    if (nb < 3) continue;
    const CellFaces& cf = faces_of(type);
    for (int fi = 0; fi < cf.nfaces; ++fi) {
      index_t fn[4];
      bool all = true;
      for (int i = 0; i < cf.size[fi]; ++i) {
        fn[i] = cn[cf.nodes[fi][i]];
        all &= on_bnd[fn[i]] != 0;
      }
      if (!all) continue;
      auto it = map.find(make_key(fn, cf.size[fi]));
      if (it == map.end()) continue;
      index_t f = it->second;
      // boundary face normal (Newell) vs direction to the cell centroid
      Vec3 cc;
      for (int i = 0; i < nn; ++i) cc += load3(xyz, cn[i]);
      cc = cc * (1.0 / nn);
      int fnn = static_cast<int>(foffsets[f + 1] - foffsets[f]);
      Vec3 nrm, fc;
      for (int i = 0; i < fnn; ++i) {
        Vec3 a = load3(xyz, fconn[foffsets[f] + i]), b = load3(xyz, fconn[foffsets[f] + (i + 1) % fnn]);
        nrm += a.cross(b);
        fc += a;
      }
      fc = fc * (1.0 / fnn);
      sign[f] = nrm.dot(cc - fc) > 0 ? 1 : -1;
    }
  }
  return sign;
}

}  // namespace cfdpost
