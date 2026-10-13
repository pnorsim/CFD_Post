#include "cfdpost/contour.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "cfdpost/cells.hpp"

namespace cfdpost {

namespace {

struct Crossing {
  lindex_t a, b;  // local ids, ordered by global id (gid[a] < gid[b])
};

struct Piece {
  // vertex list of output primitives, as crossings
  std::vector<index_t> sizes;      // polygon sizes (or 2 for segments)
  std::vector<Crossing> verts;
  std::vector<index_t> cell;       // local cell index per primitive
  std::vector<char> is_line;
};

// Per-cell scratch for the loop builder.
struct Seg {
  Crossing s, e;
  bool used;
};

inline bool same(const Crossing& x, const Crossing& y) { return x.a == y.a && x.b == y.b; }

inline Crossing make_crossing(lindex_t u, lindex_t v, const index_t* gid) {
  return gid[u] < gid[v] ? Crossing{u, v} : Crossing{v, u};
}

// Oriented segments of one face (cycle of n local node ids). pos[i]: scalar >= iso.
// Segments go from the "down" crossing (pos -> neg along the face cycle) to the
// "up" crossing; with outward faces this orients loops towards increasing scalar.
template <class Out>
void face_segments(const lindex_t* nodes, int n, const double* f, double iso, const index_t* gid, Out&& out) {
  bool pos[8];
  int ncross = 0;
  for (int i = 0; i < n; ++i) pos[i] = f[nodes[i]] >= iso;
  for (int i = 0; i < n; ++i) ncross += pos[i] != pos[(i + 1) % n];
  if (ncross == 0) return;
  if (ncross == 2) {
    Crossing up{}, down{};
    for (int i = 0; i < n; ++i) {
      int j = (i + 1) % n;
      if (pos[i] == pos[j]) continue;
      Crossing c = make_crossing(nodes[i], nodes[j], gid);
      if (pos[i])
        down = c;
      else
        up = c;
    }
    out(down, up);
    return;
  }
  // ambiguous face (alternating signs): the face centre decides which corners
  // are cut off. Sum in global-id order so both neighbouring cells agree bit-wise.
  int ord[8];
  std::iota(ord, ord + n, 0);
  std::sort(ord, ord + n, [&](int x, int y) { return gid[nodes[x]] < gid[nodes[y]]; });
  double s = 0;
  for (int i = 0; i < n; ++i) s += f[nodes[ord[i]]] - iso;
  bool centre_pos = s >= 0;
  for (int i = 0; i < n; ++i) {
    int prev = (i + n - 1) % n, next = (i + 1) % n;
    if (pos[i] == centre_pos) continue;           // corner joined to the centre region
    if (pos[prev] == pos[i] || pos[next] == pos[i]) continue;  // not an isolated corner
    Crossing cin = make_crossing(nodes[prev], nodes[i], gid);
    Crossing cout = make_crossing(nodes[i], nodes[next], gid);
    // along the cycle prev -> i -> next: entering i is "up" if i is positive
    if (pos[i])
      out(cout, cin);  // cout is down (pos->neg), cin is up
    else
      out(cin, cout);  // cin is down
  }
}

void contour_cell_3d(int type, const lindex_t* cn, const double* f, double iso, const index_t* gid,
                     const double* xyz, Piece& pc, index_t cell) {
  const CellFaces& cf = faces_of(type);
  Seg segs[24];
  int ns = 0;
  for (int fi = 0; fi < cf.nfaces; ++fi) {
    lindex_t fn[4];
    for (int i = 0; i < cf.size[fi]; ++i) fn[i] = cn[cf.nodes[fi][i]];
    face_segments(fn, cf.size[fi], f, iso, gid, [&](const Crossing& s, const Crossing& e) {
      if (ns < 24) segs[ns++] = {s, e, false};
    });
  }
  if (ns < 3) return;
  // orientation: faces are outward only for positively oriented cells
  int nn = cell_num_nodes(type);
  Vec3 x[8];
  for (int i = 0; i < nn; ++i) x[i] = load3(xyz, cn[i]);
  bool flip = cell_volume_3d(type, x) < 0;
  for (int s0 = 0; s0 < ns; ++s0) {
    if (segs[s0].used) continue;
    Crossing loop[24];
    int nl = 0;
    int cur = s0;
    bool closed = false;
    while (true) {
      segs[cur].used = true;
      loop[nl++] = segs[cur].s;
      if (same(segs[cur].e, segs[s0].s)) {
        closed = true;
        break;
      }
      int nxt = -1;
      for (int j = 0; j < ns; ++j)
        if (!segs[j].used && same(segs[j].s, segs[cur].e)) {
          nxt = j;
          break;
        }
      if (nxt < 0 || nl >= 24) break;
      cur = nxt;
    }
    if (!closed || nl < 3) continue;
    pc.sizes.push_back(nl);
    if (flip)
      for (int i = nl - 1; i >= 0; --i) pc.verts.push_back(loop[i]);
    else
      for (int i = 0; i < nl; ++i) pc.verts.push_back(loop[i]);
    pc.cell.push_back(cell);
    pc.is_line.push_back(0);
  }
}

void contour_cell_2d(const lindex_t* cn, int nn, const double* f, double iso, const index_t* gid, Piece& pc,
                     index_t cell) {
  face_segments(cn, std::min(nn, 8), f, iso, gid, [&](const Crossing& s, const Crossing& e) {
    pc.sizes.push_back(2);
    pc.verts.push_back(s);
    pc.verts.push_back(e);
    pc.cell.push_back(cell);
    pc.is_line.push_back(1);
  });
}

}  // namespace

ContourResult contour_cells(const CellArray<lindex_t>& cells, const index_t* gid, const double* xyz,
                            const double* f, double iso, const double* fields, int nf, index_t cell_gid0) {
  index_t nc = cells.size();
  int nt = num_threads();
  std::vector<Piece> pieces(nt);
#pragma omp parallel num_threads(nt)
  {
    int t = 0;
#ifdef _OPENMP
    t = omp_get_thread_num();
    int nthr = omp_get_num_threads();
#else
    int nthr = 1;
#endif
    index_t c0 = nc * t / nthr, c1 = nc * (t + 1) / nthr;  // contiguous => deterministic order
    Piece& pc = pieces[t];
    for (index_t c = c0; c < c1; ++c) {
      index_t b = cells.offsets[c];
      int nn = static_cast<int>(cells.offsets[c + 1] - b);
      const lindex_t* cn = &cells.conn[b];
      bool any_pos = false, any_neg = false;
      for (int i = 0; i < nn; ++i) {
        bool p = f[cn[i]] >= iso;
        any_pos |= p;
        any_neg |= !p;
      }
      if (!(any_pos && any_neg)) continue;
      int type = cells.types[c];
      int d = cell_dim(type);
      if (d == 3)
        contour_cell_3d(type, cn, f, iso, gid, xyz, pc, c);
      else if (d == 2)
        contour_cell_2d(cn, nn, f, iso, gid, pc, c);
    }
  }

  // concatenate pieces
  std::vector<Crossing> verts;
  std::vector<index_t> sizes, cellids;
  std::vector<char> is_line;
  for (auto& p : pieces) {
    verts.insert(verts.end(), p.verts.begin(), p.verts.end());
    sizes.insert(sizes.end(), p.sizes.begin(), p.sizes.end());
    cellids.insert(cellids.end(), p.cell.begin(), p.cell.end());
    is_line.insert(is_line.end(), p.is_line.begin(), p.is_line.end());
  }

  // unique crossings -> output points
  index_t nv = static_cast<index_t>(verts.size());
  std::vector<index_t> order(static_cast<std::size_t>(nv));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](index_t x, index_t y) {
    index_t ax = gid[verts[x].a], ay = gid[verts[y].a];
    if (ax != ay) return ax < ay;
    return gid[verts[x].b] < gid[verts[y].b];
  });
  std::vector<index_t> vid(static_cast<std::size_t>(nv));
  std::vector<Crossing> uniq;
  for (index_t i = 0; i < nv; ++i) {
    const Crossing& c = verts[order[i]];
    if (uniq.empty() || !same(uniq.back(), c)) uniq.push_back(c);
    vid[order[i]] = static_cast<index_t>(uniq.size()) - 1;
  }

  ContourResult r;
  r.nf = nf;
  index_t np = static_cast<index_t>(uniq.size());
  r.edge_a.resize(np);
  r.edge_b.resize(np);
  r.t.resize(np);
  r.xyz.resize(np * 3);
  r.fields.resize(np * nf);
#pragma omp parallel for if (np > 10000)
  for (index_t i = 0; i < np; ++i) {
    lindex_t a = uniq[i].a, b = uniq[i].b;
    double fa = f[a], fb = f[b];
    double t = fb != fa ? (iso - fa) / (fb - fa) : 0.5;
    t = std::min(1.0, std::max(0.0, t));
    r.edge_a[i] = gid[a];
    r.edge_b[i] = gid[b];
    r.t[i] = t;
    for (int d = 0; d < 3; ++d) r.xyz[3 * i + d] = xyz[3 * a + d] + t * (xyz[3 * b + d] - xyz[3 * a + d]);
    for (int q = 0; q < nf; ++q)
      r.fields[i * nf + q] = fields[a * nf + q] + t * (fields[b * nf + q] - fields[a * nf + q]);
  }

  index_t k = 0;
  for (std::size_t p = 0; p < sizes.size(); ++p) {
    CellArray<index_t>& ca = is_line[p] ? r.lines : r.polys;
    ca.types.push_back(is_line[p] ? VTK_LINE : (sizes[p] == 3 ? VTK_TRIANGLE : VTK_POLYGON));
    for (index_t j = 0; j < sizes[p]; ++j) ca.conn.push_back(vid[k + j]);
    ca.offsets.push_back(static_cast<index_t>(ca.conn.size()));
    (is_line[p] ? r.line_cell : r.poly_cell).push_back(cell_gid0 + cellids[p]);
    k += sizes[p];
  }
  return r;
}

ContourResult gather_contour(const ContourResult& loc, const Comm& comm, int root) {
  if (comm.size() == 1) return loc;
  int nf = loc.nf;
  std::vector<std::int64_t> cnt;
  auto ga = comm.gatherv(loc.edge_a, root, cnt);
  auto gb = comm.gatherv(loc.edge_b, root, cnt);
  auto gt = comm.gatherv(loc.t, root, cnt);
  std::vector<std::int64_t> npts = cnt;
  auto gx = comm.gatherv(loc.xyz, root, cnt);
  auto gf = comm.gatherv(loc.fields, root, cnt);
  std::vector<std::int64_t> c_poff, c_pconn, c_loff, c_lconn;
  auto pt = comm.gatherv(loc.polys.types, root, cnt);
  auto po = comm.gatherv(loc.polys.offsets, root, c_poff);
  auto pc = comm.gatherv(loc.polys.conn, root, c_pconn);
  auto pcell = comm.gatherv(loc.poly_cell, root, cnt);
  auto lt = comm.gatherv(loc.lines.types, root, cnt);
  auto lo = comm.gatherv(loc.lines.offsets, root, c_loff);
  auto lc = comm.gatherv(loc.lines.conn, root, c_lconn);
  auto lcell = comm.gatherv(loc.line_cell, root, cnt);
  ContourResult r;
  r.nf = nf;
  if (comm.rank() != root) return r;

  // global unique points by edge key
  index_t n = static_cast<index_t>(ga.size());
  std::vector<index_t> order(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](index_t x, index_t y) {
    return ga[x] != ga[y] ? ga[x] < ga[y] : gb[x] < gb[y];
  });
  std::vector<index_t> remap(static_cast<std::size_t>(n));
  for (index_t i = 0; i < n; ++i) {
    index_t o = order[i];
    if (r.edge_a.empty() || r.edge_a.back() != ga[o] || r.edge_b.back() != gb[o]) {
      r.edge_a.push_back(ga[o]);
      r.edge_b.push_back(gb[o]);
      r.t.push_back(gt[o]);
      for (int d = 0; d < 3; ++d) r.xyz.push_back(gx[3 * o + d]);
      for (int q = 0; q < nf; ++q) r.fields.push_back(gf[o * nf + q]);
    }
    remap[o] = static_cast<index_t>(r.edge_a.size()) - 1;
  }

  auto merge = [&](const std::vector<std::uint8_t>& types, const std::vector<index_t>& offs,
                   const std::vector<std::int64_t>& coff, const std::vector<index_t>& conn,
                   const std::vector<std::int64_t>& cconn, CellArray<index_t>& out) {
    out.types = types;
    std::size_t oo = 0, co = 0;
    index_t pbase = 0;
    for (int rk = 0; rk < comm.size(); ++rk) {
      // offsets of rank rk: coff[rk] values starting with 0
      for (std::int64_t i = 0; i + 1 < coff[rk]; ++i) {
        for (index_t j = offs[oo + i]; j < offs[oo + i + 1]; ++j)
          out.conn.push_back(remap[pbase + conn[co + j]]);
        out.offsets.push_back(static_cast<index_t>(out.conn.size()));
      }
      oo += static_cast<std::size_t>(coff[rk]);
      co += static_cast<std::size_t>(cconn[rk]);
      pbase += npts[rk];
    }
  };
  merge(pt, po, c_poff, pc, c_pconn, r.polys);
  merge(lt, lo, c_loff, lc, c_lconn, r.lines);
  r.poly_cell = pcell;
  r.line_cell = lcell;
  return r;
}

}  // namespace cfdpost
