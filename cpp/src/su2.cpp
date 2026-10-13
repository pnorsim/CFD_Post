#include "cfdpost/su2.hpp"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <numeric>

#include "cfdpost/mmap_file.hpp"

namespace cfdpost {

namespace {

inline bool is_blank(char c) { return c == ' ' || c == '\t' || c == '\r'; }

inline const char* skip_blank(const char* p, const char* e) {
  while (p < e && is_blank(*p)) ++p;
  return p;
}

inline const char* parse_int(const char* p, const char* e, index_t& v) {
  p = skip_blank(p, e);
  auto r = std::from_chars(p, e, v);
  if (r.ec != std::errc()) fail("SU2: expected an integer");
  return r.ptr;
}

inline const char* parse_double(const char* p, const char* e, double& v) {
  p = skip_blank(p, e);
  if (p < e && *p == '+') ++p;
  auto r = std::from_chars(p, e, v);
  if (r.ec != std::errc()) fail("SU2: expected a number");
  return r.ptr;
}

inline const char* line_end(const char* p, const char* e) {
  const char* n = static_cast<const char*>(std::memchr(p, '\n', static_cast<std::size_t>(e - p)));
  return n ? n : e;
}

// Offset just after the n-th newline starting at p (parallel newline count).
const char* skip_lines(const char* p, const char* e, index_t n) {
  if (n <= 0) return p;
  const std::size_t window = std::size_t(1) << 26;  // 64 MiB per thread per round
  int nt = num_threads();
  while (p < e) {
    std::size_t len = std::min(static_cast<std::size_t>(e - p), window * static_cast<std::size_t>(nt));
    std::size_t chunk = (len + nt - 1) / nt;
    std::vector<index_t> cnt(nt, 0);
#pragma omp parallel for schedule(static, 1)
    for (int t = 0; t < nt; ++t) {
      std::size_t b = std::min(len, chunk * t), en = std::min(len, chunk * (t + 1));
      cnt[t] = std::count(p + b, p + en, '\n');
    }
    for (int t = 0; t < nt; ++t) {
      std::size_t b = std::min(len, chunk * t), en = std::min(len, chunk * (t + 1));
      if (cnt[t] < n) {
        n -= cnt[t];
        continue;
      }
      const char* q = p + b;
      while (true) {
        q = static_cast<const char*>(std::memchr(q, '\n', static_cast<std::size_t>(p + en - q)));
        if (--n == 0) return q + 1;
        ++q;
      }
    }
    p += len;
  }
  if (n == 1) return e;  // last line without newline
  fail("SU2: unexpected end of file");
}

// Split [b, e) into nt chunks starting at line boundaries.
std::vector<const char*> line_chunks(const char* b, const char* e, int nt) {
  std::vector<const char*> c(nt + 1, e);
  c[0] = b;
  std::size_t len = static_cast<std::size_t>(e - b);
  for (int t = 1; t < nt; ++t) {
    const char* q = b + len * t / nt;
    if (q < c[t - 1]) q = c[t - 1];
    q = line_end(q, e);
    c[t] = q < e ? q + 1 : e;
  }
  return c;
}

struct Cursor {
  const char* p;
  const char* e;
  // next non-empty, non-comment line, returns [b, le)
  bool next(const char*& b, const char*& le) {
    while (p < e) {
      const char* l = skip_blank(p, e);
      le = line_end(l, e);
      p = le < e ? le + 1 : e;
      if (l == le || *l == '%' || *l == '\n') continue;
      b = l;
      return true;
    }
    return false;
  }
};

bool keyword(const char* b, const char* le, const char* kw, const char*& val) {
  std::size_t n = std::strlen(kw);
  if (static_cast<std::size_t>(le - b) < n || std::memcmp(b, kw, n) != 0) return false;
  val = b + n;
  return true;
}

std::string trim(const char* b, const char* e) {
  while (b < e && (is_blank(*b) || *b == '\n')) ++b;
  while (e > b && (is_blank(e[-1]) || e[-1] == '\n')) --e;
  return std::string(b, e);
}

// Parse `n` element lines in [b, ...) into cells; returns end pointer.
const char* parse_elements(const char* b, const char* fe, index_t n, CellArray<index_t>& out) {
  const char* e = skip_lines(b, fe, n);
  int nt = num_threads();
  auto ch = line_chunks(b, e, nt);
  std::vector<index_t> nlines(nt, 0), nnodes(nt, 0);
#pragma omp parallel for schedule(static, 1)
  for (int t = 0; t < nt; ++t) {
    const char* p = ch[t];
    while (p < ch[t + 1]) {
      const char* le = line_end(p, ch[t + 1]);
      const char* q = skip_blank(p, le);
      if (q < le) {
        index_t type;
        parse_int(q, le, type);
        nlines[t]++;
        nnodes[t] += cell_num_nodes(static_cast<int>(type));
      }
      p = le + 1;
    }
  }
  index_t tot = std::accumulate(nlines.begin(), nlines.end(), index_t(0));
  if (tot != n) fail("SU2: element count mismatch (blank lines inside the element block?)");
  std::vector<index_t> l0(nt + 1, 0), n0(nt + 1, 0);
  for (int t = 0; t < nt; ++t) {
    l0[t + 1] = l0[t] + nlines[t];
    n0[t + 1] = n0[t] + nnodes[t];
  }
  index_t base_cells = out.size(), base_conn = static_cast<index_t>(out.conn.size());
  out.types.resize(static_cast<std::size_t>(base_cells + n));
  out.offsets.resize(static_cast<std::size_t>(base_cells + n + 1));
  out.conn.resize(static_cast<std::size_t>(base_conn + n0[nt]));
  std::string err;
#pragma omp parallel for schedule(static, 1)
  for (int t = 0; t < nt; ++t) {
    try {
      index_t c = base_cells + l0[t], k = base_conn + n0[t];
      const char* p = ch[t];
      while (p < ch[t + 1]) {
        const char* le = line_end(p, ch[t + 1]);
        const char* q = skip_blank(p, le);
        if (q < le) {
          index_t type;
          q = parse_int(q, le, type);
          int nn = cell_num_nodes(static_cast<int>(type));
          if (nn == 0) fail("SU2: unknown element type " + std::to_string(type));
          out.types[c] = static_cast<std::uint8_t>(type);
          for (int i = 0; i < nn; ++i) q = parse_int(q, le, out.conn[k++]);
          out.offsets[c + 1] = k;
          ++c;
        }
        p = le + 1;
      }
    } catch (const std::exception& ex) {
#pragma omp critical
      err = ex.what();
    }
  }
  if (!err.empty()) fail(err);
  return e;
}

const char* parse_points(const char* b, const char* fe, index_t n, int ndim, std::vector<double>& xyz) {
  const char* e = skip_lines(b, fe, n);
  int nt = num_threads();
  auto ch = line_chunks(b, e, nt);
  std::vector<index_t> nlines(nt, 0);
#pragma omp parallel for schedule(static, 1)
  for (int t = 0; t < nt; ++t) {
    const char* p = ch[t];
    while (p < ch[t + 1]) {
      const char* le = line_end(p, ch[t + 1]);
      if (skip_blank(p, le) < le) nlines[t]++;
      p = le + 1;
    }
  }
  std::vector<index_t> l0(nt + 1, 0);
  for (int t = 0; t < nt; ++t) l0[t + 1] = l0[t] + nlines[t];
  if (l0[nt] != n) fail("SU2: point count mismatch");
  xyz.assign(static_cast<std::size_t>(3 * n), 0.0);
  std::vector<index_t> idx(static_cast<std::size_t>(n), -1);
  std::string err;
#pragma omp parallel for schedule(static, 1)
  for (int t = 0; t < nt; ++t) {
    try {
      index_t i = l0[t];
      const char* p = ch[t];
      while (p < ch[t + 1]) {
        const char* le = line_end(p, ch[t + 1]);
        const char* q = skip_blank(p, le);
        if (q < le) {
          for (int d = 0; d < ndim; ++d) q = parse_double(q, le, xyz[3 * i + d]);
          q = skip_blank(q, le);
          if (q < le && *q != '\n') {
            double v;
            parse_double(q, le, v);  // index column (may be written as a float)
            idx[i] = static_cast<index_t>(v);
          }
          ++i;
        }
        p = le + 1;
      }
    } catch (const std::exception& ex) {
#pragma omp critical
      err = ex.what();
    }
  }
  if (!err.empty()) fail(err);
  // reorder if the explicit index column is not the identity
  bool identity = true;
  for (index_t i = 0; i < n && identity; ++i) identity = idx[i] == -1 || idx[i] == i;
  if (!identity) {
    std::vector<double> o(xyz.size());
    for (index_t i = 0; i < n; ++i) {
      if (idx[i] < 0 || idx[i] >= n) fail("SU2: bad point index column");
      for (int d = 0; d < 3; ++d) o[3 * idx[i] + d] = xyz[3 * i + d];
    }
    xyz.swap(o);
  }
  return e;
}

}  // namespace

Su2Mesh read_su2(const std::string& path, const Su2ReadOptions& opt) {
  MappedFile f(path);
  Su2Mesh m;
  const char* fe = f.data() + f.size();
  Cursor cur{f.data(), fe};
  const char *b, *le, *v;
  int nzone = 1, izone = 1;
  bool in_zone = true, seen_nmark = false;
  bool seen_nelem = false, seen_npoin = false;
  while (cur.next(b, le)) {
    if (keyword(b, le, "NZONE=", v)) {
      index_t z;
      parse_int(v, le, z);
      nzone = static_cast<int>(z);
      in_zone = nzone == 1;
    } else if (keyword(b, le, "IZONE=", v)) {
      if (seen_nmark && in_zone) break;  // requested zone finished
      index_t z;
      parse_int(v, le, z);
      izone = static_cast<int>(z);
      in_zone = izone == opt.zone;
    } else if (keyword(b, le, "NDIME=", v)) {
      index_t d;
      parse_int(v, le, d);
      if (in_zone) m.ndim = static_cast<int>(d);
    } else if (keyword(b, le, "NELEM=", v)) {
      index_t n;
      parse_int(v, le, n);
      if (in_zone) {
        m.nelem = n;
        seen_nelem = true;
      }
      if (in_zone && opt.elements)
        cur.p = parse_elements(cur.p, fe, n, m.elems);
      else
        cur.p = skip_lines(cur.p, fe, n);
    } else if (keyword(b, le, "NPOIN=", v)) {
      index_t n;
      parse_int(v, le, n);  // a second number (domain points) may follow and is ignored
      if (in_zone) {
        m.npoin = n;
        seen_npoin = true;
      }
      if (in_zone && opt.points) {
        if (m.ndim == 0) fail("SU2: NPOIN before NDIME");
        cur.p = parse_points(cur.p, fe, n, m.ndim, m.xyz);
      } else {
        cur.p = skip_lines(cur.p, fe, n);
      }
    } else if (keyword(b, le, "NMARK=", v)) {
      index_t nmark;
      parse_int(v, le, nmark);
      if (in_zone) seen_nmark = true;
      for (index_t k = 0; k < nmark; ++k) {
        std::string tag;
        index_t ne = -1;
        while ((tag.empty() || ne < 0) && cur.next(b, le)) {
          if (keyword(b, le, "MARKER_TAG=", v))
            tag = trim(v, le);
          else if (keyword(b, le, "MARKER_ELEMS=", v))
            parse_int(v, le, ne);
        }
        if (tag.empty() || ne < 0) fail("SU2: incomplete marker block in " + path);
        bool want = in_zone && opt.markers &&
                    (opt.only_markers.empty() ||
                     std::find(opt.only_markers.begin(), opt.only_markers.end(), tag) != opt.only_markers.end());
        if (want) {
          Su2Marker mk;
          mk.tag = tag;
          cur.p = parse_elements(cur.p, fe, ne, mk.elems);
          m.markers.push_back(std::move(mk));
        } else {
          if (in_zone && opt.markers == false) {
            // keep tag with empty elements so callers still see the marker list
            Su2Marker mk;
            mk.tag = tag;
            m.markers.push_back(std::move(mk));
          }
          cur.p = skip_lines(cur.p, fe, ne);
        }
      }
      if (in_zone && nzone == 1) break;
    }
  }
  if (!seen_nelem || !seen_npoin) fail("SU2: missing NELEM or NPOIN in " + path + " (zone " + std::to_string(opt.zone) + ")");
  for (auto& t : opt.only_markers) {
    bool found = false;
    for (auto& mk : m.markers) found |= mk.tag == t;
    if (!found) fail("SU2: marker " + t + " not found in " + path);
  }
  return m;
}

std::vector<std::pair<std::string, index_t>> su2_marker_list(const std::string& path, int zone) {
  MappedFile f(path);
  const char* fe = f.data() + f.size();
  Cursor cur{f.data(), fe};
  const char *b, *le, *v;
  std::vector<std::pair<std::string, index_t>> out;
  bool in_zone = true;
  int izone = 1;
  while (cur.next(b, le)) {
    index_t n;
    if (keyword(b, le, "NZONE=", v)) {
      parse_int(v, le, n);
      in_zone = n == 1;
    } else if (keyword(b, le, "IZONE=", v)) {
      if (!out.empty() && in_zone) break;
      parse_int(v, le, n);
      izone = static_cast<int>(n);
      in_zone = izone == zone;
    } else if (keyword(b, le, "NELEM=", v) || keyword(b, le, "NPOIN=", v)) {
      parse_int(v, le, n);
      cur.p = skip_lines(cur.p, fe, n);
    } else if (keyword(b, le, "MARKER_TAG=", v)) {
      if (in_zone) out.emplace_back(trim(v, le), 0);
    } else if (keyword(b, le, "MARKER_ELEMS=", v)) {
      parse_int(v, le, n);
      if (in_zone && !out.empty()) out.back().second = n;
      cur.p = skip_lines(cur.p, fe, n);
    }
  }
  return out;
}

}  // namespace cfdpost
