// VTK XML reader (.vtu / .vtp) built on a memory map.
//
// SU2 writes raw appended, uncompressed data; arrays of such files are read in
// place with random access (no copy of the whole file), which is what lets MPI
// ranks load only their own cells and points. Base64 ("binary"), zlib
// compressed and ascii arrays (VTK/ParaView/pyvista defaults) are decoded once
// and cached.
#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "cfdpost/common.hpp"
#include "cfdpost/mmap_file.hpp"

namespace cfdpost {

enum class DType { Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Float32, Float64 };
int dtype_size(DType t);
std::string dtype_name(DType t);

struct VtkArrayInfo {
  std::string section;  // Points, Cells, Polys, Lines, Verts, PointData, CellData, FieldData
  std::string name;
  DType dtype = DType::Float32;
  int ncomp = 1;
  std::string format;          // appended | binary | ascii
  std::size_t offset = 0;      // appended: offset from start of appended block
  std::size_t text_begin = 0;  // inline: [text_begin, text_end) in file
  std::size_t text_end = 0;
};

class VtkXmlFile {
 public:
  explicit VtkXmlFile(const std::string& path);

  const std::string& path() const { return file_.path(); }
  const std::string& dataset_type() const { return type_; }  // UnstructuredGrid | PolyData
  index_t num_points() const { return npoints_; }
  index_t num_cells() const { return ncells_; }  // UnstructuredGrid cells, or PolyData polys+lines+verts
  const std::map<std::string, index_t>& piece_counts() const { return counts_; }
  const std::vector<VtkArrayInfo>& arrays() const { return arrays_; }

  const VtkArrayInfo* find(const std::string& section, const std::string& name) const;
  const VtkArrayInfo& get(const std::string& section, const std::string& name) const;

  // Number of scalar values stored in an array.
  index_t num_values(const VtkArrayInfo& a) const;

  // True if the array can be read in place with random access.
  bool is_direct(const VtkArrayInfo& a) const;

  // Read scalar values [first, first+n) converted to T.
  template <class T>
  void read_values(const VtkArrayInfo& a, index_t first, index_t n, T* out) const;

  // Gather tuples ids[0..n) (each a.ncomp values) converted to T.
  template <class T>
  void read_tuples(const VtkArrayInfo& a, const index_t* ids, index_t n, T* out) const;

 private:
  struct Raw {
    const char* ptr;
    index_t count;  // number of scalar values
  };
  Raw raw(const VtkArrayInfo& a) const;
  void parse_header();
  const std::vector<char>& decoded(const VtkArrayInfo& a) const;

  MappedFile file_;
  std::string type_;
  std::string header_type_ = "UInt32";
  std::string compressor_;
  index_t npoints_ = 0, ncells_ = 0;
  std::map<std::string, index_t> counts_;
  std::vector<VtkArrayInfo> arrays_;
  std::size_t appended_start_ = 0;
  bool appended_base64_ = false;

  mutable std::mutex cache_mutex_;
  mutable std::map<std::string, std::unique_ptr<std::vector<char>>> cache_;
};

// ---------------------------------------------------------------- templates
namespace detail {
template <class T, class S>
inline void convert(const char* src, index_t n, T* out) {
  const S* s = reinterpret_cast<const S*>(src);
#pragma omp parallel for schedule(static) if (n > 1000000)
  for (index_t i = 0; i < n; ++i) out[i] = static_cast<T>(s[i]);
}
template <class T, class S>
inline void gather(const char* src, int nc, const index_t* ids, index_t n, T* out) {
  const S* s = reinterpret_cast<const S*>(src);
#pragma omp parallel for schedule(static) if (n > 200000)
  for (index_t i = 0; i < n; ++i)
    for (int k = 0; k < nc; ++k) out[i * nc + k] = static_cast<T>(s[ids[i] * nc + k]);
}
#define CFDPOST_DTYPE_SWITCH(dt, S, ...)              \
  switch (dt) {                                       \
    case DType::Int8: { using S = std::int8_t; __VA_ARGS__; break; }    \
    case DType::UInt8: { using S = std::uint8_t; __VA_ARGS__; break; }  \
    case DType::Int16: { using S = std::int16_t; __VA_ARGS__; break; }  \
    case DType::UInt16: { using S = std::uint16_t; __VA_ARGS__; break; } \
    case DType::Int32: { using S = std::int32_t; __VA_ARGS__; break; }  \
    case DType::UInt32: { using S = std::uint32_t; __VA_ARGS__; break; } \
    case DType::Int64: { using S = std::int64_t; __VA_ARGS__; break; }  \
    case DType::UInt64: { using S = std::uint64_t; __VA_ARGS__; break; } \
    case DType::Float32: { using S = float; __VA_ARGS__; break; }       \
    case DType::Float64: { using S = double; __VA_ARGS__; break; }      \
  }
}  // namespace detail

template <class T>
void VtkXmlFile::read_values(const VtkArrayInfo& a, index_t first, index_t n, T* out) const {
  Raw r = raw(a);
  if (first < 0 || first + n > r.count) fail("read past end of array " + a.name + " in " + path());
  const char* p = r.ptr + first * dtype_size(a.dtype);
  CFDPOST_DTYPE_SWITCH(a.dtype, S, detail::convert<T, S>(p, n, out))
}

template <class T>
void VtkXmlFile::read_tuples(const VtkArrayInfo& a, const index_t* ids, index_t n, T* out) const {
  Raw r = raw(a);
  index_t ntup = r.count / a.ncomp;
  for (index_t i = 0; i < n; ++i)
    if (ids[i] < 0 || ids[i] >= ntup) fail("tuple id out of range in " + a.name);
  CFDPOST_DTYPE_SWITCH(a.dtype, S, detail::gather<T, S>(r.ptr, a.ncomp, ids, n, out))
}

}  // namespace cfdpost
