#include "cfdpost/vtu.hpp"

#include <cctype>
#include <charconv>
#include <cstring>

#ifdef CFDPOST_HAVE_ZLIB
#include <zlib.h>
#endif

namespace cfdpost {

int dtype_size(DType t) {
  switch (t) {
    case DType::Int8:
    case DType::UInt8: return 1;
    case DType::Int16:
    case DType::UInt16: return 2;
    case DType::Int32:
    case DType::UInt32:
    case DType::Float32: return 4;
    default: return 8;
  }
}

std::string dtype_name(DType t) {
  static const char* n[] = {"Int8", "UInt8", "Int16", "UInt16", "Int32", "UInt32", "Int64", "UInt64", "Float32", "Float64"};
  return n[static_cast<int>(t)];
}

namespace {

DType parse_dtype(const std::string& s) {
  if (s == "Int8" || s == "Char") return DType::Int8;
  if (s == "UInt8") return DType::UInt8;
  if (s == "Int16") return DType::Int16;
  if (s == "UInt16") return DType::UInt16;
  if (s == "Int32") return DType::Int32;
  if (s == "UInt32") return DType::UInt32;
  if (s == "Int64") return DType::Int64;
  if (s == "UInt64") return DType::UInt64;
  if (s == "Float32") return DType::Float32;
  if (s == "Float64") return DType::Float64;
  fail("unsupported VTK data type " + s);
}

// attributes of a tag text like: DataArray type="Float32" Name="" NumberOfComponents= "3"
std::map<std::string, std::string> parse_attrs(const char* b, const char* e) {
  std::map<std::string, std::string> out;
  const char* p = b;
  while (p < e) {
    while (p < e && (std::isspace(static_cast<unsigned char>(*p)) || *p == '/')) ++p;
    const char* k0 = p;
    while (p < e && *p != '=' && !std::isspace(static_cast<unsigned char>(*p))) ++p;
    std::string key(k0, p);
    while (p < e && std::isspace(static_cast<unsigned char>(*p))) ++p;
    if (p >= e || *p != '=') {
      ++p;
      continue;
    }
    ++p;
    while (p < e && std::isspace(static_cast<unsigned char>(*p))) ++p;
    if (p >= e) break;
    char q = *p;
    if (q != '"' && q != '\'') continue;
    const char* v0 = ++p;
    while (p < e && *p != q) ++p;
    out[key] = std::string(v0, p);
    ++p;
  }
  return out;
}

const signed char* b64_table() {
  static signed char t[256];
  static bool init = false;
  if (!init) {
    std::memset(t, -1, sizeof t);
    const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) t[static_cast<unsigned char>(a[i])] = static_cast<signed char>(i);
    init = true;
  }
  return t;
}

// Decode base64 starting at p, producing exactly nbytes of output (or until text ends).
// Returns pointer just after the consumed characters (whole 4-char groups).
const char* b64_decode(const char* p, const char* end, std::size_t nbytes, std::vector<char>& out) {
  const signed char* t = b64_table();
  out.resize(nbytes);
  std::size_t o = 0;
  int quad[4];
  int nq = 0;
  while (p < end && o < nbytes) {
    unsigned char c = static_cast<unsigned char>(*p++);
    if (c == '=') {
      quad[nq++] = 0;
    } else if (t[c] >= 0) {
      quad[nq++] = t[c];
    } else {
      continue;  // whitespace
    }
    if (nq == 4) {
      unsigned v = (quad[0] << 18) | (quad[1] << 12) | (quad[2] << 6) | quad[3];
      char bytes[3] = {static_cast<char>(v >> 16), static_cast<char>(v >> 8), static_cast<char>(v)};
      for (int i = 0; i < 3 && o < nbytes; ++i) out[o++] = bytes[i];
      nq = 0;
    }
  }
  if (o < nbytes) fail("truncated base64 data");
  return p;
}

std::uint64_t read_uint(const char* p, int hsize) {
  if (hsize == 8) {
    std::uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
  }
  std::uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

}  // namespace

VtkXmlFile::VtkXmlFile(const std::string& path) : file_(path) { parse_header(); }

void VtkXmlFile::parse_header() {
  const char* d = file_.data();
  const char* end = d + file_.size();
  const char* p = d;
  std::string section;
  bool in_piece = false;
  while (p < end) {
    const char* lt = static_cast<const char*>(std::memchr(p, '<', static_cast<std::size_t>(end - p)));
    if (!lt) break;
    const char* gt = static_cast<const char*>(std::memchr(lt, '>', static_cast<std::size_t>(end - lt)));
    if (!gt) fail("malformed XML in " + path());
    const char* nb = lt + 1;
    const char* ne = nb < gt && *nb == '/' ? nb + 1 : nb;  // keep the '/' of closing tags
    while (ne < gt && !std::isspace(static_cast<unsigned char>(*ne)) && *ne != '/' && *ne != '>') ++ne;
    std::string tag(nb, ne);
    bool selfclose = gt[-1] == '/';
    p = gt + 1;
    if (tag == "VTKFile") {
      auto at = parse_attrs(ne, gt);
      type_ = at["type"];
      if (at.count("header_type")) header_type_ = at["header_type"];
      if (at.count("compressor")) compressor_ = at["compressor"];
      if (at.count("byte_order") && at["byte_order"] != "LittleEndian")
        fail("big-endian VTK files are not supported: " + path());
    } else if (tag == "Piece") {
      if (in_piece) break;  // only the first piece is read
      in_piece = true;
      for (auto& [k, v] : parse_attrs(ne, gt))
        if (k.rfind("NumberOf", 0) == 0) counts_[k] = std::stoll(v);
      npoints_ = counts_.count("NumberOfPoints") ? counts_["NumberOfPoints"] : 0;
      if (counts_.count("NumberOfCells")) {
        ncells_ = counts_["NumberOfCells"];
      } else {
        for (const char* k : {"NumberOfPolys", "NumberOfLines", "NumberOfVerts", "NumberOfStrips"})
          if (counts_.count(k)) ncells_ += counts_[k];
      }
    } else if (tag == "/Piece") {
      in_piece = false;
    } else if (tag == "Points" || tag == "Cells" || tag == "PointData" || tag == "CellData" ||
               tag == "FieldData" || tag == "Polys" || tag == "Lines" || tag == "Verts" || tag == "Strips") {
      if (!selfclose) section = tag;
    } else if (tag == "/Points" || tag == "/Cells" || tag == "/PointData" || tag == "/CellData" ||
               tag == "/FieldData" || tag == "/Polys" || tag == "/Lines" || tag == "/Verts" || tag == "/Strips") {
      section.clear();
    } else if (tag == "DataArray") {
      auto at = parse_attrs(ne, gt);
      VtkArrayInfo a;
      a.section = section;
      a.name = at["Name"];
      if (a.section == "Points" && a.name.empty()) a.name = "Points";
      a.dtype = parse_dtype(at["type"]);
      a.ncomp = at.count("NumberOfComponents") ? std::stoi(at["NumberOfComponents"]) : 1;
      a.format = at.count("format") ? at["format"] : "ascii";
      if (a.format == "appended") {
        a.offset = std::stoull(at["offset"]);
      } else if (!selfclose) {
        // inline data ends at the next tag (VTK may append <InformationKey> children)
        static const char close[] = "</DataArray>";
        const char* c = static_cast<const char*>(memmem(p, static_cast<std::size_t>(end - p), close, sizeof close - 1));
        if (!c) fail("unterminated DataArray in " + path());
        const char* t = static_cast<const char*>(std::memchr(p, '<', static_cast<std::size_t>(c - p) + 1));
        a.text_begin = static_cast<std::size_t>(p - d);
        a.text_end = static_cast<std::size_t>(t - d);
        p = c + sizeof close - 1;
      }
      if (section != "FieldData") arrays_.push_back(a);
    } else if (tag == "AppendedData") {
      auto at = parse_attrs(ne, gt);
      appended_base64_ = at["encoding"] != "raw";
      const char* u = static_cast<const char*>(std::memchr(p, '_', static_cast<std::size_t>(end - p)));
      if (!u) fail("no '_' after AppendedData in " + path());
      appended_start_ = static_cast<std::size_t>(u + 1 - d);
      break;
    }
  }
  if (type_.empty()) fail(path() + " is not a VTK XML file");
}

const VtkArrayInfo* VtkXmlFile::find(const std::string& section, const std::string& name) const {
  for (auto& a : arrays_)
    if (a.section == section && a.name == name) return &a;
  return nullptr;
}

const VtkArrayInfo& VtkXmlFile::get(const std::string& section, const std::string& name) const {
  const VtkArrayInfo* a = find(section, name);
  if (!a) fail("array " + section + "/" + name + " not found in " + path());
  return *a;
}

bool VtkXmlFile::is_direct(const VtkArrayInfo& a) const {
  return a.format == "appended" && !appended_base64_ && compressor_.empty();
}

index_t VtkXmlFile::num_values(const VtkArrayInfo& a) const { return raw(a).count; }

VtkXmlFile::Raw VtkXmlFile::raw(const VtkArrayInfo& a) const {
  int hs = header_type_ == "UInt64" ? 8 : 4;
  if (is_direct(a)) {
    const char* p = file_.data() + appended_start_ + a.offset;
    if (appended_start_ + a.offset + hs > file_.size()) fail("array offset beyond end of " + path());
    std::uint64_t nbytes = read_uint(p, hs);
    if (appended_start_ + a.offset + hs + nbytes > file_.size()) fail("truncated array " + a.name + " in " + path());
    return {p + hs, static_cast<index_t>(nbytes / static_cast<std::uint64_t>(dtype_size(a.dtype)))};
  }
  const std::vector<char>& buf = decoded(a);
  return {buf.data(), static_cast<index_t>(buf.size() / static_cast<std::size_t>(dtype_size(a.dtype)))};
}

const std::vector<char>& VtkXmlFile::decoded(const VtkArrayInfo& a) const {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  std::string key = a.section + "/" + a.name;
  auto it = cache_.find(key);
  if (it != cache_.end()) return *it->second;

  auto out = std::make_unique<std::vector<char>>();
  const char* d = file_.data();
  const char* fend = d + file_.size();
  int hs = header_type_ == "UInt64" ? 8 : 4;

  if (a.format == "ascii") {
    const char* p = d + a.text_begin;
    const char* e = d + a.text_end;
    int sz = dtype_size(a.dtype);
    while (p < e) {
      while (p < e && std::isspace(static_cast<unsigned char>(*p))) ++p;
      if (p >= e) break;
      const char* q = p;
      while (q < e && !std::isspace(static_cast<unsigned char>(*q))) ++q;
      double v = std::strtod(std::string(p, q).c_str(), nullptr);
      std::size_t o = out->size();
      out->resize(o + static_cast<std::size_t>(sz));
      CFDPOST_DTYPE_SWITCH(a.dtype, S, {
        S s = static_cast<S>(v);
        std::memcpy(out->data() + o, &s, sizeof(S));
      })
      p = q;
    }
  } else {
    // binary (inline base64), appended base64 or appended raw (compressed)
    bool b64 = a.format == "binary" || appended_base64_;
    const char* p = a.format == "binary" ? d + a.text_begin : d + appended_start_ + a.offset;
    const char* e = a.format == "binary" ? d + a.text_end : fend;
    if (b64)
      while (p < e && std::isspace(static_cast<unsigned char>(*p))) ++p;
    auto take = [&](std::size_t n, std::vector<char>& dst) {
      if (b64) {
        p = b64_decode(p, e, n, dst);
      } else {
        if (p + n > e) fail("truncated data in " + path());
        dst.assign(p, p + n);
        p += n;
      }
    };
    std::vector<char> hdr;
    if (compressor_.empty()) {
      if (b64) {
        // uncompressed base64: header and data are encoded together in one block
        const char* start = p;
        b64_decode(p, e, static_cast<std::size_t>(hs), hdr);
        std::uint64_t nbytes = read_uint(hdr.data(), hs);
        std::vector<char> all;
        b64_decode(start, e, static_cast<std::size_t>(hs + nbytes), all);
        out->assign(all.begin() + hs, all.end());
      } else {
        take(static_cast<std::size_t>(hs), hdr);
        take(static_cast<std::size_t>(read_uint(hdr.data(), hs)), *out);
      }
    } else {
#ifdef CFDPOST_HAVE_ZLIB
      if (compressor_ != "vtkZLibDataCompressor") fail("unsupported compressor " + compressor_ + " in " + path());
      take(static_cast<std::size_t>(3 * hs), hdr);
      std::uint64_t nblocks = read_uint(hdr.data(), hs);
      std::uint64_t bsize = read_uint(hdr.data() + hs, hs);
      std::uint64_t last = read_uint(hdr.data() + 2 * hs, hs);
      std::vector<char> csz;
      if (b64) {
        // whole header (3 + nblocks words) is one base64 block: re-decode it in one go
        const char* start = a.format == "binary" ? d + a.text_begin : d + appended_start_ + a.offset;
        p = start;
        while (p < e && std::isspace(static_cast<unsigned char>(*p))) ++p;
        take(static_cast<std::size_t>((3 + nblocks) * hs), hdr);
        csz.assign(hdr.begin() + 3 * hs, hdr.end());
      } else {
        take(static_cast<std::size_t>(nblocks * hs), csz);
      }
      std::uint64_t total = nblocks == 0 ? 0 : (nblocks - 1) * bsize + (last ? last : bsize);
      out->resize(static_cast<std::size_t>(total));
      std::uint64_t ctot = 0;
      for (std::uint64_t b = 0; b < nblocks; ++b) ctot += read_uint(csz.data() + b * hs, hs);
      std::vector<char> comp;
      take(static_cast<std::size_t>(ctot), comp);
      std::uint64_t co = 0, uo = 0;
      for (std::uint64_t b = 0; b < nblocks; ++b) {
        std::uint64_t cs = read_uint(csz.data() + b * hs, hs);
        uLongf us = static_cast<uLongf>(b + 1 == nblocks && last ? last : bsize);
        if (uncompress(reinterpret_cast<Bytef*>(out->data() + uo), &us,
                       reinterpret_cast<const Bytef*>(comp.data() + co), static_cast<uLong>(cs)) != Z_OK)
          fail("zlib error in array " + a.name + " of " + path());
        co += cs;
        uo += us;
      }
#else
      fail("compressed VTK file but cfdpost was built without zlib: " + path());
#endif
    }
  }
  auto& ref = *out;
  cache_[key] = std::move(out);
  return ref;
}

}  // namespace cfdpost
