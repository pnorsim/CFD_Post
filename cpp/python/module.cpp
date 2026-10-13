// Python bindings of the cfdpost C++ core (module cfdpost._core).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cfdpost/comm.hpp"
#include "cfdpost/contour.hpp"
#include "cfdpost/partition.hpp"
#include "cfdpost/su2.hpp"
#include "cfdpost/surface.hpp"
#include "cfdpost/vtu.hpp"

namespace py = pybind11;
using namespace cfdpost;

namespace {

// Hand a std::vector to numpy without copying.
template <class T>
py::array_t<T> to_numpy(std::vector<T>&& v, std::vector<py::ssize_t> shape = {}) {
  auto* heap = new std::vector<T>(std::move(v));
  py::capsule owner(heap, [](void* p) { delete static_cast<std::vector<T>*>(p); });
  if (shape.empty()) shape = {static_cast<py::ssize_t>(heap->size())};
  return py::array_t<T>(shape, heap->data(), owner);
}

template <class T>
py::array_t<T> to_numpy_2d(std::vector<T>&& v, py::ssize_t ncomp) {
  py::ssize_t n = static_cast<py::ssize_t>(v.size()) / std::max<py::ssize_t>(ncomp, 1);
  if (ncomp == 1) return to_numpy(std::move(v), {n});
  return to_numpy(std::move(v), {n, ncomp});
}

using darr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using iarr = py::array_t<index_t, py::array::c_style | py::array::forcecast>;
using u8arr = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

py::dict cells_dict(CellArray<index_t>&& c) {
  py::dict d;
  d["types"] = to_numpy(std::move(c.types));
  d["offsets"] = to_numpy(std::move(c.offsets));
  d["conn"] = to_numpy(std::move(c.conn));
  return d;
}

py::dict contour_dict(ContourResult&& r) {
  py::dict d;
  index_t n = r.num_points();
  d["edge_a"] = to_numpy(std::move(r.edge_a));
  d["edge_b"] = to_numpy(std::move(r.edge_b));
  d["t"] = to_numpy(std::move(r.t));
  d["points"] = to_numpy(std::move(r.xyz), {static_cast<py::ssize_t>(n), 3});
  d["fields"] = to_numpy(std::move(r.fields), {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(r.nf)});
  d["polys"] = cells_dict(std::move(r.polys));
  d["lines"] = cells_dict(std::move(r.lines));
  d["poly_cell"] = to_numpy(std::move(r.poly_cell));
  d["line_cell"] = to_numpy(std::move(r.line_cell));
  return d;
}

int ncomp_of(const py::array& a, index_t nrows) {
  if (a.ndim() == 1) return 1;
  py::ssize_t k = 1;
  for (py::ssize_t i = 1; i < a.ndim(); ++i) k *= a.shape(i);
  if (a.shape(0) != nrows) throw py::value_error("array has " + std::to_string(a.shape(0)) + " rows, expected " + std::to_string(nrows));
  return static_cast<int>(k);
}

py::array read_native(const VtkXmlFile& f, const VtkArrayInfo& a, index_t first, index_t count) {
  index_t total = f.num_values(a) / a.ncomp;
  if (count < 0) count = total - first;
  if (first < 0 || first + count > total) throw py::index_error("tuple range out of bounds");
  std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(count)};
  if (a.ncomp > 1) shape.push_back(a.ncomp);
  py::array out;
  CFDPOST_DTYPE_SWITCH(a.dtype, S, {
    py::array_t<S> arr(shape);
    S* p = arr.mutable_data();
    {
      py::gil_scoped_release rel;
      f.read_values(a, first * a.ncomp, count * a.ncomp, p);
    }
    out = arr;
  })
  return out;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "cfdpost C++ core";
  py::register_exception<Error>(m, "CoreError", PyExc_RuntimeError);

  m.def("num_threads", &num_threads);
  m.def("mpi_init", &mpi_init, "Initialise MPI (thread level FUNNELED); False if already initialised");
  m.def("mpi_finalize", &mpi_finalize);
  m.def("mpi_is_initialized", &mpi_is_initialized);
  m.def("mpi_is_finalized", &mpi_is_finalized);
  m.def("mpi_library_version", &mpi_library_version);

  // ------------------------------------------------------------------ Comm
  py::class_<Comm>(m, "Comm")
      .def_static("world", &Comm::world)
      .def_static("self", &Comm::self)
      .def_property_readonly("rank", &Comm::rank)
      .def_property_readonly("size", &Comm::size)
      .def_property_readonly("is_mpi", &Comm::is_mpi)
      .def("barrier", &Comm::barrier, py::call_guard<py::gil_scoped_release>())
      .def("split", &Comm::split, py::arg("color"), py::arg("key") = 0)
      .def("allreduce", [](const Comm& c, darr a, const std::string& op) {
        std::vector<double> v(a.data(), a.data() + a.size());
        {
          py::gil_scoped_release rel;
          if (op == "sum") c.allreduce_sum(v.data(), v.size());
          else if (op == "min") c.allreduce_min(v.data(), v.size());
          else if (op == "max") c.allreduce_max(v.data(), v.size());
          else throw Error("unknown op " + op);
        }
        std::vector<py::ssize_t> shape(a.shape(), a.shape() + a.ndim());
        return to_numpy(std::move(v), shape);
      }, py::arg("array"), py::arg("op") = "sum")
      .def("bcast_bytes", [](const Comm& c, py::bytes b, int root) {
        std::string s = b;
        {
          py::gil_scoped_release rel;
          c.bcast_bytes(s, root);
        }
        return py::bytes(s);
      }, py::arg("data"), py::arg("root") = 0)
      .def("gather_bytes", [](const Comm& c, py::bytes b, int root) {
        std::string s = b;
        std::vector<std::string> out;
        {
          py::gil_scoped_release rel;
          out = c.gather_bytes(s, root);
        }
        py::list l;
        for (auto& x : out) l.append(py::bytes(x));
        return l;
      }, py::arg("data"), py::arg("root") = 0)
      .def("allgather_bytes", [](const Comm& c, py::bytes b) {
        std::string s = b;
        std::vector<std::string> out;
        {
          py::gil_scoped_release rel;
          out = c.allgather_bytes(s);
        }
        py::list l;
        for (auto& x : out) l.append(py::bytes(x));
        return l;
      })
      .def("send_bytes", [](const Comm& c, py::bytes b, int dest, int tag) {
        std::string s = b;
        py::gil_scoped_release rel;
        c.send_bytes(s, dest, tag);
      }, py::arg("data"), py::arg("dest"), py::arg("tag") = 0)
      .def("recv_bytes", [](const Comm& c, int source, int tag) {
        int src = 0, tg = 0;
        std::string s;
        {
          py::gil_scoped_release rel;
          s = c.recv_bytes(source, tag, &src, &tg);
        }
        return py::make_tuple(py::bytes(s), src, tg);
      }, py::arg("source") = MPI_ANY_SOURCE, py::arg("tag") = MPI_ANY_TAG);
  m.attr("ANY_SOURCE") = MPI_ANY_SOURCE;
  m.attr("ANY_TAG") = MPI_ANY_TAG;

  // ------------------------------------------------------------------ VTK XML
  py::class_<VtkXmlFile, std::shared_ptr<VtkXmlFile>>(m, "VtkXmlFile")
      .def(py::init<const std::string&>(), py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("path", &VtkXmlFile::path)
      .def_property_readonly("dataset_type", &VtkXmlFile::dataset_type)
      .def_property_readonly("num_points", &VtkXmlFile::num_points)
      .def_property_readonly("num_cells", &VtkXmlFile::num_cells)
      .def_property_readonly("piece_counts", &VtkXmlFile::piece_counts)
      .def("arrays", [](const VtkXmlFile& f) {
        py::list l;
        for (auto& a : f.arrays()) {
          py::dict d;
          d["section"] = a.section;
          d["name"] = a.name;
          d["type"] = dtype_name(a.dtype);
          d["ncomp"] = a.ncomp;
          d["format"] = a.format;
          d["direct"] = f.is_direct(a);
          l.append(d);
        }
        return l;
      })
      .def("read", [](const VtkXmlFile& f, const std::string& section, const std::string& name, index_t first,
                      index_t count) { return read_native(f, f.get(section, name), first, count); },
           py::arg("section"), py::arg("name"), py::arg("first") = 0, py::arg("count") = -1,
           "Read tuples [first, first+count) in the stored dtype")
      .def("read_tuples", [](const VtkXmlFile& f, const std::string& section, const std::string& name, iarr ids) {
        const VtkArrayInfo& a = f.get(section, name);
        std::vector<double> out(static_cast<std::size_t>(ids.size()) * a.ncomp);
        {
          py::gil_scoped_release rel;
          f.read_tuples(a, ids.data(), ids.size(), out.data());
        }
        return to_numpy_2d(std::move(out), a.ncomp);
      }, py::arg("section"), py::arg("name"), py::arg("ids"), "Gather tuples by id, as float64");

  // ------------------------------------------------------------------ SU2
  m.def("read_su2", [](const std::string& path, bool points, bool elements, bool markers,
                       std::vector<std::string> only_markers, int zone) {
    Su2ReadOptions o;
    o.points = points;
    o.elements = elements;
    o.markers = markers;
    o.only_markers = std::move(only_markers);
    o.zone = zone;
    Su2Mesh mesh;
    {
      py::gil_scoped_release rel;
      mesh = read_su2(path, o);
    }
    py::dict d;
    d["ndim"] = mesh.ndim;
    d["npoin"] = mesh.npoin;
    d["nelem"] = mesh.nelem;
    d["points"] = to_numpy(std::move(mesh.xyz), {static_cast<py::ssize_t>(mesh.xyz.size() / 3), 3});
    d["elements"] = cells_dict(std::move(mesh.elems));
    py::dict mk;
    for (auto& mm : mesh.markers) mk[py::str(mm.tag)] = cells_dict(std::move(mm.elems));
    d["markers"] = mk;
    return d;
  }, py::arg("path"), py::arg("points") = true, py::arg("elements") = true, py::arg("markers") = true,
        py::arg("only_markers") = std::vector<std::string>{}, py::arg("zone") = 1);
  m.def("su2_marker_list", &su2_marker_list, py::arg("path"), py::arg("zone") = 1,
        py::call_guard<py::gil_scoped_release>());

  // ------------------------------------------------------------------ Volume
  py::class_<VolumePartition, std::shared_ptr<VolumePartition>>(m, "VolumePartition")
      .def(py::init<const std::string&, Comm>(), py::arg("path"), py::arg("comm"),
           py::call_guard<py::gil_scoped_release>())
      .def(py::init([](darr xyz, u8arr types, iarr offsets, iarr conn, Comm comm) {
             if (xyz.ndim() != 2 || xyz.shape(1) != 3) throw py::value_error("xyz must be (n, 3)");
             if (offsets.size() != types.size() + 1) throw py::value_error("offsets must have ncells+1 entries");
             py::gil_scoped_release rel;
             return std::make_shared<VolumePartition>(xyz.data(), xyz.shape(0), types.data(), offsets.data(),
                                                      conn.data(), types.size(), comm);
           }),
           py::arg("xyz"), py::arg("types"), py::arg("offsets"), py::arg("conn"), py::arg("comm"))
      .def_property_readonly("comm", &VolumePartition::comm)
      .def_property_readonly("n_points", &VolumePartition::n_points)
      .def_property_readonly("n_cells", &VolumePartition::n_cells)
      .def_property_readonly("cell_range", [](const VolumePartition& p) { return py::make_tuple(p.cell_begin(), p.cell_end()); })
      .def_property_readonly("own_range", [](const VolumePartition& p) { return py::make_tuple(p.own_begin(), p.own_end()); })
      .def_property_readonly("n_owned", &VolumePartition::n_owned)
      .def_property_readonly("n_touched", &VolumePartition::n_touched)
      .def_property_readonly("dim", &VolumePartition::dim)
      .def_property_readonly("touched", [](const VolumePartition& p) { return to_numpy(std::vector<index_t>(p.touched())); })
      .def_property_readonly("xyz_touched", [](const VolumePartition& p) {
        return to_numpy(std::vector<double>(p.xyz_touched()), {static_cast<py::ssize_t>(p.n_touched()), 3});
      })
      .def("xyz_owned", [](const VolumePartition& p) {
        std::vector<double> v;
        {
          py::gil_scoped_release rel;
          v = p.xyz_owned();
        }
        return to_numpy(std::move(v), {static_cast<py::ssize_t>(p.n_owned()), 3});
      })
      .def("local_cells", [](const VolumePartition& p) {
        const auto& c = p.cells();
        CellArray<index_t> out;
        out.types = c.types;
        out.offsets = c.offsets;
        out.conn.assign(c.conn.begin(), c.conn.end());
        return cells_dict(std::move(out));
      }, "Local cells (connectivity in touched-local ids)")
      .def("set_coordinates", [](VolumePartition& p, darr xyz) {
        if (xyz.ndim() != 2 || xyz.shape(0) != p.n_owned() || xyz.shape(1) != 3)
          throw py::value_error("expected owned coordinates of shape (n_owned, 3)");
        py::gil_scoped_release rel;
        p.set_coordinates(xyz.data());
      })
      .def("point_array_names", [](const VolumePartition& p) {
        py::list l;
        if (p.file())
          for (auto& a : p.file()->arrays())
            if (a.section == "PointData") l.append(a.name);
        return l;
      })
      .def("read_point_data", [](const VolumePartition& p, const std::string& name, bool owned) {
        int nc = 1;
        std::vector<double> v;
        {
          py::gil_scoped_release rel;
          v = p.read_point_data(name, owned, nc);
        }
        return to_numpy_2d(std::move(v), nc);
      }, py::arg("name"), py::arg("owned") = true)
      .def("read_cell_data", [](const VolumePartition& p, const std::string& name) {
        int nc = 1;
        std::vector<double> v = p.read_cell_data(name, nc);
        return to_numpy_2d(std::move(v), nc);
      })
      .def("to_touched", [](const VolumePartition& p, darr a) {
        int k = ncomp_of(a, p.n_owned());
        std::vector<double> v;
        {
          py::gil_scoped_release rel;
          v = p.to_touched(a.data(), k);
        }
        std::vector<py::ssize_t> shape(a.shape(), a.shape() + a.ndim());
        shape[0] = p.n_touched();
        return to_numpy(std::move(v), shape);
      })
      .def("sum_to_owned", [](const VolumePartition& p, darr a) {
        int k = ncomp_of(a, p.n_touched());
        std::vector<double> v;
        {
          py::gil_scoped_release rel;
          v = p.sum_to_owned(a.data(), k);
        }
        std::vector<py::ssize_t> shape(a.shape(), a.shape() + a.ndim());
        shape[0] = p.n_owned();
        return to_numpy(std::move(v), shape);
      })
      .def("gradient", [](const VolumePartition& p, darr a) {
        int k = ncomp_of(a, p.n_owned());
        std::vector<double> v;
        {
          py::gil_scoped_release rel;
          v = p.gradient(a.data(), k);
        }
        std::vector<py::ssize_t> shape(a.shape(), a.shape() + a.ndim());
        shape.push_back(3);
        return to_numpy(std::move(v), shape);
      }, "Nodal gradient of an owned array (n_owned, ...) -> (n_owned, ..., 3)")
      .def("nodal_volume", [](const VolumePartition& p) {
        std::vector<double> v;
        {
          py::gil_scoped_release rel;
          v = p.nodal_volume();
        }
        return to_numpy(std::move(v));
      })
      .def("cell_volumes", [](const VolumePartition& p) { return to_numpy(p.cell_volumes()); })
      .def("contour_touched", [](const VolumePartition& p, darr scalar, double iso, darr fields, bool gather) {
        if (scalar.size() != p.n_touched()) throw py::value_error("scalar must be in touched layout");
        int nf = fields.size() == 0 ? 0 : ncomp_of(fields, p.n_touched());
        ContourResult r;
        {
          py::gil_scoped_release rel;
          r = p.contour_touched(scalar.data(), iso, fields.data(), nf, gather);
        }
        return contour_dict(std::move(r));
      }, py::arg("scalar"), py::arg("iso"), py::arg("fields"), py::arg("gather") = true);

  // ------------------------------------------------------------------ serial kernels
  m.def("contour", [](darr xyz, u8arr types, iarr offsets, iarr conn, darr scalar, double iso, darr fields) {
    index_t n = xyz.shape(0);
    if (scalar.size() != n) throw py::value_error("scalar must have one value per point");
    int nf = fields.size() == 0 ? 0 : ncomp_of(fields, n);
    CellArray<lindex_t> cells;
    cells.types.assign(types.data(), types.data() + types.size());
    cells.offsets.assign(offsets.data(), offsets.data() + offsets.size());
    cells.conn.resize(static_cast<std::size_t>(conn.size()));
    for (py::ssize_t i = 0; i < conn.size(); ++i) cells.conn[i] = static_cast<lindex_t>(conn.data()[i]);
    std::vector<index_t> gid(static_cast<std::size_t>(n));
    for (index_t i = 0; i < n; ++i) gid[i] = i;
    ContourResult r;
    {
      py::gil_scoped_release rel;
      r = contour_cells(cells, gid.data(), xyz.data(), scalar.data(), iso, fields.data(), nf, 0);
    }
    return contour_dict(std::move(r));
  }, py::arg("xyz"), py::arg("types"), py::arg("offsets"), py::arg("conn"), py::arg("scalar"), py::arg("iso"),
        py::arg("fields"));

  auto surf_in = [](darr& xyz, u8arr& types, iarr& offsets, iarr& conn, darr& cp, py::object cf, py::object sign,
                    darr& cfa, darr& sga, py::object ref) {
    SurfaceInput in;
    in.xyz = xyz.data();
    in.npts = xyz.shape(0);
    in.types = types.data();
    in.offsets = offsets.data();
    in.conn = conn.data();
    in.nfaces = types.size();
    if (cp.size() != in.npts) throw py::value_error("cp must have one value per point");
    in.cp = cp.data();
    if (!cf.is_none()) {
      cfa = cf.cast<darr>();
      if (cfa.size() != 3 * in.npts) throw py::value_error("cf must be (npts, 3)");
      in.cf = cfa.data();
    }
    if (!sign.is_none()) {
      sga = sign.cast<darr>();
      if (sga.size() != in.nfaces) throw py::value_error("sign must have one value per face");
      in.sign = sga.data();
    }
    if (!ref.is_none()) {
      auto r = ref.cast<std::vector<double>>();
      for (int i = 0; i < 3; ++i) in.ref[i] = r.at(i);
    }
    return in;
  };

  m.def("face_loads", [surf_in](darr xyz, u8arr types, iarr offsets, iarr conn, darr cp, py::object cf,
                                py::object sign, py::object ref) {
    darr cfa, sga;
    SurfaceInput in = surf_in(xyz, types, offsets, conn, cp, cf, sign, cfa, sga, ref);
    std::vector<double> v;
    {
      py::gil_scoped_release rel;
      v = face_loads(in);
    }
    return to_numpy(std::move(v), {static_cast<py::ssize_t>(in.nfaces), 15});
  }, py::arg("xyz"), py::arg("types"), py::arg("offsets"), py::arg("conn"), py::arg("cp"), py::arg("cf") = py::none(),
        py::arg("sign") = py::none(), py::arg("ref") = py::none());

  m.def("strip_loads", [surf_in](darr xyz, u8arr types, iarr offsets, iarr conn, darr cp, py::object cf,
                                 py::object sign, py::object ref, std::vector<double> axis, std::vector<double> edges,
                                 std::vector<double> chord_dir) {
    darr cfa, sga;
    SurfaceInput in = surf_in(xyz, types, offsets, conn, cp, cf, sign, cfa, sga, ref);
    if (axis.size() != 3 || chord_dir.size() != 3) throw py::value_error("axis and chord_dir must have 3 components");
    std::vector<double> v;
    {
      py::gil_scoped_release rel;
      v = strip_loads(in, axis.data(), edges, chord_dir.data());
    }
    return to_numpy(std::move(v), {static_cast<py::ssize_t>(edges.size() - 1), 20});
  }, py::arg("xyz"), py::arg("types"), py::arg("offsets"), py::arg("conn"), py::arg("cp"), py::arg("cf") = py::none(),
        py::arg("sign") = py::none(), py::arg("ref") = py::none(), py::arg("axis"), py::arg("edges"),
        py::arg("chord_dir"));

  m.def("orient_faces", [](darr xyz, u8arr vtypes, iarr voffsets, iarr vconn, u8arr ftypes, iarr foffsets,
                           iarr fconn) {
    std::vector<std::int8_t> s;
    {
      py::gil_scoped_release rel;
      s = orient_faces(xyz.data(), xyz.shape(0), vtypes.data(), voffsets.data(), vconn.data(), vtypes.size(),
                       ftypes.data(), foffsets.data(), fconn.data(), ftypes.size());
    }
    return to_numpy(std::move(s));
  });
}
