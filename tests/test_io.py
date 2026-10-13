import numpy as np
import pytest

import meshgen
from cfdpost import VtkFile, read_su2, read_vtk, marker_list
from cfdpost.grid import Cells, Grid


def test_vtu_roundtrip(tmp):
    g = meshgen.box((3, 3, 2), mix=("hex", "pyr", "wedge", "tet"), seed=1)
    g.point_data["s"] = np.arange(g.n_points, dtype=np.float32)
    g.point_data["v"] = np.random.default_rng(0).normal(size=(g.n_points, 3))
    g.cell_data["id"] = np.arange(g.n_cells, dtype=np.int32)
    g.write(tmp / "a.vtu")
    f = VtkFile(tmp / "a.vtu")
    assert all(a["direct"] for a in f.arrays())
    h = read_vtk(tmp / "a.vtu")
    np.testing.assert_array_equal(h.points, g.points)
    np.testing.assert_array_equal(h.cells.conn, g.cells.conn)
    np.testing.assert_array_equal(h.cells.types, g.cells.types)
    np.testing.assert_array_equal(h.point_data["s"], g.point_data["s"])
    np.testing.assert_array_equal(h.point_data["v"], g.point_data["v"])
    np.testing.assert_array_equal(h.cell_data["id"], g.cell_data["id"])
    ids = np.array([5, 0, 7])
    np.testing.assert_array_equal(f.point("v", ids=ids), g.point_data["v"][ids])
    np.testing.assert_array_equal(f.read("PointData", "v", 2, 3), g.point_data["v"][2:5])


def test_vtp_mixed_lines_polys(tmp):
    pts = np.random.default_rng(1).normal(size=(6, 3))
    cells = Cells(np.array([5, 3, 9], np.uint8), np.array([0, 3, 5, 9]), np.array([0, 1, 2, 3, 4, 0, 1, 2, 5]))
    g = Grid(pts, cells, {"p": np.arange(6.0)}, {"c": np.array([10.0, 20.0, 30.0])})
    g.write(tmp / "a.vtp")
    h = read_vtk(tmp / "a.vtp")
    # VTK order: lines before polys
    assert list(h.cells.types) == [3, 5, 9]
    np.testing.assert_array_equal(h.cell_data["c"], [20.0, 10.0, 30.0])


@pytest.mark.parametrize("mode", ["binary", "ascii", "appended_b64", "zlib"])
def test_read_vtk_written_files(tmp, mode):
    vtk = pytest.importorskip("vtk")
    import pyvista as pv
    g = meshgen.box((3, 2, 2), mix=("hex", "tet"), seed=2)
    g.point_data["P"] = np.linspace(0, 1, g.n_points)
    g.point_data["V"] = np.random.default_rng(3).normal(size=(g.n_points, 3)).astype(np.float32)
    ug = g.to_pyvista()
    w = vtk.vtkXMLUnstructuredGridWriter()
    w.SetInputData(ug)
    w.SetFileName(str(tmp / "b.vtu"))
    w.SetHeaderTypeToUInt32() if mode == "binary" else w.SetHeaderTypeToUInt64()
    if mode == "binary":
        w.SetDataModeToBinary()
        w.SetCompressorTypeToNone()
    elif mode == "ascii":
        w.SetDataModeToAscii()
    elif mode == "appended_b64":
        w.SetDataModeToAppended()
        w.EncodeAppendedDataOn()
        w.SetCompressorTypeToNone()
    else:
        w.SetDataModeToAppended()
        w.EncodeAppendedDataOff()
        w.SetCompressorTypeToZLib()
    w.Write()
    h = read_vtk(tmp / "b.vtu")
    np.testing.assert_allclose(h.points, g.points, rtol=1e-6 if mode == "ascii" else 0)
    np.testing.assert_array_equal(h.cells.conn, g.cells.conn)
    np.testing.assert_allclose(h.point_data["P"], g.point_data["P"], rtol=1e-6 if mode == "ascii" else 0)
    np.testing.assert_allclose(h.point_data["V"], g.point_data["V"], rtol=1e-6 if mode == "ascii" else 0)
    del pv


@pytest.mark.parametrize("index_column", [True, False])
def test_su2_reader(tmp, index_column):
    g = meshgen.box((3, 2, 2), mix=("hex", "pyr", "wedge", "tet"), seed=5)
    sides = meshgen.box_boundary((3, 2, 2))
    path = tmp / "m.su2"
    meshgen.write_su2(path, g, sides, index_column=index_column)
    m = read_su2(path)
    assert (m.ndim, m.npoin, m.nelem) == (3, g.n_points, g.n_cells)
    np.testing.assert_allclose(m.points, g.points, rtol=1e-15)
    np.testing.assert_array_equal(m.elements.types, g.cells.types)
    np.testing.assert_array_equal(m.elements.conn, g.cells.conn)
    assert m.marker_names == list(sides)
    for k, v in sides.items():
        np.testing.assert_array_equal(m.markers[k].conn, v.ravel())
    assert marker_list(path) == {k: len(v) for k, v in sides.items()}
    m2 = read_su2(path, elements=False, markers=["zmax"])
    assert m2.elements is None and list(m2.markers) == ["zmax"] and m2.nelem == g.n_cells
    with pytest.raises(RuntimeError):
        read_su2(path, markers=["nope"])


def test_su2_point_index_reordering(tmp):
    path = tmp / "r.su2"
    with open(path, "w") as f:
        f.write("NDIME= 2\nNELEM= 1\n5 0 1 2 0\nNPOIN= 3\n1.0 0.0 1\n0.0 1.0 2\n0.0 0.0 0\n"
                "NMARK= 1\nMARKER_TAG= wall\nMARKER_ELEMS= 1\n3 0 1\n")
    m = read_su2(path)
    np.testing.assert_array_equal(m.points, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert m.ndim == 2 and list(m.markers["wall"].types) == [3]
