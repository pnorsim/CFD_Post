import numpy as np
import pytest

import meshgen
from cfdpost import SurfaceSolution, cuts, loads, read_su2
from cfdpost.grid import Cells, Grid

N = (6, 5, 4)


@pytest.fixture
def case(tmp):
    """Box mesh with 4 markers and an SU2-style surface_flow.vtu (points = marker
    nodes in increasing global id, faces in a shuffled order)."""
    g = meshgen.box(N, mix=("hex", "pyr"), flip=0.3, seed=11)   # conforming boundary quads
    sides = meshgen.box_boundary(N)
    meshgen.write_su2(tmp / "m.su2", g, sides)
    mesh = read_su2(tmp / "m.su2")
    gids = mesh.boundary_nodes()
    loc = np.full(mesh.npoin, -1)
    loc[gids] = np.arange(len(gids))
    faces = np.vstack(list(sides.values()))
    perm = np.random.default_rng(0).permutation(len(faces))
    faces = faces[perm]
    xyz = mesh.points[gids]
    pd = {"Pressure_Coefficient": xyz[:, 0] + 2 * xyz[:, 2],
          "Skin_Friction_Coefficient": np.tile([0.01, 0.0, 0.0], (len(gids), 1))}
    Grid(xyz, Cells.uniform(9, loc[faces]), pd).write(tmp / "surface_flow.vtu")
    return mesh, tmp / "surface_flow.vtu", sides


def test_markers_attached(case):
    mesh, path, sides = case
    sol = SurfaceSolution.read(path, mesh)
    assert sol.marker_names == list(sides)
    for i, (name, q) in enumerate(sides.items()):
        assert np.sum(sol.marker_id == i) == len(q)
    sub = sol.select(["zmin", "zmax"])
    assert sub.grid.n_cells == len(sides["zmin"]) + len(sides["zmax"])
    np.testing.assert_allclose(sub.points, mesh.points[sub.global_ids])


def test_orientation_from_volume_mesh(case):
    mesh, path, sides = case
    sol = SurfaceSolution.read(path, mesh)
    s = sol.orientation("mesh", mesh)
    # box_boundary faces point out of the box, the fluid is inside
    assert np.all(s == -1)


def test_forces_exact_divergence_theorem(case):
    mesh, path, sides = case
    sol = SurfaceSolution.read(path, mesh)
    # treat the box as a body: normals out of the box (into a fictitious outer fluid)
    sol.set_orientation(1.0)
    r = loads.integrate(sol, cf=None)
    # int -Cp n dA over closed surface = -int grad Cp dV = -(1, 0, 2) * volume(=1)
    np.testing.assert_allclose(r.force_pressure, [-1.0, 0.0, -2.0], atol=1e-12)
    # moment of -grad(Cp) volume force about origin: -int x × (1,0,2) dV, centroid (0.5,0.5,0.5)
    np.testing.assert_allclose(r.moment, -np.cross([0.5, 0.5, 0.5], [1.0, 0.0, 2.0]), atol=1e-12)
    # only 4 of 6 sides are markers -> open; use the full set check per marker sums
    tot = sum(v.force_pressure for v in r.by_marker.values())
    np.testing.assert_allclose(tot, r.force_pressure, atol=1e-13)


def test_closed_sphere_constant_and_linear_cp():
    v, f = meshgen.icosphere(3, r=0.8)
    g = Grid(v, Cells.uniform(5, f), {"Cp": np.ones(len(v)), "CpX": v[:, 0]})
    r = loads.integrate(g, cp="Cp", cf=None, sign=1.0, ref_point=(0.3, 0.1, 0))
    np.testing.assert_allclose(r.force_pressure, 0, atol=1e-13)
    np.testing.assert_allclose(r.moment, 0, atol=1e-13)
    vol = np.einsum("ij,ij->", g.cell_centers(), g.area_vectors()) / 3   # polyhedron volume
    r = loads.integrate(g, cp="CpX", cf=None, sign=1.0)
    np.testing.assert_allclose(r.force_pressure, [-vol, 0, 0], atol=1e-13)


@pytest.mark.parametrize("nbins", [1, 7, 40])
def test_strips_sum_to_total(nbins):
    v, f = meshgen.icosphere(3, r=1.0)
    rng = np.random.default_rng(1)
    cp = v @ [0.3, -1.0, 0.5] + 0.2 * v[:, 1] ** 2
    cf = rng.normal(size=(len(v), 3)) * 0.01
    g = Grid(v, Cells.uniform(5, f), {"Cp": cp, "Cf": cf})
    tot = loads.integrate(g, cp="Cp", cf="Cf", sign=1.0, ref_point=(0.1, 0.2, 0.3))
    sl = loads.spanload(g, axis=(0, 1, 0), nbins=nbins, cp="Cp", cf="Cf", sign=1.0, ref_point=(0.1, 0.2, 0.3))
    np.testing.assert_allclose(sl.force_pressure.sum(0), tot.force_pressure, atol=1e-12)
    np.testing.assert_allclose(sl.force_viscous.sum(0), tot.force_viscous, atol=1e-12)
    np.testing.assert_allclose(sl.moment.sum(0), tot.moment, atol=1e-12)
    assert sl.area.sum() == pytest.approx(tot.area, rel=1e-12)


def test_strip_exact_on_flat_plate():
    # unit square plate z=0 in two triangles, Cp = y: strip [a,b] carries -n * (b^2-a^2)/2
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0]])
    g = Grid(v, Cells.uniform(5, [[0, 1, 2], [0, 2, 3]]), {"Cp": v[:, 1]})
    edges = np.array([0, 0.1, 0.35, 0.8, 1.0])
    sl = loads.spanload(g, edges=edges, cp="Cp", cf=None, sign=1.0)
    exp = -(edges[1:] ** 2 - edges[:-1] ** 2) / 2
    np.testing.assert_allclose(sl.force_pressure[:, 2], exp, atol=1e-15)
    np.testing.assert_allclose(sl.chord, 1.0)
    np.testing.assert_allclose(sl.area, np.diff(edges))


def test_sections_of_a_sphere():
    v, f = meshgen.icosphere(4, r=1.0)
    g = Grid(v, Cells.uniform(5, f), {"z": v[:, 2]})
    secs = cuts.sections(g, (0, 0.5, 0), (0, 1, 0))
    assert len(secs) == 1 and secs[0].closed
    s = secs[0]
    assert s.length == pytest.approx(2 * np.pi * np.sqrt(0.75), rel=2e-3)
    np.testing.assert_allclose(s.points[:, 1], 0.5)
    np.testing.assert_allclose(s.data["z"], s.points[:, 2], atol=1e-12)
    up, lo = s.split()
    assert s.points[up, 2].mean() > 0 > s.points[lo, 2].mean()


def test_wing_like_section_selection():
    # two separate spheres: pick the one nearest to the seed
    v1, f1 = meshgen.icosphere(2, 1.0)
    v2, f2 = meshgen.icosphere(2, 0.5)
    v = np.vstack([v1, v2 + [5, 0, 0]])
    f = np.vstack([f1, f2 + len(v1)])
    g = Grid(v, Cells.uniform(5, f))
    s = cuts.section_at(g, (0, 0.1, 0), (0, 1, 0), seed=(5, 0, 0))
    assert s.points[:, 0].mean() == pytest.approx(5, abs=0.05)
    le, te, c = s.chord_line()
    assert c == pytest.approx(1.0, rel=0.05)


def test_strips_include_faces_on_end_planes():
    # closed box surface: the y=0 and y=1 faces lie exactly on the first / last strip planes
    sides = meshgen.box_boundary((2, 2, 2))
    g = meshgen.box((2, 2, 2))
    v = g.points
    # all six faces of the unit cube as triangles, outward normals
    cube = Grid(v, Cells.uniform(9, np.vstack(list(sides.values()) + _y_faces((2, 2, 2)))), {"Cp": v[:, 1] + 0.3})
    tot = loads.integrate(cube, cp="Cp", cf=None, sign=1.0)
    for nb in (1, 3):
        sl = loads.spanload(cube, axis=(0, 1, 0), nbins=nb, cp="Cp", cf=None, sign=1.0)
        np.testing.assert_allclose(sl.force_pressure.sum(0), tot.force_pressure, atol=1e-13)
        assert sl.area.sum() == pytest.approx(6.0)


def _y_faces(n):
    nx, ny, nz = n
    pid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k  # noqa: E731
    lo = [[pid(i, 0, k), pid(i + 1, 0, k), pid(i + 1, 0, k + 1), pid(i, 0, k + 1)] for i in range(nx) for k in range(nz)]
    hi = [[pid(i, ny, k), pid(i, ny, k + 1), pid(i + 1, ny, k + 1), pid(i + 1, ny, k)] for i in range(nx) for k in range(nz)]
    return [np.array(lo), np.array(hi)]
