import numpy as np
import pytest

import meshgen
from cfdpost import SurfaceSolution, Volume, deform, read_su2, shock
from cfdpost.grid import Cells, Grid

GAMMA = 1.4


def normal_shock_volume(mix=("hex", "pyr", "tet"), x0=1.0, delta=0.08, normal=(1, 0, 0)):
    """Flow along `normal` decelerating from M 1.5 to 0.7 through a tanh shock at x0.
    a = 1 everywhere (rho = gamma p), so Mn = u and the Mn=1 surface is the plane
    s = x0 + delta * atanh(0.25) along the normal."""
    g = meshgen.box((40, 6, 6), lo=(0, -1, -1), hi=(2, 1, 1), mix=mix, flip=0.2, seed=2)
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    s = g.points @ n
    w = 0.5 * (1 + np.tanh((s - x0) / delta))
    u = 1.5 - 0.8 * w
    p = 1.0 + 1.2 * w
    g.point_data.update(Pressure=p, Density=GAMMA * p, Velocity=u[:, None] * n[None, :],
                        Mach=u)
    return g, x0 + delta * np.arctanh(0.25), n


@pytest.mark.parametrize("normal", [(1, 0, 0), (1, 0.4, 0.2)])
def test_lovely_haimes_finds_normal_shock(normal):
    g, xs, n = normal_shock_volume(normal=normal)
    res = shock.lovely_haimes(Volume(g), preset="default", verbose=False)
    surf = res.surface
    assert surf.n_cells > 0
    d = surf.points @ n - xs
    # within a fraction of the cell size along the shock normal (h = 0.05 x 0.33 x 0.33)
    h_n = np.abs(n) @ [0.05, 1 / 3, 1 / 3]
    assert np.abs(d).max() < 0.4 * h_n
    # polygons are oriented towards increasing Mn, i.e. upstream
    # (area-weighted: the test mesh has non-conforming hex/tet faces that can leave
    # an edge-on sliver polygon)
    S = surf.area_vectors()
    A = np.linalg.norm(S, axis=1)
    assert A[S @ n >= 0].sum() < 1e-3 * A.sum()
    np.testing.assert_allclose(surf.point_data["Mn"], 1.0, atol=1e-12)


def test_shock_filters_and_presets():
    g, xs, n = normal_shock_volume()
    g.point_data["Eddy_Viscosity"] = np.where(g.points[:, 1] > 0.5, 5000.0, 1.0)
    g.point_data["Laminar_Viscosity"] = np.ones(g.n_points)
    vol = Volume(g)
    full = shock.lovely_haimes(vol, nut_max=1e9, verbose=False).surface
    cut = shock.lovely_haimes(vol, verbose=False).surface      # mu_t/mu < 1000 drops y > 0.5
    assert cut.n_cells < full.n_cells
    assert cut.points[:, 1].max() <= 0.5 + 1e-9
    none = shock.lovely_haimes(vol, gradp_factor=1e9, verbose=False).surface
    assert none.n_cells == 0


def test_point_sensors():
    g, xs, n = normal_shock_volume(mix=("hex",))
    vol = Volume(g)
    d = shock.ducros(vol)
    near = np.abs(g.points[:, 0] - xs) < 0.05
    assert d[near].min() > 0.99                    # pure compression, no vorticity
    pg = shock.pressure_gradient_sensor(vol)
    assert pg[near].max() > 10 * np.median(pg)
    rg = shock.density_gradient(vol)
    assert np.argmax(rg) in np.flatnonzero(near)


def test_mesh_pair_and_compare(tmp):
    N = (5, 4, 3)
    g = meshgen.box(N, mix=("hex", "pyr"), seed=4)
    sides = meshgen.box_boundary(N)
    meshgen.write_su2(tmp / "base.su2", g, sides)
    disp = 0.05 * np.sin(np.pi * g.points[:, [1, 2, 0]])
    g2 = Grid(g.points + disp, g.cells)
    meshgen.write_su2(tmp / "def.su2", g2, sides)
    base, dfm = read_su2(tmp / "base.su2"), read_su2(tmp / "def.su2")
    assert base.same_topology(dfm) and base.topology_hash() == dfm.topology_hash()
    pair = deform.MeshPair(base, dfm)
    np.testing.assert_allclose(pair.displacement, disp, atol=1e-14)
    s = pair.surface(["zmax"])
    np.testing.assert_allclose(s.points, g2.points[s.point_data["GlobalIndex"]])
    # zmax faces point +z: normal displacement = dz
    np.testing.assert_allclose(s.point_data["NormalDisplacement"], disp[s.point_data["GlobalIndex"], 2], atol=1e-12)

    # two surface solutions on the two geometries, compared by global id
    def surf_vtu(mesh, path, cp):
        gids = mesh.boundary_nodes()
        loc = np.full(mesh.npoin, -1)
        loc[gids] = np.arange(len(gids))
        faces = np.vstack(list(sides.values()))
        Grid(mesh.points[gids], Cells.uniform(9, loc[faces]), {"Pressure_Coefficient": cp(mesh.points[gids])}).write(path)
    surf_vtu(base, tmp / "a.vtu", lambda x: x[:, 0])
    surf_vtu(dfm, tmp / "b.vtu", lambda x: x[:, 0] + 1)
    a = SurfaceSolution.read(tmp / "a.vtu", base)
    b = SurfaceSolution.read(tmp / "b.vtu", base)        # deformed solution tagged with baseline mesh
    c = deform.compare(a, b)
    d = c.grid.point_data["delta_Pressure_Coefficient"]
    np.testing.assert_allclose(d, 1 + disp[c.global_ids, 0], atol=1e-6)
    np.testing.assert_allclose(c.grid.point_data["Displacement"], disp[c.global_ids], atol=1e-6)
