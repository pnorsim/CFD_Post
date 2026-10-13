import numpy as np
import pytest

import meshgen
from cfdpost import Volume
from cfdpost.grid import Grid

MIX = ("hex", "pyr", "wedge", "tet")


@pytest.mark.parametrize("jitter,flip", [(0.0, 0.0), (0.3, 0.0), (0.3, 0.4)])
def test_gradient_exact_for_linear_fields(jitter, flip):
    g = meshgen.box((5, 4, 4), mix=MIX, jitter=jitter, flip=flip, seed=3)
    a = np.array([1.5, -2.0, 0.7])
    f = g.points @ a + 3.0
    V = np.stack([g.points @ [1, 2, 3], g.points @ [0, -1, 4], g.points @ [5, 0, 0]], 1)
    vol = Volume(Grid(g.points, g.cells, {"f": f, "V": V}))
    gf = vol.gradient(vol.read("f"))
    assert gf.shape == (g.n_points, 3)
    np.testing.assert_allclose(gf, np.broadcast_to(a, gf.shape), atol=1e-10)
    gv = vol.gradient(vol.read("V"))
    assert gv.shape == (g.n_points, 3, 3)
    np.testing.assert_allclose(gv[:, 0], np.broadcast_to([1, 2, 3], (g.n_points, 3)), atol=1e-10)
    np.testing.assert_allclose(gv[:, 2], np.broadcast_to([5, 0, 0], (g.n_points, 3)), atol=1e-10)


@pytest.mark.parametrize("mix,jitter", [(MIX, 0.0), (("hex", "pyr"), 0.25)])
def test_nodal_volume_sums_to_domain_volume(mix, jitter):
    # (tet-split faces next to hex faces only tile exactly when faces are planar)
    g = meshgen.box((5, 4, 3), mix=mix, jitter=jitter, flip=0.3, seed=2)
    vol = Volume(g)
    assert vol.nodal_volume().sum() == pytest.approx(1.0, rel=1e-12)


def _edge_count(cells):
    from collections import Counter
    cnt = Counter()
    for i in range(len(cells)):
        c = cells.cell(i)
        for a, b in zip(c, np.roll(c, -1)):
            cnt[(min(a, b), max(a, b))] += 1
    return cnt


@pytest.mark.parametrize("mix", [("hex", "pyr"), MIX])
def test_plane_slice_area_and_orientation(mix):
    g = meshgen.box((6, 5, 4), mix=mix, flip=0.3, seed=5)
    vol = Volume(g)
    s = vol.slice((0, 0, 0.37), (0, 0, 1))
    assert s.n_cells > 0
    S = s.area_vectors()
    assert S.sum(0) == pytest.approx([0, 0, 1.0], abs=1e-12)    # area 1, normals along +z
    assert np.all(S[:, 2] > 0)
    assert np.allclose(s.points[:, 2], 0.37)


def test_sphere_isosurface_closed_and_outward():
    g = meshgen.box((14, 14, 14), lo=(-1, -1, -1), hi=(1, 1, 1), mix=("hex", "pyr"), flip=0.3, seed=1)
    c = np.array([0.05, -0.02, 0.03])
    r = np.linalg.norm(g.points - c, axis=1)
    vol = Volume(g)
    iso = vol.contour(r, 0.6, {"r": r})
    A = iso.areas().sum()
    assert A == pytest.approx(4 * np.pi * 0.36, rel=0.02)
    np.testing.assert_allclose(iso.point_data["r"], 0.6, atol=0.05)
    # watertight: every polygon edge shared by exactly two polygons
    cnt = _edge_count(iso.cells)
    assert set(cnt.values()) == {2}
    # normals towards increasing r (outward)
    S = iso.area_vectors()
    out = np.einsum("ij,ij->i", S, iso.cell_centers() - c)
    assert np.all(out > 0)
    # enclosed volume via divergence theorem
    V = np.einsum("ij,ij->", iso.cell_centers() - c, S) / 3
    assert V == pytest.approx(4 / 3 * np.pi * 0.6 ** 3, rel=0.03)


def test_contour_mixed_types_area():
    g = meshgen.box((10, 10, 10), lo=(-1, -1, -1), hi=(1, 1, 1), mix=MIX, flip=0.3, seed=4)
    f = g.points @ np.array([1.0, 1.0, 0.0])          # oblique plane x + y = 0.3
    iso = Volume(g).contour(f, 0.3)
    # chord of the square [-1,1]^2 by x+y=0.3 runs from (-0.7,1) to (1,-0.7); height 2
    L = np.hypot(1.7, 1.7)
    assert iso.areas().sum() == pytest.approx(2 * L, rel=1e-10)
    n = iso.area_vectors().sum(0)
    assert n / np.linalg.norm(n) == pytest.approx(np.array([1, 1, 0]) / np.sqrt(2), abs=1e-12)


def test_set_coordinates_deformed_same_topology():
    g = meshgen.box((4, 4, 4), mix=MIX, seed=0)
    vol = Volume(g)
    x2 = g.points * np.array([2.0, 1.0, 1.0])            # stretched copy
    vol.set_coordinates(x2)
    assert vol.nodal_volume().sum() == pytest.approx(2.0, rel=1e-12)
    np.testing.assert_allclose(vol.xyz, x2)
    grad = vol.gradient(x2[:, 0])
    np.testing.assert_allclose(grad, np.broadcast_to([1, 0, 0], grad.shape), atol=1e-10)
