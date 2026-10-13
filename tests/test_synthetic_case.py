"""End-to-end checks on the artificial wing case and shock box of
cfdpost.synthetic (all geometry and data are generated here), plus smoke runs
of the example scripts."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import cfdpost as cp
from cfdpost.synthetic import shock_box, wing_case
from conftest import have_mpirun, mpirun

ROOT = Path(__file__).resolve().parents[1]
WT = ["wing", "tip"]


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    return wing_case(tmp_path_factory.mktemp("wing"))


@pytest.fixture(scope="module")
def mesh(case):
    return cp.read_su2(case.files["mesh"])


def test_mesh_is_valid(case, mesh):
    assert mesh.marker_names == ["wing", "tip", "symmetry", "farfield"]
    vol = cp.Volume(case.files["base_volume"])
    assert vol.partition.cell_volumes().min() > 0
    assert set(np.unique(mesh.elements.types)) == {12, 13}
    # symmetry faces lie in y = 0
    assert np.abs(mesh.points[mesh.markers["symmetry"].conn, 1]).max() < 1e-12


def test_markers_and_orientation(case, mesh):
    sol = cp.SurfaceSolution.read(case.files["base_surface"], mesh)
    assert sol.marker_names == mesh.marker_names and np.all(sol.marker_id >= 0)
    assert np.array_equal(sol.global_ids, np.arange(case.base.n_surface))
    s = sol.orientation("mesh", mesh)
    assert np.all(s == 1)                             # generated faces point into the fluid
    assert np.all(sol.orientation("closed") == s)
    # markers only (no points): tagging is topological
    light = cp.read_su2(case.files["mesh"], elements=False, points=False)
    sol2 = cp.SurfaceSolution.read(case.files["opt_surface"], light)
    assert np.array_equal(sol2.marker_id, sol.marker_id)


def test_freestream(case, mesh):
    U, aoa, beta = cp.loads.freestream_from_volume(case.files["base_volume"], mesh, marker="farfield")
    assert aoa == pytest.approx(case.flow.aoa_deg, abs=1e-6) and beta == pytest.approx(0, abs=1e-9)


def test_loads_and_spanload(case, mesh):
    w = cp.SurfaceSolution.read(case.files["base_surface"], mesh).select(WT)
    w.orientation("mesh", mesh)
    tot = cp.loads.integrate(w, ref_area=case.spec.ref_area, aoa=case.flow.aoa_deg)
    assert set(tot.by_marker) == set(WT)
    assert tot.coefficients()["CL"] > 0.3
    # independent check: diagonal triangulation of the quads (O(h^2) different)
    tris, parent = w.grid.cells.triangles()
    P = w.points
    S = 0.5 * np.cross(P[tris[:, 1]] - P[tris[:, 0]], P[tris[:, 2]] - P[tris[:, 0]])
    Fp = -(w["Pressure_Coefficient"][tris].mean(1)[:, None] * S).sum(0)
    np.testing.assert_allclose(tot.force_pressure, Fp, rtol=2e-3, atol=2e-3 * np.abs(Fp).max())
    sl = cp.loads.spanload(w, nbins=40, aoa=case.flow.aoa_deg)
    np.testing.assert_allclose(sl.force.sum(0), tot.force, atol=1e-12)
    np.testing.assert_allclose(sl.moment.sum(0), tot.moment, atol=1e-10)
    assert not sl.open_strips().any()
    eta = sl.centers / case.spec.semispan
    # strip "chord" = x-extent of the surface in the strip: on a swept, tapered wing the
    # LE and TE move by dy/2 * slope across half a strip
    sp = case.spec
    t_le = np.tan(np.radians(sp.sweep_le_deg))
    t_te = t_le - sp.root_chord * (1 - sp.taper) / sp.semispan
    expect = sp.chord(eta) + 0.5 * sl.widths * (t_le + t_te)
    np.testing.assert_allclose(sl.chord, expect, rtol=0.01)
    ell, l0 = cp.loads.elliptic(sl, case.spec.semispan)
    assert l0 > 0 and np.all(ell >= 0)


def test_open_strip_detected(case, mesh):
    # remove the lower surface between 40 % and 60 % span: those strips no longer close
    sol = cp.SurfaceSolution.read(case.files["base_surface"], mesh).select(["wing"])
    c = sol.grid.cell_centers()
    S = sol.grid.area_vectors()
    hole = (c[:, 1] > 4) & (c[:, 1] < 6) & (S[:, 2] < 0)
    g = sol.grid.extract_cells(~hole)
    sl = cp.loads.spanload(g, edges=np.linspace(0, 10, 11), sign=1.0)
    assert list(np.flatnonzero(sl.open_strips())) == [4, 5]


@pytest.mark.parametrize("eta", [0.1, 0.5, 0.85])
def test_sections_recover_geometry(case, eta):
    sol = cp.SurfaceSolution.read(case.files["base_surface"], arrays=["Pressure_Coefficient"])
    sec = cp.cuts.section_at(sol, (0, eta * case.spec.semispan, 0), (0, 1, 0))
    assert sec.closed
    assert sec.chord_line()[2] == pytest.approx(case.spec.chord(eta), rel=2e-3)
    assert sec.twist() == pytest.approx(case.spec.twist(eta), abs=0.02)
    up, lo = sec.split()
    cpv = sec.data["Pressure_Coefficient"]
    assert cpv[up].min() < cpv[lo].min()                # suction side on top


def test_extract_from_volume_matches_surface(case, mesh):
    a = cp.SurfaceSolution.from_volume(case.files["base_volume"], mesh, WT)
    b = cp.SurfaceSolution.read(case.files["base_surface"], mesh).select(WT)
    o = np.argsort(b.global_ids)
    np.testing.assert_array_equal(a.global_ids, b.global_ids[o])
    np.testing.assert_array_equal(a.points, b.points[o])
    np.testing.assert_array_equal(a["Pressure_Coefficient"], b["Pressure_Coefficient"][o])


def test_deformed_pair_and_compare(case, mesh):
    pair = cp.deform.MeshPair(mesh, case.files["mesh_opt"])
    np.testing.assert_allclose(pair.displacement, case.opt.points - case.base.points, atol=1e-12)
    surf = pair.surface(WT)
    tipd = surf.point_data["Displacement"][surf.points[:, 1] > 9.99]
    assert tipd[:, 2].mean() > 0.2                       # bending of the variant
    a = cp.SurfaceSolution.read(case.files["base_surface"], mesh)
    b = cp.SurfaceSolution.read(case.files["opt_surface"], mesh)
    c = cp.deform.compare(a, b)
    np.testing.assert_allclose(c["Displacement"], (case.opt.points - case.base.points)[c.global_ids], atol=1e-5)
    assert np.abs(c["delta_Pressure_Coefficient"]).max() > 0.1
    # volume kernels on the deformed coordinates with the baseline topology
    vol = cp.Volume(case.files["base_volume"])
    vol.set_coordinates(pair.xyz_def)
    assert vol.nodal_volume().sum() > 0


def test_shock_box(tmp_path):
    path = tmp_path / "flow.vtu"
    shock_box(path)
    res = cp.shock.lovely_haimes(cp.Volume(path), verbose=False)
    x = res.surface.points
    assert res.surface.n_cells > 100
    assert x[:, 2].min() > -0.8                          # turbulent region filtered out
    full = cp.shock.lovely_haimes(cp.Volume(path), nut_max=1e9, verbose=False).surface
    assert full.points[:, 2].min() < -0.85
    # surface follows the curved shock x = 0.8 + 0.4 (y^2 + z^2) within a couple of cells
    xs = 0.8 + 0.4 * (x[:, 1] ** 2 + x[:, 2] ** 2)
    assert np.median(np.abs(x[:, 0] - xs)) < 0.1


# ------------------------------------------------------------------ example scripts
def _run(args, cwd, mpi=0):
    env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="2",
               PYTHONPATH=str(ROOT / "python") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cmd = [sys.executable, *args]
    if mpi:
        cmd = [*mpirun(mpi, ["PYTHONPATH", "MPLBACKEND", "OMP_NUM_THREADS"]), *cmd]
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=600, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    return p.stdout


def test_examples_run(tmp_path):
    _run(["-m", "cfdpost.synthetic", "demo"], tmp_path)
    d, ex = tmp_path / "demo", ROOT / "examples"
    w = d / "wing"
    out = _run([str(ex / "spanload_compare.py"), str(w / "wing.su2"), str(w / "base/surface_flow.vtu"),
                str(w / "opt/surface_flow.vtu"), "--volume", str(w / "opt/flow.vtu"), "--markers", "wing", "tip",
                "--ref-area", "21", "--farfield-marker", "farfield", "--nbins", "40"], tmp_path)
    assert "AoA = 2.5000" in out and (tmp_path / "spanload_base_opt.png").exists()
    _run([str(ex / "wing_sections.py"), str(w / "base/surface_flow.vtu"), str(w / "opt/surface_flow.vtu"),
          "--semispan", "10"], tmp_path)
    assert (tmp_path / "sections.png").exists()
    _run([str(ex / "optimization_compare.py"), str(w / "wing.su2"), str(w / "wing_opt.su2"),
          str(w / "base/surface_flow.vtu"), str(w / "opt/surface_flow.vtu"), "--markers", "wing", "tip"], tmp_path)
    assert (tmp_path / "solution_delta.vtu").exists()
    _run([str(ex / "shock_surface.py"), str(d / "shock_box/flow.vtu"), "--slice-y", "0"], tmp_path)
    assert (tmp_path / "shock_surface.vtp").exists()
    out = _run(["-m", "cfdpost", "loads", str(w / "base/surface_flow.vtu"), "--mesh", str(w / "wing.su2"),
                "--markers", "wing", "tip", "--ref-area", "21", "--aoa", "2.5"], tmp_path)
    assert "CL=" in out


@pytest.mark.mpi
@pytest.mark.skipif(not have_mpirun(), reason="no mpirun")
def test_examples_mpi(tmp_path):
    _run(["-m", "cfdpost.synthetic", "demo"], tmp_path)
    d, ex = tmp_path / "demo", ROOT / "examples"
    w = d / "wing"
    out = _run([str(ex / "batch_cases.py"), str(w / "wing.su2"), str(w / "base"), str(w / "opt"),
                "--markers", "wing", "tip", "--ref-area", "21", "--farfield-marker", "farfield"], tmp_path, mpi=3)
    assert "case=base" in out and "case=opt" in out
    _run([str(ex / "shock_surface.py"), str(d / "shock_box/flow.vtu")], tmp_path, mpi=3)
