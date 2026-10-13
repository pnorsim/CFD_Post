"""Shock detection.

Lovely & Haimes (1999): the shock is the surface where the Mach number normal
to the pressure gradient crosses one,

    Mn = (V / a) . grad p / |grad p| = 1,      compression: V . grad p > 0,

filtered to remove weak-gradient noise (|grad p| > factor x global median) and
the viscous boundary layer / wake (mu_t / mu < limit).

Additional point sensors: Ducros, normalised pressure gradient (|grad p| h / p),
density-gradient magnitude (numerical schlieren). All functions take a
:class:`~cfdpost.volume.Volume` (or a .vtu path) and work in serial or MPI.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import fields as F
from .grid import Grid
from .volume import Volume

# preset: (Mn iso level, |grad p| factor x median, max mu_t/mu)
PRESETS = {
    "default": (1.0, 3.0, 1000.0),
    "inclusive": (1.0, 1.5, 1000.0),      # 'is any shock left?'
    "conservative": (1.05, 5.0, 100.0),   # 'a shock is certainly here'
}

SU2_NAMES = {"pressure": "Pressure", "density": "Density", "velocity": "Velocity", "mach": "Mach",
             "mu_t": "Eddy_Viscosity", "mu": "Laminar_Viscosity"}


def _volume(vol, comm=None) -> Volume:
    return vol if isinstance(vol, Volume) else Volume(vol, comm)


def _velocity(vol: Volume, names):
    if names["velocity"] in vol.point_arrays:
        return vol.read(names["velocity"])
    if "Momentum" in vol.point_arrays:
        return vol.read("Momentum") / vol.read(names["density"])[:, None]
    raise KeyError("no Velocity or Momentum array in the solution")


@dataclass
class ShockResult:
    surface: Grid                      # filtered shock surface (complete on rank 0)
    raw_surface: Grid                  # Mn iso-surface before filtering
    params: dict
    fields: dict = field(default_factory=dict)   # owned point fields (Mn, gradP_mag, ...)

    def write(self, path, raw: bool = False, comm=None):
        """Write the shock surface (.vtp) from rank 0."""
        from .mpi import get_comm
        if get_comm(comm).is_root:
            (self.raw_surface if raw else self.surface).write(path)


def lovely_haimes(vol, preset: str = "default", mn_level=None, gradp_factor=None, nut_max=None,
                  gamma: float = F.GAMMA, interpolate=("Mach", "Pressure_Coefficient"), names=None,
                  comm=None, verbose: bool = True) -> ShockResult:
    """Lovely-Haimes shock surface.

    vol          : Volume or .vtu path (SU2 volume output)
    preset       : 'default' | 'inclusive' | 'conservative' (see PRESETS)
    mn_level, gradp_factor, nut_max : override the preset values; nut_max=None
                   with no eddy viscosity in the file disables that filter
    interpolate  : extra point arrays carried onto the surface (if present)
    """
    vol = _volume(vol, comm)
    c = vol.comm
    names = {**SU2_NAMES, **(names or {})}
    lv, gf, nm = PRESETS[preset]
    lv = lv if mn_level is None else mn_level
    gf = gf if gradp_factor is None else gradp_factor
    nm = nm if nut_max is None else nut_max

    p = vol.read(names["pressure"])
    rho = vol.read(names["density"])
    V = _velocity(vol, names)
    gp = vol.gradient(p)
    gmag = np.linalg.norm(gp, axis=1)
    a = F.speed_of_sound(p, rho, gamma)
    mn = F.normal_mach(V, gp, a)
    out = {"Mn": mn, "gradP_mag": gmag}
    have_nut = names["mu_t"] in vol.point_arrays and names["mu"] in vol.point_arrays
    if have_nut:
        out["nut_ratio"] = vol.read(names["mu_t"]) / vol.read(names["mu"])
    for n in interpolate:
        if n in vol.point_arrays:
            out[n] = vol.read(n)

    med = c.median(gmag)
    cutoff = gf * med
    iso = vol.contour(mn, lv, out)
    surf = iso.threshold("gradP_mag", lower=cutoff)
    if have_nut and nm is not None:
        surf = surf.threshold("nut_ratio", upper=nm)
    params = {"preset": preset, "mn_level": lv, "gradp_factor": gf, "nut_max": nm if have_nut else None,
              "gradp_median": med, "gradp_cutoff": cutoff, "gamma": gamma}
    if verbose:
        c.print(f"Lovely-Haimes: Mn={lv:g}, |grad p| > {gf:g} x median ({cutoff:.4g}), "
                f"mu_t/mu < {nm if have_nut else 'n/a'} -> {surf.n_cells} of {iso.n_cells} polygons")
    return ShockResult(surf, iso, params, out)


def ducros(vol, names=None, eps: float = 1e-30, comm=None) -> np.ndarray:
    """Ducros sensor (div V)^2 / ((div V)^2 + |curl V|^2), set to 0 in expansions.
    Owned array in [0, 1]; ~1 in shocks, ~0 in vortical regions."""
    vol = _volume(vol, comm)
    names = {**SU2_NAMES, **(names or {})}
    gv = vol.gradient(_velocity(vol, names))
    div = F.divergence(gv)
    w = np.linalg.norm(F.vorticity(gv), axis=1)
    s = div ** 2 / (div ** 2 + w ** 2 + eps)
    return np.where(div < 0, s, 0.0)


def pressure_gradient_sensor(vol, names=None, comm=None) -> np.ndarray:
    """|grad p| h / p with h the local mesh size: the relative pressure jump
    across one cell. Mesh-independent threshold (~0.05-0.2 marks shocks)."""
    vol = _volume(vol, comm)
    names = {**SU2_NAMES, **(names or {})}
    p = vol.read(names["pressure"])
    return np.linalg.norm(vol.gradient(p), axis=1) * vol.length_scale() / p


def density_gradient(vol, names=None, log: bool = False, comm=None) -> np.ndarray:
    """|grad rho| (numerical schlieren); log10 if log=True."""
    vol = _volume(vol, comm)
    names = {**SU2_NAMES, **(names or {})}
    g = np.linalg.norm(vol.gradient(vol.read(names["density"])), axis=1)
    return np.log10(g + 1e-300) if log else g
