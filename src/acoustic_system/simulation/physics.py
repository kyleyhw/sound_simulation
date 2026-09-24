"""General-path physics for ``Simulate`` (plan Phase 5).

The fast path (fused kernels in ``calculate.py``) handles the legacy model:
pressure-release (Dirichlet, p = 0) outer walls and obstacles, uniform
wave speed. Everything else runs through the general kernel here:

* **Rigid walls** (Neumann, :math:`\\partial p/\\partial n = 0`,
  reflection +1): staircase boundaries. The ghost value across a rigid face
  equals the cell's own pressure, so the face drops out of the Laplacian.
* **Locally reacting impedance walls**: the wall's normal particle velocity
  obeys a spring-damper law :math:`R v + K_s x = p` with :math:`\\dot x = v`,
  i.e. a specific impedance :math:`Z(\\omega) = R + K_s/(j\\omega)`. With
  :math:`K_s = 0` this is a frequency-independent wall with normalised
  admittance :math:`\\beta = \\rho c / R` and normal-incidence reflection
  :math:`(1-\\beta)/(1+\\beta)`. With :math:`K_s > 0` the admittance
  :math:`\\beta(\\omega) = \\beta_\\infty\\, j\\omega\\tau/(1 + j\\omega\\tau)`,
  :math:`\\tau = R/K_s`, is high-pass: reflective below
  :math:`f_c = 1/(2\\pi\\tau)`, absorbing above (a porous absorber or
  curtain).
* **Absorbing outer boundaries**: Engquist-Majda/Mur first order, or a
  graded damping layer :math:`p_{tt} + 2\\sigma p_t = c^2 \\nabla^2 p`
  with a cubic profile.
* **Heterogeneous media**: a relative wave-speed map :math:`c(x)/c_0`.

Discretisation (one cell, air, with :math:`M` faces touching
impedance walls): the ghost across each impedance face carries
:math:`\\partial p/\\partial n = -\\rho\\,\\partial v/\\partial t`, and the
wall branch is integrated with the trapezoidal rule on half steps,

.. math::
    (R + K_s \\Delta t/2)\\, v^{n+\\frac12} = \\tfrac12 (p^{n+1} + p^n) - K_s x^n,
    \\qquad x^{n+1} = x^n + \\Delta t\\, v^{n+\\frac12}.

Eliminating :math:`v^{n+\\frac12}` gives an explicit update:

.. math::
    \\Bigl(1 + \\tfrac{q}{2} + s\\Bigr) p^{n+1} =
      2p^n - p^{n-1} + C\\,\\mathcal{L} + s\\,p^{n-1}
      - q\\,\\bigl(\\tfrac12 p^n - K_s x^n\\bigr) + Q\\,v^{n-\\frac12}

where :math:`\\mathcal{L} = \\sum_{\\text{nbrs}} p_n - K p_c`, with
:math:`K` the number of air or pressure-release neighbours (wall cells
store p = 0). Also :math:`C = (c(x)\\Delta t/\\Delta x)^2`,
:math:`Q = \\rho c\\lambda M`, :math:`q = Q/(R + K_s\\Delta t/2)`, and
:math:`s = \\sigma\\Delta t` inside the absorbing layer. With
:math:`K_s = 0` this reduces exactly to the frequency-independent
centred scheme :math:`(1+g)p^{n+1} = 2p^n + C\\mathcal{L} - (1-g)p^{n-1}`,
:math:`g = \\lambda\\beta M/2`. The browser engine
(``web/src/engine/simulation.ts``) implements the same equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
from numba import njit, prange


@dataclass(frozen=True)
class Material:
    name: str
    kind: str  # "soft" | "rigid" | "absorb"
    beta: float = 0.0  # high-frequency normalised admittance rho c / R
    fc_rel: float = 0.0  # high-pass corner f_c * dx / c (0 = frequency-independent)


# Material table shared with the browser engine (same ids).
MATERIALS: tuple[Material, ...] = (
    Material("air", "air"),
    Material("soft (p = 0)", "soft"),
    Material("rigid", "rigid"),
    Material("plaster", "absorb", 0.1),
    Material("wood", "absorb", 0.3),
    Material("absorber", "absorb", 1.0),
    Material("curtain", "absorb", 1.0, 0.03),
)
SOFT, RIGID = 1, 2


def absorption_from_beta(beta: float) -> float:
    """Normal-incidence energy absorption coefficient for admittance beta."""
    r = (1.0 - beta) / (1.0 + beta)
    return 1.0 - r * r


OUTER_KINDS = ("soft", "rigid", "absorb", "mur", "sponge")


def sponge_sigma(
    shape: tuple[int, ...], cells: int, c: float, dx: float, r0: float = 1e-4
) -> np.ndarray:
    """Cubic damping profile sigma(d) = sigma_max ((L-d)/L)^3 in the outer layer.

    sigma_max is set for a theoretical round-trip reflection r0 at normal
    incidence: r = exp(-2 int sigma/c dx) = exp(-2 sigma_max L dx / (4 c)).
    """
    lcells = max(1, int(cells))
    sigma_max = -4.0 * np.log(r0) * c / (2.0 * lcells * dx)
    d = np.full(shape, np.inf)
    for a, n in enumerate(shape):
        idx = np.arange(n)
        dist = np.minimum(idx, n - 1 - idx).astype(float)
        sh = [1] * len(shape)
        sh[a] = n
        d = np.minimum(d, dist.reshape(sh))
    f = np.clip((lcells - d) / lcells, 0.0, 1.0)
    return (sigma_max * f**3).astype(np.float32)


@dataclass
class GeneralCoefficients:
    active: np.ndarray  # uint8, 0 = held at p = 0
    k_air: np.ndarray  # uint8 neighbours contributing -p_c
    c2: np.ndarray  # float32 (c(x) dt/dx)^2
    s: np.ndarray  # float32 sigma dt (damping layer)
    qq: np.ndarray  # float32 Q (rho c lambda M)
    qa: np.ndarray  # float32 Q / a
    inv_a: np.ndarray  # float32 1 / a
    ks: np.ndarray  # float32 spring constant K_s
    mur_edges: bool


def build_coefficients(
    material: np.ndarray,
    speed: np.ndarray | None,
    outer: str,
    outer_beta: float,
    sponge_cells: int,
    coeff: float,
    c: float,
    dx: float,
    dt: float,
) -> GeneralCoefficients:
    """Vectorised per-cell coefficients for the general kernel (2D and 3D)."""
    if outer not in OUTER_KINDS:
        raise ValueError(f"outer must be one of {OUTER_KINDS}, got {outer!r}")
    shape = material.shape
    dims = material.ndim
    lam = float(np.sqrt(coeff))
    rho = 1.0
    # Per-material lookup arrays.
    kinds = np.array([{"air": 0, "soft": 1, "rigid": 2, "absorb": 3}[m.kind] for m in MATERIALS])
    betas = np.array([m.beta for m in MATERIALS])
    fcs = np.array([m.fc_rel for m in MATERIALS])
    mat = np.clip(material.astype(np.int64), 0, len(MATERIALS) - 1)

    edge = np.zeros(shape, dtype=bool)
    for a in range(dims):
        sl: list[slice | int] = [slice(None)] * dims
        sl[a] = 0
        edge[tuple(sl)] = True
        sl[a] = shape[a] - 1
        edge[tuple(sl)] = True
    held_edges = outer in ("soft", "sponge", "mur")
    active = (mat == 0) & ~(edge & held_edges)

    # Neighbour classification. Pad with a sentinel for out-of-domain faces.
    outer_code = -1
    padded = np.pad(mat, 1, constant_values=outer_code)
    k_air = np.zeros(shape, dtype=np.int64)
    beta_sum = np.zeros(shape, dtype=np.float64)
    beta_rep = np.zeros(shape, dtype=np.float64)
    fc_rep = np.zeros(shape, dtype=np.float64)
    for a in range(dims):
        for sgn in (-1, 1):
            sl = [slice(1, n + 1) for n in shape]
            sl[a] = slice(1 + sgn, shape[a] + 1 + sgn)
            nb = padded[tuple(sl)]
            is_outer = nb == outer_code
            nb_kind = np.where(is_outer, 0, kinds[np.clip(nb, 0, None)])
            contributes = (~is_outer) & ((nb_kind == 0) | (nb_kind == 1))
            if outer in ("soft", "sponge", "mur"):
                contributes |= is_outer  # never reached by active cells
            k_air += contributes
            wall_face = (~is_outer) & ((nb_kind == 2) | (nb_kind == 3))
            if outer in ("rigid", "absorb"):
                wall_face |= is_outer
            nb_beta = np.where(
                is_outer, outer_beta if outer == "absorb" else 0.0, betas[np.clip(nb, 0, None)]
            )
            nb_fc = np.where(is_outer, 0.0, fcs[np.clip(nb, 0, None)])
            beta_sum += np.where(wall_face, nb_beta, 0.0)
            upd = wall_face & (nb_beta > beta_rep)
            beta_rep = np.where(upd, nb_beta, beta_rep)
            fc_rep = np.where(upd, nb_fc, fc_rep)

    r = speed.astype(np.float64) if speed is not None else np.ones(shape)
    c_local = c * r
    c2 = coeff * r * r
    # Impedance branch: R = rho c / beta, K_s = R * 2 pi f_c (f_c in 1/time).
    # Rigid faces (beta = 0) drop out of the Laplacian and add no damping;
    # impedance faces combine through the admittance sum. The branch uses the
    # most absorbing face's material for R and K_s, scaled by an effective
    # face count M = sum(beta) / beta_rep so that, for K_s = 0, q / 2 equals
    # lambda * r * sum(beta_f) / 2 exactly.
    has_branch = (beta_sum > 0) & active
    m_eff = np.where(has_branch, beta_sum / np.maximum(beta_rep, 1e-12), 0.0)
    R = np.where(has_branch, rho * c_local / np.maximum(beta_rep, 1e-12), 1.0)
    fc_abs = fc_rep * c / dx
    ks = np.where(has_branch, R * 2.0 * np.pi * fc_abs, 0.0)
    a_coef = R + ks * dt / 2.0
    qq = np.where(has_branch, rho * c_local * lam * r * m_eff, 0.0)
    qa = np.where(has_branch, qq / a_coef, 0.0)
    s = sponge_sigma(shape, sponge_cells, c, dx) * dt if outer == "sponge" else np.zeros(shape)
    return GeneralCoefficients(
        active=active.astype(np.uint8),
        k_air=k_air.astype(np.uint8),
        c2=c2.astype(np.float32),
        s=np.asarray(s, dtype=np.float32),
        qq=qq.astype(np.float32),
        qa=qa.astype(np.float32),
        inv_a=np.where(has_branch, 1.0 / a_coef, 0.0).astype(np.float32),
        ks=ks.astype(np.float32),
        mur_edges=outer == "mur",
    )


@njit(parallel=True, fastmath=False, cache=True)
def general_step_2d(
    p, pp, pn, active, k_air, c2, s, qq, qa, inv_a, ks, v, x, dt
):  # pragma: no cover - numba
    ni, nj = p.shape
    for i in prange(ni):  # ty: ignore[not-iterable]
        for j in range(nj):
            if active[i, j] == 0:
                pn[i, j] = 0.0
                continue
            pc = p[i, j]
            acc = 0.0
            if i > 0:
                acc += p[i - 1, j]
            if i < ni - 1:
                acc += p[i + 1, j]
            if j > 0:
                acc += p[i, j - 1]
            if j < nj - 1:
                acc += p[i, j + 1]
            lap = acc - k_air[i, j] * pc
            sd = s[i, j]
            rhs = 2.0 * pc - pp[i, j] + c2[i, j] * lap + sd * pp[i, j]
            q = qa[i, j]
            if q != 0.0:
                rhs += -q * (0.5 * pc - ks[i, j] * x[i, j]) + qq[i, j] * v[i, j]
                nxt = rhs / (1.0 + 0.5 * q + sd)
                vn = (0.5 * (nxt + pc) - ks[i, j] * x[i, j]) * inv_a[i, j]
                x[i, j] += dt * vn
                v[i, j] = vn
                pn[i, j] = nxt
            else:
                pn[i, j] = rhs / (1.0 + sd)


@njit(parallel=True, fastmath=False, cache=True)
def general_step_3d(
    p, pp, pn, active, k_air, c2, s, qq, qa, inv_a, ks, v, x, dt
):  # pragma: no cover - numba
    ni, nj, nk = p.shape
    for i in prange(ni):  # ty: ignore[not-iterable]
        for j in range(nj):
            for k in range(nk):
                if active[i, j, k] == 0:
                    pn[i, j, k] = 0.0
                    continue
                pc = p[i, j, k]
                acc = 0.0
                if i > 0:
                    acc += p[i - 1, j, k]
                if i < ni - 1:
                    acc += p[i + 1, j, k]
                if j > 0:
                    acc += p[i, j - 1, k]
                if j < nj - 1:
                    acc += p[i, j + 1, k]
                if k > 0:
                    acc += p[i, j, k - 1]
                if k < nk - 1:
                    acc += p[i, j, k + 1]
                lap = acc - k_air[i, j, k] * pc
                sd = s[i, j, k]
                rhs = 2.0 * pc - pp[i, j, k] + c2[i, j, k] * lap + sd * pp[i, j, k]
                q = qa[i, j, k]
                if q != 0.0:
                    rhs += -q * (0.5 * pc - ks[i, j, k] * x[i, j, k]) + qq[i, j, k] * v[i, j, k]
                    nxt = rhs / (1.0 + 0.5 * q + sd)
                    vn = (0.5 * (nxt + pc) - ks[i, j, k] * x[i, j, k]) * inv_a[i, j, k]
                    x[i, j, k] += dt * vn
                    v[i, j, k] = vn
                    pn[i, j, k] = nxt
                else:
                    pn[i, j, k] = rhs / (1.0 + sd)


def mur_edges(p: np.ndarray, pn: np.ndarray, lam: float) -> None:
    """First-order Engquist-Majda (Mur) absorbing condition on every face.

    For the face cell 0 with inward neighbour 1 (normal incidence exact):
        p_0^{n+1} = p_1^n + (lam - 1)/(lam + 1) * (p_1^{n+1} - p_0^n).
    Corners/edges take the last face written (all faces agree to O(dx)).
    """
    k = (lam - 1.0) / (lam + 1.0)
    for a in range(p.ndim):
        n = p.shape[a]
        for face, inner in ((0, 1), (n - 1, n - 2)):
            f: list[slice | int] = [slice(None)] * p.ndim
            g: list[slice | int] = [slice(None)] * p.ndim
            f[a] = face
            g[a] = inner
            ft, gt = tuple(f), tuple(g)
            pn[ft] = p[gt] + k * (pn[gt] - p[ft])


__all__ = [
    "MATERIALS",
    "Material",
    "OUTER_KINDS",
    "absorption_from_beta",
    "build_coefficients",
    "general_step_2d",
    "general_step_3d",
    "mur_edges",
    "sponge_sigma",
    "numba",
]
