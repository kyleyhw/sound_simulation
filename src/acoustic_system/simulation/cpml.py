"""Convolutional perfectly matched layer for the second-order wave equation
(plan 5.4.2).

The sponge (``boundary="sponge"``) damps ``p_t`` inside a graded layer.
It is not matched, so it reflects at oblique incidence (about -12 dB at
60 degrees; see ``scripts/verify_physics.py``). A PML stretches each
coordinate into the complex plane, :math:`\\partial_a \\to s_a^{-1}\\partial_a`
with :math:`s_a = 1 + d_a/(\\alpha_a + j\\omega)`. The layer is then
reflectionless at every angle in the continuum, and only the
discretisation and the finite layer leave a residue.

For :math:`p_{tt} = c^2 \\nabla^2 p` the convolutional form
(Pasalic & McGarry 2010) needs two memory variables per stretched axis:

.. math::
    p_{tt} = c^2 \\sum_a \\bigl(\\partial_a^2 p + \\partial_a \\psi_a + \\zeta_a\\bigr),

    \\psi_a^n = b_a \\psi_a^{n-1} + a_a (\\partial_a p)^n
        \\quad\\text{(on half points)},

    \\zeta_a^n = b_a \\zeta_a^{n-1} + a_a (\\partial_a^2 p + \\partial_a \\psi_a)^n
        \\quad\\text{(on nodes)},

with :math:`b = e^{-(d+\\alpha)\\Delta t}` and
:math:`a = d(b-1)/(d+\\alpha)`. The damping profile is
:math:`d(x) = d_0 (x/L)^2`, :math:`d_0 = -3 c \\ln R / (2L)`, for a
theoretical normal-incidence reflection R. The frequency shift
:math:`\\alpha` falls linearly from :math:`\\alpha_0` at the interface to
0 at the outer edge. It suppresses the low-frequency and grazing
growth of the plain PML.

In index units (derivatives are neighbour differences), the extra term
added to the leap-frog right-hand side is
:math:`C \\sum_a (\\psi_a[i+\\tfrac12] - \\psi_a[i-\\tfrac12] + \\zeta_a[i])`,
where :math:`C = (c\\Delta t/\\Delta x)^2` is the same per-cell
coefficient as the Laplacian. The layer's outermost cell is held at
p = 0. Only the slabs next to CPML faces are touched each step, and the
same array code runs on NumPy or CuPy.

Walls inside the layer: faces touching a rigid or impedance cell carry no
flux in the kernel's Laplacian (the neighbour drops out of K). The CPML
uses the same convention. Its differences across such faces are zeroed
(``set_walls``), so the stretched operator stays consistent with the
kernel and stable. Impedance walls act as rigid inside the layer.
Pressure-release cells are ordinary nodes with p = 0.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class CPML:
    """State and per-step update of the CPML on the faces flagged in ``faces``.

    Parameters
    ----------
    shape
        Grid shape (2D or 3D).
    faces
        One bool per face, in the order (axis 0 low, axis 0 high, axis 1 low, ...).
    cells
        Layer thickness L in cells, including the held outer cell.
    c, dx, dt
        Wave speed, grid step and time step in the engine's units.
    r0
        Theoretical normal-incidence reflection of the layer.
    alpha0
        CFS frequency shift at the interface, in radians per time unit. The
        default is pi * f_ref with f_ref = c / (20 dx), a frequency resolved
        by 20 cells per wavelength.
    xp
        numpy or cupy.
    """

    def __init__(
        self,
        shape: Sequence[int],
        faces: Sequence[bool],
        cells: int,
        c: float,
        dx: float,
        dt: float,
        r0: float = 1e-5,
        alpha0: float | None = None,
        xp: Any = np,
    ) -> None:
        self.shape = tuple(int(n) for n in shape)
        self.dims = len(self.shape)
        self.faces = tuple(bool(f) for f in faces)
        self.L = max(2, int(cells))
        self.xp = xp
        L = self.L
        d0 = -3.0 * c * np.log(r0) / (2.0 * L * dx)
        a0 = np.pi * c / (20.0 * dx) if alpha0 is None else float(alpha0)

        def coeffs(dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            f = np.clip(dist / L, 0.0, 1.0)
            d = d0 * f**2
            al = a0 * (1.0 - f)
            b = np.exp(-(d + al) * dt)
            a = np.where(d > 0, d * (b - 1.0) / np.maximum(d + al, 1e-30), 0.0)
            b = np.where(d > 0, b, 1.0)
            return a.astype(np.float32), b.astype(np.float32)

        # Per axis: 1D profiles on nodes (n) and half points (n - 1).
        # Distance into the layer measured from the interface; the outer
        # (held) node is at depth L - 1 + ... so the layer spans L cells.
        self.axes: list[dict] = []
        for ax, n in enumerate(self.shape):
            lo, hi = self.faces[2 * ax], self.faces[2 * ax + 1]
            if not (lo or hi):
                self.axes.append({})
                continue
            node = np.arange(n, dtype=np.float64)
            half = node[:-1] + 0.5
            dn = np.zeros(n)
            dh = np.zeros(n - 1)
            # Layer occupies nodes 0..L-1 (low) and n-L..n-1 (high).
            if lo:
                dn = np.maximum(dn, (L - 1) - node)
                dh = np.maximum(dh, (L - 1) - half)
            if hi:
                dn = np.maximum(dn, node - (n - L))
                dh = np.maximum(dh, half - (n - L))
            an, bn = coeffs(dn * L / (L - 1))
            ah, bh = coeffs(dh * L / (L - 1))
            sh_n = [1] * self.dims
            sh_n[ax] = n
            sh_h = [1] * self.dims
            sh_h[ax] = n - 1
            hshape = list(self.shape)
            hshape[ax] = n - 1
            windows = []  # node windows [w0, w1) covering the layer + 1 halo node
            if lo:
                windows.append((0, min(n, L + 1)))
            if hi:
                windows.append((max(0, n - L - 1), n))
            self.axes.append(
                {
                    "an": xp.asarray(an.reshape(sh_n)),
                    "bn": xp.asarray(bn.reshape(sh_n)),
                    "ah": xp.asarray(ah.reshape(sh_h)),
                    "bh": xp.asarray(bh.reshape(sh_h)),
                    "psi": xp.zeros(hshape, dtype=np.float32),
                    "zeta": xp.zeros(self.shape, dtype=np.float32),
                    "windows": windows,
                }
            )
        self.ext = xp.zeros(self.shape, dtype=np.float32)
        self._open: list = [None] * self.dims  # per-axis half-point masks (None = all open)
        self._walls_for: object = None  # coefficient set the masks were built for

    def set_walls(self, wall: "np.ndarray | None") -> None:
        """Mark no-flux cells (rigid/impedance walls); ``None`` clears them."""
        for ax, st in enumerate(self.axes):
            if not st or wall is None or not np.any(wall):
                self._open[ax] = None
                continue
            n = self.shape[ax]
            lo = np.take(wall, range(0, n - 1), axis=ax)
            hi = np.take(wall, range(1, n), axis=ax)
            self._open[ax] = self.xp.asarray((~(lo | hi)).astype(np.float32))

    def reset(self) -> None:
        for st in self.axes:
            if st:
                st["psi"].fill(0.0)
                st["zeta"].fill(0.0)
        self.ext.fill(0.0)

    def _sl(self, ax: int, a: int, b: int) -> tuple:
        s: list[slice] = [slice(None)] * self.dims
        s[ax] = slice(a, b)
        return tuple(s)

    def update(self, p: Any) -> Any:
        """Advance the memory variables with p^n; return the extra term (no C)."""
        ext = self.ext
        for ax, st in enumerate(self.axes):
            if st:
                for w0, w1 in st["windows"]:
                    ext[self._sl(ax, w0, w1)] = 0.0
        for ax, st in enumerate(self.axes):
            if not st:
                continue
            sl = self._sl
            for w0, w1 in st["windows"]:
                # Half points h in [w0, w1 - 1): psi_h from p[h+1] - p[h].
                ph = sl(ax, w0, w1 - 1)
                dp = p[sl(ax, w0 + 1, w1)] - p[sl(ax, w0, w1 - 1)]
                if self._open[ax] is not None:
                    dp = dp * self._open[ax][ph]
                psi = st["psi"]
                psi[ph] = st["bh"][sl(ax, w0, w1 - 1)] * psi[ph] + st["ah"][sl(ax, w0, w1 - 1)] * dp
                # Interior nodes i in [w0 + 1, w1 - 1): d2 = dp[i] - dp[i - 1].
                ni = sl(ax, w0 + 1, w1 - 1)
                d2 = dp[sl(ax, 1, w1 - w0 - 1)] - dp[sl(ax, 0, w1 - w0 - 2)]
                dpsi = psi[sl(ax, w0 + 1, w1 - 1)] - psi[sl(ax, w0, w1 - 2)]
                z = st["zeta"]
                z[ni] = st["bn"][ni] * z[ni] + st["an"][ni] * (d2 + dpsi)
                ext[ni] += dpsi + z[ni]
        return ext


__all__ = ["CPML"]
