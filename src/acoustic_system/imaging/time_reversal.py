"""Time-reversal imaging of obstacles (plan Task 6.2.4).

The residual :math:`r_m[n] = y_m[n] - y^{(0)}_m[n]` between the recording
and the empty-room response (``ir.empty_room_response``) is the field
scattered by the interior obstacles. Time-reversal imaging re-emits the
reversed residuals from the mic positions in the *background* medium
(the empty box) and correlates the back-propagated field with the
incident field of the source. This is reverse-time migration (RTM), and
with the zero-lag imaging condition below it is exactly the adjoint of the
linearised forward map, i.e. the first gradient step of full-waveform
inversion (``fwi.py``).

Derivation on the discrete engine
---------------------------------
Obstacles enter ``Simulate`` as the scrub :math:`p^{n+1}(\\mathbf{x})
\\leftarrow (1 - m(\\mathbf{x}))\\,p^{n+1}(\\mathbf{x})` before injection.
Linearising about the empty room (:math:`m = 0`, incident field
:math:`P^{n} \\equiv p^{n+1}_\\text{inc}`), an occupancy perturbation
:math:`m` acts as a secondary soft source :math:`-m(\\mathbf{x})
P^n(\\mathbf{x})` at step :math:`n`, so

.. math::
    \\delta y_m[n'] = -\\sum_{\\mathbf{x}} m(\\mathbf{x}) \\sum_n P^n(\\mathbf{x})
        \\, g_{\\mathbf{x} \\to m}[n' - n].

The misfit :math:`J = \\tfrac12 \\sum_{m,n} (r_m[n] - \\delta y_m[n])^2`
has steepest-descent direction at :math:`m = 0`

.. math::
    I(\\mathbf{x}) = -\\frac{\\partial J}{\\partial m(\\mathbf{x})}
      = -\\sum_n P^n(\\mathbf{x})\\, Q^n(\\mathbf{x}),
    \\qquad Q^n(\\mathbf{x}) = \\sum_{m, j} r_m[n + j]\\, g_{m \\to \\mathbf{x}}[j],

using reciprocity :math:`g_{\\mathbf{x} \\to m} = g_{m \\to \\mathbf{x}}` of
the symmetric discrete Laplacian. :math:`Q` is computed by one engine run:
inject :math:`\\tilde r_m[k] = r_m[T-1-k]` at every mic and record the field
:math:`B^k`; then :math:`Q^n = B^{T-1-n}`. Positive :math:`I` marks cells
where adding a soft obstacle would explain the residual.

Illumination compensation divides by the source energy
:math:`\\sum_n (P^n)^2 + \\epsilon`, which removes the bright halo around
the source (a standard deconvolution imaging condition).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..simulation.setup import Driver
from ..simulation.simulate import Simulate
from .ir import SampledWaveform, empty_room_response


def backpropagate(
    grid_shape: tuple[int, int],
    mic_pos: ArrayLike,
    residual: NDArray,
    courant: float = 0.5,
) -> NDArray[np.float32]:
    """Re-emit time-reversed residuals from the mics; return :math:`Q^n`.

    ``residual`` has shape ``(n_mics, T)``. The returned array has shape
    ``(T, *grid)`` and is already re-reversed, so ``Q[n]`` is aligned with
    the incident field ``P[n]`` of :func:`ir.empty_room_response`.
    """
    residual = np.asarray(residual, dtype=np.float64)
    T = residual.shape[-1]
    sim = Simulate(grid_shape=grid_shape, courant=courant)
    sim.set_drivers(
        [
            Driver(
                position=tuple(int(c) for c in m),
                waveform=SampledWaveform(r[::-1].copy(), sim.timestep),
            )
            for m, r in zip(np.asarray(mic_pos).reshape(-1, len(grid_shape)), residual)
        ]
    )
    B = np.empty((T,) + tuple(grid_shape), dtype=np.float32)
    for k in range(T):
        sim.step()
        B[k] = sim.p
    return B[::-1]


def imaging_condition(
    P: NDArray, Q: NDArray, normalise: bool = True, eps_rel: float = 1e-3
) -> NDArray[np.float64]:
    """Zero-lag cross-correlation :math:`-\\sum_n P^n Q^n` (optionally / illumination)."""
    img = -np.einsum("nij,nij->ij", P.astype(np.float64), Q.astype(np.float64))
    if normalise:
        illum = np.einsum("nij,nij->ij", P.astype(np.float64), P.astype(np.float64))
        img = img / (illum + eps_rel * illum.max())
    return img


def rtm_pose(
    grid_shape: tuple[int, int],
    source_pos: ArrayLike,
    mic_pos: ArrayLike,
    drive: NDArray,
    recording: NDArray,
    courant: float = 0.5,
    normalise: bool = True,
    time_window: NDArray | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Time-reversal image of one pose.

    Parameters
    ----------
    recording
        ``(n_mics, T)`` recorded pressure (channel-first).
    time_window
        Optional ``(T,)`` weight applied to the residual before re-emission
        (e.g. a taper that down-weights late, multiply scattered energy).

    Returns
    -------
    image, illumination
        The imaging condition and the source illumination
        :math:`\\sum_n (P^n)^2` (useful to combine poses).
    """
    recording = np.asarray(recording, dtype=np.float64)
    y0, P = empty_room_response(grid_shape, source_pos, drive, mic_pos, courant, return_field=True)
    assert P is not None
    residual = recording - y0
    if time_window is not None:
        residual = residual * np.asarray(time_window)[None, :]
    Q = backpropagate(grid_shape, mic_pos, residual, courant)
    raw = imaging_condition(P, Q, normalise=False)
    illum = np.einsum("nij,nij->ij", P.astype(np.float64), P.astype(np.float64))
    if normalise:
        raw = raw / (illum + 1e-3 * illum.max())
    return raw, illum


def rtm_room(
    grid_shape: tuple[int, int],
    source_positions: NDArray,
    mic_positions: NDArray,
    drive: NDArray,
    recordings: NDArray,
    courant: float = 0.5,
    normalise: bool = True,
    time_window: NDArray | None = None,
) -> NDArray[np.float64]:
    """Stack the per-pose time-reversal images of one room.

    ``recordings`` has the archive layout ``(K, T, n_mics)``. Each pose image
    is scaled by its RMS before summing so a pose with a loud residual
    (source next to an obstacle) does not dominate the stack.
    """
    out = np.zeros(grid_shape, dtype=np.float64)
    for k in range(len(source_positions)):
        img, _ = rtm_pose(
            grid_shape,
            source_positions[k],
            mic_positions[k],
            drive,
            np.asarray(recordings[k]).T,
            courant,
            normalise,
            time_window,
        )
        rms = float(np.sqrt(np.mean(img**2)))
        out += img / (rms + 1e-30)
    return out


__all__ = ["backpropagate", "imaging_condition", "rtm_pose", "rtm_room"]
