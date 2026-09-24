"""Sound-field control from simulated transfer functions (plan Phase 7).

Modules (see ``docs/control.md``):

* ``transfer``: rooms in SI units, running the engine with arbitrary
  drives, and speaker-to-point transfer functions (7.1.1).
* ``metrics``: acoustic contrast, reproduction error, array effort (7.1.2).
* ``beamforming``: delay-and-sum, pressure matching, acoustic contrast
  control, time reversal, and broadband FIR design verified in the time
  domain (7.2).
* ``ctc``: two-speaker crosstalk cancellation with head tracking (7.3).
* ``anc``: FxLMS noise cancellation against the engine and quiet-zone
  size (7.4).
* ``requirements``: design in an estimated room, score in the true room
  (7.5).
* ``differentiable``: drive signals optimised through the tensor engine
  (7.6). It imports torch, so it is not imported here.
"""
