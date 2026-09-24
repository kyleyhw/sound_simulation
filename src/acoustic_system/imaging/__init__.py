"""Physics-based room imaging without machine learning (plan Phase 6).

Modules
-------
``targets``
    Observable targets from a room mask and device poses: illuminated
    boundary (6.1.1), room-outline polygons (6.1.2), signed distance
    fields (6.1.3).
``ir``
    Chirp deconvolution (Wiener / Tikhonov), the analytic 2D free-field
    Green's function, and the engine's empty-room background (6.2.1).
``image_source``
    Echo picking, echo-ellipse evidence and free-space carving (6.2.2).
``backprojection``
    Synthetic-aperture delay-and-sum migration over all poses (6.2.3).
``time_reversal``
    Re-emission of reversed residuals and the zero-lag imaging condition,
    the adjoint of the linearised forward map (6.2.4).
``fwi``
    Full-waveform inversion of an occupancy map with ``TorchFDTD`` (6.2.5,
    proof of concept).
``crlb``
    Cramér-Rao bounds for echo range and TDOA bearing, and the laptop
    design chart (6.4).
``room_params``
    T60 (Schroeder), direct-to-reverberant ratio and Eyring absorption
    (6.5).
``pipeline``
    Runs every imager on one archive room.

See ``docs/imaging.md`` for the maths and the evaluation results.
"""
