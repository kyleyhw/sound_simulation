"""Generate a paired active-sensing dataset for Phase 2.

Each sample is a triplet $(\\mathbf{s}(t),\\, u(t),\\, M)$:

- $\\mathbf{s}(t) \\in \\mathbb{R}^{T \\times M}$: pressure recorded at $M$
  microphones as a function of step index. Defaults to a stereo pair
  ($M = 2$) placed about a random centre with random orientation, mimicking
  the binaural arrangement on a stock laptop.
- $u(t)$: the source-audio waveform injected at the driver.
- $M$ (boolean grid): the obstacle mask of the room.

The CNN target in Task 2.1.2 is the inverse mapping $(\\mathbf{s}, u) \\mapsto M$:
given a multi-channel microphone recording and a reference of the source
audio that was played, reconstruct the room geometry. Producing diverse
$(\\mathbf{s}, u, M)$ triplets is what this script does.

Wire-side layout per HDF5 group (single-pose, ``--poses-per-room 1``,
the default — identical to the original format)::

    /sample_NNNN/
        attrs: grid_shape, wavespeed, timestep, gridstep, courant,
               driver_position, sensor_positions (n_mics, dims),
               n_mics, mic_spacing, audio_path (or 'synthetic'),
               record_step, sim_duration_steps
        sensor    (float32, shape (T_rec, n_mics))   -- channel-last
        source    (float32, shape (N_audio_samples,))
        obstacles (uint8,   shape grid_shape)

Multi-pose layout (``--poses-per-room K`` with K > 1, Task 2.1.4): the
room (obstacle mask) and the source audio are shared across K poses;
each pose is an independent (driver, mic-pair) placement — the physical
picture is a laptop carried to K spots in the same room, chirping and
recording at each. Additional/changed fields::

    /sample_NNNN/
        attrs: poses_per_room = K,
               driver_positions   (K, dims)          -- one row per pose
               sensor_positions   (K, n_mics, dims)
        sensor    (float32, shape (K, T_rec, n_mics))

The inverse problem then conditions on all K measurements jointly,
$p(M \\mid \\{\\mathbf{s}_k\\}_{k=1}^K, u)$: each pose constrains the
map from a different geometric vantage, which is the information the
single-pose experiments (tests/reports/training_2026_05_14*.md) showed
was missing.

Usage::

    python scripts/generate_active_sensing.py \\
        --output dataset.hdf5 \\
        --num-samples 100 \\
        --audio-dir /path/to/wavs \\
        --grid 200 \\
        --duration 800 \\
        --n-mics 2 \\
        --mic-spacing 16

If ``--audio-dir`` is omitted (or contains no ``*.wav`` files), a linear
chirp from ``f_start`` to ``f_end`` is synthesised per sample so the
pipeline remains end-to-end runnable without external assets.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Optional

import h5py
import numpy as np

# Make the package importable when running from the repo root.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from acoustic_system.simulation.dataset import (  # noqa: E402
    generate_random_obstacles,
    pick_mic_positions,
    random_free_position,
    run_with_sensors,
    synthetic_chirp,
)
from acoustic_system.simulation.setup import Driver, Sensor  # noqa: E402
from acoustic_system.simulation.simulate import Simulate  # noqa: E402
from acoustic_system.simulation.waveforms import AudioFileWaveform  # noqa: E402


def list_audio_files(audio_dir: Optional[str]) -> list[pathlib.Path]:
    if not audio_dir:
        return []
    p = pathlib.Path(audio_dir)
    if not p.is_dir():
        return []
    return sorted(p.rglob("*.wav"))


def build_source(
    audio_files: list[pathlib.Path],
    rng: np.random.Generator,
    sim_dt: float,
    sim_duration_steps: int,
    args: argparse.Namespace,
) -> tuple[AudioFileWaveform, np.ndarray, float, str]:
    """Pick (or synthesise) an audio source for this sample.

    Returns
    -------
    waveform : AudioFileWaveform
        Ready to be wrapped in a ``Driver``.
    samples : np.ndarray
        The raw float32 sample buffer that will be saved alongside the
        sensor recording — this is the "$u(t)$" reference the CNN
        receives at training time.
    sample_rate : float
        Native audio sample rate (in physical seconds). Saved to attrs.
    label : str
        Either the resolved WAV path or the literal "synthetic".
    """
    # ``sim_time_per_second = 1.0`` means audio plays at native speed in
    # the dimensionless sandbox (c=1, dx=1). For a physical-units
    # simulation the caller should override --sim-time-per-second.
    if audio_files:
        # rng.choice does not accept a list[pathlib.Path] (none of its
        # numpy-typed overloads cover that signature). Use integer indexing
        # against rng.integers instead — semantically identical, type-clean.
        path = audio_files[int(rng.integers(len(audio_files)))]
        wf = AudioFileWaveform(
            path=str(path),
            amplitude=args.audio_amplitude,
            delay=args.delay,
            sim_time_per_second=args.sim_time_per_second,
        )
        # Pull the loaded samples back out for the on-disk reference copy.
        # `_samples` is private to keep it out of HDF5 attrs in the
        # driver-metadata dump; saving it explicitly here as its own
        # dataset is exactly how it should be used downstream.
        return wf, wf._samples.copy(), wf._native_fs, str(path)

    # Synthetic chirp fallback. Pick a sample count large enough that the
    # source is active for most of the simulation run.
    audio_dur_s = sim_duration_steps * sim_dt
    n_samples = int(args.synth_sample_rate * audio_dur_s)
    samples = synthetic_chirp(
        duration_samples=n_samples,
        sample_rate=args.synth_sample_rate,
        f_start=args.synth_f_start,
        f_end=args.synth_f_end,
        amplitude=1.0,
    )
    wf = AudioFileWaveform.from_samples(
        samples=samples,
        sample_rate=args.synth_sample_rate,
        amplitude=args.audio_amplitude,
        delay=args.delay,
        sim_time_per_second=args.sim_time_per_second,
    )
    return wf, samples, float(args.synth_sample_rate), "synthetic"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        help="HDF5 output path. Created if absent; overwritten if exists.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of triplets to generate."
    )
    parser.add_argument(
        "--audio-dir",
        default=None,
        help="Directory of source WAV files. Missing or empty falls back to synthetic chirps.",
    )
    parser.add_argument("--grid", type=int, default=200, help="Square grid side.")
    parser.add_argument(
        "--duration",
        type=int,
        default=800,
        help="Number of FDTD steps per sample.",
    )
    parser.add_argument(
        "--record-step",
        type=int,
        default=1,
        help="Record every Nth step; the sensor timeseries has duration/record_step samples.",
    )
    parser.add_argument(
        "--n-obstacles",
        type=int,
        default=3,
        help="Number of random axis-aligned rectangles per room.",
    )
    parser.add_argument("--obstacle-min", type=int, default=5)
    parser.add_argument("--obstacle-max", type=int, default=30)
    parser.add_argument("--audio-amplitude", type=float, default=5.0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--sim-time-per-second", type=float, default=1.0)
    parser.add_argument(
        "--synth-sample-rate",
        type=float,
        default=200.0,
        help="Sample rate for synthetic-chirp fallback (in audio-time units).",
    )
    parser.add_argument(
        "--synth-f-start",
        type=float,
        default=0.02,
        help="Start frequency for synthetic chirp (in sim-time units; keep well below sim Nyquist 1/(2*dt_sim)).",
    )
    parser.add_argument(
        "--synth-f-end",
        type=float,
        default=0.4,
        help="End frequency for synthetic chirp.",
    )
    parser.add_argument("--wavespeed", type=float, default=1.0)
    parser.add_argument("--gridstep", type=float, default=1.0)
    parser.add_argument("--courant", type=float, default=0.5)
    parser.add_argument(
        "--n-mics",
        type=int,
        default=2,
        choices=[1, 2],
        help=(
            "Number of microphones per sample (default 2). "
            "Stock-laptop hardware bound: 2-3 channels max; only 1-2 implemented."
        ),
    )
    parser.add_argument(
        "--mic-spacing",
        type=float,
        default=16.0,
        help=(
            "Distance in cells between the stereo mic pair (n_mics=2). "
            "Random orientation per sample. Ignored when n_mics=1."
        ),
    )
    parser.add_argument(
        "--poses-per-room",
        type=int,
        default=1,
        help=(
            "Number of independent (driver, mic-pair) poses recorded in each "
            "room (Task 2.1.4). 1 (default) writes the original single-pose "
            "layout byte-for-byte; K > 1 writes a pose-major 'sensor' dataset "
            "of shape (K, T_rec, n_mics) with plural position attrs."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample status to stderr.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    audio_files = list_audio_files(args.audio_dir)
    if args.audio_dir and not audio_files:
        print(
            f"[active-sensing] WARNING: --audio-dir={args.audio_dir!r} had no *.wav files; "
            "falling back to synthetic chirps.",
            file=sys.stderr,
        )

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a throwaway Simulate just to compute dt under the chosen
    # courant so the synthetic-source duration can be tuned to the real
    # sim length.
    probe = Simulate(
        grid_shape=(args.grid, args.grid),
        wavespeed=args.wavespeed,
        gridstep=args.gridstep,
        courant=args.courant,
    )
    dt = probe.timestep

    t0 = time.perf_counter()
    with h5py.File(out_path, "w") as hf:
        hf.attrs["save_type"] = "active_sensing"
        hf.attrs["created_at"] = time.strftime("%Y-%m-%d_%H-%M-%S")
        hf.attrs["seed"] = int(args.seed)
        hf.attrs["audio_dir"] = args.audio_dir or ""
        hf.attrs["num_samples_requested"] = int(args.num_samples)

        n_poses = int(args.poses_per_room)
        if n_poses < 1:
            raise SystemExit("--poses-per-room must be >= 1")
        hf.attrs["poses_per_room"] = n_poses

        for s in range(int(args.num_samples)):
            obstacle_mask = generate_random_obstacles(
                grid_shape=(args.grid, args.grid),
                n_obstacles=args.n_obstacles,
                min_size=args.obstacle_min,
                max_size=args.obstacle_max,
                rng=rng,
            )
            # Per-pose placements. The RNG draw order (driver, then mics,
            # pose by pose, THEN source) reduces exactly to the original
            # order at K=1, so seeded single-pose archives reproduce the
            # pre-multi-pose files bit-for-bit.
            driver_positions = []
            mic_positions_per_pose = []
            for _ in range(n_poses):
                driver_positions.append(
                    random_free_position(
                        grid_shape=(args.grid, args.grid),
                        obstacle_mask=obstacle_mask,
                        rng=rng,
                    )
                )
                mic_positions_per_pose.append(
                    pick_mic_positions(
                        grid_shape=(args.grid, args.grid),
                        obstacle_mask=obstacle_mask,
                        n_mics=args.n_mics,
                        spacing=args.mic_spacing,
                        rng=rng,
                    )
                )
            # One source per room, shared by all poses: the physical story
            # is one device playing the same excitation at K spots.
            wf, source_samples, source_fs, source_label = build_source(
                audio_files=audio_files,
                rng=rng,
                sim_dt=dt,
                sim_duration_steps=args.duration,
                args=args,
            )
            # One Simulate per room, reset between poses. reset() zeroes
            # the fields and the clock but keeps geometry, so each pose
            # sees a numerically fresh run in the same room without paying
            # obstacle re-upload.
            sim = Simulate(
                grid_shape=(args.grid, args.grid),
                drivers=[],
                wavespeed=args.wavespeed,
                gridstep=args.gridstep,
                courant=args.courant,
            )
            # Push obstacles in batch — the engine's set_obstacle accepts
            # an iterable of (i, j) tuples and updates the hot-loop guard
            # once.
            mask_idx = np.argwhere(obstacle_mask)
            if mask_idx.size:
                sim.set_obstacle([tuple(int(c) for c in row) for row in mask_idx])

            recordings_per_pose = []
            for k in range(n_poses):
                sim.reset()
                sim.set_drivers([Driver(position=driver_positions[k], waveform=wf)])
                sensors = [Sensor(position=p) for p in mic_positions_per_pose[k]]
                recordings_per_pose.append(
                    run_with_sensors(
                        sim=sim,
                        duration=args.duration,
                        sensors=sensors,
                        record_step=args.record_step,
                    )
                )

            grp = hf.create_group(f"sample_{s:04d}")
            grp.attrs["grid_shape"] = (args.grid, args.grid)
            grp.attrs["wavespeed"] = float(args.wavespeed)
            grp.attrs["gridstep"] = float(args.gridstep)
            grp.attrs["timestep"] = float(sim.timestep)
            grp.attrs["courant"] = float(args.courant)
            if n_poses == 1:
                # Original single-pose layout, byte-for-byte.
                grp.attrs["driver_position"] = list(driver_positions[0])
                # Sensor positions: one row per mic, columns are spatial dims.
                grp.attrs["sensor_positions"] = np.asarray(
                    mic_positions_per_pose[0], dtype=np.int32
                )
            else:
                grp.attrs["poses_per_room"] = n_poses
                grp.attrs["driver_positions"] = np.asarray(driver_positions, dtype=np.int32)
                # (K, n_mics, dims): pose-major stack of the per-pose mic rows.
                grp.attrs["sensor_positions"] = np.asarray(mic_positions_per_pose, dtype=np.int32)
            grp.attrs["n_mics"] = int(args.n_mics)
            grp.attrs["mic_spacing"] = float(args.mic_spacing)
            grp.attrs["audio_path"] = source_label
            grp.attrs["audio_native_fs"] = float(source_fs)
            grp.attrs["sim_time_per_second"] = float(args.sim_time_per_second)
            grp.attrs["audio_amplitude"] = float(args.audio_amplitude)
            grp.attrs["sim_duration_steps"] = int(args.duration)
            grp.attrs["record_step"] = int(args.record_step)
            # Channel-last: sensor[..., t, m] = pressure at mic m, step t;
            # multi-pose archives carry a leading pose axis.
            sensor_out = (
                recordings_per_pose[0] if n_poses == 1 else np.stack(recordings_per_pose, axis=0)
            )
            grp.create_dataset("sensor", data=sensor_out, compression="gzip")
            grp.create_dataset("source", data=source_samples, compression="gzip")
            grp.create_dataset(
                "obstacles",
                data=obstacle_mask.astype(np.uint8),
                compression="gzip",
            )
            if args.verbose:
                elapsed = time.perf_counter() - t0
                mic_str = ", ".join(str(tuple(p)) for p in mic_positions_per_pose[0])
                print(
                    f"[active-sensing] sample {s + 1:4d}/{args.num_samples} "
                    f"poses={n_poses} driver0={driver_positions[0]} mics0=[{mic_str}] "
                    f"obstacles={int(obstacle_mask.sum())} "
                    f"peak_rec={float(np.max(np.abs(sensor_out))):.3e} "
                    f"elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                )

    total_s = time.perf_counter() - t0
    print(
        f"[active-sensing] wrote {args.num_samples} samples to {out_path} "
        f"({total_s:.1f}s, {total_s / max(args.num_samples, 1):.2f}s/sample)"
    )


if __name__ == "__main__":
    main()
