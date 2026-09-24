"""Score a real-room capture set exported from the web Lab (plan 9.7/9.8).

Input: the ``captures.json`` written by the Lab page's *Export* button
(format ``acoustic-sandbox-captures``, version 1). Each capture holds the
measured impulse responses (base64 float32, one per mic channel), the
browser's own estimates, and whatever ground truth the user entered: a
tape-measured distance to the nearest reflector and the room's box
dimensions.

For each capture this recomputes the estimates offline with
``acoustic_system.utils.room_ir`` (a NumPy port of the browser DSP) and
reports:

* first-echo distance vs the tape measurement (error in cm), per capture
  and as mean absolute error / median over the set. The Phase 9 target is
  "within a few cm";
* broadband T30 (and T20), and the Eyring absorption alpha it implies for
  the entered room, plus Sabine/Eyring T60 at ``--alpha`` if given;
* agreement between the browser and the offline estimates (a regression
  check on the two implementations).

Usage::

    uv run python scripts/eval_real_captures.py captures.json [--alpha 0.15] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

from acoustic_system.utils.room_ir import (
    alpha_from_t60,
    b64_to_f32,
    decay_metrics,
    eyring,
    find_echoes,
    sabine,
)


def evaluate(doc: dict, alpha: float | None = None, c: float = 343.0) -> dict:
    if doc.get("format") != "acoustic-sandbox-captures":
        raise ValueError("not a Lab capture export (format != acoustic-sandbox-captures)")
    rows = []
    for cap in doc["captures"]:
        fs = float(cap["sampleRate"])
        irs = [b64_to_f32(s) for s in cap["irs"]]
        h = irs[0]
        direct, echoes = find_echoes(h, fs, c=c)
        dm = decay_metrics(h, fs, direct)
        row: dict = {
            "label": cap.get("label", cap.get("id")),
            "channels": len(irs),
            "sample_rate": fs,
            "first_echo_m": echoes[0].distance if echoes else None,
            "echoes_m": [round(e.distance, 4) for e in echoes],
            "t30": dm["t30"],
            "t20": dm["t20"],
            "edt": dm["edt"],
            "equalized": bool(cap.get("equalized", False)),
            "latency_ms": cap.get("estimates", {}).get("latencyMs"),
        }
        tape = cap.get("measuredDistance")
        if tape is not None and row["first_echo_m"] is not None:
            row["tape_m"] = float(tape)
            row["distance_error_cm"] = 100 * (row["first_echo_m"] - float(tape))
        room = cap.get("room")
        t = dm["t30"] or dm["t20"]
        if room and t:
            dims = (room["lx"], room["ly"], room["lz"])
            row["alpha_implied"] = alpha_from_t60(*dims, t, c=c)
            if alpha is not None:
                row["t60_sabine"] = sabine(*dims, alpha, c=c)
                row["t60_eyring"] = eyring(*dims, alpha, c=c)
                row["t30_vs_eyring_pct"] = 100 * (t / row["t60_eyring"] - 1)
        est = cap.get("estimates", {})
        if est.get("firstEchoDistance") is not None and row["first_echo_m"] is not None:
            row["browser_first_echo_delta_mm"] = 1000 * (
                row["first_echo_m"] - est["firstEchoDistance"]
            )
        if est.get("t30") and dm["t30"]:
            row["browser_t30_delta_pct"] = 100 * (dm["t30"] / est["t30"] - 1)
        rows.append(row)
    errs = [abs(r["distance_error_cm"]) for r in rows if "distance_error_cm" in r]
    summary = {
        "n_captures": len(rows),
        "n_with_tape": len(errs),
        "distance_mae_cm": statistics.fmean(errs) if errs else None,
        "distance_median_abs_cm": statistics.median(errs) if errs else None,
        "within_5cm": sum(e <= 5 for e in errs),
    }
    return {"captures": rows, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("captures", type=pathlib.Path)
    ap.add_argument(
        "--alpha", type=float, default=None, help="assumed mean absorption for Sabine/Eyring"
    )
    ap.add_argument("--c", type=float, default=343.0, help="speed of sound (m/s)")
    ap.add_argument("--json", type=pathlib.Path, default=None, help="write the full results here")
    args = ap.parse_args(argv)
    res = evaluate(json.loads(args.captures.read_text()), alpha=args.alpha, c=args.c)
    print(
        f"{'label':<28} {'echo (m)':>9} {'tape (m)':>9} {'err (cm)':>9} {'T30 (s)':>8} {'alpha':>6}"
    )
    for r in res["captures"]:

        def f(v, fmt):
            return format(v, fmt) if isinstance(v, (int, float)) else "—"

        print(
            f"{str(r['label'])[:28]:<28} {f(r['first_echo_m'], '9.3f')} {f(r.get('tape_m'), '9.3f')} "
            f"{f(r.get('distance_error_cm'), '9.1f')} {f(r['t30'], '8.2f')} {f(r.get('alpha_implied'), '6.2f')}"
        )
    s = res["summary"]
    if s["n_with_tape"]:
        print(
            f"\ndistance: MAE {s['distance_mae_cm']:.1f} cm, median |err| {s['distance_median_abs_cm']:.1f} cm, "
            f"{s['within_5cm']}/{s['n_with_tape']} within 5 cm"
        )
    if args.json:
        args.json.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
