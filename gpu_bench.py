#!/usr/bin/env python3
"""
DustyFans GPU benchmark — modern path (CUDA / current driver).

Captures the before/after story that the whole project rests on:
  - idle + load temperatures
  - the throttle-vs-temp behaviour (clocks dropping as the card hits limits)
  - a relative-performance number (sustained FP32 matmul throughput, TFLOPS)

Method consistency over absolute precision (see project doc 3.5):
identical VM, identical run config, 3 runs averaged.

Telemetry is sampled at 1 Hz from nvidia-smi (driver-agnostic, so the same
logger works on the legacy VM too). The load generator here is CUDA-specific
(PyTorch); the legacy path will swap in an era-appropriate load.

Output: a single JSON result + the raw 1 Hz samples as CSV.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time

# nvidia-smi throttle/clock-event reason bits -> human names.
# Thermal/power ones are the interesting part of the dirty-vs-clean story.
THROTTLE_BITS = {
    0x0001: "idle",
    0x0002: "app_clocks_setting",
    0x0004: "sw_power_cap",
    0x0008: "hw_slowdown",
    0x0010: "sync_boost",
    0x0020: "sw_thermal_slowdown",
    0x0040: "hw_thermal_slowdown",
    0x0080: "hw_power_brake",
    0x0100: "display_clocks_setting",
}
# Bits that mean "the card is being held back by heat or power" — the headline.
THERMAL_POWER_BITS = 0x0004 | 0x0008 | 0x0020 | 0x0040 | 0x0080

SAMPLE_FIELDS = [
    "clocks.sm",
    "clocks.mem",
    "temperature.gpu",
    "power.draw",
    "fan.speed",
    "utilization.gpu",
    "clocks_event_reasons.active",
]


def decode_throttle(mask):
    if mask == 0:
        return []
    return [name for bit, name in THROTTLE_BITS.items() if mask & bit]


def query_gpu(index):
    """One nvidia-smi sample as a dict. Returns None on parse failure."""
    q = ",".join(SAMPLE_FIELDS)
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={index}",
                f"--query-gpu={q}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    parts = [p.strip() for p in out.split(",")]
    if len(parts) != len(SAMPLE_FIELDS):
        return None

    def num(x):
        try:
            return float(x)
        except ValueError:
            return None

    mask_raw = parts[6]
    mask = int(mask_raw, 16) if mask_raw.lower().startswith("0x") else int(mask_raw)
    return {
        "clock_sm": num(parts[0]),
        "clock_mem": num(parts[1]),
        "temp": num(parts[2]),
        "power": num(parts[3]),
        "fan": num(parts[4]),
        "util": num(parts[5]),
        "throttle_mask": mask,
        "throttle": decode_throttle(mask),
    }


class Sampler(threading.Thread):
    """Background 1 Hz telemetry logger. Tags each sample with the active phase."""

    def __init__(self, index, hz=1.0):
        super().__init__(daemon=True)
        self.index = index
        self.interval = 1.0 / hz
        self.samples = []
        self.phase = "init"
        self._stop = threading.Event()

    def set_phase(self, phase):
        self.phase = phase

    def run(self):
        while not self._stop.is_set():
            t0 = time.monotonic()
            s = query_gpu(self.index)
            if s:
                s["t"] = round(time.monotonic(), 2)
                s["phase"] = self.phase
                self.samples.append(s)
            dt = time.monotonic() - t0
            self._stop.wait(max(0.0, self.interval - dt))

    def stop(self):
        self._stop.set()


def run_glmark2(display, extra_args):
    """Universal cross-era relative-perf score. OpenGL, so it runs on every
    card from a GTX 660 to an RTX 2070 — this is the fleet-wide ranking number.
    Returns the integer glmark2 score, or None on failure."""
    import re

    env = dict(os.environ, DISPLAY=display)
    try:
        out = subprocess.check_output(
            ["glmark2"] + extra_args, text=True, env=env, timeout=1800,
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  glmark2 failed: {e}", file=sys.stderr)
        return None
    m = re.search(r"glmark2 Score:\s*(\d+)", out)
    return int(m.group(1)) if m else None


def run_load(matrix, seconds):
    """Sustained FP32 matmul. Returns achieved TFLOPS (relative-perf metric)."""
    import torch

    dev = torch.device("cuda")
    a = torch.randn(matrix, matrix, device=dev, dtype=torch.float32)
    b = torch.randn(matrix, matrix, device=dev, dtype=torch.float32)
    flops_per_iter = 2.0 * matrix**3  # multiply-add
    torch.cuda.synchronize()

    iters = 0
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        c = a @ b
        a = c  # chain so the optimizer can't elide work
        iters += 1
        if iters % 50 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    return (flops_per_iter * iters) / elapsed / 1e12


def summarize_phase(samples, phase, tail_secs=None):
    """Aggregate samples for a phase. tail_secs -> only the last N seconds
    (used for load steady-state, ignoring the ramp)."""
    rows = [s for s in samples if s["phase"] == phase]
    if tail_secs and rows:
        cutoff = rows[-1]["t"] - tail_secs
        rows = [s for s in rows if s["t"] >= cutoff]
    if not rows:
        return None

    def col(key):
        return [s[key] for s in rows if s.get(key) is not None]

    temps = col("temp")
    masks = [s["throttle_mask"] for s in rows]
    thermal_power = any(m & THERMAL_POWER_BITS for m in masks)
    reasons = sorted({r for s in rows for r in s["throttle"] if r != "idle"})
    sm = col("clock_sm")
    return {
        "n": len(rows),
        "temp_mean": round(sum(temps) / len(temps), 1) if temps else None,
        "temp_max": max(temps) if temps else None,
        "clock_sm_mean": round(sum(sm) / len(sm)) if sm else None,
        "clock_sm_min": min(sm) if sm else None,
        "power_mean": round(sum(col("power")) / len(col("power")), 1) if col("power") else None,
        "fan_max": max(col("fan")) if col("fan") else None,
        "thermal_or_power_throttled": thermal_power,
        "throttle_reasons": reasons,
    }


def one_run(sampler, args, run_idx):
    sampler.set_phase(f"idle{run_idx}")
    time.sleep(args.idle_secs)

    sampler.set_phase(f"load{run_idx}")
    tflops = run_load(args.matrix, args.load_secs)

    sampler.set_phase(f"cooldown{run_idx}")
    time.sleep(args.cooldown_secs)

    idle = summarize_phase(sampler.samples, f"idle{run_idx}")
    load = summarize_phase(sampler.samples, f"load{run_idx}", tail_secs=args.steady_secs)
    return {
        "run": run_idx,
        "tflops_fp32": round(tflops, 2),
        "idle": idle,
        "load_steady": load,
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="card label, e.g. unit 0001 / before")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--idle-secs", type=int, default=30)
    ap.add_argument("--load-secs", type=int, default=120)
    ap.add_argument("--steady-secs", type=int, default=30, help="tail of load used for steady-state")
    ap.add_argument("--cooldown-secs", type=int, default=20)
    ap.add_argument("--matrix", type=int, default=8192)
    ap.add_argument("--display", default=":0", help="X display for glmark2")
    ap.add_argument("--glmark2-args", default="--off-screen",
                    help="args passed to glmark2 (default: full off-screen suite)")
    ap.add_argument("--no-glmark2", action="store_true", help="skip the universal GL score")
    ap.add_argument("--out", default=None, help="JSON output path (default: ./<label>.json)")
    args = ap.parse_args()

    name = query_gpu(args.gpu)
    if name is None:
        print("ERROR: cannot read GPU via nvidia-smi", file=sys.stderr)
        sys.exit(1)
    gpu_name = subprocess.check_output(
        ["nvidia-smi", f"--id={args.gpu}", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()

    sampler = Sampler(args.gpu)
    sampler.start()
    runs = []
    try:
        for i in range(1, args.runs + 1):
            print(f"[run {i}/{args.runs}] idle {args.idle_secs}s -> load {args.load_secs}s ...", flush=True)
            r = one_run(sampler, args, i)
            print(f"  tflops={r['tflops_fp32']}  idle_temp={r['idle']['temp_mean'] if r['idle'] else '?'}"
                  f"  load_temp={r['load_steady']['temp_max'] if r['load_steady'] else '?'}"
                  f"  throttled={r['load_steady']['thermal_or_power_throttled'] if r['load_steady'] else '?'}", flush=True)
            runs.append(r)
        glmark2_score = None
        if not args.no_glmark2:
            print("[glmark2] universal GL score ...", flush=True)
            sampler.set_phase("glmark2")
            glmark2_score = run_glmark2(args.display, args.glmark2_args.split())
            print(f"  glmark2_score={glmark2_score}", flush=True)
    finally:
        sampler.stop()
        sampler.join(timeout=3)

    summary = {
        "glmark2_score": glmark2_score,
        "tflops_fp32": mean([r["tflops_fp32"] for r in runs]),
        "idle_temp_mean": mean([r["idle"]["temp_mean"] for r in runs if r["idle"]]),
        "load_temp_max": max([r["load_steady"]["temp_max"] for r in runs if r["load_steady"]], default=None),
        "load_clock_sm_mean": mean([r["load_steady"]["clock_sm_mean"] for r in runs if r["load_steady"]]),
        "thermal_or_power_throttled": any(
            r["load_steady"]["thermal_or_power_throttled"] for r in runs if r["load_steady"]
        ),
        "throttle_reasons": sorted({
            x for r in runs if r["load_steady"] for x in r["load_steady"]["throttle_reasons"]
        }),
    }

    # Decimated load-phase feed (last run's load) — this is what charts plot: the
    # temp climb + clock sag that tell the throttle story. Relative t from load
    # start; ~25 points. throttle_spans marks thermal/power-limited intervals.
    load_phase = f"load{args.runs}"
    load_samples = [s for s in sampler.samples if s["phase"] == load_phase]
    samples_decimated = []
    throttle_spans = []
    if load_samples:
        t0 = load_samples[0]["t"]
        step = max(1, len(load_samples) // 25)
        samples_decimated = [
            {"t": round(s["t"] - t0, 1), "temp": s["temp"], "clock_sm": s["clock_sm"], "power": s["power"]}
            for s in load_samples[::step]
        ]
        cur = None
        for s in load_samples:
            thr = bool(s["throttle_mask"] & THERMAL_POWER_BITS)
            rt = round(s["t"] - t0, 1)
            if thr and cur is None:
                cur = rt
            elif not thr and cur is not None:
                throttle_spans.append({"start": cur, "end": rt})
                cur = None
        if cur is not None:
            throttle_spans.append({"start": cur, "end": round(load_samples[-1]["t"] - t0, 1)})

    result = {
        "label": args.label,
        "gpu": gpu_name,
        "config": {
            "runs": args.runs,
            "idle_secs": args.idle_secs,
            "load_secs": args.load_secs,
            "steady_secs": args.steady_secs,
            "matrix": args.matrix,
        },
        "runs": runs,
        "summary": summary,
        "samples_decimated": samples_decimated,
        "throttle_spans": throttle_spans,
    }

    out = args.out or f"./{args.label.replace(' ', '_').replace('/', '_')}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    csv_path = os.path.splitext(out)[0] + "_samples.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "phase", "temp", "clock_sm", "clock_mem", "power", "fan", "util", "throttle_mask", "throttle"])
        for s in sampler.samples:
            w.writerow([s["t"], s["phase"], s["temp"], s["clock_sm"], s["clock_mem"],
                        s["power"], s["fan"], s["util"], hex(s["throttle_mask"]), "|".join(s["throttle"])])

    print(f"\nwrote {out} and {csv_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
