#!/usr/bin/env python3
"""
DustyFans GPU benchmark — modern path (CUDA / current driver).

Captures the before/after story that the whole project rests on:
  - idle + load temperatures (incl. memory-junction temp where the card exposes it)
  - the throttle-vs-temp behaviour (clocks dropping as the card hits limits)
  - a relative-performance number (sustained FP32 matmul throughput, TFLOPS)
  - WIDE 1 Hz telemetry (pstate, power limit, PCIe link gen/width, mem util, ...)
  - optional room ambient (--ambient-cmd) + isolated wall power (--shelly-ip)

Method consistency over absolute precision (see project doc 3.5):
identical rig, identical run config, 3 runs averaged.

Telemetry is sampled at 1 Hz from nvidia-smi (driver-agnostic, so the same logger
works on the legacy rig too — fields the card doesn't expose come back NULL). The
load generator here is CUDA-specific (PyTorch, cu124 for Pascal+); the legacy path
swaps in an era-appropriate load. VRAM-integrity (memtest_vulkan) and LLM tok/s
(llama.cpp) steps attach to the same JSON (added separately).

Output: a single JSON result + the raw 1 Hz samples as CSV (one phase per run of
this script; before/after are separate invocations).
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

# nvidia-smi throttle/clock-event reason bits -> human names.
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

# --- Vulkan tooling (VRAM + LLM). The bench box exposes 3 Vulkan devices (NVIDIA +
# AMD iGPU + llvmpipe), so PIN to the NVIDIA GPU by exposing only its ICD — that
# makes device selection deterministic + fleet-repeatable. All paths env-overridable.
NVIDIA_VK_ICD = os.environ.get("BENCH_VK_ICD", "/usr/share/vulkan/icd.d/nvidia_icd.json")
MEMTEST_BIN   = os.environ.get("BENCH_MEMTEST", "/opt/memtest_vulkan/memtest_vulkan")
LLAMA_DIR     = os.environ.get("BENCH_LLAMA_DIR", "/opt/llama/llama-b9496")
LLAMA_BENCH   = os.path.join(LLAMA_DIR, "llama-bench")
MODELS_DIR    = os.environ.get("BENCH_MODELS_DIR", "/opt/bench/models")
# LLM rung ladder: (label, filename, min_vram_mib_to_fit, params_hint_b). Larger rungs
# are VRAM-gated — a card below the gate records oom=True without running. The
# 'universal' rung runs on every card (the cross-card comparable axis).
LLM_RUNGS = [
    ("universal", "universal-qwen2.5-0.5b-q4km.gguf", 0,     0.5),
    ("7b-q4",     "7b-qwen2.5-7b-q4km.gguf",          6000,  7.0),
    ("13b-q4",    "13b-qwen2.5-13b-q4km.gguf",        11000, 13.0),
]

# Widened query. Order matters — parsed positionally in query_gpu(). pstate is a
# string ("P0".."P12"); everything else is numeric or "[N/A]" (card doesn't expose).
SAMPLE_FIELDS = [
    "clocks.sm",                  # 0
    "clocks.mem",                 # 1
    "temperature.gpu",            # 2
    "temperature.memory",         # 3  (N/A on Pascal/GDDR5X; present on GDDR6X)
    "power.draw",                 # 4
    "enforced.power.limit",       # 5
    "fan.speed",                  # 6
    "utilization.gpu",            # 7
    "utilization.memory",         # 8
    "pstate",                     # 9  (string)
    "pcie.link.gen.current",      # 10
    "pcie.link.width.current",    # 11
    "clocks_event_reasons.active",# 12 (throttle bitmask)
]


def decode_throttle(mask):
    if mask == 0:
        return []
    return [name for bit, name in THROTTLE_BITS.items() if mask & bit]


def _num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def query_gpu(index):
    """One widened nvidia-smi sample as a dict. Returns None on parse failure.
    Fields the card doesn't expose (e.g. temperature.memory on Pascal) -> None."""
    q = ",".join(SAMPLE_FIELDS)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={index}", f"--query-gpu={q}",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    parts = [p.strip() for p in out.split(",")]
    if len(parts) != len(SAMPLE_FIELDS):
        return None

    mask_raw = parts[12]
    try:
        mask = int(mask_raw, 16) if mask_raw.lower().startswith("0x") else int(mask_raw)
    except ValueError:
        mask = 0

    def pstate(v):
        v = v.strip()
        return v if v and not v.startswith("[") else None  # "[N/A]" -> None

    return {
        "clock_sm": _num(parts[0]),
        "clock_mem": _num(parts[1]),
        "temp": _num(parts[2]),
        "temp_mem": _num(parts[3]),
        "power": _num(parts[4]),
        "power_limit": _num(parts[5]),
        "fan": _num(parts[6]),
        "util": _num(parts[7]),
        "util_mem": _num(parts[8]),
        "pstate": pstate(parts[9]),
        "pcie_gen_cur": _num(parts[10]),
        "pcie_width_cur": _num(parts[11]),
        "voltage_mv": None,           # not exposed via --query-gpu (card/driver dependent)
        "throttle_mask": mask,
        "throttle": decode_throttle(mask),
    }


def poll_shelly(ip, timeout=0.6):
    """Active power (W) from a Shelly plug metering the eGPU PSU. Gen2 RPC first,
    then Gen1 /status. Best-effort: any failure -> None (never blocks the loop)."""
    if not ip:
        return None
    try:
        with urllib.request.urlopen(f"http://{ip}/rpc/Switch.GetStatus?id=0", timeout=timeout) as r:
            d = json.load(r)
        v = _num(d.get("apower"))
        if v is not None:
            return v
    except Exception:
        pass
    try:
        with urllib.request.urlopen(f"http://{ip}/status", timeout=timeout) as r:
            d = json.load(r)
        meters = d.get("meters") or []
        return _num(meters[0].get("power")) if meters else None
    except Exception:
        return None


def poll_ambient(cmd, timeout=2.0):
    """Ambient °C from a user-supplied command that prints a number (e.g. a TEMPer
    reader script). Lets any ambient source plug in without hardcoding the device."""
    if not cmd:
        return None
    try:
        out = subprocess.check_output(cmd, shell=True, text=True, timeout=timeout).strip()
        return _num(out.split()[0]) if out else None
    except Exception:
        return None


class Sampler(threading.Thread):
    """Background 1 Hz telemetry logger. Tags each sample with the active phase,
    and (if configured) attaches ambient °C + isolated wall watts per sample."""

    def __init__(self, index, hz=1.0, shelly_ip=None, ambient_cmd=None):
        super().__init__(daemon=True)
        self.index = index
        self.interval = 1.0 / hz
        self.samples = []
        self.phase = "init"
        self.shelly_ip = shelly_ip
        self.ambient_cmd = ambient_cmd
        self._stop_evt = threading.Event()

    def set_phase(self, phase):
        self.phase = phase

    def run(self):
        while not self._stop_evt.is_set():
            t0 = time.monotonic()
            s = query_gpu(self.index)
            if s:
                s["ambient_c"] = poll_ambient(self.ambient_cmd)
                s["wall_power_w"] = poll_shelly(self.shelly_ip)
                s["t"] = round(time.monotonic(), 2)
                s["phase"] = self.phase
                self.samples.append(s)
            dt = time.monotonic() - t0
            self._stop_evt.wait(max(0.0, self.interval - dt))

    def stop(self):
        self._stop_evt.set()


def run_glmark2(display, extra_args):
    """Universal cross-era relative-perf score. OpenGL, so it runs on every card
    from a GTX 660 to an RTX 4090 — the fleet-wide ranking number. Int score or None."""
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
    temp_mem = col("temp_mem")
    ambient = col("ambient_c")
    wall = col("wall_power_w")
    masks = [s["throttle_mask"] for s in rows]
    thermal_power = any(m & THERMAL_POWER_BITS for m in masks)
    reasons = sorted({r for s in rows for r in s["throttle"] if r != "idle"})
    sm = col("clock_sm")

    def avg(xs, nd=1):
        return round(sum(xs) / len(xs), nd) if xs else None

    return {
        "n": len(rows),
        "temp_mean": avg(temps),
        "temp_max": max(temps) if temps else None,
        "temp_mem_max": max(temp_mem) if temp_mem else None,
        "clock_sm_mean": round(sum(sm) / len(sm)) if sm else None,
        "clock_sm_min": min(sm) if sm else None,
        "power_mean": avg(col("power")),
        "fan_max": max(col("fan")) if col("fan") else None,
        "ambient_c_mean": avg(ambient),
        "wall_power_mean": avg(wall),
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


def file_sha256_cached(path):
    """sha256 of a model file, cached to a <path>.sha256 sidecar (these are GBs)."""
    side = path + ".sha256"
    try:
        if os.path.exists(side):
            return open(side).read().split()[0]
    except OSError:
        pass
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    try:
        with open(side, "w") as f:
            f.write(digest)
    except OSError:
        pass
    return digest


def run_vram(secs):
    """memtest_vulkan VRAM-integrity pass, PINNED to the NVIDIA GPU. Time-boxed via
    `timeout --signal=INT` (memtest stops + reports cleanly on Ctrl-C). HONEST: errors
    mean bad/marginal silicon — cleaning never repairs that. Returns the phase fields."""
    env = dict(os.environ, VK_DRIVER_FILES=NVIDIA_VK_ICD)
    base = {"vram_test_tool": "memtest_vulkan", "vram_tested_mib": None,
            "vram_test_secs": float(secs)}
    if not os.path.exists(MEMTEST_BIN):
        return {**base, "vram_test_status": "unsupported", "vram_errors": None}
    try:
        p = subprocess.run(["timeout", "--signal=INT", str(secs), MEMTEST_BIN],
                           capture_output=True, text=True, env=env, timeout=secs + 30)
        out = p.stdout + p.stderr
    except Exception:
        return {**base, "vram_test_status": "skipped", "vram_errors": None}
    on_nvidia = "NVIDIA" in out                       # the VK pin should guarantee this
    passed = ("no any errors" in out) or ("testing PASSed" in out)
    if not on_nvidia:
        return {**base, "vram_test_status": "skipped", "vram_errors": None}
    return {**base, "vram_test_status": "ok" if passed else "errors",
            "vram_errors": 0 if passed else None}


def run_llm(mem_total_mib, models_dir):
    """llama.cpp tok/s for each rung that fits VRAM, PINNED to the NVIDIA GPU. SPEED
    ONLY (prefill + generation tok/s); never quality. Rungs above the card's VRAM gate
    record oom=True without running; missing model files are skipped (noted)."""
    results = []
    if not os.path.exists(LLAMA_BENCH):
        return results
    env = dict(os.environ, VK_DRIVER_FILES=NVIDIA_VK_ICD, LD_LIBRARY_PATH=LLAMA_DIR)
    for label, fname, min_vram, params_b in LLM_RUNGS:
        path = os.path.join(models_dir, fname)
        row = {"model_label": label, "model_repo": fname, "model_params_b": params_b,
               "runner": "llama.cpp", "oom": False, "fully_offloaded": None,
               "pp_tok_s": None, "tg_tok_s": None}
        if mem_total_mib is not None and mem_total_mib < min_vram:
            results.append({**row, "oom": True,
                            "note": f"VRAM-gated: needs ~{min_vram}MiB, card has {mem_total_mib}MiB"})
            continue
        if not os.path.exists(path):
            results.append({**row, "note": "model file not present (rung skipped)"})
            continue
        try:
            out = subprocess.check_output(
                [LLAMA_BENCH, "-m", path, "-ngl", "999", "-p", "512", "-n", "128", "-o", "json"],
                text=True, env=env, timeout=1800, stderr=subprocess.DEVNULL)
            arr = json.loads(out)
        except Exception as e:
            results.append({**row, "note": f"llama-bench failed: {str(e)[:160]}"})
            continue
        pp = next((x for x in arr if x.get("n_prompt") and not x.get("n_gen")), None)
        tg = next((x for x in arr if x.get("n_gen") and not x.get("n_prompt")), None)
        ref = pp or tg or {}
        nparams = ref.get("model_n_params")
        results.append({**row,
            "model_quant": ref.get("model_type"),
            "model_params_b": round(nparams / 1e9, 1) if nparams else params_b,
            "model_sha256": file_sha256_cached(path),
            "runner_build": f"b{ref.get('build_number', '?')}",
            "n_gpu_layers": ref.get("n_gpu_layers"),
            "fully_offloaded": True,
            "pp_tok_s": round(pp["avg_ts"], 2) if pp else None,
            "tg_tok_s": round(tg["avg_ts"], 2) if tg else None,
            "pp_n": pp["n_prompt"] if pp else None,
            "tg_n": tg["n_gen"] if tg else None,
            "vram_used_mib": round(ref["model_size"] / 1048576) if ref.get("model_size") else None,
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="card label, e.g. 'unit 0001 before'")
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
    ap.add_argument("--shelly-ip", default=os.environ.get("BENCH_SHELLY_IP"),
                    help="Shelly plug IP (eGPU PSU) for isolated wall power; else NULL")
    ap.add_argument("--ambient-cmd", default=os.environ.get("BENCH_AMBIENT_CMD"),
                    help="shell command printing room ambient °C; else NULL")
    ap.add_argument("--vram", action="store_true", help="run memtest_vulkan VRAM-integrity pass")
    ap.add_argument("--vram-secs", type=int, default=120, help="VRAM test duration")
    ap.add_argument("--llm", action="store_true", help="run the llama.cpp tok/s rung ladder")
    ap.add_argument("--models-dir", default=MODELS_DIR, help="dir holding the rung GGUF models")
    ap.add_argument("--out", default=None, help="JSON output path (default: ./<label>.json)")
    args = ap.parse_args()

    if query_gpu(args.gpu) is None:
        print("ERROR: cannot read GPU via nvidia-smi", file=sys.stderr)
        sys.exit(1)
    gpu_name = subprocess.check_output(
        ["nvidia-smi", f"--id={args.gpu}", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    mem_total_raw = subprocess.check_output(
        ["nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", f"--id={args.gpu}", "--query-gpu=gpu_uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    mem_total_mib = int(float(mem_total_raw)) if _num(mem_total_raw) is not None else None

    sampler = Sampler(args.gpu, shelly_ip=args.shelly_ip, ambient_cmd=args.ambient_cmd)
    sampler.start()
    runs = []
    vram_result = None
    llm_results = []
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
        if args.vram:
            print("[vram] memtest_vulkan (pinned to NVIDIA) ...", flush=True)
            sampler.set_phase("vram")
            vram_result = run_vram(args.vram_secs)
            print(f"  vram_test_status={vram_result.get('vram_test_status')}", flush=True)
        if args.llm:
            print("[llm] llama.cpp rung ladder (pinned to NVIDIA) ...", flush=True)
            sampler.set_phase("llm")
            llm_results = run_llm(mem_total_mib, args.models_dir)
            for r in llm_results:
                print(f"  llm {r['model_label']}: pp={r.get('pp_tok_s')} tg={r.get('tg_tok_s')} oom={r.get('oom')}", flush=True)
    finally:
        sampler.stop()
        sampler.join(timeout=3)

    # Energy over the load phases (Wh) from isolated wall power, if metered.
    load_wall = [s["wall_power_w"] for s in sampler.samples
                 if s["phase"].startswith("load") and s.get("wall_power_w") is not None]
    wall_energy_wh = round(sum(load_wall) / 3600.0, 2) if load_wall else None

    summary = {
        "glmark2_score": glmark2_score,
        "tflops_fp32": mean([r["tflops_fp32"] for r in runs]),
        "idle_temp_mean": mean([r["idle"]["temp_mean"] for r in runs if r["idle"]]),
        "load_temp_max": max([r["load_steady"]["temp_max"] for r in runs if r["load_steady"]], default=None),
        "load_temp_mean": mean([r["load_steady"]["temp_mean"] for r in runs if r["load_steady"]]),
        "load_clock_sm_mean": mean([r["load_steady"]["clock_sm_mean"] for r in runs if r["load_steady"]]),
        "load_clock_sm_min": min([r["load_steady"]["clock_sm_min"] for r in runs if r["load_steady"] and r["load_steady"]["clock_sm_min"] is not None], default=None),
        "power_mean": mean([r["load_steady"]["power_mean"] for r in runs if r["load_steady"]]),
        "fan_max": max([r["load_steady"]["fan_max"] for r in runs if r["load_steady"] and r["load_steady"]["fan_max"] is not None], default=None),
        "temp_mem_max": max([r["load_steady"]["temp_mem_max"] for r in runs if r["load_steady"] and r["load_steady"]["temp_mem_max"] is not None], default=None),
        "ambient_c_mean": mean([r["load_steady"]["ambient_c_mean"] for r in runs if r["load_steady"]]),
        "wall_power_mean": mean([r["load_steady"]["wall_power_mean"] for r in runs if r["load_steady"]]),
        "wall_energy_wh": wall_energy_wh,
        "thermal_or_power_throttled": any(
            r["load_steady"]["thermal_or_power_throttled"] for r in runs if r["load_steady"]
        ),
        "throttle_reasons": sorted({
            x for r in runs if r["load_steady"] for x in r["load_steady"]["throttle_reasons"]
        }),
    }

    # Decimated load-phase feed (last run's load) — what the public charts plot: the
    # temp climb + clock sag that tell the throttle story. ~25 points, relative t.
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
        "gpu_uuid": gpu_uuid,
        "memory_total_mib": mem_total_mib,
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
        "vram": vram_result,   # memtest_vulkan VRAM-integrity (--vram)
        "llm": llm_results,    # llama.cpp tok/s rung ladder (--llm)
    }

    out = args.out or f"./{args.label.replace(' ', '_').replace('/', '_')}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    csv_path = os.path.splitext(out)[0] + "_samples.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "phase", "temp", "temp_mem", "clock_sm", "clock_mem", "power",
                    "power_limit", "fan", "util", "util_mem", "pstate", "pcie_gen_cur",
                    "pcie_width_cur", "voltage_mv", "ambient_c", "wall_power_w",
                    "throttle_mask", "throttle"])
        for s in sampler.samples:
            w.writerow([s["t"], s["phase"], s["temp"], s["temp_mem"], s["clock_sm"],
                        s["clock_mem"], s["power"], s["power_limit"], s["fan"], s["util"],
                        s["util_mem"], s["pstate"], s["pcie_gen_cur"], s["pcie_width_cur"],
                        s["voltage_mv"], s["ambient_c"], s["wall_power_w"],
                        hex(s["throttle_mask"]), "|".join(s["throttle"])])

    print(f"\nwrote {out} and {csv_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
