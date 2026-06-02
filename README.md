# GPU Clean Benchmark — methodology

The open benchmarking methodology behind **[DustyFans](https://dustyfans.com)**: we tear down, clean, and
re-paste/re-pad used NVIDIA GPUs, and benchmark each one **before and after** to answer a simple question —
*is cleaning a used GPU actually worth it?*

The data is only meaningful if you can see how it was produced. So this is the method, in the open, with the
script we run. The interesting answer is the honest one: sometimes cleaning drops temperatures a lot, sometimes
barely, and sometimes a card doesn't improve at all. We publish all of it.

## What we measure

Per card, in two phases — **before** (dirty, as received) and **after** (cleaned) — we capture:

- **Temperatures** — idle and sustained-load GPU temperature (max and steady-state mean).
- **Clock behaviour + throttle reasons** — SM/memory clocks sampled over the run, plus the GPU's active
  *throttle reasons* (thermal slowdown, power cap, hardware slowdown). This is the core story: a dirty card
  often **thermal-throttles** — clocks step down as it hits its temperature limit — while the same card,
  cleaned, tends to hold higher clocks and instead bump into its **power** limit. You can see that shift in
  the data, not just in a single temperature number.
- **Relative performance**
  - **glmark2 score** — an OpenGL benchmark that runs on *every* card from the GTX 600 series through current
    GPUs. This is the **universal cross-era number** we rank by.
  - **FP32 throughput (TFLOPS)** — a sustained matmul load, on cards new enough to run it (a secondary,
    modern-only number).
- **Power and fan** — average board power and peak fan speed under load.

From the two phases we compute **deltas** — temperature drop, performance change, clock change — *including the
unimpressive and negative ones*.

## How we run it

- **Two phases, three runs each, averaged.** The goal is **consistency of method**, not laboratory precision.
- **Identical configuration** for before and after (same host, same OS image, same driver, same run settings) so
  the before/after comparison is apples-to-apples.
- **~1 Hz telemetry** is logged for the whole run (temp, clocks, power, fan, utilisation, throttle bitmask),
  which is what powers the temperature-vs-time and clock-vs-temperature curves.
- Each run is: a short **idle baseline**, then a **sustained load** held to thermal steady state, then capture.

### Cross-era fairness

A new card will always out-score an old one on raw numbers, so absolute boards (hottest, top score) are paired
with **per-era / per-series** views and **normalised** measures — *percentage* improvement and
**performance-per-watt** — which compare a Kepler-era card and a modern one fairly.

## Honesty & limitations

- These are **standardised synthetic measurements**, not real-world game FPS.
- We do **not** currently normalise for **ambient room temperature**; we hold method and configuration constant
  instead. Absolute temperatures will carry some ambient variance — the *deltas* and the *throttle-reason
  shifts* are the robust signal.
- "Relative performance" is a consistent score for ranking, not a vendor spec.

## The script

[`gpu_bench.py`](./gpu_bench.py) is the harness for the modern path. It needs a Linux machine with an NVIDIA GPU
and driver, and:

- `nvidia-smi` (telemetry — driver-provided)
- `python3` + [PyTorch](https://pytorch.org) (the sustained FP32 load + TFLOPS)
- [`glmark2`](https://github.com/glmark2/glmark2) run against a headless X server on the GPU (the universal score)

```bash
python3 gpu_bench.py --label "before" --runs 3 --idle-secs 30 --load-secs 120
python3 gpu_bench.py --label "after"  --runs 3 --idle-secs 30 --load-secs 120
```

It writes a JSON summary plus the raw 1 Hz samples as CSV. A legacy-driver variant (for Kepler-era cards that
can't run current CUDA) emits the same `glmark2` universal score and the same phase/run/sample shape.

## License

MIT — see [LICENSE](./LICENSE). Use it, adapt it, check our numbers.

---

Built for [DustyFans](https://dustyfans.com) by [Hake Hardware](https://github.com/hakehardware).
