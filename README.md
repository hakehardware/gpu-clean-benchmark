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
- **Sustained-load soak (the stress test)** — beyond the short runs above, each phase also holds a ~15-minute
  sustained load so the whole cooler reaches heat-soak equilibrium — exactly where a dirty card's degraded heat
  transfer shows up. We record **time-to-throttle** (how long before *thermal* throttling begins — power-capping
  from the start is normal and does not count), the **sustained throttle %** and worst held clock, **stability**
  (crashes / compute artifacts / driver Xid events), and **fresh-vs-soaked retention** — the graphics score cool
  vs after the soak (a clean card holds it; a dirty one sheds performance as it heats).
- **Wide 1 Hz telemetry** — beyond temp/clock/power/fan we also log memory-junction temperature (where the card
  exposes it), memory-controller utilisation, performance state, the enforced power limit, and PCIe link
  gen/width. Where sensors are fitted we also log **room ambient temperature** and **isolated wall power** (a
  smart plug on the GPU's PSU only) — the latter is the one honest power number for older cards `nvidia-smi`
  can't report board power for.
- **VRAM integrity (run hot)** — a [`memtest_vulkan`](https://github.com/GpuZelenograd/memtest_vulkan) pass at the
  **end of the soak**, when the card is heat-soaked and memory-junction temperature is near its peak (marginal
  memory can pass cold and fault hot — so we test it hot). Vulkan compute, so it runs on every card; records
  pass / error count, each phase. **This is an honesty check, not a
  cleaning metric:** VRAM errors mean failing or marginal memory silicon, and cleaning a card does **not** repair
  that. We capture it before *and* after so an electrically sick card shows up as sick — never sold as if cleaning
  fixed it.
- **LLM throughput (tokens/sec)** — how fast the card runs local language models, via
  [`llama.cpp`](https://github.com/ggml-org/llama.cpp)'s `llama-bench`. We report **speed only** (prefill and
  generation tok/s) — *never* model quality, which is a property of the model, not the GPU. A fixed **universal**
  tiny model runs on every card (the cross-card comparable number); larger models run on a **VRAM-gated ladder**
  (e.g. a 7B on an 8 GB card, bigger models on bigger cards), and one that doesn't fit is recorded as such, not
  silently skipped. Pinned model checksums + a pinned runner build make each number reproducible.

From the two phases we compute **deltas** — temperature drop, performance change, clock change — *including the
unimpressive and negative ones*.

## How we run it

- **Two phases, three runs each, averaged.** The goal is **consistency of method**, not laboratory precision.
- **Identical configuration** for before and after (same host, same OS image, same driver, same run settings) so
  the before/after comparison is apples-to-apples.
- **~1 Hz telemetry** is logged for the whole run (temp, clocks, power, fan, utilisation, throttle bitmask),
  which is what powers the temperature-vs-time and clock-vs-temperature curves.
- Each run is: a short **idle baseline**, then a **sustained load** held to thermal steady state, then capture.
- **LLM after the thermal runs; VRAM hot at the end of the soak** — the llama.cpp tok/s ladder runs once the short
  bench is captured; the memtest_vulkan VRAM pass runs at the end of the sustained soak (peak memory temperature).
  Both Vulkan tools are **pinned to the NVIDIA GPU** (our open-air rig also exposes an integrated GPU as a Vulkan
  device), so device selection is deterministic across the batch.
- Cards run in an **open-air eGPU rig** (GPU on a riser with its own PSU), which keeps the thermal baseline
  consistent and makes swaps fast across a large batch.

### Cross-era fairness

A new card will always out-score an old one on raw numbers, so absolute boards (hottest, top score) are paired
with **per-era / per-series** views and **normalised** measures — *percentage* improvement and
**performance-per-watt** — which compare a Kepler-era card and a modern one fairly.

## Honesty & limitations

- These are **standardised synthetic measurements**, not real-world game FPS.
- We hold method + configuration constant rather than normalising for **ambient room temperature**. Where an
  ambient sensor is fitted we now **log** it next to each sample so the variance is visible, but we don't correct
  for it — the *deltas* and the *throttle-reason shifts* remain the robust signal.
- **VRAM errors and LLM speed are reported as-is.** A VRAM error is a property of the card's silicon (cleaning
  won't fix it); LLM throughput is a *speed* number, not a statement about model quality.
- "Relative performance" is a consistent score for ranking, not a vendor spec.

## The script

[`gpu_bench.py`](./gpu_bench.py) is the harness for the modern path. It needs a Linux machine with an NVIDIA GPU
and driver, and:

- `nvidia-smi` (telemetry — driver-provided)
- `python3` + [PyTorch](https://pytorch.org) (the sustained FP32 load + TFLOPS)
- [`glmark2`](https://github.com/glmark2/glmark2) run against a headless X server on the GPU (the universal score)
- *(optional)* [`memtest_vulkan`](https://github.com/GpuZelenograd/memtest_vulkan) for `--vram`, and
  [`llama.cpp`](https://github.com/ggml-org/llama.cpp) (`llama-bench`, Vulkan build) + GGUF models for `--llm` —
  both Vulkan, so the same tools work across card generations (incl. the legacy path)

```bash
# per phase (before, then after) — short bench: temps, throttle, glmark2, FP32, LLM tok/s
python3 gpu_bench.py --label "before" --runs 3 --idle-secs 30 --load-secs 120 --llm
# per phase — sustained heat soak: throttle ceiling, stability, fresh-vs-soaked retention, hot VRAM
python3 gpu_bench.py --label "before" --soak-only --soak-secs 900 --vram
```

`--soak-only` runs the sustained heat soak — and with `--vram` it adds the hot memtest_vulkan pass at the end;
`--llm` runs the tok/s ladder over the models in `--models-dir`. Where a
metering smart plug / ambient probe are fitted, `--shelly-ip` and `--ambient-cmd` add isolated wall power and room
ambient to the telemetry. It writes a JSON summary plus the raw 1 Hz samples as CSV. A legacy-driver variant (for
Kepler-era cards that can't run current CUDA) emits the same `glmark2` universal score, the same VRAM/LLM (Vulkan)
numbers, and the same phase/run/sample shape.

## License

MIT — see [LICENSE](./LICENSE). Use it, adapt it, check our numbers.

---

Built for [DustyFans](https://dustyfans.com) by [Hake Hardware](https://github.com/hakehardware).
