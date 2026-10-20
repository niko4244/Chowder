# Kaggle int8 witness lane

The int8 placement (`expert_placement: cpu-transient-grad-int8`) ran out of
**host** memory on the Windows box: 28.6 GB private after load, **44.5 GiB**
during the second backward, free RAM 0.4 GiB, step 2 paging for 29 minutes.
This lane moves the run to a Kaggle session in two kernels:

- **capacity** (`probe.py`) — what a session gives the placement: RAM, disk, GPU,
  bf16, and the 18.8 GB grid held resident at 19.89 GB peak RSS. Measured on
  2026-10-20; see [RESULTS-2026-10-20.md](RESULTS-2026-10-20.md).
- **step** (`run_step.py`) — the real thing: the worker subprocess, the real
  checkpoint, `max_steps` steps, with the child's peak RSS and the host's lowest
  available memory sampled while it runs.

Both refuse to claim gates. The pre-registration's rows stay `not measured`
until a run finishes and the writer records a witness beside its adapter.

It answers three things with measurements:

- host RAM total and available, scratch disk, GPU model, compute capability,
  and whether that card supports **bf16** — the spec is bf16, and T4/P100 do not;
- the int8 grid's real size (40 layers x 160 experts x 2048 hidden x 512
  intermediate = 20.1 GB int8 + scales), and how many layers the session's RAM
  can actually hold;
- per-layer host RSS and device peak walking those layers through the worker's
  own `TransientLayerExpertsInt8`, which is where the Windows run grew by 16 GB
  during backward.

**It measures no gate.** The pre-registration's rows stay `not measured` until a
real reap160 run finishes and writes `placement-witnesses.json`. A capacity probe
that fits says a run *could* fit here; it does not say the placement passed.

## 0. The checkpoint dataset (once)

`Qwen3.6-REAP160` is local-only (43 GB at `G:/llm-cold/bf16/`); it is not on the
Hub and no dataset carried it, so it is uploaded once and reused:

```bash
cd /g/llm-cold/bf16/Qwen3.6-REAP160      # dataset-metadata.json lives here (isPrivate: true)
kaggle datasets create -p . -q
```

It is private. Nothing local can confirm what the site did with `isPrivate`, so
check the dataset page.

## 2. Generate the upload directories

From the repo root:

```bash
python kaggle/int8_witness/make_dataset.py --owner nikmarco --out /c/llm-work/moe-int8/kaggle/chowder-moe-int8
python kaggle/int8_witness/make_kernel.py  --owner nikmarco --kind capacity --out /c/llm-work/moe-int8/kaggle/kernels/capacity
python kaggle/int8_witness/make_kernel.py  --owner nikmarco --kind step     --out /c/llm-work/moe-int8/kaggle/kernels/step
```

`make_dataset.py` exports **the committed tree** (`git archive HEAD`), not the
worktree: the checkout carries other agents' uncommitted edits, and a kernel must
run the commit its evidence names. It prints that commit and the digests of the
files that matter.

## 3. Create the private code dataset (before a kernel is pushed)

```bash
kaggle datasets create -p /c/llm-work/moe-int8/kaggle/chowder-moe-int8
kaggle datasets status nikmarco/chowder-moe-int8
```

Check on kaggle.com that it is **Private**: `isPrivate` in the metadata is a
request, and nothing local can confirm what the site did with it.

## 4. Push and watch a kernel

```bash
kaggle kernels push -p /c/llm-work/moe-int8/kaggle/kernels/capacity   # the probe
kaggle kernels push -p /c/llm-work/moe-int8/kaggle/kernels/step       # real steps
kaggle kernels status nikmarco/chowder-int8-placement-capacity-probe
kaggle kernels status nikmarco/chowder-int8-witness-step
```

The site slugs a kernel from its title, so a metadata id that does not resolve to
that slug is refused with a 409 on the second push; `make_kernel.py` carries the
resolved slugs.

## 5. Read the results

```bash
kaggle kernels output nikmarco/chowder-int8-placement-capacity-probe -p /c/llm-work/moe-int8/kaggle/out/capacity
kaggle kernels output nikmarco/chowder-int8-witness-step -p /c/llm-work/moe-int8/kaggle/out/step
```

`int8_capacity.json` carries `environment`, `plan`, `probe` and one `verdict`
string, with per-layer `samples` showing whether `rss_gb` climbs across layers.
`real_step.json` carries the worker's exit code, wall time, `peak_child_rss_gb`,
`min_host_available_gb`, the worker's own telemetry, and `witness_written`. A
one-step run can populate the gradient witness; smoke and router balance need the
full 20 steps and stay `not measured` until a run does them.

## Notes

- The probe chooses its layer count from the session's *available* RAM and
  records `layers_probed`; a truncated probe must never be read as the full grid.
- Both fill the int8 buffers rather than zeroing them, because lazily mapped pages
  would under-count RSS, and both hold the grid resident for the whole walk —
  freeing each layer would measure a run that does not exist.
- Inputs mount at `/kaggle/input/datasets/<owner>/<slug>` on the current image;
  both kernels find their data by content so a layout change fails loudly with
  what was actually mounted instead of a guessed path.
- GPU quota is shared: 30 GPU-hours a week, 12 hours per session. The probe is
  minutes; the 20-step run measured 1031 s/step on the 16 GB Blackwell card, and
  a T4 at compute capability 7.5 emulates bf16, so it is not faster.
- The shapes this account can request through the CLI are T4 and P100; a T4
  reports `is_bf16_supported()` true under torch 2.11, but that is emulation.
