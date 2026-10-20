# Kaggle int8 placement capacity probe

The int8 placement (`expert_placement: cpu-transient-grad-int8`) ran out of
**host** memory on the Windows box: 28.6 GB private after load, **44.5 GiB**
during the second backward, free RAM 0.4 GiB, step 2 paging for 29 minutes.
This lane asks whether a Kaggle session can host the run before a 43 GB
checkpoint is uploaded to it.

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

## 0. Generate the upload directories

From the repo root:

```bash
python kaggle/int8_witness/make_dataset.py --owner nikmarco --out /c/llm-work/moe-int8/kaggle/chowder-moe-int8
python kaggle/int8_witness/make_kernel.py  --owner nikmarco --out /c/llm-work/moe-int8/kaggle/kernels/capacity
```

`make_dataset.py` exports **the committed tree** (`git archive HEAD`), not the
worktree: the checkout carries other agents' uncommitted edits, and a kernel must
run the commit its evidence names. It prints that commit and the digests of the
files that matter.

## 1. Create the private code dataset (before the kernel is pushed)

```bash
kaggle datasets create -p /c/llm-work/moe-int8/kaggle/chowder-moe-int8
kaggle datasets status nikmarco/chowder-moe-int8
```

Check on kaggle.com that it is **Private**: `isPrivate` in the metadata is a
request, and nothing local can confirm what the site did with it.

## 2. Push and watch the probe

```bash
kaggle kernels push -p /c/llm-work/moe-int8/kaggle/kernels/capacity
kaggle kernels status nikmarco/chowder-int8-placement-capacity
```

## 3. Read the verdict

```bash
kaggle kernels output nikmarco/chowder-int8-placement-capacity -p /c/llm-work/moe-int8/kaggle/out/capacity
```

`int8_capacity.json` carries `environment`, `plan`, `probe` and one `verdict`
string. The per-layer `samples` are the point: if `rss_gb` climbs across layers
on Linux too, the 16 GB growth was not WDDM paging and no free-tier session
(about 30 GB RAM) will complete the run.

## Notes

- The probe chooses its layer count from the session's *available* RAM and
  records `layers_probed`; a truncated probe must never be read as the full grid.
- It fills the int8 buffers rather than zeroing them, because lazily mapped pages
  would under-count RSS.
- The 43 GB checkpoint is not on the Hub and no dataset carries it; the real
  witness run needs a private dataset of `Qwen3.6-REAP160` first. This probe
  costs one short GPU session and decides whether that upload is worth it.
- GPU quota is shared: 30 GPU-hours a week, 12 hours per session. The probe is
  minutes; the 20-step run measured 1031 s/step on the 16 GB card.
