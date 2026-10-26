"""Render the REAP128_v2 measured-vs-reference scorecard chart.

Top panel: what this repository actually measured, both quantized arms on the
full frozen populations (GSM8K 1,319 and HumanEval 164 items) with sustained
source-bound generation speed.

Bottom panel: the 11 screenshot task environments. REAP has no measured score
for any of them, so each cell is printed as UNKNOWN rather than zero. Vendor
reference numbers are plotted as markers, hollow and starred where the vendor's
benchmark version or tool setting does not match the screenshot row.

Usage: python scripts/reap_v2_frontier_scorecard_chart.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parents[1]
DATA = HERE / 'docs' / 'quals' / 'reap-v2-frontier-scorecard-20261026.json'
OUT = HERE / 'docs' / 'quals' / 'reap-v2-frontier-scorecard-20261026.png'

INK = '#1b1b1b'
MUTED = '#6b6b6b'
MEASURED = '#1f6feb'
MEASURED_ALT = '#8ab4f8'
UNKNOWN_BAND = '#efefef'


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def draw_measured(ax, data: dict) -> None:
    arms = [('reap128_v2_q4_k_m', 'Q4_K_M', MEASURED),
            ('reap128_v2_q8_0', 'Q8_0', MEASURED_ALT)]
    suites = [('gsm8k', 'GSM8K\n1,319 items'), ('humaneval', 'HumanEval\n164 items')]
    width = 0.34
    ticks = []
    for index, (suite_key, label) in enumerate(suites):
        speeds = []
        for offset, (arm_key, _, colour) in enumerate(arms):
            suite = data['measured'][arm_key]['suites'][suite_key]
            pct = suite['accuracy'] * 100
            x = index + (offset - 0.5) * width
            ax.bar(x, pct, width * 0.92, color=colour, edgecolor=INK, linewidth=0.7, zorder=3)
            ax.text(x, pct + 1.4, f'{pct:.2f}%\n{suite["correct"]}/{suite["n"]}',
                    ha='center', va='bottom', fontsize=8.5, color=INK, linespacing=1.25)
            speeds.append(f'{suite["generated_tokens_per_wall_second"]:.1f}')
        ticks.append(f'{label}\n{float(speeds[0]):.1f} / {float(speeds[1]):.1f} tok/s (Q4 / Q8)')
    ax.set_xticks(range(len(suites)))
    ax.set_xticklabels(ticks, fontsize=8.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel('accuracy (%)', fontsize=9)
    ax.grid(axis='y', color='#dddddd', linewidth=0.6, zorder=0)
    handles = [Patch(facecolor=colour, edgecolor=INK, label=f'REAP128_v2 {label}') for _, label, colour in arms]
    ax.legend(handles=handles, fontsize=8, frameon=False, ncol=2,
              loc='upper center', bbox_to_anchor=(0.5, -0.20))
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)


def draw_categories(ax, data: dict) -> None:
    categories = [row['name'] for row in data['screenshot_categories']]
    modes = [row['mode'] for row in data['screenshot_categories']]
    refs = data['reference_models']
    styles = {
        'qwen3.8-27b': dict(marker='o', color='#b3261e', label=refs['qwen3.8-27b']['label']),
        'opus-5.5': dict(marker='D', color='#6a1b9a', label=refs['opus-5.5']['label']),
    }
    ys = list(range(len(categories)))[::-1]
    for y, name in zip(ys, categories):
        ax.axhspan(y - 0.42, y + 0.42, color=UNKNOWN_BAND, zorder=0)
        ax.text(1.2, y, 'REAP: UNKNOWN', fontsize=7.5, color=MUTED, va='center', style='italic')
        for key, style in styles.items():
            score = refs[key]['scores'].get(name)
            if not score:
                continue
            filled = score['harness_match']
            ax.plot(score['value'], y, marker=style['marker'], markersize=7.5,
                    markerfacecolor=style['color'] if filled else 'white',
                    markeredgecolor=style['color'], markeredgewidth=1.6,
                    linestyle='none', zorder=4)
            suffix = '' if filled else '*'
            ax.annotate(f'{score["value"]:g}{suffix}', (score['value'], y),
                        textcoords='offset points', xytext=(0, 9), ha='center',
                        fontsize=7.5, color=style['color'])
    ax.set_yticks(ys)
    ax.set_yticklabels([f'{n}\n({m})' for n, m in zip(categories, modes)], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(categories) - 0.3)
    ax.set_xlabel('published vendor reference score (%) — not a same-harness comparison', fontsize=9)
    ax.grid(axis='x', color='#dddddd', linewidth=0.6, zorder=0)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    handles = [Line2D([], [], marker=s['marker'], color=s['color'], linestyle='none',
                      markersize=7, label=s['label']) for s in styles.values()]
    handles.append(Line2D([], [], marker='o', markerfacecolor='white', markeredgecolor=MUTED,
                          linestyle='none', markersize=7, label='hollow + * = version or tool setting differs'))
    handles.append(Patch(facecolor=UNKNOWN_BAND, label='no REAP measurement exists'))
    ax.legend(handles=handles, fontsize=7.5, loc='lower right', ncol=2,
              frameon=True, facecolor='white', edgecolor='none', framealpha=1.0)


def render(data: dict, out: pathlib.Path) -> pathlib.Path:
    fig = plt.figure(figsize=(13.2, 13.4), dpi=150)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.6], hspace=0.62,
                            left=0.185, right=0.975, top=0.895, bottom=0.105)
    ax_a = fig.add_subplot(grid[0])
    ax_b = fig.add_subplot(grid[1])
    draw_measured(ax_a, data)
    draw_categories(ax_b, data)
    fig.suptitle('Chowder REAP128_v2 Q4_K_M / Q8_0 — measured results vs published frontier references',
                 fontsize=14, x=0.012, ha='left', y=0.975, color=INK)
    fig.text(0.012, 0.948,
             'Frozen deterministic selector: status BLOCKED, chosen null, release not approved. '
             'No REAP score is claimed for any screenshot benchmark.',
             fontsize=8.5, color=MUTED, ha='left')
    ax_a.set_title('A. Actually measured in this repository — same harness, full frozen populations',
                   fontsize=11, loc='left', color=INK, pad=10)
    ax_b.set_title('B. Screenshot task environments — REAP cells unmeasured, marks are vendor-published',
                   fontsize=11, loc='left', color=INK, pad=10)
    fig.text(0.012, 0.030,
             'Panel A: Linux Kaggle 2xT4, frozen evaluator and decoding; speeds are sustained source-bound generation. '
             'Status is PROVISIONAL vs the Windows\nserving-owner gate, and the HumanEval figure is the original scorer '
             'boolean, not a new isolated-sandbox execution.',
             fontsize=7.5, color=MUTED, ha='left', va='bottom')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=pathlib.Path, default=DATA)
    parser.add_argument('--out', type=pathlib.Path, default=OUT)
    args = parser.parse_args()
    print(render(load(args.data), args.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
