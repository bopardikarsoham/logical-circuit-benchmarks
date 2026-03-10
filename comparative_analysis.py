"""
Phase 3: Comparative Analysis — Steane [[7,1,3]] Code
======================================================
Three analyses:

  3a — Logical error rate vs noise (Memory circuit only with pymatching)
       Bell + GHZ use direct observable sampling without decoding.
       Reason: sequential single-ancilla syndrome extraction creates
       cross-block Y-errors that exceed pymatching's 15-detector limit.
       This is documented as a known limitation of the sequential ancilla
       architecture. Memory circuit uses pymatching correctly.

  3b — Logical error rate vs syndrome rounds (all three circuits)
       Uses direct observable sampling (no decoder needed).
       Measures raw logical error accumulation vs rounds.

  3c — Circuit overhead table
       Qubits, detectors, instructions per circuit.
       Quantifies the resource cost of each circuit.
"""

import stim
import sinter
import pymatching
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict
from steane import steane_memory_circuit
from bell_ghz import bell_state_circuit, ghz_state_circuit

NOISE_RATES  = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.02]
ROUNDS_LIST  = [1, 3, 5, 7, 10, 15]
FIXED_NOISE  = 0.001
FIXED_ROUNDS = 5
N_SHOTS      = 50_000
MAX_ERRORS   = 500
N_WORKERS    = max(1, os.cpu_count() - 1)

STYLES = {
    "Memory": {"color": "steelblue",  "marker": "o"},
    "Bell"  : {"color": "darkorange", "marker": "s"},
    "GHZ"   : {"color": "seagreen",   "marker": "^"},
}

os.makedirs("results", exist_ok=True)


def build_circuit(name, rounds, noise):
    if name == "Memory":
        return steane_memory_circuit(rounds=rounds, noise=noise)
    elif name == "Bell":
        return bell_state_circuit(
            noise=noise, syndrome_rounds_per_gate=max(1, rounds // 3))
    elif name == "GHZ":
        return ghz_state_circuit(
            noise=noise, syndrome_rounds_per_gate=max(1, rounds // 4))


def sample_logical_error_rate(circuit: stim.Circuit, shots: int) -> float:
    """
    Sample logical error rate directly from observable flips.
    No decoder used — measures raw (unдекoded) logical error rate.
    Used for Bell and GHZ where pymatching cannot decode due to
    cross-block hyper-errors from sequential ancilla reuse.
    """
    sampler = circuit.compile_detector_sampler()
    _, obs = sampler.sample(shots=shots, separate_observables=True)
    # any observable flip = logical error in that shot
    return float(np.mean(np.any(obs, axis=1)))


# ── Analysis 3a: Logical error rate vs noise ──────────────────────────────────

def run_noise_sweep():
    print("[3a] Noise sweep")
    print("     Memory: decoded with pymatching")
    print("     Bell/GHZ: raw observable flip rate (no decoder)")

    # Memory: use sinter + pymatching
    print("\n  Running Memory with sinter...")
    memory_tasks = [
        sinter.Task(
            circuit=steane_memory_circuit(rounds=FIXED_ROUNDS, noise=p),
            json_metadata={"circuit": "Memory", "p": p},
        )
        for p in NOISE_RATES
    ]
    memory_stats = sinter.collect(
        num_workers=N_WORKERS,
        tasks=memory_tasks,
        decoders=["pymatching"],
        max_shots=N_SHOTS,
        max_errors=MAX_ERRORS,
        print_progress=True,
    )

    # Bell + GHZ: direct sampling
    print("\n  Running Bell + GHZ with direct sampling...")
    bell_ghz_data = {}
    for name in ["Bell", "GHZ"]:
        xs, ys = [], []
        for p in NOISE_RATES:
            c    = build_circuit(name, FIXED_ROUNDS, p)
            rate = sample_logical_error_rate(c, shots=N_SHOTS)
            xs.append(p)
            ys.append(rate)
            print(f"    {name} p={p:.4f} → {rate:.4f}")
        bell_ghz_data[name] = (xs, ys)

    return memory_stats, bell_ghz_data


def plot_noise_sweep(memory_stats, bell_ghz_data):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Memory from sinter
    xs_m, ys_m, yerr_lo, yerr_hi = [], [], [], []
    for s in sorted(memory_stats, key=lambda x: x.json_metadata["p"]):
        if s.shots == 0:
            continue
        p    = s.json_metadata["p"]
        rate = s.errors / s.shots
        xs_m.append(p)
        ys_m.append(max(rate, 1e-7))
        n, k = s.shots, s.errors
        if k == 0:
            yerr_lo.append(0); yerr_hi.append(0)
        else:
            lo = k/n - 1.96*np.sqrt(k/n*(1-k/n)/n)
            hi = k/n + 1.96*np.sqrt(k/n*(1-k/n)/n)
            yerr_lo.append(max(rate-lo, 0))
            yerr_hi.append(max(hi-rate, 0))

    ax.errorbar(xs_m, ys_m, yerr=[yerr_lo, yerr_hi],
                fmt='o-', color=STYLES["Memory"]["color"],
                label='Memory (pymatching decoded)',
                capsize=4, linewidth=2, markersize=7)

    # Bell + GHZ direct
    for name in ["Bell", "GHZ"]:
        xs, ys = bell_ghz_data[name]
        ax.plot(xs, [max(y, 1e-7) for y in ys],
                f'{STYLES[name]["marker"]}--',
                color=STYLES[name]["color"],
                label=f'{name} (raw, undeсoded)',
                linewidth=2, markersize=7, alpha=0.8)

    ax.plot(NOISE_RATES, NOISE_RATES, 'k--', alpha=0.4,
            label='Physical (no encoding)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Physical Error Rate p', fontsize=13)
    ax.set_ylabel('Logical Error Rate per Shot', fontsize=13)
    ax.set_title(
        'Steane [[7,1,3]]: Logical Error Rate vs Physical Noise\n'
        'Memory (decoded) vs Bell/GHZ (raw) — circuit-level depolarizing',
        fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    path = "results/3a_error_rate_vs_noise.png"
    fig.savefig(path, dpi=150)
    print(f"\n  Saved: {path}")
    plt.show()


# ── Analysis 3b: Logical error rate vs rounds ─────────────────────────────────

def run_rounds_sweep():
    print("\n[3b] Rounds sweep — direct sampling all three circuits")
    data = {}
    for name in ["Memory", "Bell", "GHZ"]:
        xs, ys = [], []
        for r in ROUNDS_LIST:
            c    = build_circuit(name, r, FIXED_NOISE)
            rate = sample_logical_error_rate(c, shots=N_SHOTS)
            xs.append(r)
            ys.append(rate)
            print(f"  {name} rounds={r} → {rate:.4f}")
        data[name] = (xs, ys)
    return data


def plot_rounds_sweep(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (xs, ys) in data.items():
        ax.plot(xs, ys,
                f'{STYLES[name]["marker"]}-',
                color=STYLES[name]["color"],
                label=f'Logical {name}',
                linewidth=2, markersize=7)
    ax.set_xlabel('Syndrome Rounds', fontsize=13)
    ax.set_ylabel('Logical Error Rate per Shot', fontsize=13)
    ax.set_title(
        f'Steane [[7,1,3]]: Logical Error Rate vs Syndrome Rounds\n'
        f'p={FIXED_NOISE} — raw observable flip rate',
        fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    path = "results/3b_error_rate_vs_rounds.png"
    fig.savefig(path, dpi=150)
    print(f"\n  Saved: {path}")
    plt.show()


# ── Analysis 3c: Overhead table ───────────────────────────────────────────────

def compute_overhead_table():
    print("\n[3c] Circuit overhead table")
    rows = []
    for name in ["Memory", "Bell", "GHZ"]:
        c = build_circuit(name, FIXED_ROUNDS, FIXED_NOISE)
        rows.append({
            "Circuit"     : name,
            "Phys.Qubits" : c.num_qubits,
            "Log.Qubits"  : c.num_observables,
            "Detectors"   : c.num_detectors,
            "Instructions": len(list(c.flattened())),
        })
    return rows


def print_overhead_table(rows):
    print("\n" + "="*65)
    print("  3c — CIRCUIT OVERHEAD TABLE")
    print(f"  (noise p={FIXED_NOISE}, rounds={FIXED_ROUNDS})")
    print("="*65)
    print(f"  {'Circuit':<10} {'Phys.Qubits':>12} {'Log.Qubits':>11} "
          f"{'Detectors':>10} {'Instructions':>13}")
    print(f"  {'-'*60}")
    for r in rows:
        print(f"  {r['Circuit']:<10} "
              f"{r['Phys.Qubits']:>12} "
              f"{r['Log.Qubits']:>11} "
              f"{r['Detectors']:>10} "
              f"{r['Instructions']:>13}")
    print("="*65)


def plot_overhead(rows):
    names  = [r["Circuit"] for r in rows]
    colors = [STYLES[n]["color"] for n in names]
    metrics = {
        "Physical Qubits" : [r["Phys.Qubits"]   for r in rows],
        "Detectors"       : [r["Detectors"]      for r in rows],
        "Instructions"    : [r["Instructions"]   for r in rows],
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (metric, values) in zip(axes, metrics.items()):
        bars = ax.bar(names, values, color=colors, alpha=0.85, edgecolor='black')
        ax.set_title(metric, fontsize=12)
        ax.set_ylabel("Count", fontsize=11)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(values)*0.02,
                    str(val), ha='center', va='bottom', fontsize=11)
        ax.set_ylim(0, max(values)*1.2)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle(
        f'Steane [[7,1,3]]: Circuit Overhead — Memory vs Bell vs GHZ\n'
        f'(rounds={FIXED_ROUNDS})',
        fontsize=13)
    fig.tight_layout()
    path = "results/3c_overhead_table.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 3: Comparative Analysis")
    print("  Steane [[7,1,3]] — Memory vs Bell vs GHZ")
    print("=" * 60)

    memory_stats, bell_ghz_data = run_noise_sweep()
    plot_noise_sweep(memory_stats, bell_ghz_data)

    rounds_data = run_rounds_sweep()
    plot_rounds_sweep(rounds_data)

    overhead_rows = compute_overhead_table()
    print_overhead_table(overhead_rows)
    plot_overhead(overhead_rows)

    print("\n✓ Phase 3 complete. Results in results/")