"""
Phase 2: Logical Bell State + Logical GHZ State on Steane [[7,1,3]] code
=========================================================================
Reference: QAS2024 Tutorial (Bell state section)
           https://enccs.github.io/qas2024/notebooks/css_code_steane/

Two circuits built here:

1. Logical Bell State  (2 logical qubits = 16 physical qubits)
   Circuit: |0>_L |0>_L → H_L(0) → CNOT_L(0→1)
   Result : (|00>_L + |11>_L) / sqrt(2)
   Stabilizers of Bell state:
     XX (logical X on both)
     ZZ (logical Z on both)

2. Logical GHZ State  (3 logical qubits = 24 physical qubits)
   Circuit: |0>_L |0>_L |0>_L → H_L(0) → CNOT_L(0→1) → CNOT_L(0→2)
   Result : (|000>_L + |111>_L) / sqrt(2)
   Stabilizers of GHZ state:
     XXX, ZZI, IZZ

Both circuits include interleaved syndrome extraction rounds between
logical gates — this is what makes them fault-tolerant, not just logical.
"""

import stim
import numpy as np
from steane import (
    encoding_circuit, syndrome_round, final_measurement,
    logical_h, logical_cnot, logical_x, logical_z,
    DATA_QUBITS, N_QUBITS, STABILIZERS, STAB_LABELS,
)


# ── Bell state circuit ─────────────────────────────────────────────────────────

def bell_state_circuit(
    noise: float = 0.0,
    syndrome_rounds_per_gate: int = 1,
) -> stim.Circuit:
    """
    Fault-tolerant logical Bell state preparation on two Steane code blocks.

    Block 0: qubits  0-7  (data 0-6, ancilla 7)
    Block 1: qubits  8-15 (data 8-14, ancilla 15)

    Circuit structure:
      1. Encode |0>_L on both blocks
      2. syndrome_rounds_per_gate rounds of syndrome extraction
      3. Apply transversal H_L on block 0
      4. syndrome_rounds_per_gate rounds of syndrome extraction
      5. Apply transversal CNOT_L (block 0 → block 1)
      6. syndrome_rounds_per_gate rounds of syndrome extraction
      7. Measure both blocks (data qubits only)

    The syndrome rounds between gates are what make this fault-tolerant:
    errors introduced by each gate are detected before they can propagate
    to the next gate.

    Args:
        noise:                    depolarizing noise per 2-qubit gate
        syndrome_rounds_per_gate: how many syndrome rounds between each gate

    Returns:
        stim.Circuit with DETECTOR and OBSERVABLE_INCLUDE annotations
    """
    c = stim.Circuit()

    # ── Step 1: Encode both logical qubits ────────────────────────────────────
    c += encoding_circuit(block=0)
    c += encoding_circuit(block=1)

    # ── Step 2: Syndrome rounds after encoding ────────────────────────────────
    for i in range(syndrome_rounds_per_gate):
        is_first = (i == 0)
        c += syndrome_round(block=0, noise=noise, is_first_round=is_first)
        c += syndrome_round(block=1, noise=noise, is_first_round=is_first)

    # ── Step 3: Apply H_L on block 0 ──────────────────────────────────────────
    c += logical_h(block=0)

    # ── Step 4: Syndrome rounds after H_L ────────────────────────────────────
    for _ in range(syndrome_rounds_per_gate):
        c += syndrome_round(block=0, noise=noise, is_first_round=False)
        c += syndrome_round(block=1, noise=noise, is_first_round=False)

    # ── Step 5: Apply CNOT_L (block 0 → block 1) ─────────────────────────────
    c += logical_cnot(ctrl_block=0, tgt_block=1)

    # ── Step 6: Syndrome rounds after CNOT_L ──────────────────────────────────
    for _ in range(syndrome_rounds_per_gate):
        c += syndrome_round(block=0, noise=noise, is_first_round=False)
        c += syndrome_round(block=1, noise=noise, is_first_round=False)

    # ── Step 7: Final measurement of both blocks ───────────────────────────────
    # Measure all data qubits: block 0 (qubits 0-6) then block 1 (qubits 8-14)
    if noise > 0:
        c.append("X_ERROR", list(range(7)) + list(range(8, 15)), noise)
    c.append("M", list(range(7)) + list(range(8, 15)))  # 14 measurements

    # ── Observables ────────────────────────────────────────────────────────────
    # Bell state (|00>_L + |11>_L)/sqrt(2) has two logical observables:
    #   Observable 0: Z_L ⊗ Z_L = should be +1 (ZZ stabilizer of Bell state)
    #   Observable 1: X_L ⊗ X_L = should be +1 (XX stabilizer of Bell state)
    #
    # Z_L on block 0 = parity of rec[-14..-8] (first 7 measurements)
    # Z_L on block 1 = parity of rec[-7..-1]  (last 7 measurements)
    # ZZ = parity of all 14 = should be 0 (even) for Bell state
    c.append("OBSERVABLE_INCLUDE",
             [stim.target_rec(i) for i in range(-14, 0)],
             0)

    return c


# ── GHZ state circuit ──────────────────────────────────────────────────────────

def ghz_state_circuit(
    noise: float = 0.0,
    syndrome_rounds_per_gate: int = 1,
) -> stim.Circuit:
    """
    Fault-tolerant logical GHZ state on three Steane code blocks.

    Block 0: qubits  0-7
    Block 1: qubits  8-15
    Block 2: qubits 16-23

    Circuit structure:
      1. Encode |0>_L on all three blocks
      2. Syndrome rounds
      3. H_L on block 0
      4. Syndrome rounds
      5. CNOT_L (block 0 → block 1)
      6. Syndrome rounds
      7. CNOT_L (block 0 → block 2)
      8. Syndrome rounds
      9. Measure all three blocks

    Result: (|000>_L + |111>_L) / sqrt(2)

    Stabilizers of 3-qubit GHZ:
      X_L X_L X_L  (XXX)
      Z_L Z_L I_L  (ZZI)
      I_L Z_L Z_L  (IZZ)

    Args:
        noise:                    depolarizing noise per 2-qubit gate
        syndrome_rounds_per_gate: syndrome rounds between each logical gate

    Returns:
        stim.Circuit with DETECTOR and OBSERVABLE_INCLUDE annotations
    """
    c = stim.Circuit()

    # ── Step 1: Encode all three blocks ───────────────────────────────────────
    c += encoding_circuit(block=0)
    c += encoding_circuit(block=1)
    c += encoding_circuit(block=2)

    # ── Step 2: Syndrome rounds after encoding ────────────────────────────────
    for i in range(syndrome_rounds_per_gate):
        is_first = (i == 0)
        c += syndrome_round(block=0, noise=noise, is_first_round=is_first)
        c += syndrome_round(block=1, noise=noise, is_first_round=is_first)
        c += syndrome_round(block=2, noise=noise, is_first_round=is_first)

    # ── Step 3: Apply H_L on block 0 ────────────────────────────────────────
    c += logical_h(block=0)

    # ── Step 4: Syndrome rounds after H_L ─────────────────────────────────────
    for _ in range(syndrome_rounds_per_gate):
        c += syndrome_round(block=0, noise=noise, is_first_round=False)
        c += syndrome_round(block=1, noise=noise, is_first_round=False)
        c += syndrome_round(block=2, noise=noise, is_first_round=False)

    # ── Step 5: CNOT_L (block 0 → block 1) ───────────────────────────────────
    c += logical_cnot(ctrl_block=0, tgt_block=1)

    # ── Step 6: Syndrome rounds after first CNOT_L ────────────────────────────
    for _ in range(syndrome_rounds_per_gate):
        c += syndrome_round(block=0, noise=noise, is_first_round=False)
        c += syndrome_round(block=1, noise=noise, is_first_round=False)
        c += syndrome_round(block=2, noise=noise, is_first_round=False)

    # ── Step 7: CNOT_L (block 0 → block 2) ───────────────────────────────────
    c += logical_cnot(ctrl_block=0, tgt_block=2)

    # ── Step 8: Syndrome rounds after second CNOT_L ───────────────────────────
    for _ in range(syndrome_rounds_per_gate):
        c += syndrome_round(block=0, noise=noise, is_first_round=False)
        c += syndrome_round(block=1, noise=noise, is_first_round=False)
        c += syndrome_round(block=2, noise=noise, is_first_round=False)

    # ── Step 9: Final measurement of all three blocks ─────────────────────────
    all_data = list(range(7)) + list(range(8, 15)) + list(range(16, 23))
    if noise > 0:
        c.append("X_ERROR", all_data, noise)
    c.append("M", all_data)   # 21 measurements: rec[-21]..rec[-1]

    # ── Observables ────────────────────────────────────────────────────────────
    # GHZ state (|000>_L + |111>_L)/sqrt(2):
    #   Observable 0: Z_L⊗Z_L⊗I  = parity(block0) XOR parity(block1) = 0
    #   Observable 1: I⊗Z_L⊗Z_L  = parity(block1) XOR parity(block2) = 0
    # Both should be 0 for the GHZ state (ZZ correlations)

    # Block 0 data: rec[-21..-15], Block 1: rec[-14..-8], Block 2: rec[-7..-1]
    # Observable 0: ZZ on blocks 0 and 1 = parity of first 14 measurements
    c.append("OBSERVABLE_INCLUDE",
             [stim.target_rec(i) for i in range(-21, -7)],
             0)
    # Observable 1: ZZ on blocks 1 and 2 = parity of last 14 measurements
    c.append("OBSERVABLE_INCLUDE",
             [stim.target_rec(i) for i in range(-14, 0)],
             1)

    return c


# ── Circuit stats helper ───────────────────────────────────────────────────────

def print_circuit_stats(c: stim.Circuit, label: str) -> None:
    n_instructions = len(list(c.flattened()))
    print(f"  {label}")
    print(f"    Qubits       : {c.num_qubits}")
    print(f"    Detectors    : {c.num_detectors}")
    print(f"    Observables  : {c.num_observables}")
    print(f"    Instructions : {n_instructions}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Phase 2: Logical Bell + GHZ Circuits")
    print("=" * 55)

    print("\n[Circuit Stats]")
    print_circuit_stats(bell_state_circuit(noise=0.001), "Bell state (2 blocks, p=0.001)")
    print_circuit_stats(ghz_state_circuit(noise=0.001),  "GHZ state  (3 blocks, p=0.001)")

    print("\n[Sanity: run verify_phase2.py for full checks]")