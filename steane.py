"""
Steane [[7,1,3]] Code — Phase 1: Memory Circuit
=================================================
Stabilizer reference : Wikipedia / errorcorrectionzoo.org / Nielsen & Chuang p.456
Encoding circuit     : rlftqc (github.com/remmyzen/rlftqc) — verified stim circuit

Stabilizer generators (H_X = H_Z = parity check matrix of [7,4,3] Hamming code):
    IIIXXXX   X on qubits 3,4,5,6
    IXXIIXX   X on qubits 1,2,5,6
    XIXIXIX   X on qubits 0,2,4,6
    IIIZZZZ   Z on qubits 3,4,5,6
    IZZIIZZ   Z on qubits 1,2,5,6   ← note: IZZIIZZ not IZZIZZI
    ZIZIZIZ   Z on qubits 0,2,4,6

All 6 mutually commute — verified (each X/Z pair overlaps on even # of qubits).

Logical operators:
    X_L = XXXXXXX  (transversal X on all 7)
    Z_L = ZZZZZZZ  (transversal Z on all 7)

Qubit layout per block (8 qubits total):
    Data    : 0, 1, 2, 3, 4, 5, 6
    Ancilla : 7
"""

import stim
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_QUBITS = list(range(7))
ANCILLA     = 7
N_QUBITS    = 8   # per block (7 data + 1 ancilla)

# Stabilizer supports (0-indexed)
X_STAB_SUPPORTS = [
    [3, 4, 5, 6],   # IIIXXXX
    [1, 2, 5, 6],   # IXXIIXX
    [0, 2, 4, 6],   # XIXIXIX
]
Z_STAB_SUPPORTS = [
    [3, 4, 5, 6],   # IIIZZZZ
    [1, 2, 5, 6],   # IZZIIZZ
    [0, 2, 4, 6],   # ZIZIZIZ
]

# Verified PauliStrings — all 6 mutually commute
STABILIZERS = [
    stim.PauliString("IIIXXXX"),
    stim.PauliString("IXXIIXX"),
    stim.PauliString("XIXIXIX"),
    stim.PauliString("IIIZZZZ"),
    stim.PauliString("IZZIIZZ"),
    stim.PauliString("ZIZIZIZ"),
]
STAB_LABELS = ["X0","X1","X2","Z0","Z1","Z2"]

# Logical operators (transversal = weight 7)
LOG_X = stim.PauliString("XXXXXXX")
LOG_Z = stim.PauliString("ZZZZZZZ")

# For the logical observable we use parity of all 7 data qubits (= Z_L)
LOGICAL_Z_SUPPORT = DATA_QUBITS


# ── Encoding circuit ───────────────────────────────────────────────────────────

def encoding_circuit(block: int = 0) -> stim.Circuit:
    """
    Prepare logical |0>_L.

    Encoding circuit from rlftqc (github.com/remmyzen/rlftqc),
    verified against stabilizers IIIXXXX / IXXIIXX / XIXIXIX:

        H 0 1 3
        CX 0 6
        CX 1 5
        CX 0 4
        CX 3 4
        CX 3 5
        CX 5 6
        CX 0 2
        CX 1 2

    This prepares the +1 eigenstate of all X stabilizers.
    The Z stabilizers are automatically satisfied for |0>_L since
    all physical qubits start in |0> and the circuit only involves
    H and CNOT gates (no Z rotations).
    """
    s = block * N_QUBITS
    c = stim.Circuit()

    c.append("R", [s + q for q in range(N_QUBITS)])
    c.append("TICK")

    c.append("H",    [s+0, s+1, s+3])
    c.append("CNOT", [s+0, s+6])
    c.append("CNOT", [s+1, s+5])
    c.append("CNOT", [s+0, s+4])
    c.append("CNOT", [s+3, s+4])
    c.append("CNOT", [s+3, s+5])
    c.append("CNOT", [s+5, s+6])
    c.append("CNOT", [s+0, s+2])
    c.append("CNOT", [s+1, s+2])
    c.append("TICK")

    return c


# ── Syndrome extraction ────────────────────────────────────────────────────────

def syndrome_round(
    block: int = 0,
    noise: float = 0.0,
    is_first_round: bool = False,
) -> stim.Circuit:
    """
    One full syndrome extraction round — measures all 6 stabilizers.

    Uses the single ancilla qubit (qubit 7) sequentially.
    Measurement order: X0, X1, X2, Z0, Z1, Z2  (6 measurements).

    X stabilizer:  R → H → CNOTs(anc→data) → H → M
    Z stabilizer:  R → CNOTs(data→anc) → M

    Detectors:
      First round  : fires if syndrome ≠ 0 (checks |0>_L baseline)
      Later rounds : fires if syndrome CHANGES round-to-round
    """
    s   = block * N_QUBITS
    c   = stim.Circuit()
    N_S = 6

    # ── X stabilizers ──────────────────────────────────────────────────────────
    for support in X_STAB_SUPPORTS:
        c.append("R", [s + ANCILLA])
        c.append("H", [s + ANCILLA])
        c.append("TICK")
        for d in support:
            c.append("CNOT", [s + ANCILLA, s + d])
            if noise > 0:
                c.append("DEPOLARIZE2", [s + ANCILLA, s + d], noise)
        c.append("TICK")
        c.append("H", [s + ANCILLA])
        if noise > 0:
            c.append("X_ERROR", [s + ANCILLA], noise)
        c.append("M", [s + ANCILLA])

    # ── Z stabilizers ──────────────────────────────────────────────────────────
    for support in Z_STAB_SUPPORTS:
        c.append("R", [s + ANCILLA])
        c.append("TICK")
        for d in support:
            c.append("CNOT", [s + d, s + ANCILLA])
            if noise > 0:
                c.append("DEPOLARIZE2", [s + d, s + ANCILLA], noise)
        c.append("TICK")
        if noise > 0:
            c.append("X_ERROR", [s + ANCILLA], noise)
        c.append("M", [s + ANCILLA])

    # ── Detectors ──────────────────────────────────────────────────────────────
    for i in range(N_S):
        cur = stim.target_rec(i - N_S)
        if is_first_round:
            c.append("DETECTOR", [cur])
        else:
            prev = stim.target_rec(i - 2 * N_S)
            c.append("DETECTOR", [cur, prev])

    return c


# ── Final measurement ──────────────────────────────────────────────────────────

def final_measurement(block: int = 0, noise: float = 0.0) -> stim.Circuit:
    """
    Measure all 7 data qubits and declare the logical Z observable.

    Final detectors: for each Z stabilizer, the parity of its data qubits
    in the final measurement must match the last syndrome round result.

    Logical Z observable = parity of all 7 data qubits (transversal Z_L).
    """
    s = block * N_QUBITS
    c = stim.Circuit()

    if noise > 0:
        c.append("X_ERROR", [s + q for q in DATA_QUBITS], noise)
    c.append("M", [s + q for q in DATA_QUBITS])   # rec[-7] .. rec[-1]

    # Last syndrome round had 6 measurements before the 7 data measurements.
    # Z syndromes were at positions -10 (Z0), -9 (Z1), -8 (Z2)
    # relative to end of the 7 data measurements.
    for i, support in enumerate(Z_STAB_SUPPORTS):
        z_rec  = stim.target_rec(-(7 + 3 - i))
        d_recs = [stim.target_rec(-(7 - q)) for q in support]
        c.append("DETECTOR", [z_rec] + d_recs)

    # Logical observable: parity of all 7 data qubits = transversal Z_L
    c.append("OBSERVABLE_INCLUDE",
             [stim.target_rec(-(7 - q)) for q in LOGICAL_Z_SUPPORT],
             0)
    return c


# ── Full memory experiment ─────────────────────────────────────────────────────

def steane_memory_circuit(
    rounds: int = 5,
    noise: float = 0.001,
    block: int = 0,
) -> stim.Circuit:
    """
    Full Steane memory experiment:
      encode |0>_L → syndrome rounds → final measurement + observable

    Standard QEC benchmark used for threshold estimation with sinter/pymatching.
    """
    c  = stim.Circuit()
    c += encoding_circuit(block=block)
    c += syndrome_round(block=block, noise=noise, is_first_round=True)
    for _ in range(rounds - 1):
        c += syndrome_round(block=block, noise=noise, is_first_round=False)
    c += final_measurement(block=block, noise=noise)
    return c


# ── Logical gates ──────────────────────────────────────────────────────────────

def logical_h(block: int = 0) -> stim.Circuit:
    """Transversal H on all 7 data qubits. Valid: Steane is self-dual CSS."""
    s = block * N_QUBITS
    c = stim.Circuit()
    c.append("H", [s + q for q in DATA_QUBITS])
    return c

def logical_x(block: int = 0) -> stim.Circuit:
    """Transversal X on all 7 data qubits (= X_L)."""
    s = block * N_QUBITS
    c = stim.Circuit()
    c.append("X", [s + q for q in DATA_QUBITS])
    return c

def logical_z(block: int = 0) -> stim.Circuit:
    """Transversal Z on all 7 data qubits (= Z_L)."""
    s = block * N_QUBITS
    c = stim.Circuit()
    c.append("Z", [s + q for q in DATA_QUBITS])
    return c

def logical_s(block: int = 0) -> stim.Circuit:
    """Transversal S on all 7 data qubits. Valid for Steane code."""
    s = block * N_QUBITS
    c = stim.Circuit()
    c.append("S", [s + q for q in DATA_QUBITS])
    return c

def logical_cnot(ctrl_block: int = 0, tgt_block: int = 1) -> stim.Circuit:
    """Transversal CNOT between two blocks. Valid for all CSS codes."""
    c = stim.Circuit()
    sc = ctrl_block * N_QUBITS
    st = tgt_block  * N_QUBITS
    for q in DATA_QUBITS:
        c.append("CNOT", [sc + q, st + q])
    return c