"""
Phase 1 Verification — run with: python verify_phase1.py
All checks must PASS before moving to Phase 2.
"""

import stim
import numpy as np
from steane import (
    encoding_circuit, syndrome_round, final_measurement,
    steane_memory_circuit, logical_h, logical_x, logical_z,
    STABILIZERS, STAB_LABELS, LOG_X, LOG_Z,
    DATA_QUBITS, ANCILLA,
)

results = {}


# ── CHECK 1: Stabilizers mutually commute ─────────────────────────────────────
print("--- CHECK 1: Stabilizer commutativity ---")
ok = True
for i in range(len(STABILIZERS)):
    for j in range(i+1, len(STABILIZERS)):
        if not STABILIZERS[i].commutes(STABILIZERS[j]):
            print(f"  ✗ {STAB_LABELS[i]} and {STAB_LABELS[j]} do not commute")
            ok = False
print(f"  {'✓ All 6 stabilizers commute' if ok else '✗ FAILED'}")
results["1. Stabilizer commutativity"] = ok


# ── CHECK 2: Logical operators valid ──────────────────────────────────────────
print("\n--- CHECK 2: Logical operator validity ---")
ok_comm = True
for label, stab in zip(STAB_LABELS, STABILIZERS):
    if not LOG_X.commutes(stab):
        print(f"  ✗ X_L does not commute with {label}")
        ok_comm = False
    if not LOG_Z.commutes(stab):
        print(f"  ✗ Z_L does not commute with {label}")
        ok_comm = False
if ok_comm:
    print("  ✓ X_L and Z_L commute with all stabilizers")
ok_anti = not LOG_X.commutes(LOG_Z)
print(f"  {'✓ X_L and Z_L anticommute' if ok_anti else '✗ X_L and Z_L should anticommute'}")
ok = ok_comm and ok_anti
results["2. Logical operator validity"] = ok


# ── CHECK 3: Encoding satisfies all 6 stabilizers ────────────────────────────
print("\n--- CHECK 3: Encoding circuit satisfies stabilizers ---")
sim = stim.TableauSimulator()
sim.do(encoding_circuit(block=0))
ok = True
for label, stab in zip(STAB_LABELS, STABILIZERS):
    padded = stim.PauliString(str(stab) + "I")   # pad to 8 qubits
    val = sim.peek_observable_expectation(padded)
    print(f"  {label}: {'✓ +1' if val == 1 else f'✗ got {val}'}")
    if val != 1:
        ok = False
print(f"  {'✓ All +1' if ok else '✗ FAILED'}")
results["3. Encoding satisfies stabilizers"] = ok


# ── CHECK 4: Zero-noise memory circuit ───────────────────────────────────────
print("\n--- CHECK 4: Zero-noise memory circuit ---")
circuit = steane_memory_circuit(rounds=5, noise=0.0)
sampler = circuit.compile_detector_sampler()
det_s, obs_s = sampler.sample(shots=1000, separate_observables=True)
det_fires  = int(np.sum(det_s))
obs_errors = int(np.sum(obs_s))
print(f"  Detectors   : {circuit.num_detectors}")
print(f"  Det. fires  : {det_fires} / {1000 * circuit.num_detectors}  (expect 0)")
print(f"  Obs. errors : {obs_errors} / 1000  (expect 0)")
ok = (det_fires == 0 and obs_errors == 0)
print(f"  {'✓ PASSED' if ok else '✗ FAILED'}")
results["4. Zero-noise memory circuit"] = ok


# ── CHECK 5: Logical X maps |0>_L to |1>_L ───────────────────────────────────
print("\n--- CHECK 5: Logical X maps |0>_L to |1>_L ---")
# Use TableauSimulator directly.
# Z_L = ZZZZZZZ should give +1 on |0>_L and -1 on |1>_L.
# We pad to 8 qubits (ancilla = I).
zl = stim.PauliString("ZZZZZZZI")

sim0 = stim.TableauSimulator()
sim0.do(encoding_circuit(block=0))
val_before = sim0.peek_observable_expectation(zl)

sim1 = stim.TableauSimulator()
sim1.do(encoding_circuit(block=0))
sim1.do(logical_x(block=0))
val_after = sim1.peek_observable_expectation(zl)

print(f"  Z_L on |0>_L        : {val_before}   (expect +1)")
print(f"  Z_L on X_L|0>_L     : {val_after}    (expect -1)")
ok = (val_before == 1 and val_after == -1)
print(f"  {'✓ PASSED' if ok else '✗ FAILED'}")
results["5. Logical X maps |0>_L to |1>_L"] = ok


# ── CHECK 6: Logical H·H = Identity ──────────────────────────────────────────
print("\n--- CHECK 6: H·H = Identity ---")
# Apply H twice: stabilizers should be unchanged.
# Check all 6 stabilizers still give +1 after H·H.
sim_hh = stim.TableauSimulator()
sim_hh.do(encoding_circuit(block=0))
sim_hh.do(logical_h(block=0))
sim_hh.do(logical_h(block=0))
ok = True
for label, stab in zip(STAB_LABELS, STABILIZERS):
    padded = stim.PauliString(str(stab) + "I")
    val = sim_hh.peek_observable_expectation(padded)
    if val != 1:
        print(f"  ✗ {label} not +1 after H·H: got {val}")
        ok = False
print(f"  {'✓ All stabilizers +1 after H·H' if ok else '✗ FAILED'}")
results["6. Logical H·H = Identity"] = ok


# ── CHECK 7: Logical H maps X_L ↔ Z_L ────────────────────────────────────────
print("\n--- CHECK 7: Logical H maps X_L <-> Z_L ---")
# After encoding |0>_L and applying H_L:
#   Z_L expectation should become 0 (we're now in |+>_L, eigenstate of X_L)
#   X_L expectation should become +1
xl = stim.PauliString("XXXXXXXI")
zl = stim.PauliString("ZZZZZZZI")

sim_h = stim.TableauSimulator()
sim_h.do(encoding_circuit(block=0))
sim_h.do(logical_h(block=0))

xl_val = sim_h.peek_observable_expectation(xl)
zl_val = sim_h.peek_observable_expectation(zl)

print(f"  X_L after H_L on |0>_L : {xl_val}   (expect +1, now in |+>_L)")
print(f"  Z_L after H_L on |0>_L : {zl_val}   (expect  0, not eigenstate)")
ok = (xl_val == 1 and zl_val == 0)
print(f"  {'✓ PASSED' if ok else '✗ FAILED'}")
results["7. H maps X_L <-> Z_L"] = ok


# ── CHECK 8: Noisy circuit produces detections ───────────────────────────────
print("\n--- CHECK 8: Noisy circuit behaviour ---")
noisy = steane_memory_circuit(rounds=5, noise=0.001)
det_s, obs_s = noisy.compile_detector_sampler().sample(shots=1000, separate_observables=True)
det_fires  = int(np.sum(det_s))
obs_errors = int(np.sum(obs_s))
print(f"  Det. fires  : {det_fires}   (expect > 0)")
print(f"  Obs. errors : {obs_errors}  (expect small, << 500)")
ok = (det_fires > 0 and obs_errors < 100)
print(f"  {'✓ PASSED' if ok else '✗ REVIEW'}")
results["8. Noisy circuit detects errors"] = ok


# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  PHASE 1 VERIFICATION SUMMARY")
print("="*55)
for name, passed in results.items():
    print(f"  {'✓' if passed else '✗'} {name}")
all_ok = all(results.values())
print(f"\n  {'ALL CHECKS PASSED ✓ — ready for Phase 2' if all_ok else 'SOME FAILED ✗'}")
print("="*55)