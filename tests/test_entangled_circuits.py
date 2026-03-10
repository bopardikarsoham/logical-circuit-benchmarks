"""
Phase 2 Verification — run with: python verify_phase2.py
"""

import stim
import numpy as np
from steane import (
    encoding_circuit, logical_h, logical_cnot,
    DATA_QUBITS, N_QUBITS,
)
from bell_ghz import bell_state_circuit, ghz_state_circuit

results = {}


# ── Helper ─────────────────────────────────────────────────────────────────────

def check_stabilizer(sim: stim.TableauSimulator,
                     pauli_str: str, label: str,
                     expect: int) -> bool:
    val = sim.peek_observable_expectation(stim.PauliString(pauli_str))
    ok  = (val == expect)
    print(f"  {label}: got {val:+d}  (expect {expect:+d})  {'✓' if ok else '✗'}")
    return ok


# ── CHECK 1: Bell state stabilizers ───────────────────────────────────────────
print("--- CHECK 1: Bell state stabilizers ---")
# After H_L(0) then CNOT_L(0→1) on |00>_L:
# Logical stabilizers of (|00>+|11>)/sqrt(2):
#   X_L X_L → +1
#   Z_L Z_L → +1
# In physical terms (transversal):
#   X on all 14 data qubits → +1
#   Z on all 14 data qubits → +1

sim = stim.TableauSimulator()
sim.do(encoding_circuit(block=0))
sim.do(encoding_circuit(block=1))
sim.do(logical_h(block=0))
sim.do(logical_cnot(ctrl_block=0, tgt_block=1))

ok = True
# X_L X_L = X on all 14 data qubits (blocks 0 and 1), ancillas = I
xl_xl = "X" * 7 + "I" + "X" * 7 + "I"   # 16 chars
ok &= check_stabilizer(sim, xl_xl, "X_L⊗X_L", +1)

# Z_L Z_L = Z on all 14 data qubits
zl_zl = "Z" * 7 + "I" + "Z" * 7 + "I"
ok &= check_stabilizer(sim, zl_zl, "Z_L⊗Z_L", +1)

# Also verify the individual qubits are entangled:
# Z_L on block 0 alone should be 0 (not in eigenstate individually)
zl_i = "Z" * 7 + "I" * 9
ok_entangled = (sim.peek_observable_expectation(stim.PauliString(zl_i)) == 0)
print(f"  Z_L⊗I  : got {sim.peek_observable_expectation(stim.PauliString(zl_i)):+d}"
      f"  (expect 0 — qubits are entangled, not in Z eigenstate)  "
      f"{'✓' if ok_entangled else '✗'}")
ok &= ok_entangled

print(f"  {'✓ PASSED' if ok else '✗ FAILED'}")
results["1. Bell state stabilizers"] = ok


# ── CHECK 2: Bell state measurement correlations ──────────────────────────────
print("\n--- CHECK 2: Bell state measurement correlations ---")
# Sample from the Bell state circuit (no noise).
# Measure logical Z on each block: should always get 00 or 11, never 01 or 10.
# Logical Z_L = parity of all 7 data qubits in a block.

c = stim.Circuit()
c += encoding_circuit(block=0)
c += encoding_circuit(block=1)
c += logical_h(block=0)
c += logical_cnot(ctrl_block=0, tgt_block=1)
# Measure all data qubits of both blocks
c.append("M", list(range(7)) + list(range(8, 15)))

sampler  = c.compile_sampler()
samples  = sampler.sample(shots=1000).astype(int)   # shape (1000, 14)

# Logical Z_L = parity of 7 data qubits per block
parity_b0 = np.sum(samples[:, :7],  axis=1) % 2   # block 0
parity_b1 = np.sum(samples[:, 7:],  axis=1) % 2   # block 1

always_equal   = np.all(parity_b0 == parity_b1)
roughly_half_0 = 400 < np.sum(parity_b0 == 0) < 600   # ~50% in each state

print(f"  Shots where Z_L(0)=Z_L(1) : {np.sum(parity_b0 == parity_b1)}/1000  (expect 1000)")
print(f"  Shots with Z_L = 0        : {np.sum(parity_b0 == 0)}/1000  (expect ~500)")
print(f"  Shots with Z_L = 1        : {np.sum(parity_b0 == 1)}/1000  (expect ~500)")

ok = always_equal and roughly_half_0
print(f"  {'✓ PASSED' if ok else '✗ FAILED'}")
results["2. Bell state correlations"] = ok


# ── CHECK 3: GHZ state stabilizers ────────────────────────────────────────────
print("\n--- CHECK 3: GHZ state stabilizers ---")
# After H_L(0), CNOT_L(0→1), CNOT_L(0→2) on |000>_L:
# Logical stabilizers of (|000>+|111>)/sqrt(2):
#   X_L X_L X_L → +1
#   Z_L Z_L I_L → +1
#   I_L Z_L Z_L → +1

sim3 = stim.TableauSimulator()
sim3.do(encoding_circuit(block=0))
sim3.do(encoding_circuit(block=1))
sim3.do(encoding_circuit(block=2))
sim3.do(logical_h(block=0))
sim3.do(logical_cnot(ctrl_block=0, tgt_block=1))
sim3.do(logical_cnot(ctrl_block=0, tgt_block=2))

# 24 qubits total: 8 per block (7 data + 1 ancilla)
ok = True

xxx = "X"*7 + "I" + "X"*7 + "I" + "X"*7 + "I"   # 24 chars
ok &= check_stabilizer(sim3, xxx, "X_L⊗X_L⊗X_L", +1)

zzi = "Z"*7 + "I" + "Z"*7 + "I" + "I"*7 + "I"
ok &= check_stabilizer(sim3, zzi, "Z_L⊗Z_L⊗I_L", +1)

izz = "I"*7 + "I" + "Z"*7 + "I" + "Z"*7 + "I"
ok &= check_stabilizer(sim3, izz, "I_L⊗Z_L⊗Z_L", +1)

# Individual Z_L should be 0 (entangled)
zii = "Z"*7 + "I"*17
entangled = (sim3.peek_observable_expectation(stim.PauliString(zii)) == 0)
print(f"  Z_L⊗I⊗I: got {sim3.peek_observable_expectation(stim.PauliString(zii)):+d}"
      f"  (expect 0 — entangled)  {'✓' if entangled else '✗'}")
ok &= entangled

print(f"  {'✓ PASSED' if ok else '✗ FAILED'}")
results["3. GHZ state stabilizers"] = ok


# ── CHECK 4: GHZ measurement correlations ────────────────────────────────────
print("\n--- CHECK 4: GHZ state measurement correlations ---")
# All three blocks should always give same Z_L outcome: 000 or 111.

c3 = stim.Circuit()
c3 += encoding_circuit(block=0)
c3 += encoding_circuit(block=1)
c3 += encoding_circuit(block=2)
c3 += logical_h(block=0)
c3 += logical_cnot(ctrl_block=0, tgt_block=1)
c3 += logical_cnot(ctrl_block=0, tgt_block=2)
c3.append("M", list(range(7)) + list(range(8, 15)) + list(range(16, 23)))

samples3 = c3.compile_sampler().sample(shots=1000).astype(int)
p0 = np.sum(samples3[:, :7],   axis=1) % 2
p1 = np.sum(samples3[:, 7:14], axis=1) % 2
p2 = np.sum(samples3[:, 14:],  axis=1) % 2

all_equal     = np.all((p0 == p1) & (p1 == p2))
roughly_half  = 400 < np.sum(p0 == 0) < 600

print(f"  All three Z_L equal  : {np.sum((p0==p1)&(p1==p2))}/1000  (expect 1000)")
print(f"  Shots with Z_L = 0   : {np.sum(p0 == 0)}/1000  (expect ~500)")
print(f"  Shots with Z_L = 1   : {np.sum(p0 == 1)}/1000  (expect ~500)")

ok = all_equal and roughly_half
print(f"  {'✓ PASSED' if ok else '✗ FAILED'}")
results["4. GHZ correlations"] = ok


# ── CHECK 5: Full Bell circuit with syndrome rounds (zero noise) ───────────────
print("\n--- CHECK 5: Bell circuit with syndrome rounds (zero noise) ---")
bell = bell_state_circuit(noise=0.0, syndrome_rounds_per_gate=1)
det_s, obs_s = bell.compile_detector_sampler().sample(
    shots=500, separate_observables=True)
det_fires  = int(np.sum(det_s))
obs_errors = int(np.sum(obs_s))
print(f"  Qubits     : {bell.num_qubits}")
print(f"  Detectors  : {bell.num_detectors}")
print(f"  Det. fires : {det_fires}  (expect 0)")
print(f"  Obs errors : {obs_errors}  (expect 0)")
ok = (det_fires == 0 and obs_errors == 0)
print(f"  {'✓ PASSED' if ok else '✗ FAILED'}")
results["5. Bell circuit zero-noise"] = ok


# ── CHECK 6: Full GHZ circuit with syndrome rounds (zero noise) ───────────────
print("\n--- CHECK 6: GHZ circuit with syndrome rounds (zero noise) ---")
ghz = ghz_state_circuit(noise=0.0, syndrome_rounds_per_gate=1)
det_s, obs_s = ghz.compile_detector_sampler().sample(
    shots=500, separate_observables=True)
det_fires  = int(np.sum(det_s))
obs_errors = int(np.sum(obs_s))
print(f"  Qubits     : {ghz.num_qubits}")
print(f"  Detectors  : {ghz.num_detectors}")
print(f"  Det. fires : {det_fires}  (expect 0)")
print(f"  Obs errors : {obs_errors}  (expect 0)")
ok = (det_fires == 0 and obs_errors == 0)
print(f"  {'✓ PASSED' if ok else '✗ FAILED'}")
results["6. GHZ circuit zero-noise"] = ok


# ── CHECK 7: Noisy Bell circuit ───────────────────────────────────────────────
print("\n--- CHECK 7: Noisy Bell circuit (p=0.001) ---")
bell_noisy = bell_state_circuit(noise=0.001, syndrome_rounds_per_gate=1)
det_s, obs_s = bell_noisy.compile_detector_sampler().sample(
    shots=1000, separate_observables=True)
print(f"  Det. fires : {int(np.sum(det_s))}   (expect > 0)")
print(f"  Obs errors : {int(np.sum(obs_s))}  (expect small)")
ok = (int(np.sum(det_s)) > 0 and int(np.sum(obs_s)) < 200)
print(f"  {'✓ PASSED' if ok else '✗ REVIEW'}")
results["7. Noisy Bell circuit"] = ok


# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  PHASE 2 VERIFICATION SUMMARY")
print("="*55)
for name, passed in results.items():
    print(f"  {'✓' if passed else '✗'} {name}")
all_ok = all(results.values())
print(f"\n  {'ALL CHECKS PASSED ✓ — ready for Phase 3' if all_ok else 'SOME FAILED ✗'}")
print("="*55)