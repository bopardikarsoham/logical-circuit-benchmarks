"""
Run this standalone — no steane.py needed.
Just paste and run: python debug_stabilizers.py
"""
import stim

# The 6 stabilizers exactly as strings
stab_strings = [
    "XXXXIII",   # X0: X on 0,1,2,3
    "IXXIXXI",   # X1: X on 1,2,4,5
    "IIIXXXX",   # X2: X on 3,4,5,6
    "ZZZZIII",   # Z0: Z on 0,1,2,3
    "IZZIZZI",   # Z1: Z on 1,2,4,5
    "IIIZZZZ",   # Z2: Z on 3,4,5,6
]
labels = ["X0", "X1", "X2", "Z0", "Z1", "Z2"]

print("=== Step 1: Print each stabilizer and its length ===")
for lbl, s in zip(labels, stab_strings):
    ps = stim.PauliString(s)
    print(f"  {lbl}: '{s}'  len={len(s)}  parsed='{ps}'")

print("\n=== Step 2: Check all pairs for commutativity ===")
stabs = [stim.PauliString(s) for s in stab_strings]
all_ok = True
for i in range(len(stabs)):
    for j in range(i+1, len(stabs)):
        comm = stabs[i].commutes(stabs[j])
        if not comm:
            print(f"  ✗ {labels[i]} and {labels[j]} DO NOT commute")
            print(f"      {labels[i]} = {stabs[i]}")
            print(f"      {labels[j]} = {stabs[j]}")
            # Show overlap
            s1 = stab_strings[i]
            s2 = stab_strings[j]
            overlap = [(k, s1[k], s2[k]) for k in range(7)
                       if s1[k] != 'I' and s2[k] != 'I']
            print(f"      Overlap qubits (both non-I): {overlap}")
            print(f"      Count of overlap: {len(overlap)} "
                  f"({'even=commute' if len(overlap)%2==0 else 'ODD=anticommute ← problem'})")
            all_ok = False

if all_ok:
    print("  ✓ All pairs commute")

print("\n=== Step 3: Manual overlap check X0 vs Z2 ===")
print("  X0 = XXXXIII → X at positions 0,1,2,3")
print("  Z2 = IIIZZZZ → Z at positions 3,4,5,6")
print("  Overlap: position 3 only → count=1 (ODD) → anticommute")
print()
print("  This means the standard Steane stabilizer strings")
print("  IIIXXXX and IIIZZZZ are WRONG for this code.")
print()
print("=== Step 4: Try the OTHER standard convention ===")
# Many textbooks use a different row ordering of H.
# The correct H matrix for [7,4,3] Hamming is:
#   row1: 1 1 0 1 1 0 0  → qubits 0,1,3,4
#   row2: 0 1 1 1 0 1 0  → qubits 1,2,3,5
#   row3: 0 0 0 1 1 1 1  → qubits 3,4,5,6  ← same as before
# OR the systematic form:
#   row1: 1 0 0 1 0 1 1
#   row2: 0 1 0 1 1 0 1
#   row3: 0 0 1 0 1 1 1

alt_conventions = [
    # Convention A — from Nielsen & Chuang
    ["XXXXIII", "XXIXXII", "XIXIXIX",
     "ZZZZIII", "ZZIIZZI", "ZIZIZIZ"],
    # Convention B — from QAS2024 exercise code
    ["XXXXIII", "IXXIXXI", "IIIXXXX",
     "ZZZZIII", "IZZIZZI", "IIIZZZZ"],
    # Convention C — from errorcorrectionzoo / Steane original
    ["IIIXXXX", "IXXIXXI", "XIXIXIX",
     "IIIZZZZ", "IZZIZZI", "ZIZIZIZ"],
]

conv_labels = ["Nielsen&Chuang", "QAS2024", "ErrCorrectZoo"]

for conv_label, conv in zip(conv_labels, alt_conventions):
    stabs_c = [stim.PauliString(s) for s in conv]
    ok = True
    for i in range(len(stabs_c)):
        for j in range(i+1, len(stabs_c)):
            if not stabs_c[i].commutes(stabs_c[j]):
                ok = False
                break
    print(f"  Convention '{conv_label}': {'✓ all commute' if ok else '✗ some anticommute'}")
    if ok:
        print(f"    Stabilizers: {conv}")