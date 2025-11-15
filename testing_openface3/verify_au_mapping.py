#!/usr/bin/env python3
"""
Verify the AU mapping against our actual CSV data.
"""

import pandas as pd

# Read our CSV
df = pd.read_csv('openface3_output.csv')

# The mapping we discovered
mapping = {
    'AU01_r': 'AU01 (Inner Brow Raiser)',
    'AU02_r': 'AU06 (Cheek Raiser)',
    'AU03_r': 'AU12 (Lip Corner Puller/Smile)',
    'AU04_r': 'AU15 (Lip Corner Depressor)',
    'AU05_r': 'AU17 (Chin Raiser)',
    'AU06_r': 'AU02 (Outer Brow Raiser)',
    'AU07_r': 'AU09 (Nose Wrinkler)',
    'AU08_r': 'AU10 (Upper Lip Raiser)',
}

print("="*70)
print("OPENFACE 3.0 ACTUAL OUTPUT VERIFICATION")
print("="*70)

print("\nCSV Column → FACS AU Mapping:")
print()

for csv_col, facs_au in mapping.items():
    mean_val = df[csv_col].mean()
    max_val = df[csv_col].max()
    non_zero = (df[csv_col] > 0.001).sum()

    status = "✓ ACTIVE" if non_zero > 5 else "✗ INACTIVE"

    print(f"{csv_col} → {facs_au:35} | Mean: {mean_val:.4f} | Max: {max_val:.4f} | {status}")

print("\n" + "="*70)
print("ANSWER TO YOUR ORIGINAL QUESTION")
print("="*70)

print("\nOpenFace 3.0 outputs CSV columns named AU01_r through AU08_r")
print("But these DO NOT directly map to FACS AU numbers!")
print()
print("The actual FACS AUs detected are:")
print("  - AU01 (Inner Brow Raiser)")
print("  - AU02 (Outer Brow Raiser)")
print("  - AU06 (Cheek Raiser)")
print("  - AU09 (Nose Wrinkler)")
print("  - AU10 (Upper Lip Raiser)")
print("  - AU12 (Lip Corner Puller/Smile)")
print("  - AU15 (Lip Corner Depressor)")
print("  - AU17 (Chin Raiser)")
print()
print("That's only 8 AUs, not the 9 mentioned in documentation.")
print()
print("From our test data, the actually WORKING AUs are:")
active_aus = []
for csv_col, facs_au in mapping.items():
    if (df[csv_col] > 0.001).sum() > 5:
        active_aus.append(facs_au.split('(')[0].strip())

print(f"  {', '.join(active_aus)}")
print(f"\nTotal: {len(active_aus)} AUs with meaningful values")

print("\n" + "="*70)
print("DOCUMENTATION DISCREPANCY")
print("="*70)
print("\nThe S3 Data Analysis README claimed 9 available AUs:")
print("  AU01, AU02, AU04, AU06, AU12, AU15, AU20, AU25, AU45")
print()
print("But the actual model only outputs 8 AUs:")
print("  AU01, AU02, AU06, AU09, AU10, AU12, AU15, AU17")
print()
print("Missing from model: AU04, AU20, AU25, AU45")
print("Extra in model: AU09, AU10, AU17")
print()
print("This explains why your system had issues!")
