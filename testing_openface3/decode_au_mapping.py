#!/usr/bin/env python3
"""
Decode the AU mapping from OpenFace 3.0 model outputs to FACS AU numbers.

From the training code, we found:
eval_au = [0, 4, 8, 10, 11, 1, 6, 7]

This selects 8 AUs from a larger list. Common AU dataset orderings:
"""

# Common AU orderings in datasets like BP4D, DISFA
# BP4D uses: AU1, AU2, AU4, AU6, AU7, AU10, AU12, AU14, AU15, AU17, AU23, AU24
# DISFA uses: AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26

# Most comprehensive ordering (similar to OpenFace 2.2):
standard_aus = [
    'AU01',  # 0
    'AU02',  # 1
    'AU04',  # 2
    'AU05',  # 3
    'AU06',  # 4
    'AU07',  # 5
    'AU09',  # 6
    'AU10',  # 7
    'AU12',  # 8
    'AU14',  # 9
    'AU15',  # 10
    'AU17',  # 11
    'AU20',  # 12
    'AU23',  # 13
    'AU25',  # 14
    'AU26',  # 15
    'AU45',  # 16
]

# OpenFace 3.0 selection indices
eval_au = [0, 4, 8, 10, 11, 1, 6, 7]

# Map model output indices to FACS AU numbers
print("="*70)
print("OPENFACE 3.0 AU MAPPING")
print("="*70)
print("\nModel outputs 8 AUs in this order:")
print()

for model_idx, dataset_idx in enumerate(eval_au):
    au_number = standard_aus[dataset_idx]
    print(f"  Model Output AU{model_idx+1:02d}_r  →  FACS {au_number}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

selected_aus = [standard_aus[i] for i in eval_au]
print(f"\nOpenFace 3.0 detects these {len(selected_aus)} AUs:")
print(f"  {', '.join(selected_aus)}")

print("\n\nFull FACS names:")
au_names = {
    'AU01': 'Inner Brow Raiser',
    'AU02': 'Outer Brow Raiser',
    'AU04': 'Brow Lowerer',
    'AU05': 'Upper Lid Raiser',
    'AU06': 'Cheek Raiser',
    'AU07': 'Lid Tightener',
    'AU09': 'Nose Wrinkler',
    'AU10': 'Upper Lip Raiser',
    'AU12': 'Lip Corner Puller (Smile)',
    'AU14': 'Dimpler',
    'AU15': 'Lip Corner Depressor',
    'AU17': 'Chin Raiser',
    'AU20': 'Lip Stretcher',
    'AU23': 'Lip Tightener',
    'AU25': 'Lips Part',
    'AU26': 'Jaw Drop',
    'AU45': 'Blink',
}

for au in selected_aus:
    print(f"  {au}: {au_names[au]}")

# Now map back to our CSV
print("\n" + "="*70)
print("CSV COLUMN INTERPRETATION")
print("="*70)
print("\nOur OpenFace 3.0 CSV columns should be interpreted as:")

csv_mapping = {
    'AU01_r': selected_aus[0],
    'AU02_r': selected_aus[1],
    'AU03_r': selected_aus[2],
    'AU04_r': selected_aus[3],
    'AU05_r': selected_aus[4],
    'AU06_r': selected_aus[5],
    'AU07_r': selected_aus[6],
    'AU08_r': selected_aus[7],
}

for csv_col, facs_au in csv_mapping.items():
    print(f"  CSV column '{csv_col}' = FACS {facs_au} ({au_names[facs_au]})")
