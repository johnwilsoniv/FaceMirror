#!/usr/bin/env python3
"""
Correlate OpenFace 3.0 raw outputs with OpenFace 2.2 labeled AUs.

This script will determine which OF3 index corresponds to which AU by
finding the highest correlation with OF2.2's known-good AU labels.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# The 8 AUs that OpenFace 3.0 should detect (DISFA subset)
# Order is currently unknown - that's what we're trying to determine!
EXPECTED_OF3_AUS = ['AU01', 'AU02', 'AU04', 'AU06', 'AU12', 'AU15', 'AU20', 'AU25']


def extract_of3_raw_vectors(csv_path):
    """
    Extract the raw 8-dimensional OF3 vectors from CSV.

    Currently the CSV has our assumed mapping, but we need to reconstruct
    the original 8-dimensional vectors to test all possible mappings.
    """
    df = pd.read_csv(csv_path)

    # Our current assumed mapping (which we suspect is wrong)
    current_mapping = {
        0: 'AU01_r',  # Inner Brow Raiser
        1: 'AU02_r',  # Outer Brow Raiser
        2: 'AU04_r',  # Brow Lowerer
        3: 'AU06_r',  # Cheek Raiser
        4: 'AU12_r',  # Lip Corner Puller (Smile) - SUSPECTED WRONG
        5: 'AU15_r',  # Lip Corner Depressor
        6: 'AU20_r',  # Lip Stretcher
        7: 'AU25_r',  # Lips Part
    }

    # Extract the 8 values in their current order
    raw_vectors = np.zeros((len(df), 8))
    for idx, au_name in current_mapping.items():
        if au_name in df.columns:
            raw_vectors[:, idx] = df[au_name].values

    return raw_vectors, df['frame'].values


def load_of22_aus(csv_path):
    """Load OpenFace 2.2 AU labels (ground truth)"""
    df = pd.read_csv(csv_path)

    # Get all intensity AUs from OF2.2
    au_cols = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]

    au_data = {}
    for col in au_cols:
        au_num = col.replace('AU', '').replace('_r', '')
        au_data[f'AU{au_num}'] = df[col].values

    return au_data, df['frame'].values


def compute_correlation_matrix(of3_vectors, of22_aus, of3_frames, of22_frames):
    """
    Compute correlation between each OF3 output index and each OF2.2 AU.

    Returns:
        correlation_matrix: (8, N_AUs) array of correlations
        au_names: List of OF2.2 AU names
    """
    # Align frames (both should have same frames, but verify)
    common_frames = np.intersect1d(of3_frames, of22_frames)
    print(f"Common frames: {len(common_frames)}")

    # Get indices for common frames
    of3_indices = np.isin(of3_frames, common_frames)
    of22_indices = np.isin(of22_frames, common_frames)

    of3_aligned = of3_vectors[of3_indices]

    # Focus on the 8 AUs that OF3 should detect
    au_names = []
    of22_aligned_list = []

    for au_name in EXPECTED_OF3_AUS:
        if au_name in of22_aus:
            au_names.append(au_name)
            of22_aligned_list.append(of22_aus[au_name][of22_indices])

    of22_aligned = np.column_stack(of22_aligned_list)

    # Compute correlation matrix
    n_of3_outputs = 8
    n_aus = len(au_names)
    correlation_matrix = np.zeros((n_of3_outputs, n_aus))
    p_values = np.zeros((n_of3_outputs, n_aus))

    for i in range(n_of3_outputs):
        for j in range(n_aus):
            # Remove NaN values
            mask = ~(np.isnan(of3_aligned[:, i]) | np.isnan(of22_aligned[:, j]))
            if mask.sum() > 10:  # Need at least 10 valid points
                x = of3_aligned[mask, i]
                y = of22_aligned[mask, j]

                # Check if either variable is constant
                if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                    correlation_matrix[i, j] = np.nan
                    p_values[i, j] = np.nan
                else:
                    corr, p = pearsonr(x, y)
                    correlation_matrix[i, j] = corr
                    p_values[i, j] = p
            else:
                correlation_matrix[i, j] = np.nan
                p_values[i, j] = np.nan

    return correlation_matrix, p_values, au_names


def find_best_mapping(correlation_matrix, au_names):
    """
    Find the best AU mapping for each OF3 output index.
    """
    print("\n" + "="*80)
    print("BEST AU MAPPING FOR EACH OF3 OUTPUT INDEX")
    print("="*80)

    best_mapping = {}

    for idx in range(8):
        correlations = correlation_matrix[idx, :]

        # Check if all NaN
        if np.all(np.isnan(correlations)):
            print(f"\nOF3 Output Index {idx}:")
            print(f"  ❌ No valid correlations (all NaN)")
            best_mapping[idx] = {
                'au': 'UNKNOWN',
                'correlation': 0.0,
                'inverted': False
            }
            continue

        # Find best positive and negative correlation
        best_pos_idx = np.nanargmax(correlations)
        best_neg_idx = np.nanargmin(correlations)

        best_pos_corr = correlations[best_pos_idx]
        best_neg_corr = correlations[best_neg_idx]

        print(f"\nOF3 Output Index {idx}:")
        print(f"  Best positive match: {au_names[best_pos_idx]} (r={best_pos_corr:.3f})")
        print(f"  Best negative match: {au_names[best_neg_idx]} (r={best_neg_corr:.3f})")

        # Choose the stronger correlation (absolute value)
        if abs(best_pos_corr) > abs(best_neg_corr):
            best_mapping[idx] = {
                'au': au_names[best_pos_idx],
                'correlation': best_pos_corr,
                'inverted': False
            }
            print(f"  ✓ BEST MATCH: {au_names[best_pos_idx]} (r={best_pos_corr:.3f})")
        else:
            best_mapping[idx] = {
                'au': au_names[best_neg_idx],
                'correlation': best_neg_corr,
                'inverted': True
            }
            print(f"  ⚠️  INVERTED MATCH: {au_names[best_neg_idx]} (r={best_neg_corr:.3f}) - INVERTED!")

    return best_mapping


def plot_correlation_heatmap(correlation_matrix, au_names, output_path):
    """Create a heatmap of correlations"""
    plt.figure(figsize=(12, 8))

    # Create labels for OF3 indices
    of3_labels = [f"OF3[{i}]" for i in range(8)]

    # Plot heatmap
    sns.heatmap(
        correlation_matrix,
        xticklabels=au_names,
        yticklabels=of3_labels,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt='.2f',
        square=True,
        cbar_kws={'label': 'Pearson Correlation'}
    )

    plt.title('OpenFace 3.0 Output Indices vs OpenFace 2.2 AU Labels\n(Finding Correct AU Mapping)',
              fontsize=14, fontweight='bold')
    plt.xlabel('OpenFace 2.2 AU Labels (Ground Truth)', fontsize=12)
    plt.ylabel('OpenFace 3.0 Output Index', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved correlation heatmap to: {output_path}")
    plt.close()


def generate_corrected_mapping_code(best_mapping):
    """Generate Python code for the corrected AU mapping"""
    print("\n" + "="*80)
    print("CORRECTED AU MAPPING CODE")
    print("="*80)
    print("\n# Corrected OpenFace 3.0 AU mapping (based on correlation with OF2.2):")
    print("self.of3_au_mapping = {")

    for idx in range(8):
        mapping = best_mapping[idx]
        au_name = mapping['au']
        corr = mapping['correlation']
        inverted = mapping['inverted']

        comment = f"  # r={corr:.3f}"
        if inverted:
            comment += " - INVERTED!"

        # Extract AU number
        au_num = au_name.replace('AU', '')

        print(f"    {idx}: 'AU{au_num}_r',{comment}")

    print("}")

    # Check for inversions
    inversions = [idx for idx, m in best_mapping.items() if m['inverted']]
    if inversions:
        print("\n⚠️  WARNING: The following indices have NEGATIVE correlation (inverted):")
        for idx in inversions:
            mapping = best_mapping[idx]
            print(f"    Index {idx} -> {mapping['au']} (r={mapping['correlation']:.3f})")
        print("\nYou may need to negate these values or investigate further!")


def main():
    print("="*80)
    print("CORRELATING OF3 OUTPUTS WITH OF2.2 AU LABELS")
    print("="*80)

    # File paths
    of3_csv = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_left_mirroredOP3ORIG.csv"
    of22_csv = "/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_left_mirroredOP22.csv"
    output_heatmap = "of3_of22_correlation_matrix.png"

    print(f"\nLoading OF3 outputs: {of3_csv}")
    of3_vectors, of3_frames = extract_of3_raw_vectors(of3_csv)
    print(f"  Loaded {len(of3_vectors)} frames with 8 outputs each")

    print(f"\nLoading OF2.2 AU labels: {of22_csv}")
    of22_aus, of22_frames = load_of22_aus(of22_csv)
    print(f"  Loaded {len(of22_frames)} frames")
    print(f"  Available AUs: {list(of22_aus.keys())}")

    print("\nComputing correlation matrix...")
    correlation_matrix, p_values, au_names = compute_correlation_matrix(
        of3_vectors, of22_aus, of3_frames, of22_frames
    )

    print(f"  Correlation matrix shape: {correlation_matrix.shape}")
    print(f"  Comparing {correlation_matrix.shape[0]} OF3 outputs with {correlation_matrix.shape[1]} AUs")

    # Find best mapping
    best_mapping = find_best_mapping(correlation_matrix, au_names)

    # Plot heatmap
    plot_correlation_heatmap(correlation_matrix, au_names, output_heatmap)

    # Generate corrected code
    generate_corrected_mapping_code(best_mapping)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the correlation heatmap to understand the mappings")
    print("2. Use the corrected mapping code to update openface3_to_18au_adapter.py")
    print("3. Handle any inverted outputs appropriately")
    print("4. Re-run validation to confirm the fix")
    print("="*80)


if __name__ == "__main__":
    main()
