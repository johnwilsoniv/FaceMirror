#!/usr/bin/env python3
"""
Analyze systematic transformations from OF3 to OF2.2
Goal: Find if there's a scaling/mapping that can convert OF3 outputs to OF2.2-like values

DO NOT MAKE CHANGES - ANALYSIS ONLY
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

print("="*80)
print("Analyzing OF3 → OF2.2 Transformation Potential")
print("="*80)

# Load both sides for more data
of3_left = pd.read_csv('/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_left_mirroredOP3ORIG.csv')
of3_right = pd.read_csv('/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_right_mirroredvOP3ORIG.csv')
of22_left = pd.read_csv('/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_left_mirroredOP22.csv')
of22_right = pd.read_csv('/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Three-Way Comparison/IMG_0942_right_mirroredOP22.csv')

# Combine for more robust analysis
of3_combined = pd.concat([of3_left, of3_right], ignore_index=True)
of22_combined = pd.concat([of22_left, of22_right], ignore_index=True)

common_aus = ['AU01_r', 'AU02_r', 'AU04_r', 'AU06_r', 'AU12_r', 'AU15_r', 'AU20_r', 'AU25_r', 'AU45_r']

print(f"\nAnalyzing {len(of3_combined)} frames across both sides")
print(f"Common AUs: {len(common_aus)}")

print("\n" + "="*80)
print("1. LINEAR SCALING ANALYSIS")
print("="*80)

results = []
for au in common_aus:
    of3_vals = of3_combined[au].values
    of22_vals = of22_combined[au].values

    # Remove NaN
    mask = ~(np.isnan(of3_vals) | np.isnan(of22_vals))
    of3_clean = of3_vals[mask].reshape(-1, 1)
    of22_clean = of22_vals[mask]

    if len(of3_clean) < 10:
        continue

    # Fit linear model: OF2.2 = a * OF3 + b
    model = LinearRegression()
    model.fit(of3_clean, of22_clean)

    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(of3_clean, of22_clean)

    # Predict and calculate error
    of22_pred = model.predict(of3_clean)
    mae = np.mean(np.abs(of22_clean - of22_pred))

    results.append({
        'AU': au,
        'slope': slope,
        'intercept': intercept,
        'r2': r2,
        'mae': mae,
        'of3_mean': np.mean(of3_clean),
        'of22_mean': np.mean(of22_clean),
        'of3_std': np.std(of3_clean),
        'of22_std': np.std(of22_clean)
    })

    print(f"\n{au}:")
    print(f"  Transform: OF2.2 ≈ {slope:.3f} * OF3 + {intercept:.3f}")
    print(f"  R² = {r2:.3f} (how well linear model fits)")
    print(f"  MAE = {mae:.3f} (mean absolute error)")

    if r2 > 0.7:
        print(f"  ✓ GOOD linear relationship - transformation feasible")
    elif r2 > 0.5:
        print(f"  ⚠ MODERATE linear relationship - transformation may help")
    else:
        print(f"  ✗ POOR linear relationship - transformation unlikely to work")

results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("2. SUMMARY OF TRANSFORMATION POTENTIAL")
print("="*80)

print(f"\nAUs with good transformation potential (R² > 0.5):")
good_aus = results_df[results_df['r2'] > 0.5]
if len(good_aus) > 0:
    for _, row in good_aus.iterrows():
        print(f"  {row['AU']}: R²={row['r2']:.3f}, Transform: {row['slope']:.2f}*OF3 + {row['intercept']:.2f}")
else:
    print("  None found")

print(f"\nAUs with poor transformation potential (R² < 0.5):")
poor_aus = results_df[results_df['r2'] <= 0.5]
for _, row in poor_aus.iterrows():
    print(f"  {row['AU']}: R²={row['r2']:.3f}")

print("\n" + "="*80)
print("3. NON-LINEAR TRANSFORMATION ANALYSIS")
print("="*80)

# Try polynomial and other transforms
for au in common_aus:
    of3_vals = of3_combined[au].values
    of22_vals = of22_combined[au].values

    mask = ~(np.isnan(of3_vals) | np.isnan(of22_vals))
    of3_clean = of3_vals[mask]
    of22_clean = of22_vals[mask]

    if len(of3_clean) < 10:
        continue

    # Try different transformations
    transforms = {
        'linear': of3_clean,
        'squared': of3_clean ** 2,
        'sqrt': np.sqrt(np.abs(of3_clean)),
        'log': np.log(of3_clean + 1),  # +1 to avoid log(0)
    }

    best_r2 = 0
    best_transform = 'linear'

    for name, transformed in transforms.items():
        try:
            corr = np.corrcoef(transformed, of22_clean)[0, 1] ** 2
            if corr > best_r2:
                best_r2 = corr
                best_transform = name
        except:
            pass

    if au in ['AU01_r', 'AU12_r', 'AU20_r', 'AU45_r']:  # Show key AUs
        print(f"\n{au}:")
        print(f"  Best transformation: {best_transform} (R²={best_r2:.3f})")

print("\n" + "="*80)
print("4. DISTRIBUTION ANALYSIS")
print("="*80)

print("\nChecking if OF3 and OF2.2 have similar distributions:")
for au in ['AU01_r', 'AU12_r', 'AU20_r', 'AU45_r']:
    of3_vals = of3_combined[au].dropna()
    of22_vals = of22_combined[au].dropna()

    print(f"\n{au}:")
    print(f"  OF3:  range=[{of3_vals.min():.3f}, {of3_vals.max():.3f}], median={of3_vals.median():.3f}")
    print(f"  OF2.2: range=[{of22_vals.min():.3f}, {of22_vals.max():.3f}], median={of22_vals.median():.3f}")

    # Check if ranges are comparable
    of3_range = of3_vals.max() - of3_vals.min()
    of22_range = of22_vals.max() - of22_vals.min()

    if abs(of3_range - of22_range) / of22_range < 0.3:
        print(f"  ✓ Similar ranges - scaling may not be needed")
    else:
        print(f"  ⚠ Different ranges ({of3_range:.2f} vs {of22_range:.2f}) - scaling needed")

print("\n" + "="*80)
print("5. VISUALIZE SCATTER PLOTS FOR KEY AUs")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('OF3 vs OF2.2 - Scatter Plot Analysis', fontsize=16)

key_aus = ['AU01_r', 'AU12_r', 'AU20_r', 'AU45_r']
for idx, au in enumerate(key_aus):
    ax = axes.flatten()[idx]

    of3_vals = of3_combined[au].values
    of22_vals = of22_combined[au].values

    mask = ~(np.isnan(of3_vals) | np.isnan(of22_vals))
    of3_clean = of3_vals[mask]
    of22_clean = of22_vals[mask]

    # Scatter plot
    ax.scatter(of3_clean, of22_clean, alpha=0.3, s=10)

    # Fit line
    if len(of3_clean) > 0:
        model = LinearRegression()
        model.fit(of3_clean.reshape(-1, 1), of22_clean)

        # Plot fitted line
        x_range = np.array([of3_clean.min(), of3_clean.max()])
        y_pred = model.predict(x_range.reshape(-1, 1))
        ax.plot(x_range, y_pred, 'r-', linewidth=2,
                label=f'y={model.coef_[0]:.2f}x+{model.intercept_:.2f}')

        r2 = model.score(of3_clean.reshape(-1, 1), of22_clean)
        ax.text(0.05, 0.95, f'R²={r2:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('OpenFace 3.0')
    ax.set_ylabel('OpenFace 2.2')
    ax.set_title(au)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of3_to_of22_scatter.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nScatter plots saved to: {output_path}")
plt.close()

print("\n" + "="*80)
print("6. CONCLUSION & RECOMMENDATIONS")
print("="*80)

avg_r2 = results_df['r2'].mean()
good_count = len(results_df[results_df['r2'] > 0.5])
total_count = len(results_df)

print(f"\nAverage R² across all AUs: {avg_r2:.3f}")
print(f"AUs with R² > 0.5: {good_count}/{total_count}")

print("\nOptions:")

if avg_r2 > 0.7:
    print("  1. ✓ LINEAR SCALING FEASIBLE")
    print("     Apply per-AU linear transforms: OF2.2_pred = slope * OF3 + intercept")
    print("     This should get you close to OF2.2 values")
elif avg_r2 > 0.5:
    print("  1. ⚠ LINEAR SCALING PARTIALLY FEASIBLE")
    print("     Some AUs can be transformed, others cannot")
    print("     May need per-AU handling")
else:
    print("  1. ✗ LINEAR SCALING NOT FEASIBLE")
    print("     The relationship is too non-linear or there's no systematic relationship")

print("\n  2. ALTERNATIVE APPROACHES:")
print("     a. Stick with OpenFace 2.2 (you said it works well)")
print("     b. Retrain your downstream pipeline with OF3 data")
print("     c. Use OF3 for speed (ONNX) but calibrate thresholds for your use case")
print("     d. Investigate WHY OF3 isn't working well in your pipeline")

print("\n  3. QUESTIONS TO ANSWER:")
print("     - What specific issues are you seeing with OF3 in your pipeline?")
print("     - Are the predictions wrong, or just scaled differently?")
print("     - Could your pipeline's thresholds/models be retrained for OF3?")

print("\n" + "="*80)
