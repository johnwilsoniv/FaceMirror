"""
Verify which AUs are actually used for each zone based on paralysis_config.py
and compare with OpenFace 3.0 functional AUs
"""

# From paralysis_config.py - ACTUAL AUs used in training
ZONE_AUS_CONFIG = {
    'upper': ['AU01_r', 'AU02_r'],
    'mid': ['AU45_r', 'AU07_r', 'AU06_r'],
    'lower': ['AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r']
}

# From our analysis - OpenFace 3.0 functional AUs
FUNCTIONAL_AUS_OF3 = ['AU01', 'AU02', 'AU04', 'AU06', 'AU12', 'AU15', 'AU20', 'AU25', 'AU45']
NON_FUNCTIONAL_AUS_OF3 = ['AU05', 'AU07', 'AU09', 'AU10', 'AU14', 'AU16', 'AU17', 'AU23', 'AU26']

# From combined results analysis - OpenFace 2.2 functional AUs
FUNCTIONAL_AUS_OF2 = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10',
                      'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU45']

print("="*100)
print("ACTUAL AU USAGE IN PARALYSIS DETECTION (from paralysis_config.py)")
print("="*100)

for zone, aus in ZONE_AUS_CONFIG.items():
    # Remove the '_r' suffix for comparison
    aus_clean = [au.replace('_r', '') for au in aus]

    print(f"\n{zone.upper()} FACE:")
    print(f"  Configured AUs ({len(aus_clean)}): {', '.join(aus_clean)}")

    # Check which are functional in OF3.0
    functional_in_of3 = [au for au in aus_clean if au in FUNCTIONAL_AUS_OF3]
    non_functional_in_of3 = [au for au in aus_clean if au in NON_FUNCTIONAL_AUS_OF3]

    retention_rate = (len(functional_in_of3) / len(aus_clean) * 100) if aus_clean else 0

    print(f"  ✓ Functional in OF3.0 ({len(functional_in_of3)}/{len(aus_clean)}): {', '.join(functional_in_of3)}")
    if non_functional_in_of3:
        print(f"  ✗ NON-Functional in OF3.0 ({len(non_functional_in_of3)}/{len(aus_clean)}): {', '.join(non_functional_in_of3)}")
    print(f"  Retention Rate: {retention_rate:.1f}%")

    # Impact assessment
    if retention_rate == 100:
        impact = "✓ NO IMPACT - All AUs maintained"
    elif retention_rate >= 75:
        impact = "⚠ LOW IMPACT - Minor AU loss"
    elif retention_rate >= 50:
        impact = "⚠ MODERATE IMPACT - Significant AU loss"
    else:
        impact = "🔴 SEVERE IMPACT - Critical AU loss"

    print(f"  {impact}")

print("\n" + "="*100)
print("DETAILED ANALYSIS BY ZONE")
print("="*100)

# Upper Face Detailed Analysis
print("\n### UPPER FACE ###")
upper_aus = [au.replace('_r', '') for au in ZONE_AUS_CONFIG['upper']]
print(f"Used AUs: {', '.join(upper_aus)}")
print(f"OpenFace 2.2 Status: All functional")
print(f"OpenFace 3.0 Status:")
for au in upper_aus:
    status = "✓ FUNCTIONAL" if au in FUNCTIONAL_AUS_OF3 else "✗ NON-FUNCTIONAL"
    print(f"  {au}: {status}")

# Mid Face Detailed Analysis
print("\n### MID FACE ###")
mid_aus = [au.replace('_r', '') for au in ZONE_AUS_CONFIG['mid']]
print(f"Used AUs: {', '.join(mid_aus)}")
print(f"OpenFace 2.2 Status: All functional")
print(f"OpenFace 3.0 Status:")
for au in mid_aus:
    status = "✓ FUNCTIONAL" if au in FUNCTIONAL_AUS_OF3 else "✗ NON-FUNCTIONAL"
    print(f"  {au}: {status}")

# Lower Face Detailed Analysis
print("\n### LOWER FACE ###")
lower_aus = [au.replace('_r', '') for au in ZONE_AUS_CONFIG['lower']]
print(f"Used AUs: {', '.join(lower_aus)}")
print(f"OpenFace 2.2 Status: All functional")
print(f"OpenFace 3.0 Status:")
for au in lower_aus:
    status = "✓ FUNCTIONAL" if au in FUNCTIONAL_AUS_OF3 else "✗ NON-FUNCTIONAL"
    print(f"  {au}: {status}")

print("\n" + "="*100)
print("CORRECTED IMPACT SUMMARY")
print("="*100)

print("""
UPPER FACE: 100% retention (✓ NO IMPACT)
  - All 2 configured AUs are functional in OF3.0
  - AU01 ✓, AU02 ✓
  - Expected F1: Should maintain ~0.83 (minimal drop from value reduction)

MID FACE: 66.7% retention (⚠ MODERATE-TO-SEVERE IMPACT)
  - 2 out of 3 configured AUs are functional in OF3.0
  - AU45 ✓, AU06 ✓, AU07 ✗
  - AU07 was the STRONGEST feature (mean=1.397 in OF2.2, now 0.0 in OF3.0)
  - AU06 values dropped 93% (mean: 0.857→0.059)
  - Expected F1: Likely drop from 0.92 to 0.70-0.80

LOWER FACE: 44.4% retention (🔴 SEVERE IMPACT)
  - 4 out of 9 configured AUs are functional in OF3.0
  - AU12 ✓, AU15 ✓, AU20 ✓, AU25 ✓
  - AU10 ✗, AU14 ✗, AU17 ✗, AU23 ✗, AU26 ✗
  - Lost 5 critical AUs for lower face paralysis detection
  - Expected F1: Likely drop from 0.82 to 0.60-0.70

KEY CORRECTION:
  - Upper face is in MUCH better shape than previously estimated
  - Upper face only uses AU01 and AU02, both of which are functional in OF3.0
  - No need for AU04 or AU05 (which I incorrectly included in earlier analysis)
""")

print("\n" + "="*100)
print("BONUS AUs AVAILABLE BUT NOT CURRENTLY USED")
print("="*100)

# Check which OF3.0 functional AUs are NOT currently used in any zone
all_configured_aus = []
for aus in ZONE_AUS_CONFIG.values():
    all_configured_aus.extend([au.replace('_r', '') for au in aus])

bonus_aus = [au for au in FUNCTIONAL_AUS_OF3 if au not in all_configured_aus]

if bonus_aus:
    print(f"\nFunctional in OF3.0 but NOT configured: {', '.join(bonus_aus)}")
    print("\nThese could potentially be added through feature engineering:")
    for au in bonus_aus:
        print(f"  {au}: Consider adding as interaction term or zone-crossover feature")
else:
    print("\nNo bonus AUs available - all OF3.0 functional AUs are already used")
