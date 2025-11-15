#!/usr/bin/env python3
"""
OpenFace 3.0 AU Column Mapping

This file provides the mapping from OpenFace 3.0 CSV column names
to actual FACS Action Unit numbers.

IMPORTANT: OpenFace 3.0 CSV columns are misleadingly named!
Column "AU02_r" does NOT correspond to FACS AU02, it's actually AU06.
"""

# ============================================================================
# CSV COLUMN → FACS AU MAPPING
# ============================================================================

OPENFACE3_CSV_TO_FACS_AU = {
    # CSV Column: (FACS AU, AU Name)
    'AU01_r': ('AU01', 'Inner Brow Raiser'),
    'AU02_r': ('AU06', 'Cheek Raiser'),
    'AU03_r': ('AU12', 'Lip Corner Puller (Smile)'),
    'AU04_r': ('AU15', 'Lip Corner Depressor'),
    'AU05_r': ('AU17', 'Chin Raiser'),
    'AU06_r': ('AU02', 'Outer Brow Raiser'),
    'AU07_r': ('AU09', 'Nose Wrinkler'),
    'AU08_r': ('AU10', 'Upper Lip Raiser'),
}

# Simplified version - just AU numbers
OPENFACE3_CSV_TO_AU_NUMBER = {
    'AU01_r': 'AU01',
    'AU02_r': 'AU06',
    'AU03_r': 'AU12',
    'AU04_r': 'AU15',
    'AU05_r': 'AU17',
    'AU06_r': 'AU02',
    'AU07_r': 'AU09',
    'AU08_r': 'AU10',
}

# Reverse mapping - FACS AU to CSV column
FACS_AU_TO_OPENFACE3_CSV = {
    'AU01': 'AU01_r',
    'AU02': 'AU06_r',  # ← Note: AU02 is in column AU06_r!
    'AU06': 'AU02_r',  # ← Note: AU06 is in column AU02_r!
    'AU09': 'AU07_r',
    'AU10': 'AU08_r',
    'AU12': 'AU03_r',
    'AU15': 'AU04_r',
    'AU17': 'AU05_r',
}

# Full list of AUs that OpenFace 3.0 actually detects
OPENFACE3_AVAILABLE_AUS = [
    'AU01',  # Inner Brow Raiser
    'AU02',  # Outer Brow Raiser
    'AU06',  # Cheek Raiser
    'AU09',  # Nose Wrinkler
    'AU10',  # Upper Lip Raiser
    'AU12',  # Lip Corner Puller (Smile)
    'AU15',  # Lip Corner Depressor
    'AU17',  # Chin Raiser
]

# AUs that were observed to work in testing (non-zero values)
OPENFACE3_WORKING_AUS = [
    'AU01',  # Inner Brow Raiser
    'AU09',  # Nose Wrinkler (most active)
    'AU12',  # Lip Corner Puller (Smile)
    'AU15',  # Lip Corner Depressor
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def translate_csv_to_facs(csv_column_name):
    """
    Convert OpenFace 3.0 CSV column name to FACS AU number.

    Args:
        csv_column_name (str): CSV column name (e.g., 'AU02_r')

    Returns:
        str: FACS AU number (e.g., 'AU06')

    Example:
        >>> translate_csv_to_facs('AU02_r')
        'AU06'
    """
    return OPENFACE3_CSV_TO_AU_NUMBER.get(csv_column_name, None)


def translate_facs_to_csv(facs_au):
    """
    Convert FACS AU number to OpenFace 3.0 CSV column name.

    Args:
        facs_au (str): FACS AU number (e.g., 'AU06')

    Returns:
        str: CSV column name (e.g., 'AU02_r')

    Example:
        >>> translate_facs_to_csv('AU06')
        'AU02_r'
    """
    return FACS_AU_TO_OPENFACE3_CSV.get(facs_au, None)


def get_au_name(facs_au_or_csv_column):
    """
    Get the full name of an Action Unit.

    Args:
        facs_au_or_csv_column (str): Either a FACS AU number or CSV column name

    Returns:
        str: Full AU name

    Example:
        >>> get_au_name('AU06')
        'Cheek Raiser'
        >>> get_au_name('AU02_r')
        'Cheek Raiser'
    """
    # If it's a CSV column, translate first
    if facs_au_or_csv_column in OPENFACE3_CSV_TO_FACS_AU:
        return OPENFACE3_CSV_TO_FACS_AU[facs_au_or_csv_column][1]

    # Otherwise look up by FACS AU
    for csv_col, (au_num, au_name) in OPENFACE3_CSV_TO_FACS_AU.items():
        if au_num == facs_au_or_csv_column:
            return au_name

    return None


def is_au_available(facs_au):
    """
    Check if a FACS AU is available in OpenFace 3.0.

    Args:
        facs_au (str): FACS AU number (e.g., 'AU06')

    Returns:
        bool: True if AU is available, False otherwise

    Example:
        >>> is_au_available('AU06')
        True
        >>> is_au_available('AU45')
        False
    """
    return facs_au in OPENFACE3_AVAILABLE_AUS


def print_mapping_table():
    """Print a formatted table of the CSV to FACS AU mapping."""
    print("=" * 80)
    print("OpenFace 3.0 CSV Column → FACS AU Mapping")
    print("=" * 80)
    print(f"{'CSV Column':<15} {'→':<3} {'FACS AU':<10} {'AU Name':<40}")
    print("-" * 80)

    for csv_col, (facs_au, au_name) in OPENFACE3_CSV_TO_FACS_AU.items():
        working = "✓" if facs_au in OPENFACE3_WORKING_AUS else "✗"
        print(f"{csv_col:<15} {'→':<3} {facs_au:<10} {au_name:<40} {working}")

    print("=" * 80)
    print(f"\nTotal AUs in model: {len(OPENFACE3_AVAILABLE_AUS)}")
    print(f"Working AUs (non-zero): {len(OPENFACE3_WORKING_AUS)}")
    print(f"\nWorking: {', '.join(OPENFACE3_WORKING_AUS)}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    # Print the mapping table
    print_mapping_table()

    print("\n" + "=" * 80)
    print("Usage Examples")
    print("=" * 80)

    # Example 1: Translate CSV column to FACS AU
    csv_col = 'AU02_r'
    facs_au = translate_csv_to_facs(csv_col)
    print(f"\nExample 1: CSV column '{csv_col}' → FACS {facs_au}")
    print(f"  Full name: {get_au_name(csv_col)}")

    # Example 2: Translate FACS AU to CSV column
    facs_au = 'AU06'
    csv_col = translate_facs_to_csv(facs_au)
    print(f"\nExample 2: FACS {facs_au} → CSV column '{csv_col}'")
    print(f"  Full name: {get_au_name(facs_au)}")

    # Example 3: Check AU availability
    print(f"\nExample 3: Checking AU availability")
    print(f"  AU06 available? {is_au_available('AU06')}")
    print(f"  AU07 available? {is_au_available('AU07')}")
    print(f"  AU45 available? {is_au_available('AU45')}")

    # Example 4: Process a DataFrame
    print(f"\nExample 4: Renaming DataFrame columns")
    print("  import pandas as pd")
    print("  df = pd.read_csv('openface3_output.csv')")
    print("  df.rename(columns=OPENFACE3_CSV_TO_AU_NUMBER, inplace=True)")
    print("  # Now columns are correctly named AU01, AU06, AU12, etc.")
