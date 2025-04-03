# check_features.py (Prints ALL features)
import joblib
import os
import sys

# --- Configuration ---
model_dir = 'models'
# Select the feature list you want to check:
feature_list_filename = 'lower_face_features.list'
# feature_list_filename = 'mid_face_features.list'
# feature_list_filename = 'upper_face_features.list'
# --- End Configuration ---

file_path = os.path.join(model_dir, feature_list_filename)

print(f"--- Checking Feature List: {file_path} ---")

if not os.path.exists(file_path):
    print(f"\nERROR: File not found at '{file_path}'")
    sys.exit(1)

try:
    features = joblib.load(file_path)
    print(f"Successfully loaded.")
    print(f"Type: {type(features)}")

    if isinstance(features, list):
        print(f"Total Features: {len(features)}")

        # --- MODIFICATION: Print ALL features ---
        print(f"\nAll Features:")
        max_index_width = len(str(len(features) - 1)) # Determine width for index padding
        for i, f in enumerate(features):
            print(f"  {i:{max_index_width}d}: {f}") # Print index (padded) and feature name
        # --- END MODIFICATION ---

        # Check specifically for expert labels or target names
        expert_keywords = ['expert', 'target'] # Keywords to look for
        expert_found = [f for f in features if any(keyword in f.lower() for keyword in expert_keywords)]

        if expert_found:
            print(f"\n\n*** WARNING: Found features possibly related to expert labels: ***")
            for f in expert_found:
                print(f"  - {f}")
        else:
            print("\n\nCheck PASSED: No features containing 'expert' or 'target' found.")

    else:
        print(f"\nERROR: Loaded object is not a list!")
        sys.exit(1)

except Exception as e:
    print(f"\nERROR: Could not load or process file: {e}")
    sys.exit(1)

print(f"\n--- Check Complete ---")