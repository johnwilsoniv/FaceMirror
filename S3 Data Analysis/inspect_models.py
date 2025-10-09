import joblib
import os

zones = ['lower', 'mid', 'upper']

for zone in zones:
    print(f"\n{'='*60}")
    print(f"{zone.upper()} FACE")
    print('='*60)

    scaler_path = f'models/{zone}_face_scaler.pkl'
    features_path = f'models/{zone}_face_features.list'

    # Load scaler
    scaler = joblib.load(scaler_path)
    print(f"Scaler expects: {scaler.n_features_in_} features")

    # Load feature list
    with open(features_path, 'r') as f:
        feature_list = [line.strip() for line in f if line.strip()]
    print(f"Feature list has: {len(feature_list)} features")

    # Check if scaler has feature names
    if hasattr(scaler, 'feature_names_in_'):
        scaler_features = list(scaler.feature_names_in_)
        print(f"Scaler has stored feature names: {len(scaler_features)}")

        # Find missing features
        missing = set(scaler_features) - set(feature_list)
        extra = set(feature_list) - set(scaler_features)

        if missing:
            print(f"\n{len(missing)} features MISSING from .list file:")
            for i, feat in enumerate(sorted(missing)[:20], 1):
                print(f"  {i}. {feat}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")

        if extra:
            print(f"\n{len(extra)} EXTRA features in .list file:")
            for i, feat in enumerate(sorted(extra)[:20], 1):
                print(f"  {i}. {feat}")
            if len(extra) > 20:
                print(f"  ... and {len(extra) - 20} more")

        if not missing and not extra:
            print("\nFeature lists MATCH perfectly!")
    else:
        print("Scaler doesn't have feature_names_in_ attribute")
