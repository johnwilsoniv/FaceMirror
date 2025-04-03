"""
Two-stage ML model training for lower face paralysis detection.
Trains separate models for paralysis detection and severity classification.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import xgboost as xgb
import warnings
import os

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_two_stage_models():
    """
    Train the two-stage model system:
    1. Binary classifier for paralysis detection
    2. Severity classifier for partial vs. complete
    """
    logger.info("Starting two-stage model training process")

    # Load datasets
    try:
        results_df = pd.read_csv("combined_results.csv")
        expert_df = pd.read_csv("FPRS FP Key.csv")

        logger.info(f"Loaded {len(results_df)} detection results and {len(expert_df)} expert grades")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

    # Rename columns for consistent joining
    expert_df = expert_df.rename(columns={
        'Patient': 'Patient ID',
        'Paralysis - Left Lower Face': 'Expert_Left_Lower_Face',
        'Paralysis - Right Lower Face': 'Expert_Right_Lower_Face'
    })

    # Merge datasets on Patient ID
    merged_df = pd.merge(results_df, expert_df, on='Patient ID', how='inner')
    logger.info(f"Merged dataset contains {len(merged_df)} patients")

    # --------------------------
    # Stage 1: Binary Classification Model (Paralysis vs. No Paralysis)
    # --------------------------
    logger.info("Training Stage 1: Binary Classification Model")

    # Extract features for binary detection
    binary_features_left = extract_binary_features(merged_df, 'Left')
    binary_features_right = extract_binary_features(merged_df, 'Right')

    # Process binary targets (None=0, Partial/Complete=1)
    binary_targets_left = process_binary_targets(merged_df['Expert_Left_Lower_Face'])
    binary_targets_right = process_binary_targets(merged_df['Expert_Right_Lower_Face'])

    # Combine sides into one dataset
    binary_features_left['side'] = 0  # 0 for left
    binary_features_right['side'] = 1  # 1 for right

    binary_features = pd.concat([binary_features_left, binary_features_right], ignore_index=True)
    binary_targets = np.concatenate([binary_targets_left, binary_targets_right])

    # Handle missing values
    binary_features = binary_features.fillna(0)

    # Check class distribution
    unique, counts = np.unique(binary_targets, return_counts=True)
    binary_class_dist = dict(zip(['No Paralysis', 'Paralysis'], counts))
    logger.info(f"Binary class distribution: {binary_class_dist}")

    # Train binary model
    binary_model, binary_scaler, binary_feature_importance = train_binary_model(binary_features, binary_targets)

    # --------------------------
    # Stage 2: Severity Classification Model (Partial vs. Complete Paralysis)
    # --------------------------
    logger.info("Training Stage 2: Severity Classification Model")

    # Extract features for severity classification
    severity_features_left = extract_severity_features(merged_df, 'Left')
    severity_features_right = extract_severity_features(merged_df, 'Right')

    # Process severity targets (Partial=0, Complete=1, filter out None)
    severity_targets_left, severity_filter_left = process_severity_targets(merged_df['Expert_Left_Lower_Face'])
    severity_targets_right, severity_filter_right = process_severity_targets(merged_df['Expert_Right_Lower_Face'])

    # Filter features to only include cases with paralysis
    severity_features_left = severity_features_left[severity_filter_left].reset_index(drop=True)
    severity_features_right = severity_features_right[severity_filter_right].reset_index(drop=True)

    # Combine sides into one dataset
    severity_features_left['side'] = 0  # 0 for left
    severity_features_right['side'] = 1  # 1 for right

    severity_features = pd.concat([severity_features_left, severity_features_right], ignore_index=True)
    severity_targets = np.concatenate([
        severity_targets_left[severity_filter_left],
        severity_targets_right[severity_filter_right]
    ])

    # Handle missing values
    severity_features = severity_features.fillna(0)

    # Check class distribution
    unique, counts = np.unique(severity_targets, return_counts=True)
    severity_class_dist = dict(zip(['Partial', 'Complete'], counts))
    logger.info(f"Severity class distribution: {severity_class_dist}")

    # Train severity model
    severity_model, severity_scaler, severity_feature_importance = train_severity_model(severity_features,
                                                                                        severity_targets)

    # --------------------------
    # Save models and scalers
    # --------------------------
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save binary model
    joblib.dump(binary_model, 'models/lower_face_binary_model.pkl')
    joblib.dump(binary_scaler, 'models/lower_face_binary_scaler.pkl')
    binary_feature_importance.to_csv('models/binary_feature_importance.csv', index=False)

    logger.info("Binary classifier model saved to models/lower_face_binary_model.pkl")

    # Save severity model
    joblib.dump(severity_model, 'models/lower_face_severity_model.pkl')
    joblib.dump(severity_scaler, 'models/lower_face_severity_scaler.pkl')
    severity_feature_importance.to_csv('models/severity_feature_importance.csv', index=False)

    logger.info("Severity classifier model saved to models/lower_face_severity_model.pkl")

    return {
        'binary_model': binary_model,
        'binary_scaler': binary_scaler,
        'binary_feature_importance': binary_feature_importance,
        'severity_model': severity_model,
        'severity_scaler': severity_scaler,
        'severity_feature_importance': severity_feature_importance
    }


def extract_binary_features(df, side):
    """
    Extract features optimized for binary paralysis detection.
    Focus on asymmetry metrics, BS action features, and combined AU behavior.

    Args:
        df (pandas.DataFrame): Merged dataframe with detection results and expert grades
        side (str): 'Left' or 'Right'

    Returns:
        pandas.DataFrame: Features for binary classification
    """
    features = pd.DataFrame()

    # Actions to extract features from - prioritize BS (most important for detection)
    actions = ['BS', 'SS', 'SO', 'SE']

    # Get opposite side name
    opposite_side = 'Right' if side == 'Left' else 'Left'

    for action in actions:
        # Build column names for AU values
        au12_col = f"{action}_{side} AU12_r"
        au25_col = f"{action}_{side} AU25_r"
        au12_norm_col = f"{action}_{side} AU12_r (Normalized)"
        au25_norm_col = f"{action}_{side} AU25_r (Normalized)"

        # Opposite side columns
        au12_opp_col = f"{action}_{opposite_side} AU12_r"
        au25_opp_col = f"{action}_{opposite_side} AU25_r"
        au12_opp_norm_col = f"{action}_{opposite_side} AU12_r (Normalized)"
        au25_opp_norm_col = f"{action}_{opposite_side} AU25_r (Normalized)"

        # 1. Strong asymmetry features
        # Ratio metrics
        if au12_col in df.columns and au12_opp_col in df.columns:
            features[f"{action}_AU12_ratio"] = calculate_ratio(df[au12_col], df[au12_opp_col])
        else:
            features[f"{action}_AU12_ratio"] = 1.0

        if au25_col in df.columns and au25_opp_col in df.columns:
            features[f"{action}_AU25_ratio"] = calculate_ratio(df[au25_col], df[au25_opp_col])
        else:
            features[f"{action}_AU25_ratio"] = 1.0

        # Percent difference metrics
        if au12_col in df.columns and au12_opp_col in df.columns:
            features[f"{action}_AU12_percent_diff"] = calculate_percent_diff(df[au12_col], df[au12_opp_col])
        else:
            features[f"{action}_AU12_percent_diff"] = 0.0

        if au25_col in df.columns and au25_opp_col in df.columns:
            features[f"{action}_AU25_percent_diff"] = calculate_percent_diff(df[au25_col], df[au25_opp_col])
        else:
            features[f"{action}_AU25_percent_diff"] = 0.0

        # Store raw values
        if au12_col in df.columns:
            features[f"{action}_AU12_side_value"] = df[au12_col]
        else:
            features[f"{action}_AU12_side_value"] = 0.0

        if au12_opp_col in df.columns:
            features[f"{action}_AU12_other_value"] = df[au12_opp_col]
        else:
            features[f"{action}_AU12_other_value"] = 0.0

        # Extreme asymmetry flag (>120% difference)
        features[f"{action}_AU12_extreme_asymmetry"] = (
                features[f"{action}_AU12_percent_diff"] > 120).astype(int)
        features[f"{action}_AU25_extreme_asymmetry"] = (
                features[f"{action}_AU25_percent_diff"] > 120).astype(int)

        # 2. Action-specific features (emphasize BS)
        if action == 'BS':
            # BS is most important for paralysis detection
            if au12_col in df.columns and au12_opp_col in df.columns:
                features['BS_AU12_absolute_diff'] = abs(df[au12_col] - df[au12_opp_col])
                features['BS_AU12_side_to_other_ratio'] = (
                        df[au12_col] / df[au12_opp_col].replace(0, 0.1))

            # Add AU25_r features for BS
            if au25_col in df.columns and au25_opp_col in df.columns:
                features['BS_AU25_absolute_diff'] = abs(df[au25_col] - df[au25_opp_col])
                features['BS_AU25_side_to_other_ratio'] = (
                        df[au25_col] / df[au25_opp_col].replace(0, 0.1))

        # 3. Combined AU behavior features
        if au12_col in df.columns and au25_col in df.columns:
            # AU interaction metrics
            features[f"{action}_AU12_AU25_interaction"] = df[au12_col] * df[au25_col]
            features[f"{action}_AU12_AU25_ratio"] = (
                    df[au12_col] / df[au25_col].replace(0, 0.1))

            # Same for other side
            if au12_opp_col in df.columns and au25_opp_col in df.columns:
                # Interaction pattern difference between sides
                side_interaction = df[au12_col] * df[au25_col]
                other_interaction = df[au12_opp_col] * df[au25_opp_col]

                interaction_avg = (side_interaction + other_interaction) / 2
                interaction_avg = interaction_avg.replace(0, 0.1)

                features[f"{action}_interaction_asymmetry"] = (
                        abs(side_interaction - other_interaction) / interaction_avg)

    # 4. Cross-action features
    for au in ['AU12', 'AU25']:
        # Calculate max asymmetry across actions
        asymmetry_columns = [f"{action}_{au}_percent_diff" for action in actions
                             if f"{action}_{au}_percent_diff" in features.columns]
        if asymmetry_columns:
            features[f"max_{au}_asymmetry"] = features[asymmetry_columns].max(axis=1)

        # Calculate max ratio across actions (min/max ratio, lower = more asymmetry)
        ratio_columns = [f"{action}_{au}_ratio" for action in actions
                         if f"{action}_{au}_ratio" in features.columns]
        if ratio_columns:
            features[f"min_{au}_ratio"] = features[ratio_columns].min(axis=1)

    # Add current detection as a feature
    lower_face_col = f"{side} Lower Face Paralysis"
    if lower_face_col in df.columns:
        features['current_detection'] = df[lower_face_col].map(
            {'None': 0, 'Partial': 1, 'Complete': 2, 'none': 0, 'partial': 1, 'complete': 2})
    else:
        features['current_detection'] = 0

    return features


def extract_severity_features(df, side):
    """
    Extract features optimized for severity classification (partial vs. complete).
    Focus on functional movement metrics and fine-grained asymmetry features.

    Args:
        df (pandas.DataFrame): Merged dataframe with detection results and expert grades
        side (str): 'Left' or 'Right'

    Returns:
        pandas.DataFrame: Features for severity classification
    """
    features = pd.DataFrame()

    # Actions to extract features from - prioritize BS
    actions = ['BS', 'SS', 'SO', 'SE']

    # Get opposite side name
    opposite_side = 'Right' if side == 'Left' else 'Left'

    for action in actions:
        # Build column names for AU values
        au12_col = f"{action}_{side} AU12_r"
        au25_col = f"{action}_{side} AU25_r"
        au12_norm_col = f"{action}_{side} AU12_r (Normalized)"
        au25_norm_col = f"{action}_{side} AU25_r (Normalized)"

        # Opposite side columns
        au12_opp_col = f"{action}_{opposite_side} AU12_r"
        au25_opp_col = f"{action}_{opposite_side} AU25_r"
        au12_opp_norm_col = f"{action}_{opposite_side} AU12_r (Normalized)"
        au25_opp_norm_col = f"{action}_{opposite_side} AU25_r (Normalized)"

        # 1. Functional movement measures
        # Store absolute activation values (critical for severity)
        if au12_col in df.columns:
            features[f"{action}_AU12_absolute_activation"] = df[au12_col]

            # Functional threshold features
            features[f"{action}_AU12_below_threshold_04"] = (df[au12_col] < 0.4).astype(int)
            features[f"{action}_AU12_below_threshold_08"] = (df[au12_col] < 0.8).astype(int)
            features[f"{action}_AU12_below_threshold_12"] = (df[au12_col] < 1.2).astype(int)

            # Movement capacity feature
            features[f"{action}_AU12_movement_capacity"] = df[au12_col].apply(
                lambda x: min(1.0, x / 3.0))
        else:
            features[f"{action}_AU12_absolute_activation"] = 0.0
            features[f"{action}_AU12_below_threshold_04"] = 0
            features[f"{action}_AU12_below_threshold_08"] = 0
            features[f"{action}_AU12_below_threshold_12"] = 0
            features[f"{action}_AU12_movement_capacity"] = 0.0

        if au25_col in df.columns:
            features[f"{action}_AU25_absolute_activation"] = df[au25_col]

            # Functional threshold features
            features[f"{action}_AU25_below_threshold_04"] = (df[au25_col] < 0.4).astype(int)
            features[f"{action}_AU25_below_threshold_08"] = (df[au25_col] < 0.8).astype(int)
            features[f"{action}_AU25_below_threshold_12"] = (df[au25_col] < 1.2).astype(int)

            # Movement capacity feature
            features[f"{action}_AU25_movement_capacity"] = df[au25_col].apply(
                lambda x: min(1.0, x / 3.0))
        else:
            features[f"{action}_AU25_absolute_activation"] = 0.0
            features[f"{action}_AU25_below_threshold_04"] = 0
            features[f"{action}_AU25_below_threshold_08"] = 0
            features[f"{action}_AU25_below_threshold_12"] = 0
            features[f"{action}_AU25_movement_capacity"] = 0.0

        # 2. Fine-grained asymmetry features
        if au12_col in df.columns and au12_opp_col in df.columns:
            # Basic asymmetry metrics
            features[f"{action}_AU12_percent_diff"] = calculate_percent_diff(
                df[au12_col], df[au12_opp_col])
            features[f"{action}_AU12_ratio"] = calculate_ratio(
                df[au12_col], df[au12_opp_col])

            # Severity-weighted asymmetry scores
            features[f"{action}_AU12_weighted_asym"] = features[f"{action}_AU12_percent_diff"] * 1.5

            # Extreme asymmetry flags with specific thresholds
            features[f"{action}_AU12_extreme_asym"] = (
                    features[f"{action}_AU12_percent_diff"] > 150).astype(int)

            # Lower threshold flags
            features[f"{action}_AU12_severe_asym"] = (
                    features[f"{action}_AU12_percent_diff"] > 80).astype(int)
            features[f"{action}_AU12_moderate_asym"] = (
                    features[f"{action}_AU12_percent_diff"] > 50).astype(int)

            # Ratio-based severity scores
            features[f"{action}_AU12_severity_score"] = 1.0 - features[f"{action}_AU12_ratio"]

        if au25_col in df.columns and au25_opp_col in df.columns:
            # Basic asymmetry metrics
            features[f"{action}_AU25_percent_diff"] = calculate_percent_diff(
                df[au25_col], df[au25_opp_col])
            features[f"{action}_AU25_ratio"] = calculate_ratio(
                df[au25_col], df[au25_opp_col])

            # Severity-weighted asymmetry scores
            features[f"{action}_AU25_weighted_asym"] = features[f"{action}_AU25_percent_diff"] * 1.0

            # Extreme asymmetry flags with specific thresholds
            features[f"{action}_AU25_extreme_asym"] = (
                    features[f"{action}_AU25_percent_diff"] > 130).astype(int)

            # Lower threshold flags
            features[f"{action}_AU25_severe_asym"] = (
                    features[f"{action}_AU25_percent_diff"] > 80).astype(int)
            features[f"{action}_AU25_moderate_asym"] = (
                    features[f"{action}_AU25_percent_diff"] > 50).astype(int)

            # Ratio-based severity scores
            features[f"{action}_AU25_severity_score"] = 1.0 - features[f"{action}_AU25_ratio"]

        # 3. Confidence modifiers
        # Calculate consistency across multiple indicators
        if all(col in features.columns for col in [
            f"{action}_AU12_below_threshold_04",
            f"{action}_AU12_severe_asym",
            f"{action}_AU12_extreme_asym"
        ]):
            au12_metrics = [
                features[f"{action}_AU12_below_threshold_04"],
                features[f"{action}_AU12_severe_asym"],
                features[f"{action}_AU12_extreme_asym"]
            ]

            # Calculate consistency scores
            features[f"{action}_AU12_consistency"] = sum(au12_metrics) / 3

        if all(col in features.columns for col in [
            f"{action}_AU25_below_threshold_04",
            f"{action}_AU25_severe_asym",
            f"{action}_AU25_extreme_asym"
        ]):
            au25_metrics = [
                features[f"{action}_AU25_below_threshold_04"],
                features[f"{action}_AU25_severe_asym"],
                features[f"{action}_AU25_extreme_asym"]
            ]

            # Calculate consistency scores
            features[f"{action}_AU25_consistency"] = sum(au25_metrics) / 3

        if all(col in features.columns for col in [
            f"{action}_AU12_consistency", f"{action}_AU25_consistency"
        ]):
            features[f"{action}_overall_consistency"] = (
                                                                features[f"{action}_AU12_consistency"] + features[
                                                            f"{action}_AU25_consistency"]) / 2

        # Borderline case detection
        if au12_col in df.columns and f"{action}_AU12_percent_diff" in features.columns:
            # Borderline cases: moderate asymmetry with some movement
            features[f"{action}_borderline_case"] = (
                    (features[f"{action}_AU12_percent_diff"] > 50) &
                    (features[f"{action}_AU12_percent_diff"] < 90) &
                    (df[au12_col] > 0.4) &
                    (df[au12_col] < 1.2)
            ).astype(int)

        # 4. Secondary pattern features
        # Check for compensatory movement patterns
        au14_col = f"{action}_{side} AU14_r"
        au14_opp_col = f"{action}_{opposite_side} AU14_r"

        if au14_col in df.columns and au14_opp_col in df.columns:
            # Compensatory movement: increased AU14_r on affected side
            features[f"{action}_AU14_compensatory"] = (
                    df[au14_col] > df[au14_opp_col] * 1.5).astype(int)

    # 5. Cross-action comparisons
    for au in ['AU12', 'AU25']:
        # Calculate averages across actions
        for metric in ['absolute_activation', 'severity_score']:
            cols = [f"{action}_{au}_{metric}" for action in actions
                    if f"{action}_{au}_{metric}" in features.columns]
            if cols:
                features[f"avg_{au}_{metric}"] = features[cols].mean(axis=1)

        # Calculate variance in asymmetry
        pd_cols = [f"{action}_{au}_percent_diff" for action in actions
                   if f"{action}_{au}_percent_diff" in features.columns]
        if pd_cols and len(pd_cols) > 1:
            features[f"{au}_pd_variance"] = features[pd_cols].var(axis=1)

        # Max asymmetry across actions
        if pd_cols:
            features[f"max_{au}_percent_diff"] = features[pd_cols].max(axis=1)

    # Add current detection as a feature
    lower_face_col = f"{side} Lower Face Paralysis"
    if lower_face_col in df.columns:
        features['current_detection'] = df[lower_face_col].map(
            {'None': 0, 'Partial': 1, 'Complete': 2, 'none': 0, 'partial': 1, 'complete': 2})
    else:
        features['current_detection'] = 0

    return features


def process_binary_targets(target_series):
    """
    Convert text labels to binary targets (0=None, 1=Partial/Complete).

    Args:
        target_series (pandas.Series): Series of text labels

    Returns:
        numpy.ndarray: Binary target values
    """
    binary_targets = []

    for label in target_series:
        # Handle NaN values and other non-string types
        if pd.isna(label):
            binary_targets.append(0)  # Treat NaN as No paralysis
        # Convert to string and then check
        elif str(label).lower() in ['none', 'no', 'n/a', '', 'normal']:
            binary_targets.append(0)  # No paralysis
        else:
            binary_targets.append(1)  # Any paralysis (partial or complete)

    return np.array(binary_targets)


def process_severity_targets(target_series):
    """
    Convert text labels to severity targets and filter for paralysis cases.

    Args:
        target_series (pandas.Series): Series of text labels

    Returns:
        tuple: (severity targets array, boolean filter for paralysis cases)
    """
    severity_targets = []
    has_paralysis = []

    for label in target_series:
        # Handle NaN values
        if pd.isna(label):
            severity_targets.append(-1)
            has_paralysis.append(False)
        # Convert to string and then check
        elif str(label).lower() in ['none', 'no', 'n/a', '', 'normal']:
            # No paralysis - we'll filter these out
            severity_targets.append(-1)
            has_paralysis.append(False)
        elif str(label).lower() in ['partial', 'mild', 'moderate']:
            # Partial paralysis
            severity_targets.append(0)
            has_paralysis.append(True)
        else:
            # Complete paralysis
            severity_targets.append(1)
            has_paralysis.append(True)

    return np.array(severity_targets), has_paralysis


def train_binary_model(features, targets):
    """
    Train the binary classification model for paralysis detection.

    Args:
        features (pandas.DataFrame): Feature data
        targets (numpy.ndarray): Binary target labels

    Returns:
        tuple: (trained model, scaler, feature importance dataframe)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.25, random_state=42, stratify=targets)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model using XGBoost for binary classification
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=1.0
    )

    # Fit model
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    logger.info("Binary Classification Results:")
    logger.info(classification_report(y_test, y_pred))

    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': importances
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        logger.info("Top 10 features for binary classification:")
        logger.info(feature_importance.head(10))
    else:
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': 0.0
        })

    return model, scaler, feature_importance


def train_severity_model(features, targets):
    """
    Train the severity classification model for partial vs. complete paralysis.

    Args:
        features (pandas.DataFrame): Feature data
        targets (numpy.ndarray): Severity target labels

    Returns:
        tuple: (trained model, scaler, feature importance dataframe)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.25, random_state=42, stratify=targets)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model using Gradient Boosting for severity classification
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )

    # Fit model
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)

    logger.info("Severity Classification Results:")
    logger.info(classification_report(y_test, y_pred))

    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': importances
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        logger.info("Top 10 features for severity classification:")
        logger.info(feature_importance.head(10))
    else:
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': 0.0
        })

    return model, scaler, feature_importance


def calculate_ratio(val1, val2):
    """Calculate ratio between values (min/max)."""
    # Make copies to avoid warnings
    val1_copy = val1.copy()
    val2_copy = val2.copy()

    # Replace zeros with small value to avoid division by zero
    val1_copy = val1_copy.replace(0, 0.0001)
    val2_copy = val2_copy.replace(0, 0.0001)

    min_vals = pd.Series([min(a, b) for a, b in zip(val1_copy, val2_copy)])
    max_vals = pd.Series([max(a, b) for a, b in zip(val1_copy, val2_copy)])

    # Calculate ratio (capped at 1.0)
    ratio = min_vals / max_vals
    ratio[ratio > 1.0] = 1.0  # Cap at 1.0 in case of floating point issues

    return ratio


def calculate_percent_diff(val1, val2):
    """Calculate percent difference between values."""
    # Make copies to avoid warnings
    val1_copy = val1.copy()
    val2_copy = val2.copy()

    # Calculate absolute difference
    abs_diff = abs(val1_copy - val2_copy)

    # Calculate average of the two values
    avg = (val1_copy + val2_copy) / 2

    # Replace zeros with small value to avoid division by zero
    avg = avg.replace(0, 0.0001)

    # Calculate percent difference
    percent_diff = (abs_diff / avg) * 100

    # Cap at 200% for extreme differences
    percent_diff[percent_diff > 200] = 200

    return percent_diff


if __name__ == "__main__":
    train_two_stage_models()