# impact_predictor.py (v2 - Dual Pipeline Support)
import pandas as pd
import numpy as np
import logging

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import joblib
from copy import deepcopy
import xgboost as xgb  # Added for type checking if needed
from sklearn.ensemble import VotingClassifier, RandomForestClassifier  # Added for type checking if needed

logger = logging.getLogger(__name__)


class ImpactPredictor:
    """Predicts the impact of potential label changes"""

    def __init__(self, item_key, model, scaler, feature_names, item_config, item_class_map):
        self.item_key = item_key
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.item_config = item_config  # Specific config for the item
        self.item_name_display = self.item_config.get('name', item_key.capitalize())
        self.item_class_map = item_class_map  # e.g. PARALYSIS_MAP or SYNKINESIS_MAP
        self.numerical_label_map = {v: k for k, v in self.item_class_map.items()}  # Name to number

    def analyze_potential_changes(self, proposed_changes_df, features_df,
                                  current_labels_numeric, metadata_df):  # Expect numeric labels
        logger.info(f"[{self.item_name_display}] Analyzing {len(proposed_changes_df)} proposed changes...")

        # Create modified labels (numeric)
        modified_labels_numeric = current_labels_numeric.copy()
        changes_applied_count = 0
        for _, change in proposed_changes_df.iterrows():
            mask = (metadata_df['Patient ID'] == change['Patient ID']) & \
                   (metadata_df['Side'] == change['Side'])
            if mask.any():
                idx = metadata_df[mask].index[0]
                # proposed_changes_df New_Label is string name, convert to numeric
                new_label_num = self.numerical_label_map.get(change['New_Label'], -1)
                if new_label_num != -1 and idx < len(modified_labels_numeric):
                    modified_labels_numeric[idx] = new_label_num
                    changes_applied_count += 1
                # else: logger.warning(f"Could not map new label '{change['New_Label']}' or index out of bounds for patient {change['Patient ID']}/{change['Side']}") # Reduced
        # logger.debug(f"Applied {changes_applied_count} changes to create modified_labels_numeric.") # Reduced

        impact_results_summary = self._quick_validation(
            features_df, current_labels_numeric, modified_labels_numeric
        )

        change_impacts = []
        for _, change in proposed_changes_df.iterrows():
            mask = (metadata_df['Patient ID'] == change['Patient ID']) & \
                   (metadata_df['Side'] == change['Side'])
            if mask.any():
                idx = metadata_df[mask].index[0]
                if idx >= len(features_df):  # Boundary check
                    # logger.warning(f"Index {idx} out of bounds for features_df. Skipping change impact for {change['Patient ID']}/{change['Side']}.") # Reduced
                    continue

                features_single_scaled = self.scaler.transform(features_df.iloc[[idx]])
                pred_proba_single = self.model.predict_proba(features_single_scaled)[0]
                pred_class_num = self.model.predict(features_single_scaled)[0]

                old_label_num = current_labels_numeric[idx]
                new_label_str = change['New_Label']
                new_label_num = self.numerical_label_map.get(new_label_str, -1)

                impact = {
                    'Patient ID': change['Patient ID'], 'Side': change['Side'],
                    'Current_Label_Num': old_label_num,
                    'Current_Label_Name': self.item_class_map.get(old_label_num, 'Unknown'),
                    'Proposed_Label_Num': new_label_num,
                    'Proposed_Label_Name': new_label_str if new_label_num != -1 else 'Invalid',
                    'Model_Pred_Num': pred_class_num,
                    'Model_Pred_Name': self.item_class_map.get(pred_class_num, 'Unknown'),
                    'Model_Agrees_With_Change': pred_class_num == new_label_num if new_label_num != -1 else False,
                    'Confidence_In_Proposed': pred_proba_single[
                        new_label_num] if new_label_num != -1 and new_label_num < len(pred_proba_single) else 0.0,
                    'Change_Type': self._classify_change(old_label_num, new_label_num)
                }
                change_impacts.append(impact)
        change_impacts_df = pd.DataFrame(change_impacts)
        if not change_impacts_df.empty:  # Add overall impact only if there are per-change results
            change_impacts_df['Overall_F1_Improvement'] = impact_results_summary.get('f1_improvement', 0.0)
            change_impacts_df['Overall_F1_Original'] = impact_results_summary.get('f1_original', 0.0)
            change_impacts_df['Overall_F1_Modified'] = impact_results_summary.get('f1_modified', 0.0)
        return change_impacts_df

    def _quick_validation(self, features, original_labels_numeric, modified_labels_numeric, n_folds=3):
        # logger.info(f"[{self.item_name_display}] Running quick validation...") # Reduced

        n_changes = np.sum(original_labels_numeric != modified_labels_numeric)
        if n_changes == 0: return {'f1_improvement': 0.0, 'significant': False, 'f1_original': 0.0, 'f1_modified': 0.0}
        # logger.info(f"[{self.item_name_display}] Testing {n_changes} label changes...") # Reduced

        unique_orig_labels, counts_orig = np.unique(original_labels_numeric, return_counts=True)
        can_stratify = len(unique_orig_labels) > 1 and all(c >= n_folds for c in counts_orig)

        if can_stratify:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = list(kf.split(features, original_labels_numeric))  # Ensure it's a list for re-use if needed
        else:
            # logger.warning(f"Cannot stratify for quick_validation in {self.item_name_display}. Using KFold.") # Reduced
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = list(kf.split(features))

        original_scores, modified_scores = [], []
        for train_idx, val_idx in splits:
            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            scaler_temp = deepcopy(self.scaler);
            X_train_scaled = scaler_temp.fit_transform(X_train);
            X_val_scaled = scaler_temp.transform(X_val)

            # Original
            y_train_orig, y_val_orig = original_labels_numeric[train_idx], original_labels_numeric[val_idx]
            model_orig = deepcopy(self.model)  # Use a fresh copy for each fold
            try:
                # Handle different model types (CalibratedCV vs base XGB/RF/Voting)
                # For CalibratedClassifierCV, fit the base_estimator or estimator
                # For VotingClassifier, it's fitted directly.
                # For XGBoost/RandomForest, fitted directly.
                estimator_to_fit = model_orig
                if hasattr(model_orig, 'estimator') and model_orig.estimator is not None:  # CalibratedCV with prefit
                    estimator_to_fit = model_orig.estimator
                elif hasattr(model_orig,
                             'base_estimator') and model_orig.base_estimator is not None:  # CalibratedCV with CV
                    # This case is tricky as base_estimator is a template.
                    # For quick validation, we'll just refit the outer CalibratedCV on train fold.
                    # Or, more simply, fit the main model object if it's not CalibratedCV
                    pass  # Fit model_orig directly

                if isinstance(estimator_to_fit, (xgb.XGBClassifier, RandomForestClassifier, VotingClassifier)):
                    estimator_to_fit.fit(X_train_scaled,
                                         y_train_orig)  # Fit the actual underlying model if not CalibratedCV
                elif isinstance(model_orig, CalibratedClassifierCV) and model_orig.cv != 'prefit':
                    # If it's CalibratedCV with actual CV (e.g., integer), refit the whole thing
                    model_orig.fit(X_train_scaled, y_train_orig)
                    # If it's prefit CalibratedCV, estimator_to_fit (base) was fitted.
                # Prediction is always done on the main model object.
                pred_orig = model_orig.predict(X_val_scaled)
                original_scores.append(f1_score(y_val_orig, pred_orig, average='weighted', zero_division=0))
            except Exception as e:
                logger.warning(f"Error in original validation fold: {e}"); original_scores.append(0.0)

            # Modified
            y_train_mod, y_val_mod = modified_labels_numeric[train_idx], modified_labels_numeric[val_idx]
            model_mod = deepcopy(self.model)  # Fresh copy
            try:
                estimator_to_fit_mod = model_mod
                if hasattr(model_mod, 'estimator') and model_mod.estimator is not None:
                    estimator_to_fit_mod = model_mod.estimator
                elif hasattr(model_mod, 'base_estimator') and model_mod.base_estimator is not None:
                    pass
                if isinstance(estimator_to_fit_mod, (xgb.XGBClassifier, RandomForestClassifier, VotingClassifier)):
                    estimator_to_fit_mod.fit(X_train_scaled, y_train_mod)
                elif isinstance(model_mod, CalibratedClassifierCV) and model_mod.cv != 'prefit':
                    model_mod.fit(X_train_scaled, y_train_mod)
                pred_mod = model_mod.predict(X_val_scaled)
                modified_scores.append(f1_score(y_val_mod, pred_mod, average='weighted', zero_division=0))
            except Exception as e:
                logger.warning(f"Error in modified validation fold: {e}"); modified_scores.append(0.0)

        avg_original = np.mean(original_scores) if original_scores else 0.0
        avg_modified = np.mean(modified_scores) if modified_scores else 0.0
        improvement = avg_modified - avg_original
        significant = improvement > self.item_config.get('training', {}).get('review_validation', {}).get(
            'min_improvement_threshold', 0.01)
        # logger.info(f"[{self.item_name_display}] Orig F1: {avg_original:.4f}, Mod F1: {avg_modified:.4f}, Improv: {improvement:.4f}") # Reduced
        return {'f1_original': avg_original, 'f1_modified': avg_modified, 'f1_improvement': improvement,
                'significant': significant, 'confidence': 'low' if n_folds < 5 else 'medium'}

    def _classify_change(self, old_label_num, new_label_num):
        if old_label_num == new_label_num: return "No_Change"
        old_name = self.item_class_map.get(old_label_num, f'Num_{old_label_num}')
        new_name = self.item_class_map.get(new_label_num, f'Num_{new_label_num}')
        return f"{old_name}_to_{new_name}"

    def estimate_performance_impact(self, features_df, current_labels_numeric,
                                    proposed_changes_df, metadata_df):
        # logger.info(f"[{self.item_name_display}] Estimating performance impact...") # Reduced

        change_analysis_df = self.analyze_potential_changes(
            proposed_changes_df, features_df, current_labels_numeric, metadata_df
        )
        summary = {
            'total_changes': len(proposed_changes_df),
            'model_agrees': change_analysis_df[
                'Model_Agrees_With_Change'].sum() if 'Model_Agrees_With_Change' in change_analysis_df else 0,
            'model_disagrees': (~change_analysis_df[
                'Model_Agrees_With_Change']).sum() if 'Model_Agrees_With_Change' in change_analysis_df else 0,
            'avg_confidence_in_changes': change_analysis_df[
                'Confidence_In_Proposed'].mean() if 'Confidence_In_Proposed' in change_analysis_df else 0.0,
            'expected_f1_improvement': change_analysis_df['Overall_F1_Improvement'].iloc[
                0] if not change_analysis_df.empty and 'Overall_F1_Improvement' in change_analysis_df else 0.0
        }

        current_dist = pd.Series(current_labels_numeric).value_counts(normalize=True)
        modified_labels_numeric_temp = current_labels_numeric.copy()  # Temp copy for dist calc
        for _, change in proposed_changes_df.iterrows():
            mask = (metadata_df['Patient ID'] == change['Patient ID']) & (metadata_df['Side'] == change['Side'])
            if mask.any():
                idx = metadata_df[mask].index[0]
                new_label_num = self.numerical_label_map.get(change['New_Label'], -1)  # New_Label is string name
                if new_label_num != -1 and idx < len(modified_labels_numeric_temp): modified_labels_numeric_temp[
                    idx] = new_label_num
        new_dist = pd.Series(modified_labels_numeric_temp).value_counts(normalize=True)

        dist_shift = {}
        all_possible_numeric_labels = sorted(list(self.item_class_map.keys()))  # e.g. [0,1] or [0,1,2]
        for label_num in all_possible_numeric_labels:
            old_prop = current_dist.get(label_num, 0.0);
            new_prop = new_dist.get(label_num, 0.0)
            dist_shift[self.item_class_map.get(label_num,
                                               f"Num_{label_num}")] = new_prop - old_prop  # Store shift by label name
        summary['distribution_shift_details'] = dist_shift
        summary['max_distribution_shift_magnitude'] = max(abs(v) for v in dist_shift.values()) if dist_shift else 0.0

        recommendations = []
        # Use review_config_global for thresholds
        validation_thresholds = REVIEW_CONFIG_GLOBAL.get('validation', {})
        min_f1_improv = validation_thresholds.get('min_improvement_threshold', 0.01)
        max_dist_shift_allowed = REVIEW_CONFIG_GLOBAL.get('change_limits', {}).get('max_distribution_shift', 0.05)

        if summary['expected_f1_improvement'] > min_f1_improv:
            recommendations.append("Changes likely to improve model performance")
        elif summary['expected_f1_improvement'] < -min_f1_improv:
            recommendations.append("WARNING: Changes may hurt model performance")
        if summary['model_agrees'] > summary['model_disagrees']:
            recommendations.append("Model generally agrees with proposed changes")
        else:
            recommendations.append("Model disagrees with many proposed changes - review carefully")
        if summary['max_distribution_shift_magnitude'] > max_dist_shift_allowed: recommendations.append(
            "WARNING: Significant distribution shift detected")
        summary['recommendations'] = recommendations if recommendations else [
            "No strong signals for/against changes based on automated checks."]
        return summary

    def identify_high_impact_changes(self, review_candidates_df, features_df,
                                     current_labels_numeric, metadata_df, top_k=20):
        # logger.info(f"[{self.item_name_display}] Identifying high-impact changes...") # Reduced
        if review_candidates_df is None or review_candidates_df.empty:
            logger.warning(
                f"[{self.item_name_display}] review_candidates_df is empty. Cannot identify high impact changes.")
            return pd.DataFrame()

        misclassified = review_candidates_df[review_candidates_df.get('Is_Correct', True) == False].copy()
        if misclassified.empty:
            # logger.warning(f"[{self.item_name_display}] No misclassified cases found in review_candidates_df.") # Reduced
            return pd.DataFrame()

        impact_scores = []
        for _, row in misclassified.iterrows():
            patient_id, side = row['Patient ID'], row['Side']
            model_pred_name = row.get('Predicted_Label_Name', 'Unknown')
            if model_pred_name == 'Unknown': continue

            potential_change_df = pd.DataFrame([{'Patient ID': patient_id, 'Side': side, 'New_Label': model_pred_name}])
            mask = (metadata_df['Patient ID'] == patient_id) & (metadata_df['Side'] == side)
            if mask.any():
                idx = metadata_df[mask].index[0]
                # For single point impact, we use confidence and influence directly rather than re-running full validation
                # Ensure Prob_{model_pred_name} column exists by constructing it carefully
                safe_model_pred_name = model_pred_name.replace(' ', '_')
                confidence_col_name = f'Prob_{safe_model_pred_name}'
                confidence = row.get(confidence_col_name, 0.0)

                influence = row.get('Influence_Score (Delta_F1)', 0.0)
                is_critical = 'CRITICAL' in str(row.get('Flag_Reason', '')).upper() or \
                              'CRITICAL_ERROR' in str(row.get('Error_Type', '')).upper()

                # Revised impact score: prioritize fixing critical errors, then high confidence errors, then influence.
                impact_score = 0.0
                if is_critical: impact_score += 1000
                impact_score += confidence * 100  # Confidence in the model's (wrong) prediction
                if influence > 0:
                    impact_score += influence * 500  # If removing this (if it were error) would help a lot
                elif influence < 0:
                    impact_score += abs(
                        influence) * 100  # If it's an influential correct point that was misflagged somehow

                impact_scores.append({
                    'Patient ID': patient_id, 'Side': side,
                    'Current_Label_Name': row.get('Expert_Label_Name', 'Unknown'),
                    'Suggested_Change_To_Name': model_pred_name,
                    'Model_Confidence_In_Suggestion': confidence,
                    'Estimated_F1_Gain_If_Corrected': influence,  # This is delta_F1 if point *removed*
                    'Impact_Score': impact_score,
                    'Is_Critical_Error_Pattern': is_critical,
                    'Review_Tier': row.get('Review_Tier', 4)
                })
        impact_df = pd.DataFrame(impact_scores)
        if not impact_df.empty:
            impact_df = impact_df.sort_values('Impact_Score', ascending=False)
            return impact_df.head(top_k)
        return impact_df