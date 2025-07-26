# review_advisor.py (v2.3.1 - Fix training_ground_truth rename and bool coalesce)
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ReviewAdvisor:
    """Consolidates review data and provides actionable recommendations"""

    def __init__(self, item_key, item_config, item_class_map, review_config_global):
        self.item_key = item_key
        self.item_config = item_config
        self.item_name_display = self.item_config.get('name', item_key.capitalize())
        self.item_class_map = item_class_map  # e.g., {0: 'None', 1: 'Synkinesis'}
        self.review_config_global = review_config_global

    def generate_prioritized_review_list(self,
                                         review_candidates_df,
                                         error_details_df,
                                         consistency_df=None,
                                         training_ground_truth_df=None):  # NEW parameter
        logger.info(f"[{self.item_name_display}] Generating prioritized review list...")

        # Prepare input DFs by ensuring Patient ID/Side are strings and adding suffixes
        if review_candidates_df is None or review_candidates_df.empty:
            cand_df_processed = pd.DataFrame()
        else:
            cand_df_processed = review_candidates_df.copy()
            if 'Patient ID' in cand_df_processed.columns: cand_df_processed['Patient ID'] = cand_df_processed[
                'Patient ID'].astype(str).str.strip()
            if 'Side' in cand_df_processed.columns: cand_df_processed['Side'] = cand_df_processed['Side'].astype(
                str).str.strip()
            cand_df_processed = cand_df_processed.rename(
                columns=lambda c: c + '_cand' if c not in ['Patient ID', 'Side'] else c)

        if error_details_df is None or error_details_df.empty:
            error_df_processed = pd.DataFrame()
        else:
            error_df_processed = error_details_df.copy()
            if 'Patient ID' in error_df_processed.columns: error_df_processed['Patient ID'] = error_df_processed[
                'Patient ID'].astype(str).str.strip()
            if 'Side' in error_df_processed.columns: error_df_processed['Side'] = error_df_processed['Side'].astype(
                str).str.strip()
            error_df_processed = error_df_processed.rename(
                columns=lambda c: c + '_error' if c not in ['Patient ID', 'Side'] else c)

        if consistency_df is None or consistency_df.empty:
            consistency_df_processed = pd.DataFrame()
        else:
            consistency_df_processed = consistency_df.copy()
            if 'Patient ID' in consistency_df_processed.columns: consistency_df_processed['Patient ID'] = \
            consistency_df_processed['Patient ID'].astype(str).str.strip()
            if 'Side' in consistency_df_processed.columns: consistency_df_processed['Side'] = consistency_df_processed[
                'Side'].astype(str).str.strip()
            consistency_df_processed = consistency_df_processed.rename(
                columns=lambda c: c + '_consist' if c not in ['Patient ID', 'Side'] else c)

        # MODIFIED MERGE LOGIC: Start with training_ground_truth_df if available
        if training_ground_truth_df is not None and not training_ground_truth_df.empty and \
                'Patient ID' in training_ground_truth_df.columns and 'Side' in training_ground_truth_df.columns:
            final_df = training_ground_truth_df.copy()
            # The column 'Expert_Label_Name_train_gt' is already named correctly from generate_review_package.py
            # No rename needed here. Just ensure it exists.
            if 'Expert_Label_Name_train_gt' not in final_df.columns:
                logger.warning(
                    f"[{self.item_name_display}] Expected 'Expert_Label_Name_train_gt' not found in provided training_ground_truth_df. Setting to NaN.")
                final_df['Expert_Label_Name_train_gt'] = np.nan

            if not error_df_processed.empty:
                final_df = pd.merge(final_df, error_df_processed, on=['Patient ID', 'Side'], how='left',
                                    suffixes=('', '_dup_error'))
            if not cand_df_processed.empty:
                final_df = pd.merge(final_df, cand_df_processed, on=['Patient ID', 'Side'], how='left',
                                    suffixes=('', '_dup_cand'))
            if not consistency_df_processed.empty:
                final_df = pd.merge(final_df, consistency_df_processed, on=['Patient ID', 'Side'], how='left',
                                    suffixes=('', '_dup_consist'))
        else:
            logger.warning(
                f"[{self.item_name_display}] training_ground_truth_df not provided or invalid. Using original merge logic based on errors/candidates first.")
            if not error_df_processed.empty:
                final_df = error_df_processed.copy()
                if not cand_df_processed.empty:
                    final_df = pd.merge(final_df, cand_df_processed, on=['Patient ID', 'Side'], how='outer',
                                        suffixes=('', '_dup_cand'))
            elif not cand_df_processed.empty:
                final_df = cand_df_processed.copy()
            else:
                final_df = pd.DataFrame(columns=['Patient ID', 'Side'])

            if not consistency_df_processed.empty:
                if not final_df.empty and 'Patient ID' in final_df.columns and 'Side' in final_df.columns:
                    final_df = pd.merge(final_df, consistency_df_processed, on=['Patient ID', 'Side'], how='outer',
                                        suffixes=('', '_dup_consist'))
                elif final_df.empty:
                    final_df = consistency_df_processed.copy()

        if final_df.empty or 'Patient ID' not in final_df.columns:
            logger.warning(
                f"[{self.item_name_display}] Final_df is empty or missing key columns after merges. No recommendations generated.")
            return pd.DataFrame()

        final_df.drop_duplicates(subset=['Patient ID', 'Side'], keep='first', inplace=True)

        # --- Coalesce columns ---
        processed_cols = set(final_df.columns)

        prob_col_bases = [name.replace(' ', '_') for name in self.item_class_map.values()]

        cols_to_coalesce_defs = {
            'Expert_Label_Name': (
            'Expert_Label_Name_error', 'Expert_Label_Name_cand', 'Expert_Label_Name_train_gt', 'Unknown'),
            'Predicted_Label_Name': ('Predicted_Label_Name_error', 'Predicted_Label_Name_cand', None, 'Unknown'),
            'Is_Correct': ('Is_Correct_error', 'Is_Correct_cand', None, False),
            'Model_Confidence': ('Model_Confidence_error', 'Model_Confidence_cand', None, 0.0),
            'Error_Type': ('Error_Type_error', 'Error_Type_cand', None, 'N/A'),
            'Entropy': (None, 'Entropy_cand', 'Entropy_error', 0.0),
            'Margin': (None, 'Margin_cand', 'Margin_error', 1.0),
            'Prob_True_Label': (None, 'Prob_True_Label_cand', 'Prob_True_Label_error', 1.0),
            'Influence_Score (Delta_F1)': (None, 'Influence_Score (Delta_F1)_cand', None, 0.0),
            'Flag_Reason': (None, 'Flag_Reason_cand', None, ""),
            'Inconsistency_Score': (None, None, 'Inconsistency_Score_consist', 0.0),
            'Num_Inconsistent_Similar': (None, None, 'Num_Inconsistent_Similar_consist', 0),
            'Avg_Similarity': (None, None, 'Avg_Similarity_consist', 0.0),
            'Similar_Inconsistent_Patients': (None, None, 'Similar_Inconsistent_Patients_consist', "")
        }
        for base_name in prob_col_bases:
            prob_col = f'Prob_{base_name}'
            cols_to_coalesce_defs[prob_col] = (f'{prob_col}_error', f'{prob_col}_cand', None, 0.0)

        all_source_cols_to_ensure = set()
        # Ensure Expert_Label_Name_train_gt exists if it's a source, even if training_ground_truth_df was empty/problematic
        if 'Expert_Label_Name_train_gt' not in final_df.columns:
            final_df['Expert_Label_Name_train_gt'] = np.nan

        for final_col_name, sources_config in cols_to_coalesce_defs.items():
            for src_col_name_part in sources_config[:-1]:
                if src_col_name_part:
                    all_source_cols_to_ensure.add(src_col_name_part)

        for col_to_ensure in all_source_cols_to_ensure:
            if col_to_ensure not in final_df.columns:
                final_df[col_to_ensure] = np.nan

        for final_col_name, sources_config in cols_to_coalesce_defs.items():
            src_error_col, src_cand_col, src_third_col, default_value = sources_config

            if final_col_name not in final_df.columns: final_df[final_col_name] = np.nan

            # Coalesce: Start with current value then fill with third source, then candidate, then error
            if src_third_col and src_third_col in final_df.columns:
                final_df[final_col_name] = final_df[final_col_name].fillna(final_df[src_third_col])
            if src_cand_col and src_cand_col in final_df.columns:
                final_df[final_col_name] = final_df[final_col_name].fillna(final_df[src_cand_col])
            if src_error_col and src_error_col in final_df.columns:
                final_df[final_col_name] = final_df[final_col_name].fillna(final_df[src_error_col])

            if isinstance(default_value, (int, float)):
                final_df[final_col_name] = pd.to_numeric(final_df[final_col_name], errors='coerce').fillna(
                    default_value)
            elif isinstance(default_value, bool):
                temp_series = final_df[final_col_name].copy()
                bool_map = {'TRUE': True, 'True': True, 'true': True, True: True,
                            'FALSE': False, 'False': False, 'false': False, False: False}
                # If the series is object/string, try mapping known string bools first
                if pd.api.types.is_object_dtype(temp_series.dtype) or pd.api.types.is_string_dtype(temp_series.dtype):
                    temp_series = temp_series.map(bool_map).fillna(temp_series)  # Replace known strings, keep others

                # Now fill remaining NaNs with the boolean default
                temp_series = temp_series.fillna(default_value)

                # Attempt to convert to bool; if it fails due to unmapped strings, log and use default
                try:
                    final_df[final_col_name] = temp_series.astype(bool)
                except TypeError:  # Handle cases where unmapped strings might cause astype(bool) to fail
                    logger.warning(
                        f"Could not cleanly convert column '{final_col_name}' to boolean due to mixed types not in bool_map. Applying default '{default_value}'. Problematic values: {temp_series[~temp_series.isin([True, False])].unique()}")
                    # Set to default for all rows if conversion fails for any
                    final_df[final_col_name] = default_value
            else:  # String
                final_df[final_col_name] = final_df[final_col_name].fillna(default_value)
                if default_value in ['Unknown', 'N/A'] and '' in final_df[final_col_name].unique():
                    final_df[final_col_name] = final_df[final_col_name].replace('', default_value)

            processed_cols.add(final_col_name)

        if 'Expert_Label_Name' in final_df and 'Predicted_Label_Name' in final_df:
            mask_unknown_labels = (final_df['Expert_Label_Name'] == 'Unknown') | \
                                  (final_df['Predicted_Label_Name'] == 'Unknown')
            # Only set Is_Correct to False for these if it wasn't already False from coalescing
            final_df.loc[mask_unknown_labels & (
                        final_df['Is_Correct'] != False), 'Is_Correct'] = False  # Use & with non-False check

            mask_known_labels = (~mask_unknown_labels)
            # Only calculate if Is_Correct is currently NaN (or not False due to unknowns) for these known_label rows
            # This preserves Is_Correct values that came from _error or _cand files
            final_df.loc[mask_known_labels & final_df['Is_Correct'].isna(), 'Is_Correct'] = \
                (final_df.loc[mask_known_labels & final_df['Is_Correct'].isna(), 'Expert_Label_Name'] == \
                 final_df.loc[mask_known_labels & final_df['Is_Correct'].isna(), 'Predicted_Label_Name'])
        else:
            final_df['Is_Correct'] = False
        final_df['Is_Correct'] = final_df['Is_Correct'].fillna(False).astype(bool)

        all_cols_after_coalesce = final_df.columns.tolist()
        cols_to_drop_final = [
            col for col in all_cols_after_coalesce
            if (col.endswith('_cand') or col.endswith('_error') or col.endswith('_consist') or \
                col.endswith('_dup_error') or col.endswith('_dup_cand') or col.endswith('_dup_consist') or \
                col == 'Expert_Label_Name_train_gt') and \
               col not in processed_cols
        ]
        legacy_suffixed = ['Expert_Label_error', 'Model_Prediction_error', 'Zone_error',
                           'Expert_Label_cand', 'Model_Prediction_cand', 'Zone_cand', 'PriorityScore_cand']
        for lc in legacy_suffixed:
            if lc in final_df.columns and lc not in processed_cols:
                cols_to_drop_final.append(lc)

        if cols_to_drop_final:
            cols_to_drop_unique_final = list(set(cols_to_drop_final))
            final_df.drop(columns=cols_to_drop_unique_final, inplace=True, errors='ignore')

        if final_df.empty:
            logger.warning(
                f"[{self.item_name_display}] Final_df is empty before scoring. No recommendations generated.")
            return pd.DataFrame()

        numeric_cols_for_scoring = [
                                       'Model_Confidence', 'Entropy', 'Margin', 'Inconsistency_Score',
                                       'Influence_Score (Delta_F1)', 'Prob_True_Label'
                                   ] + [f'Prob_{base_name}' for base_name in prob_col_bases]

        for col in numeric_cols_for_scoring:
            if col not in final_df.columns:
                if col.startswith("Prob_"):
                    final_df[col] = 0.0
                elif col in ['Model_Confidence', 'Entropy', 'Inconsistency_Score', 'Influence_Score (Delta_F1)']:
                    final_df[col] = 0.0
                elif col == 'Margin' or col == 'Prob_True_Label':
                    final_df[col] = 1.0
            else:
                default_fill = 0.0
                if col == 'Margin' or col == 'Prob_True_Label': default_fill = 1.0
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(default_fill)

        final_df['Review_Priority_Score'] = final_df.apply(self._calculate_priority_score, axis=1)
        final_df['Review_Tier'] = final_df.apply(self._assign_tier, axis=1)
        final_df['Recommendation'] = final_df.apply(self._generate_recommendation, axis=1)
        final_df['Review_Context'] = final_df.apply(self._create_review_context, axis=1)

        final_df = final_df.sort_values(
            ['Review_Tier', 'Review_Priority_Score'],
            ascending=[True, False]
        ).reset_index(drop=True)

        logger.info(f"[{self.item_name_display}] Generated {len(final_df)} review recommendations.")
        return final_df

    def _safe_get_float(self, row, key, default_value):
        val = row.get(key)
        if pd.isna(val): return default_value
        try:
            return float(val)
        except (ValueError, TypeError):
            return default_value

    def _calculate_priority_score(self, row):
        score = 0.0
        is_correct = row.get('Is_Correct', True)
        flag_reason = str(row.get('Flag_Reason', ''))
        error_type_str = str(row.get('Error_Type', ''))

        model_confidence = self._safe_get_float(row, 'Model_Confidence', 0.0)
        entropy = self._safe_get_float(row, 'Entropy', 0.0)
        margin = self._safe_get_float(row, 'Margin', 1.0)
        inconsistency_score = self._safe_get_float(row, 'Inconsistency_Score', 0.0)
        influence_score = self._safe_get_float(row, 'Influence_Score (Delta_F1)', 0.0)
        prob_true_label = self._safe_get_float(row, 'Prob_True_Label', 1.0)

        priority_weights = self.review_config_global.get('priority_weights', {})
        default_weights = {'critical_error_base': 50.0, 'misclassification_base': 30.0,
                           'confidence_in_error_multiplier': 20.0, 'entropy_multiplier': 10.0,
                           'margin_multiplier': 10.0, 'low_true_label_prob_threshold': 0.5,
                           'low_true_label_prob_multiplier': 20.0, 'inconsistency_multiplier': 15.0,
                           'positive_influence_multiplier': 100.0, 'negative_influence_incorrect_multiplier': 50.0}

        if 'CRITICAL' in flag_reason.upper() or 'CRITICAL_ERROR' in error_type_str.upper():
            score += priority_weights.get('critical_error_base', default_weights['critical_error_base'])
        elif not is_correct:
            score += priority_weights.get('misclassification_base', default_weights['misclassification_base'])

        if not is_correct: score += model_confidence * priority_weights.get('confidence_in_error_multiplier',
                                                                            default_weights[
                                                                                'confidence_in_error_multiplier'])

        score += entropy * priority_weights.get('entropy_multiplier', default_weights['entropy_multiplier'])
        score += (1.0 - margin) * priority_weights.get('margin_multiplier', default_weights['margin_multiplier'])

        threshold = priority_weights.get('low_true_label_prob_threshold',
                                         default_weights['low_true_label_prob_threshold'])
        if not is_correct and prob_true_label < threshold:
            score += (threshold - prob_true_label) * priority_weights.get('low_true_label_prob_multiplier',
                                                                          default_weights[
                                                                              'low_true_label_prob_multiplier'])

        score += inconsistency_score * priority_weights.get('inconsistency_multiplier',
                                                            default_weights['inconsistency_multiplier'])

        if influence_score > 0:
            score += influence_score * priority_weights.get('positive_influence_multiplier',
                                                            default_weights['positive_influence_multiplier'])
        elif influence_score < 0:
            if not is_correct: score += abs(influence_score) * priority_weights.get(
                'negative_influence_incorrect_multiplier', default_weights['negative_influence_incorrect_multiplier'])
        return score

    def _assign_tier(self, row):
        is_correct = row.get('Is_Correct', True)
        flag_reason = str(row.get('Flag_Reason', ''))
        error_type_str = str(row.get('Error_Type', ''))
        model_confidence = self._safe_get_float(row, 'Model_Confidence', 0.0)
        inconsistency_score = self._safe_get_float(row, 'Inconsistency_Score', 0.0)
        entropy = self._safe_get_float(row, 'Entropy', 0.0)
        margin = self._safe_get_float(row, 'Margin', 1.0)

        tier_thresholds_cfg = self.review_config_global.get('review_tiers_config', {}).get('thresholds', {})
        default_thresholds = {'high_confidence_error': 0.9, 'inconsistency_tier2': 0.5,
                              'entropy_tier3': 0.8, 'margin_tier3': 0.2}

        if 'CRITICAL' in flag_reason.upper() or 'CRITICAL_ERROR' in error_type_str.upper(): return 1
        if not is_correct and model_confidence > tier_thresholds_cfg.get('high_confidence_error', default_thresholds[
            'high_confidence_error']): return 1

        current_tier = 4
        if not is_correct: current_tier = 2

        if inconsistency_score > tier_thresholds_cfg.get('inconsistency_tier2',
                                                         default_thresholds['inconsistency_tier2']):
            current_tier = min(current_tier, 2)

        if current_tier > 2:
            if (entropy > tier_thresholds_cfg.get('entropy_tier3', default_thresholds['entropy_tier3'])) or \
                    (margin < tier_thresholds_cfg.get('margin_tier3', default_thresholds['margin_tier3'])):
                current_tier = min(current_tier, 3)

        if not is_correct and flag_reason and current_tier == 4:
            current_tier = 3

        return current_tier

    def _generate_recommendation(self, row):
        recommendations = []
        is_correct = row.get('Is_Correct', True)
        flag_reason = str(row.get('Flag_Reason', ''))
        error_type_str = str(row.get('Error_Type', ''))
        model_confidence = self._safe_get_float(row, 'Model_Confidence', 0.0)
        inconsistency_score = self._safe_get_float(row, 'Inconsistency_Score', 0.0)
        entropy = self._safe_get_float(row, 'Entropy', 0.0)
        influence_score = self._safe_get_float(row, 'Influence_Score (Delta_F1)', 0.0)
        predicted_label = row.get('Predicted_Label_Name', 'Unknown')
        expert_label = row.get('Expert_Label_Name', 'Unknown')
        similar_inconsistent_patients = row.get('Similar_Inconsistent_Patients', '')

        reco_thresholds_cfg = self.review_config_global.get('recommendation_thresholds', {})
        default_reco_thresholds = {'high_confidence_error_threshold': 0.9, 'inconsistency_reco_threshold': 0.1,
                                   'entropy_reco_threshold': 0.7, 'influence_reco_threshold': 0.01}

        if 'CRITICAL' in flag_reason.upper() or 'CRITICAL_ERROR' in error_type_str.upper():
            recommendations.append(
                f"CRITICAL: Review immediately - {error_type_str.replace('_', ' ')} (Expert: {expert_label}, Model: {predicted_label})")
        elif not is_correct and expert_label != 'Unknown' and predicted_label != 'Unknown':
            reco_text = f"Model predicted '{predicted_label}' (conf: {model_confidence:.2f}), expert was '{expert_label}'. Verify expert label."
            if model_confidence > reco_thresholds_cfg.get('high_confidence_error_threshold',
                                                          default_reco_thresholds['high_confidence_error_threshold']):
                reco_text = f"Model HIGHLY CONFIDENT ({model_confidence:.2f}) in incorrect prediction '{predicted_label}', expert was '{expert_label}'. Verify expert label."
            recommendations.append(reco_text)
        elif not is_correct and expert_label == 'Unknown' and predicted_label == 'Unknown':
            recommendations.append(
                "Labels are 'Unknown' but flagged as incorrect. Investigate source or manual override.")

        if inconsistency_score > reco_thresholds_cfg.get('inconsistency_reco_threshold',
                                                         default_reco_thresholds['inconsistency_reco_threshold']):
            reco_text = f"Inconsistent with similar patients (Score: {inconsistency_score:.2f})"
            if similar_inconsistent_patients and similar_inconsistent_patients != '': reco_text += f": {similar_inconsistent_patients}"
            recommendations.append(reco_text)

        if entropy > reco_thresholds_cfg.get('entropy_reco_threshold',
                                             default_reco_thresholds['entropy_reco_threshold']):
            recommendations.append("High model uncertainty (entropy). Case may be ambiguous.")

        threshold_influence = reco_thresholds_cfg.get('influence_reco_threshold',
                                                      default_reco_thresholds['influence_reco_threshold'])
        if influence_score > threshold_influence:
            recommendations.append(
                f"High positive influence - changing this label (if incorrect) could improve F1 by ~{influence_score:.3f}")
        elif influence_score < -threshold_influence:
            recommendations.append(
                f"High negative influence - this point is helpful to model. Ensure label correct (Delta_F1 if removed: {influence_score:.3f})")

        if not recommendations:
            if flag_reason:
                recommendations.append(f"Flagged from training analysis: {flag_reason}. General review.")
            elif is_correct:
                recommendations.append("Appears correct. Review for QA or if other context applies.")
            else:
                recommendations.append("Review for general quality check or dataset inclusion.")
        return "; ".join(list(dict.fromkeys(recommendations)))

    def _create_review_context(self, row):
        context = {
            'item_name': self.item_name_display,
            'side': row.get('Side', 'Unknown'),
            'current_label': row.get('Expert_Label_Name', 'Unknown'),
            'model_prediction': row.get('Predicted_Label_Name', 'Unknown'),
            'probabilities': {}
        }
        for class_idx_map, class_name_map_val in self.item_class_map.items():
            safe_class_name = class_name_map_val.replace(' ', '_')
            prob_col = f'Prob_{safe_class_name}'
            prob_val = self._safe_get_float(row, prob_col, 0.0)
            context['probabilities'][class_name_map_val] = prob_val

        context['flags'] = []
        if row.get('Is_Correct') == False: context['flags'].append('Misclassified')
        error_type_str = str(row.get('Error_Type', ''))
        if 'CRITICAL_ERROR' in error_type_str.upper() or 'CRITICAL' in str(row.get('Flag_Reason', '')).upper():
            context['flags'].append('Critical_Error')

        consistency_threshold_context = self.review_config_global.get('recommendation_thresholds', {}).get(
            'inconsistency_reco_threshold', 0.1)
        entropy_threshold_context = self.review_config_global.get('recommendation_thresholds', {}).get(
            'entropy_reco_threshold', 0.7)
        tier_thresholds = self.review_config_global.get('review_tiers_config', {}).get('thresholds', {})
        margin_threshold_display = tier_thresholds.get('margin_tier3', 0.2)

        inconsistency_score_val = self._safe_get_float(row, 'Inconsistency_Score', 0.0)
        entropy_val = self._safe_get_float(row, 'Entropy', 0.0)
        margin_val = self._safe_get_float(row, 'Margin', 1.0)

        if inconsistency_score_val > consistency_threshold_context: context['flags'].append('Potential_Inconsistency')
        if entropy_val > entropy_threshold_context: context['flags'].append('High_Uncertainty_Entropy')
        if margin_val < margin_threshold_display: context['flags'].append('High_Uncertainty_Margin')

        return json.dumps(context)

    def create_review_report(self, review_df, output_path):
        logger.info(f"[{self.item_name_display}] Creating review report...")
        report_lines = []
        report_lines.append(f"REVIEW RECOMMENDATIONS REPORT - {self.item_name_display}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)

        if review_df is None or review_df.empty:
            report_lines.append("\nNo review recommendations generated.")
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write('\n'.join(report_lines))
            except Exception as e:
                logger.error(f"Could not write empty review report: {e}")
            return

        report_lines.append("\nSUMMARY STATISTICS");
        report_lines.append("-" * 40)
        report_lines.append(f"Total cases for review: {len(review_df)}")
        if 'Review_Tier' in review_df.columns:
            tier_counts = review_df['Review_Tier'].value_counts().sort_index()
            report_lines.append("\nCases by Tier:")
            for tier, count in tier_counts.items():
                tier_desc = self.review_config_global.get('review_tiers', {}).get(tier, {}).get('name',
                                                                                                f"Tier {tier} (Unknown)")
                report_lines.append(f"  {tier_desc}: {count} cases")
        else:
            report_lines.append("\nCases by Tier: 'Review_Tier' column not found.")

        report_lines.append("\n" + "=" * 80);
        report_lines.append("DETAILED RECOMMENDATIONS BY TIER (Top 20 per Tier)");
        report_lines.append("=" * 80)
        if 'Review_Tier' in review_df.columns:
            for tier_val in sorted(review_df['Review_Tier'].unique()):
                tier_df = review_df[review_df['Review_Tier'] == tier_val].head(20)
                tier_desc_cfg = self.review_config_global.get('review_tiers', {}).get(tier_val, {})
                tier_name_report = tier_desc_cfg.get('name', f"TIER {tier_val} (Unknown Description)")
                report_lines.append(f"\n{tier_name_report.upper()}");
                report_lines.append("-" * len(tier_name_report))
                for idx, row in tier_df.iterrows():
                    report_lines.append(f"\nPatient: {row.get('Patient ID', 'N/A')}, Side: {row.get('Side', 'N/A')}")
                    report_lines.append(f"  Current Label: {row.get('Expert_Label_Name', 'N/A')}")
                    report_lines.append(f"  Model Prediction: {row.get('Predicted_Label_Name', 'N/A')}")
                    report_lines.append(f"  Is Correct: {row.get('Is_Correct', 'N/A')}")
                    probs_display = []
                    for class_idx_map, class_name_map_val in self.item_class_map.items():
                        safe_class_name = class_name_map_val.replace(' ', '_')
                        prob_col = f'Prob_{safe_class_name}'
                        if prob_col in row and pd.notna(row[prob_col]):
                            probs_display.append(
                                f"{class_name_map_val}: {self._safe_get_float(row, prob_col, 0.0):.3f}")
                    report_lines.append(
                        f"  Probabilities: {', '.join(probs_display) if probs_display else 'Not available'}")
                    report_lines.append(
                        f"  Priority Score: {self._safe_get_float(row, 'Review_Priority_Score', np.nan):.2f}")
                    report_lines.append(f"  Recommendation: {row.get('Recommendation', 'N/A')}")
                    if pd.notna(row.get('Similar_Inconsistent_Patients')) and row.get(
                            'Similar_Inconsistent_Patients') != '':
                        report_lines.append(f"  Inconsistent With: {row.get('Similar_Inconsistent_Patients')}")
                    if pd.notna(row.get('Flag_Reason')) and row.get('Flag_Reason') != '':
                        report_lines.append(f"  Training Analysis Flags: {row.get('Flag_Reason')}")
        else:
            report_lines.append("\n'Review_Tier' column not found. Cannot provide detailed tier breakdown.")

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write('\n'.join(report_lines))
        except Exception as e:
            logger.error(f"Could not write review report to {output_path}: {e}")

    def export_review_spreadsheet(self, review_df, output_path):
        if review_df is None or review_df.empty:
            prob_col_bases_empty = [name.replace(' ', '_') for name in self.item_class_map.values()]
            empty_prob_cols = [f'Prob_{b}' for b in prob_col_bases_empty]
            empty_cols_list = ['Patient ID', 'Side', 'Review_Status', 'Reviewer_Notes', 'Review_Tier',
                               'Review_Priority_Score',
                               'Expert_Label_Name', 'Predicted_Label_Name', 'Is_Correct'] + sorted(empty_prob_cols) + [
                                  'Model_Confidence', 'Error_Type', 'Entropy', 'Margin', 'Prob_True_Label',
                                  'Inconsistency_Score', 'Num_Inconsistent_Similar', 'Avg_Similarity',
                                  'Similar_Inconsistent_Patients',
                                  'Influence_Score (Delta_F1)', 'Flag_Reason', 'Recommendation', 'Review_Context']
            export_df = pd.DataFrame(columns=list(dict.fromkeys(empty_cols_list)))
        else:
            prob_col_bases_df = [name.replace(' ', '_') for name in self.item_class_map.values()]
            prob_cols_present = [f'Prob_{b}' for b in prob_col_bases_df if f'Prob_{b}' in review_df.columns]
            export_columns_ordered = ['Patient ID', 'Side', 'Review_Tier', 'Review_Priority_Score',
                                      'Expert_Label_Name', 'Predicted_Label_Name', 'Is_Correct'] + sorted(
                prob_cols_present) + [
                                         'Model_Confidence', 'Error_Type', 'Entropy', 'Margin', 'Prob_True_Label',
                                         'Inconsistency_Score', 'Num_Inconsistent_Similar', 'Avg_Similarity',
                                         'Similar_Inconsistent_Patients',
                                         'Influence_Score (Delta_F1)', 'Flag_Reason', 'Recommendation',
                                         'Review_Context']
            export_columns_final = [col for col in export_columns_ordered if col in review_df.columns]
            additional_cols = [col for col in review_df.columns if
                               col not in export_columns_final and col not in ['Review_Status', 'Reviewer_Notes']]
            export_columns_final.extend(additional_cols)
            export_columns_final = list(dict.fromkeys(export_columns_final))
            export_df = review_df[export_columns_final].copy()

            numeric_cols_to_round = ['Review_Priority_Score', 'Model_Confidence', 'Entropy', 'Margin',
                                     'Prob_True_Label',
                                     'Inconsistency_Score', 'Influence_Score (Delta_F1)',
                                     'Avg_Similarity'] + prob_cols_present
            for col in numeric_cols_to_round:
                if col in export_df.columns: export_df[col] = pd.to_numeric(export_df[col], errors='coerce').round(4)

            if 'Review_Status' not in export_df.columns: export_df.insert(min(2, len(export_df.columns)),
                                                                          'Review_Status', '')
            if 'Reviewer_Notes' not in export_df.columns: export_df.insert(min(3, len(export_df.columns)),
                                                                           'Reviewer_Notes', '')
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                export_df.to_excel(writer, sheet_name='Review_Recommendations', index=False)
                worksheet = writer.sheets['Review_Recommendations']
                for column_cells in worksheet.columns:
                    try:
                        length = max(len(str(cell.value if cell.value is not None else "")) for cell in column_cells)
                        adjusted_width = min(max(length + 2, 10), 50)
                        worksheet.column_dimensions[column_cells[0].column_letter].width = adjusted_width
                    except Exception as e_col_width:
                        logger.debug(f"Could not set column width for {column_cells[0].column_letter}: {e_col_width}")
        except Exception as e:
            logger.error(f"Could not write review spreadsheet to {output_path}: {e}", exc_info=True)