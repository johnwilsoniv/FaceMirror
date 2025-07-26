# consistency_checker.py (v2 - Dual Pipeline Support)
import pandas as pd
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
import os

logger = logging.getLogger(__name__)


class ConsistencyChecker:
    """Identifies label inconsistencies in the expert key"""

    def __init__(self, item_key, item_config, item_class_map,
                 similarity_threshold=0.95):  # Added item_class_map, similarity_threshold
        self.item_key = item_key
        self.item_config = item_config
        self.item_name_display = self.item_config.get('name', item_key.capitalize())
        self.item_class_map = item_class_map  # Store for potential use in report patterns
        self.similarity_threshold = similarity_threshold  # Set during init

    def find_label_inconsistencies(self, features_df, labels_df, metadata_df,
                                   similarity_threshold=None):
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold  # Use instance's threshold

        # logger.info(f"[{self.item_name_display}] Finding label inconsistencies (Thresh: {similarity_threshold:.2f})...") # Reduced logging

        if isinstance(labels_df, np.ndarray):  # Ensure labels_df is a Series with proper index
            labels_df = pd.Series(labels_df, name='Label', dtype=object)  # Ensure dtype object for mixed labels
            if features_df is not None and not features_df.empty and len(labels_df) == len(features_df):
                labels_df.index = features_df.index
            # else: logger.warning(f"[{self.item_name_display}] Could not align labels_df index with features_df.") # Reduced

        if features_df is None or labels_df is None or metadata_df is None:
            logger.error(
                "One or more input DataFrames (features, labels, metadata) is None for find_label_inconsistencies.")
            return pd.DataFrame()
        if not (len(features_df) == len(labels_df) == len(metadata_df)):
            logger.error(f"Mismatched lengths: Feat:{len(features_df)}, Lab:{len(labels_df)}, Meta:{len(metadata_df)}")
            return pd.DataFrame()
        if features_df.empty:
            # logger.warning(f"[{self.item_name_display}] Features DataFrame empty. Cannot compute similarity.") # Reduced
            return pd.DataFrame()

        similarity_matrix = cosine_similarity(features_df)
        inconsistencies = []
        n_samples = len(features_df)

        for i in range(n_samples):
            similar_mask = similarity_matrix[i] > similarity_threshold
            similar_indices = np.where(similar_mask)[0]
            similar_indices = similar_indices[similar_indices != i]  # Exclude self

            if len(similar_indices) > 0:
                current_label_val = labels_df.iloc[i]
                for j in similar_indices:
                    if j < len(labels_df):  # Boundary check
                        if labels_df.iloc[j] != current_label_val:
                            # Map numerical labels to names using self.item_class_map for reporting if needed later
                            current_label_name = self.item_class_map.get(current_label_val, str(current_label_val))
                            other_label_name = self.item_class_map.get(labels_df.iloc[j], str(labels_df.iloc[j]))

                            inconsistencies.append({
                                'Patient_ID_1': metadata_df.iloc[i]['Patient ID'],
                                'Side_1': metadata_df.iloc[i]['Side'],
                                'Label_1_Num': current_label_val, 'Label_1_Name': current_label_name,
                                # Store both num and name
                                'Patient_ID_2': metadata_df.iloc[j]['Patient ID'],
                                'Side_2': metadata_df.iloc[j]['Side'],
                                'Label_2_Num': labels_df.iloc[j], 'Label_2_Name': other_label_name,
                                # Store both num and name
                                'Similarity': similarity_matrix[i, j],
                                'Zone_Item': self.item_name_display  # Use generic term
                            })
                    # else: logger.warning(f"Index {j} out of bounds for labels_df for sample {i}.") # Reduced

        inconsistency_df = pd.DataFrame(inconsistencies)
        if not inconsistency_df.empty:
            inconsistency_df['pair_key'] = inconsistency_df.apply(
                lambda row: tuple(
                    sorted([f"{row['Patient_ID_1']}_{row['Side_1']}", f"{row['Patient_ID_2']}_{row['Side_2']}"])),
                axis=1
            )
            inconsistency_df = inconsistency_df.drop_duplicates(subset='pair_key').drop('pair_key', axis=1)
        # logger.info(f"[{self.item_name_display}] Found {len(inconsistency_df)} inconsistent pairs.") # Reduced
        return inconsistency_df

    def create_inconsistency_summary(self, features_df, labels_df, metadata_df,
                                     inconsistency_df=None):
        # logger.info(f"[{self.item_name_display}] Creating inconsistency summary...") # Reduced

        if isinstance(labels_df, np.ndarray):  # Ensure labels_df is a Series
            labels_df = pd.Series(labels_df, name='Label', dtype=object)
            if features_df is not None and not features_df.empty and len(labels_df) == len(features_df):
                labels_df.index = features_df.index

        if inconsistency_df is None:
            # logger.info(f"[{self.item_name_display}] Inconsistency DataFrame not provided to summary, calculating it now...") # Reduced
            inconsistency_df = self.find_label_inconsistencies(features_df, labels_df,
                                                               metadata_df)  # Uses instance's similarity_threshold

        summary_data = []
        if metadata_df is None or metadata_df.empty:
            logger.warning(f"[{self.item_name_display}] Metadata_df is empty. Cannot create inconsistency summary.")
            return pd.DataFrame(
                columns=['Patient ID', 'Side', 'Inconsistency_Score', 'Num_Inconsistent_Similar', 'Avg_Similarity',
                         'Similar_Inconsistent_Patients'])

        for idx, row in metadata_df.iterrows():
            patient_id = row['Patient ID'];
            side = row['Side']
            if inconsistency_df is not None and not inconsistency_df.empty:
                mask1 = (inconsistency_df['Patient_ID_1'] == patient_id) & (inconsistency_df['Side_1'] == side)
                mask2 = (inconsistency_df['Patient_ID_2'] == patient_id) & (inconsistency_df['Side_2'] == side)
                patient_inconsistencies = inconsistency_df[mask1 | mask2]
                if not patient_inconsistencies.empty:
                    similar_patients = []
                    for _, inc_row in patient_inconsistencies.iterrows():
                        if inc_row['Patient_ID_1'] == patient_id:
                            similar_patients.append(f"{inc_row['Patient_ID_2']}_{inc_row['Side_2']}")
                        else:
                            similar_patients.append(f"{inc_row['Patient_ID_1']}_{inc_row['Side_1']}")
                    avg_similarity = patient_inconsistencies['Similarity'].mean()
                    inconsistency_score = len(patient_inconsistencies) * avg_similarity
                    summary_data.append(
                        {'Patient ID': patient_id, 'Side': side, 'Inconsistency_Score': inconsistency_score,
                         'Num_Inconsistent_Similar': len(patient_inconsistencies), 'Avg_Similarity': avg_similarity,
                         'Similar_Inconsistent_Patients': ', '.join(similar_patients[:5])})
                else:
                    summary_data.append({'Patient ID': patient_id, 'Side': side, 'Inconsistency_Score': 0.0,
                                         'Num_Inconsistent_Similar': 0, 'Avg_Similarity': 0.0,
                                         'Similar_Inconsistent_Patients': ''})
            else:
                summary_data.append(
                    {'Patient ID': patient_id, 'Side': side, 'Inconsistency_Score': 0.0, 'Num_Inconsistent_Similar': 0,
                     'Avg_Similarity': 0.0, 'Similar_Inconsistent_Patients': ''})
        return pd.DataFrame(summary_data)

    def generate_consistency_report(self, inconsistency_df, cross_zone_df_placeholder,
                                    output_path):  # cross_zone_df not used for now
        # logger.info(f"Generating consistency report for {self.item_name_display}...") # Reduced
        report_lines = []
        report_lines.append(f"CONSISTENCY ANALYSIS REPORT - {self.item_name_display}")
        report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80);
        report_lines.append("\nFEATURE-BASED INCONSISTENCIES (within item)");
        report_lines.append("-" * 40)

        if inconsistency_df is None or inconsistency_df.empty:
            report_lines.append(f"No feature-based inconsistencies found for this item ({self.item_name_display}).")
        else:
            report_lines.append(f"Found {len(inconsistency_df)} inconsistent pairs")
            # Use Label_1_Name and Label_2_Name for reporting if they exist, otherwise fallback
            label1_col = 'Label_1_Name' if 'Label_1_Name' in inconsistency_df.columns else 'Label_1_Num'
            label2_col = 'Label_2_Name' if 'Label_2_Name' in inconsistency_df.columns else 'Label_2_Num'

            if label1_col in inconsistency_df.columns and label2_col in inconsistency_df.columns:
                label_combos = inconsistency_df.groupby([label1_col, label2_col]).size()
                report_lines.append("\nInconsistency patterns (Label Names):")
                for (label1, label2), count in label_combos.items(): report_lines.append(
                    f"  {label1} <-> {label2}: {count} pairs")

            if 'Similarity' in inconsistency_df.columns:
                report_lines.append("\nTop 10 most similar inconsistent pairs:")
                top_pairs = inconsistency_df.nlargest(10, 'Similarity')
                for _, row in top_pairs.iterrows():
                    report_lines.append(
                        f"\n  Patient {row.get('Patient_ID_1', 'N/A')} ({row.get('Side_1', 'N/A')}) - {row.get(label1_col, 'N/A')}")
                    report_lines.append(
                        f"  Patient {row.get('Patient_ID_2', 'N/A')} ({row.get('Side_2', 'N/A')}) - {row.get(label2_col, 'N/A')}")
                    report_lines.append(f"  Similarity: {row.get('Similarity', np.nan):.4f}")

        # Cross-zone consistency part would need more significant refactoring to be generic
        # For now, it's omitted or simplified.
        if cross_zone_df_placeholder is not None and not cross_zone_df_placeholder.empty:
            report_lines.append("\n\nCROSS-ITEM INCONSISTENCIES (Placeholder - requires specific logic)")

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write('\n'.join(report_lines))
            # logger.info(f"Consistency report saved to {output_path}") # Reduced
        except Exception as e:
            logger.error(f"Could not write consistency report to {output_path}: {e}")