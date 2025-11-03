# facial_au_visualizer.py

"""
Visualization module for facial AU analysis.
Creates visualizations of AU values and paralysis.
V1.50: Removed all synkinesis and hypertonicity detection code - paralysis detection only.
"""

import os
import cv2
import numpy as np
import logging
import pandas as pd

# Use non-interactive backend for thread-safe plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
import re # Keep re import as it is used in get_au_region
import json # For debug printing if needed

try:
    # Import constants for paralysis detection only
    from facial_au_constants import (
        ALL_AU_COLUMNS, AU_NAMES, FACIAL_ZONES,
        ASYMMETRY_THRESHOLDS, EXPERT_KEY_MAPPING,
        standardize_paralysis_label, standardize_binary_label,
        SEVERITY_ABBREVIATIONS,
        PARALYSIS_FINDINGS_KEYS, BOOL_FINDINGS_KEYS,
        ZONE_SPECIFIC_ACTIONS
    )
    logger = logging.getLogger(__name__)
except ImportError as e:
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.critical(f"CRITICAL ERROR: Failed to import required modules/functions: {e}.", exc_info=True)
    raise ImportError(f"Could not import necessary modules/functions: {e}") from e

class FacialAUVisualizer:

    def __init__(self):
        """Initialize the visualizer - paralysis detection only."""
        self.au_names = AU_NAMES
        self.facial_zones = FACIAL_ZONES
        # Synkinesis detection removed - paralysis detection only
        self.color_scheme = {
            'left_raw': '#4682B4', 'left_norm': '#B0C4DE',
            'right_raw': '#B22222', 'right_norm': '#FA8072',
            'key_au_highlight': '#F0F0F0', 'background': '#FFFFFF', 'grid': '#E5E5E5',
            'text': '#212121', 'threshold_line': '#757575', 'table_border': '#CCCCCC',
            'header_bg': '#E3F2FD', 'header_text': '#0D47A1',
            'complete_severity_text': '#D32F2F', 'partial_severity_text': '#FF7043',
            'none_severity_text': '#757575', 'error_severity_text': '#9E9E9E',
            'detected_yes_text': '#000000', 'detected_no_text': '#757575',
            'textbox_bg': '#F8F8F8', 'textbox_border': '#D0D0D0',
            'contradiction_marker': '#D32F2F',
            'complete_severity_rgba': to_rgba('#D32F2F', alpha=0.7),
            'partial_severity_rgba': to_rgba('#FF7043', alpha=0.7),
            'normal_severity_rgba': to_rgba('#F5F5F5', alpha=0.8),
            'min_value_threshold': 0.01
        }
        self.sev_abbr = SEVERITY_ABBREVIATIONS
        self.confidence_colormap_colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]

    def _plot_clinical_findings_panel(self, ax, results, patient_id, contradictions=None):
        if contradictions is None: contradictions = {}
        debug_contradictions_exist = bool(contradictions)

        ax.clear();
        ax.axis('off');
        severity_map = {'None': 0, 'Partial': 1, 'Complete': 2, 'Error': -1}

        paralysis_details = {'Left': {}, 'Right': {}};
        first_action_key = next((k for k, v in results.items() if k != 'patient_summary' and isinstance(v, dict)), None)
        if first_action_key and 'paralysis' in results[first_action_key]:
            zones_data = results[first_action_key]['paralysis'].get('zones', {});
            for side_key, side_zones in zones_data.items(): side_cap = side_key.capitalize(); paralysis_details[
                side_cap] = {}; [paralysis_details[side_cap].__setitem__(zone_key.capitalize(), severity) for
                                 zone_key, severity in side_zones.items()];
        else:
            logger.warning(
                f"VISUALIZER_CONSOLIDATION ({patient_id}): Could not find action with paralysis info."); paralysis_details = {
                'Left': {'Upper': 'Error', 'Mid': 'Error', 'Lower': 'Error'},
                'Right': {'Upper': 'Error', 'Mid': 'Error', 'Lower': 'Error'}}

        # Synkinesis detection removed - paralysis detection only

        paralysis_text_lines = [];
        p_finding_width = 7;
        p_side_width = 7;

        # Paralysis Table
        paralysis_text_lines.append(f"{'Zone':<{p_finding_width}} {'L':<{p_side_width}} {'R':<{p_side_width}}");
        paralysis_text_lines.append("-" * (p_finding_width + p_side_width * 2 + 1))
        for row_label, zone_cap_key in [('Upper', 'Upper'), ('Mid', 'Mid'), ('Lower', 'Lower')]:
            line_parts = [f"{row_label:<{p_finding_width}}"];
            for side in ['Left', 'Right']:
                algo_key = f"{side} {zone_cap_key} Face Paralysis";
                algo_severity = paralysis_details.get(side, {}).get(zone_cap_key, 'Error')
                if algo_severity not in self.sev_abbr: algo_severity = 'Error'
                has_contradiction = False;
                expert_severity = contradictions.get(algo_key);
                if expert_severity is not None and algo_severity != expert_severity: has_contradiction = True
                cell_text = self.sev_abbr.get(algo_severity, 'E');
                if has_contradiction:
                    expert_abbr = self.sev_abbr.get(expert_severity, '?'); cell_text += f" [{expert_abbr}!]"
                else:
                    cell_text += "     ";  # Padding for alignment
                line_parts.append(f"{cell_text:<{p_side_width}}")
            paralysis_text_lines.append(" ".join(line_parts));
        paralysis_summary_text = "\n".join(paralysis_text_lines)

        textbox_props = dict(boxstyle='round,pad=0.3', fc=self.color_scheme['textbox_bg'], alpha=0.95,
                             ec=self.color_scheme['textbox_border'])
        title_props = dict(fontsize=9, fontweight='bold', color=self.color_scheme['header_text'], ha='center', va='top')
        text_props = dict(fontsize=7.5, family='monospace', color=self.color_scheme['text'], ha='center', va='top',
                          bbox=textbox_props, linespacing=1.25)

        title_y = 0.98 # Y position for the title
        text_y_offset = 0.22
        text_start_y = title_y - text_y_offset # Calculated starting Y for the table

        ax.text(0.50, title_y, "Paralysis", transform=ax.transAxes, **title_props)
        ax.text(0.50, text_start_y, paralysis_summary_text, transform=ax.transAxes, **text_props)

    def create_baseline_visualization(self, analyzer, au_values_left, au_values_right, frame_num,
                                      patient_output_dir, frame_path, action_descriptions,
                                      results=None, contradictions=None):
        if results is None: results = {}
        action = 'BL'; patient_id = analyzer.patient_id
        logger.debug(f"Generating Baseline visualization for {patient_id}")
        fig = plt.figure(figsize=(18, 9)); gs = gridspec.GridSpec(2, 2, height_ratios=[6, 1], width_ratios=[1.2, 0.8], wspace=0.25, hspace=0.05)
        ax_raw = fig.add_subplot(gs[:, 0]); ax_img = fig.add_subplot(gs[0, 1]); ax_findings = fig.add_subplot(gs[1, 1])
        fig.patch.set_facecolor(self.color_scheme['background'])
        all_aus = sorted([au for au in ALL_AU_COLUMNS if au.endswith('_r')]); significant_aus_raw = [au for au in all_aus if (au_values_left.get(au, 0) > 0.05 or au_values_right.get(au, 0) > 0.05)]
        significant_aus = sorted(list(set(significant_aus_raw)), key=lambda au: int(re.search(r'\d+', au).group()) if re.search(r'\d+', au) else 99)
        if not significant_aus: logger.warning(f"({patient_id} - BL) No significant baseline AUs. Skipping BL plot."); plt.close(fig); return None
        x = np.arange(len(significant_aus)); width_baseline = 0.38
        left_raw = [float(au_values_left.get(au, 0.0) if pd.notna(au_values_left.get(au)) else 0.0) for au in significant_aus]
        right_raw = [float(au_values_right.get(au, 0.0) if pd.notna(au_values_right.get(au)) else 0.0) for au in significant_aus]
        plt.sca(ax_raw); ax_raw.set_facecolor(self.color_scheme['background'])
        bars1 = ax_raw.bar(x - width_baseline / 2, left_raw, width_baseline, label='Left Raw', color=self.color_scheme['left_raw'], edgecolor='black', linewidth=0.5)
        bars2 = ax_raw.bar(x + width_baseline / 2, right_raw, width_baseline, label='Right Raw', color=self.color_scheme['right_raw'], edgecolor='black', linewidth=0.5)
        self._add_bar_labels(ax_raw, bars1); self._add_bar_labels(ax_raw, bars2)
        ax_raw.set_title(f"Raw Baseline AU Values", fontsize=11, fontweight='bold', color=self.color_scheme['text'])
        ax_raw.set_ylabel("Raw Intensity", fontsize=9, color=self.color_scheme['text']); ax_raw.set_xlabel("Action Units", fontsize=9, color=self.color_scheme['text'])
        tick_labels = [f"{au}\n{self.au_names.get(au, '').split('(')[0].strip()}" for au in significant_aus]; ax_raw.set_xticks(x); ax_raw.set_xticklabels(tick_labels, rotation=45, fontsize=8, ha='right', color=self.color_scheme['text'])
        ax_raw.legend(fontsize=8, loc='upper right'); ax_raw.grid(axis='y', linestyle='--', alpha=0.3, color=self.color_scheme['grid']); ax_raw.set_ylim(bottom=0)
        ax_raw.spines['bottom'].set_color(self.color_scheme['text']); ax_raw.spines['left'].set_color(self.color_scheme['text']); ax_raw.tick_params(axis='x', colors=self.color_scheme['text'], labelsize=8); ax_raw.tick_params(axis='y', colors=self.color_scheme['text'], labelsize=8)
        plt.sca(ax_img)
        if frame_path and os.path.exists(frame_path): frame_img = cv2.imread(frame_path);
        if frame_path and os.path.exists(frame_path) and frame_img is not None: img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB); ax_img.imshow(img_rgb); ax_img.set_title(f"Frame {frame_num}", fontsize=10, color=self.color_scheme['text'], pad=2)
        else: ax_img.text(0.5, 0.5, "BL Frame Unavailable", ha='center', va='center', fontsize=10, color=self.color_scheme['text']); logger.warning(f"BL frame path not found or failed load: {frame_path}")
        ax_img.axis('off')
        self._plot_clinical_findings_panel(ax_findings, results, patient_id, contradictions)
        fig.suptitle(f"{patient_id} - Baseline Analysis", fontsize=14, fontweight='bold', color=self.color_scheme['text'])
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        output_filename = f"BL_{patient_id}_AUs.png"; output_path = os.path.join(patient_output_dir, output_filename)
        try: plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor()); logger.info(f"Saved BL viz: {output_path}")
        except Exception as e: logger.error(f"Failed save BL viz {output_path}: {e}", exc_info=True); output_path = None
        finally: plt.close(fig)
        return output_path

    def create_au_visualization(self, analyzer, au_values_left, au_values_right, norm_au_values_left,
                                norm_au_values_right, action, frame_num, patient_output_dir,
                                frame_path, action_descriptions, action_to_aus, results, contradictions=None):
        if results is None: results = {}
        patient_id = analyzer.patient_id; logger.debug(f"Generating AU viz for {patient_id} - {action}")
        fig = plt.figure(figsize=(18, 9)); gs = gridspec.GridSpec(2, 2, height_ratios=[6, 1], width_ratios=[1.2, 0.8], wspace=0.25, hspace=0.05)
        ax_main = fig.add_subplot(gs[:, 0]); ax_img = fig.add_subplot(gs[0, 1]); ax_findings = fig.add_subplot(gs[1, 1])
        fig.patch.set_facecolor(self.color_scheme['background'])
        all_aus = sorted([au for au in ALL_AU_COLUMNS if au.endswith('_r')]); norm_left_dict = norm_au_values_left if isinstance(norm_au_values_left, dict) else {}; norm_right_dict = norm_au_values_right if isinstance(norm_au_values_right, dict) else {}; raw_left_dict = au_values_left if isinstance(au_values_left, dict) else {}; raw_right_dict = au_values_right if isinstance(au_values_right, dict) else {}
        significant_aus_candidates = set([au for au, val in norm_left_dict.items() if pd.notna(val) and val > 0.01] + [au for au, val in norm_right_dict.items() if pd.notna(val) and val > 0.01] + [au for au, val in raw_left_dict.items() if pd.notna(val) and val > 0.05] + [au for au, val in raw_right_dict.items() if pd.notna(val) and val > 0.05])
        significant_aus = [au for au in significant_aus_candidates if au in ALL_AU_COLUMNS]
        if not significant_aus: logger.warning(f"({patient_id} - {action}) No significant AUs. Skipping plot."); plt.close(fig); return None
        def get_au_region(au_name): match = re.search(r'\d+', au_name); num = int(match.group()) if match else 999; return 'Upper' if num <= 5 else ('Mid' if num <= 10 else ('Lower' if num <= 20 else 'Mouth/Other'))
        significant_aus = sorted(significant_aus, key=lambda au: (get_au_region(au), int(re.search(r'\d+', au).group()) if re.search(r'\d+', au) else 99))
        x = np.arange(len(significant_aus)); width_action = 0.2
        left_raw = [float(raw_left_dict.get(au, 0.0) if pd.notna(raw_left_dict.get(au)) else 0.0) for au in significant_aus]; right_raw = [float(raw_right_dict.get(au, 0.0) if pd.notna(raw_right_dict.get(au)) else 0.0) for au in significant_aus]
        left_norm = [float(norm_left_dict.get(au, 0.0) if pd.notna(norm_left_dict.get(au)) else 0.0) for au in significant_aus]; right_norm = [float(norm_right_dict.get(au, 0.0) if pd.notna(norm_right_dict.get(au)) else 0.0) for au in significant_aus]
        key_aus = action_to_aus.get(action, []); action_info = results.get(action, {}); paralysis_info = action_info.get('paralysis', {})
        plt.sca(ax_main); ax_main.set_facecolor(self.color_scheme['background'])
        bars_lr = ax_main.bar(x - 1.5*width_action, left_raw, width_action, label='Left Raw', color=self.color_scheme['left_raw'], edgecolor='black', linewidth=0.5)
        bars_ln = ax_main.bar(x - 0.5*width_action, left_norm, width_action, label='Left Norm', color=self.color_scheme['left_norm'], edgecolor='black', linewidth=0.5)
        bars_rr = ax_main.bar(x + 0.5*width_action, right_raw, width_action, label='Right Raw', color=self.color_scheme['right_raw'], edgecolor='black', linewidth=0.5)
        bars_rn = ax_main.bar(x + 1.5*width_action, right_norm, width_action, label='Right Norm', color=self.color_scheme['right_norm'], edgecolor='black', linewidth=0.5)
        self._add_bar_labels(ax_main, bars_lr); self._add_bar_labels(ax_main, bars_ln); self._add_bar_labels(ax_main, bars_rr); self._add_bar_labels(ax_main, bars_rn)
        ax_main.set_title(f"AU Values", fontsize=11, fontweight='bold', color=self.color_scheme['text'])
        ax_main.set_ylabel("Intensity", fontsize=9, color=self.color_scheme['text']); ax_main.set_xlabel("Action Units", fontsize=9, color=self.color_scheme['text'])
        tick_labels = [f"{au}\n{self.au_names.get(au, '').split('(')[0].strip()}" for au in significant_aus]; ax_main.set_xticks(x); ax_main.set_xticklabels(tick_labels, rotation=45, fontsize=8, ha='right', color=self.color_scheme['text'])
        key_au_indices = [i for i, au in enumerate(significant_aus) if au in key_aus]
        for idx in key_au_indices:
             if 0 <= idx < len(ax_main.get_xticklabels()): ax_main.get_xticklabels()[idx].set_bbox(dict(facecolor=self.color_scheme['key_au_highlight'], alpha=0.8, boxstyle='round,pad=0.2'))
             else: logger.warning(f"Index {idx} out of bounds for xticklabels in action plot {action}.")
        ax_main.legend(fontsize=8, loc='upper right'); ax_main.grid(axis='y', linestyle='--', alpha=0.3, color=self.color_scheme['grid'])
        ax_main.spines['bottom'].set_color(self.color_scheme['text']); ax_main.spines['left'].set_color(self.color_scheme['text']); ax_main.tick_params(axis='x', colors=self.color_scheme['text'], labelsize=8); ax_main.tick_params(axis='y', colors=self.color_scheme['text'], labelsize=8)
        max_y_val = max(left_raw + right_raw + left_norm + right_norm + [0.1]) if (left_raw or right_raw or left_norm or right_norm) else 0.1; ax_main.set_ylim(bottom=0, top=max_y_val * 1.1)
        # self._add_detection_markers(ax_main, 'paralysis', paralysis_info, significant_aus, bars_lr, bars_ln, bars_rr, bars_rn, x, width_action) # <--- DISABLED BAR COLOR CHANGE

        plt.sca(ax_img)
        if frame_path and os.path.exists(frame_path): frame_img = cv2.imread(frame_path);
        if frame_path and os.path.exists(frame_path) and frame_img is not None: img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB); ax_img.imshow(img_rgb); ax_img.set_title(f"Frame {frame_num}", fontsize=10, color=self.color_scheme['text'], pad=2)
        else: ax_img.text(0.5, 0.5, "Frame Unavailable", ha='center', va='center', fontsize=10, color=self.color_scheme['text']); logger.warning(f"Frame path not found/loaded for {action}: {frame_path}")
        ax_img.axis('off')
        self._plot_clinical_findings_panel(ax_findings, results, patient_id, contradictions)
        fig.suptitle(f"{patient_id} - {action_descriptions.get(action, action)}", fontsize=14, fontweight='bold', color=self.color_scheme['text'])
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        output_filename = f"{action}_{patient_id}_AUs.png"
        output_path = os.path.join(patient_output_dir, output_filename)
        try: plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor()); logger.info(f"Saved AU viz: {output_path}")
        except Exception as e: logger.error(f"Failed save AU viz {output_path}: {e}", exc_info=True); output_path = None
        finally: plt.close(fig)
        return output_path

    def _add_detection_markers(self, ax, detection_type, info_dict, aus_plotted, bars_left_raw, bars_left_norm, bars_right_raw, bars_right_norm, x_coords, bar_width):
        # ... (no changes in this method body) ...
        if not info_dict or not aus_plotted: return
        if detection_type == 'paralysis':
            paralysis_zones = info_dict.get('zones', {})
            detected_in_action = any(sev != 'None' for side_zones in paralysis_zones.values() for sev in side_zones.values())
            if not detected_in_action: return
            for i, au in enumerate(aus_plotted):
                current_zone = None;
                for zone_name, zone_aus in self.facial_zones.items():
                    if au in zone_aus: current_zone = zone_name; break
                if not current_zone: continue
                left_severity = paralysis_zones.get('left', {}).get(current_zone, 'None'); fill_color_left = None
                if left_severity == 'Complete': fill_color_left = self.color_scheme['complete_severity_rgba']
                elif left_severity == 'Partial': fill_color_left = self.color_scheme['partial_severity_rgba']
                if fill_color_left:
                    if i < len(bars_left_raw): bars_left_raw[i].set_facecolor(fill_color_left)
                    if i < len(bars_left_norm): bars_left_norm[i].set_facecolor(fill_color_left)
                right_severity = paralysis_zones.get('right', {}).get(current_zone, 'None'); fill_color_right = None
                if right_severity == 'Complete': fill_color_right = self.color_scheme['complete_severity_rgba']
                elif right_severity == 'Partial': fill_color_right = self.color_scheme['partial_severity_rgba']
                if fill_color_right:
                     if i < len(bars_right_raw): bars_right_raw[i].set_facecolor(fill_color_right)
                     if i < len(bars_right_norm): bars_right_norm[i].set_facecolor(fill_color_right)

    def create_symmetry_visualization(self, analyzer, patient_output_dir, patient_id, results, action_descriptions):
        # ... (no changes in this method body) ...
        if not results: logger.error(f"({patient_id}) No results for symmetry plot."); return None
        os.makedirs(patient_output_dir, exist_ok=True); fig = plt.figure(figsize=(18, 15)); gs = gridspec.GridSpec(4, 1, height_ratios=[3, 3, 3, 2]); fig.patch.set_facecolor(self.color_scheme['background'])
        zone_titles = {'upper': 'Upper Face Asymmetry', 'mid': 'Mid Face Asymmetry', 'lower': 'Lower Face Asymmetry'}; zone_plot_config = { 'upper': {'metric': 'Asym_Ratio', 'lower_is_worse': True}, 'mid': {'metric': 'Asym_Ratio', 'lower_is_worse': True}, 'lower': {'metric': 'Asym_PercDiff', 'lower_is_worse': False} }; zone_data = {zone: {'actions': [], 'metric_values': []} for zone in zone_plot_config}; all_action_metrics = {}
        for action, info in results.items():
            if action == 'patient_summary' or not isinstance(info, dict): continue
            action_metrics = {}
            for zone in zone_plot_config.keys():
                if zone not in zone_plot_config: logger.warning(f"Zone '{zone}' not found in zone_plot_config. Skipping."); continue
                metric_config = zone_plot_config[zone]; metric_base = metric_config['metric']
                au_for_metric = None
                if zone == 'upper': au_for_metric = 'AU01_r'
                elif zone == 'mid': au_for_metric = 'AU45_r'
                elif zone == 'lower': au_for_metric = 'AU12_r'
                else: logger.warning(f"Unsupported zone '{zone}' for symmetry metric AU selection."); continue
                if not au_for_metric: continue;
                metric_val = None;
                left_norm_dict = info.get('left', {}).get('normalized_au_values', {})
                right_norm_dict = info.get('right', {}).get('normalized_au_values', {})
                left_norm_val = None; right_norm_val = None
                if left_norm_dict and right_norm_dict and au_for_metric in left_norm_dict and au_for_metric in right_norm_dict:
                    left_norm_val = left_norm_dict.get(au_for_metric); right_norm_val = right_norm_dict.get(au_for_metric)
                if left_norm_val is not None and right_norm_val is not None and pd.notna(left_norm_val) and pd.notna(right_norm_val):
                    if metric_base == 'Asym_Ratio': metric_val = self._calculate_ratio_scalar(pd.Series([left_norm_val]), pd.Series([right_norm_val]))
                    elif metric_base == 'Asym_PercDiff': metric_val = self._calculate_percent_diff_scalar(pd.Series([left_norm_val]), pd.Series([right_norm_val]))
                action_metrics[zone] = metric_val if metric_val is not None else np.nan
            all_action_metrics[action] = action_metrics
        local_zone_actions = ZONE_SPECIFIC_ACTIONS if 'ZONE_SPECIFIC_ACTIONS' in globals() else {}
        if local_zone_actions:
            for zone, specific_actions in local_zone_actions.items():
                if zone not in zone_plot_config: continue
                for action in specific_actions:
                    if action in all_action_metrics and zone in all_action_metrics[action]: zone_data[zone]['actions'].append(action_descriptions.get(action, action)); zone_data[zone]['metric_values'].append(all_action_metrics[action].get(zone, np.nan))
        else: logger.warning("ZONE_SPECIFIC_ACTIONS missing/empty. Symmetry plot might be empty.")
        paralyzed_zones = {'left': {}, 'right': {}}; severity_map = {'None': 0, 'Partial': 1, 'Complete': 2, 'Error': -1}
        for action, info in results.items():
             if action == 'patient_summary' or not isinstance(info, dict): continue
             if 'paralysis' in info:
                 zones_dict = info.get('paralysis', {}).get('zones', {});
                 for side in ['left', 'right']:
                     if side not in paralyzed_zones: paralyzed_zones[side] = {}
                     side_zones = zones_dict.get(side, {});
                     for zone, severity in side_zones.items(): current_level = severity_map.get(paralyzed_zones[side].get(zone, 'None'), 0); new_level = severity_map.get(severity, 0);
                     if new_level > current_level: paralyzed_zones[side][zone] = severity
        for i, zone in enumerate(zone_plot_config.keys()):
            ax = fig.add_subplot(gs[i]); ax.set_facecolor(self.color_scheme['background']); config = zone_plot_config[zone]; metric_base = config['metric']
            if not zone_data[zone]['actions']: ax.text(0.5, 0.5, f"No primary actions for {zone} zone", ha='center', va='center', color=self.color_scheme['text']); ax.set_title(zone_titles[zone], color=self.color_scheme['text']); ax.axis('off'); continue
            actions = zone_data[zone]['actions']; metric_values = zone_data[zone]['metric_values']
            if not actions: ax.text(0.5, 0.5, f"Metric data missing for {zone}", ha='center', va='center', color=self.color_scheme['text']); ax.set_title(zone_titles[zone], color=self.color_scheme['text']); ax.axis('off'); continue
            x_coords = np.arange(len(actions)); width_sym = 0.6; colors = []; local_asym_thresh = ASYMMETRY_THRESHOLDS if 'ASYMMETRY_THRESHOLDS' in globals() else {}; partial_thresh_pd = local_asym_thresh.get(zone, {}).get('partial', {}).get('percent_diff', 60); complete_thresh_pd = local_asym_thresh.get(zone, {}).get('complete', {}).get('percent_diff', partial_thresh_pd * 1.5); partial_thresh_ratio = local_asym_thresh.get(zone, {}).get('partial', {}).get('ratio', 0.6); complete_thresh_ratio = local_asym_thresh.get(zone, {}).get('complete', {}).get('ratio', 0.4)
            for val in metric_values:
                color_rgba = self.color_scheme['normal_severity_rgba'];
                if pd.notna(val):
                    if metric_base == 'Asym_Ratio':
                        if val < complete_thresh_ratio: color_rgba = self.color_scheme['complete_severity_rgba']
                        elif val < partial_thresh_ratio: color_rgba = self.color_scheme['partial_severity_rgba']
                    elif metric_base == 'Asym_PercDiff':
                        if val > complete_thresh_pd: color_rgba = self.color_scheme['complete_severity_rgba']
                        elif val > partial_thresh_pd: color_rgba = self.color_scheme['partial_severity_rgba']
                colors.append(color_rgba)
            metric_values_plot = [v if pd.notna(v) else 0 for v in metric_values] # Replace NaN with 0 for plotting
            bars = ax.bar(x_coords, metric_values_plot, width_sym, color=colors, edgecolor=self.color_scheme['text'], linewidth=0.5);
            self._add_bar_labels(ax, bars)
            if metric_base == 'Asym_Ratio': ax.axhline(partial_thresh_ratio, color=self.color_scheme['partial_severity_rgba'][0:3], linestyle='--', linewidth=1, label=f'Partial Thr ({partial_thresh_ratio:.2f})'); ax.axhline(complete_thresh_ratio, color=self.color_scheme['complete_severity_rgba'][0:3], linestyle=':', linewidth=1, label=f'Complete Thr ({complete_thresh_ratio:.2f})'); ax.set_ylim(0, 1.1)
            elif metric_base == 'Asym_PercDiff': ax.axhline(partial_thresh_pd, color=self.color_scheme['partial_severity_rgba'][0:3], linestyle='--', linewidth=1, label=f'Partial Thr ({partial_thresh_pd:.0f}%)'); ax.axhline(complete_thresh_pd, color=self.color_scheme['complete_severity_rgba'][0:3], linestyle=':', linewidth=1, label=f'Complete Thr ({complete_thresh_pd:.0f}%)'); numeric_vals = [m for m in metric_values_plot if pd.notna(m)] or [0]; ax.set_ylim(bottom=0, top=max(numeric_vals + [complete_thresh_pd * 1.1, 50]))
            left_sev = paralyzed_zones.get('left', {}).get(zone, 'None'); right_sev = paralyzed_zones.get('right', {}).get(zone, 'None'); title = f"{zone_titles[zone]} ({metric_base})"; paralysis_title_info = [f"{s.upper()[0]}:{sev}" for s, sev in [('L', left_sev), ('R', right_sev)] if sev not in ['None', 'Error', 'NA']];
            if paralysis_title_info: title += f" - [Overall: {'; '.join(paralysis_title_info)}]";
            ax.set_title(title, fontsize=14, color=self.color_scheme['text']); ax.set_ylabel(metric_base, color=self.color_scheme['text']); ax.set_xticks(x_coords); ax.set_xticklabels(actions, rotation=45, ha='right', color=self.color_scheme['text']); ax.legend(fontsize=9); ax.grid(axis='y', linestyle='--', alpha=0.7, color=self.color_scheme['grid']); ax.spines['bottom'].set_color(self.color_scheme['text']); ax.spines['left'].set_color(self.color_scheme['text']); ax.tick_params(axis='x', colors=self.color_scheme['text']); ax.tick_params(axis='y', colors=self.color_scheme['text'])

        # Synkinesis detection removed - paralysis detection only
        ax_synk = fig.add_subplot(gs[3]); ax_synk.set_facecolor(self.color_scheme['background'])
        ax_synk.text(0.5, 0.5, "Paralysis Detection Only", ha='center', va='center', fontsize=14, color=self.color_scheme['text']); ax_synk.axis('off')
        summary_text = f"SUMMARY - {patient_id}\n" + "="*(12 + len(patient_id)) + "\nParalysis:\n"
        paralysis_found = any(p not in ['None', 'Error', 'NA'] for side_dict in paralyzed_zones.values() for p in side_dict.values())
        summary_text += "  None Detected\n" if not paralysis_found else ""
        for side in ['left', 'right']: findings = [f"{z.capitalize()}:{s}" for z,s in paralyzed_zones.get(side, {}).items() if s not in ['None', 'Error', 'NA']]; summary_text += f"  {side.capitalize()}: {', '.join(findings) if findings else 'None'}\n"
        fig.text(0.5, 0.02, summary_text, ha='center', va='bottom', fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc=self.color_scheme['background'], alpha=0.9, ec=self.color_scheme['grid']), family='monospace', color=self.color_scheme['text'])
        fig.suptitle(f'Facial Symmetry Analysis - {patient_id}', fontsize=16, fontweight='bold', color=self.color_scheme['text'])
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        output_path = os.path.join(patient_output_dir, f"symmetry_{patient_id}.png")
        try: plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor()); logger.info(f"({patient_id}) Saved symmetry viz: {output_path}")
        except Exception as e: logger.error(f"({patient_id}) Failed save symmetry plot: {e}"); output_path = None
        finally: plt.close(fig)
        return output_path

    def create_patient_dashboard(self, analyzer, patient_output_dir, patient_id, results, action_descriptions,
                                 frame_paths, contradictions=None):
        if contradictions is None: contradictions = {}

        logger.info(
            f"Generating dashboard for {patient_id} (Contradictions found: {sum(1 for v in contradictions.values() if v is not None)})")
        dashboard_filename = f"dashboard_{patient_id}.html";
        dashboard_path = os.path.join(patient_output_dir, dashboard_filename)

        symmetry_plot_filename = f"symmetry_{patient_id}.png";
        symmetry_plot_path_rel = symmetry_plot_filename if os.path.exists(
            os.path.join(patient_output_dir, symmetry_plot_filename)) else None

        action_plots_rel = {}
        action_order = ['BL', 'RE', 'FR', 'ES', 'BK', 'ET', 'WN', 'SS', 'BS', 'SO', 'SE', 'PL', 'LT', 'BC']
        analyzed_actions = sorted(
            [action for action in results if action != 'patient_summary' and isinstance(results.get(action), dict)])
        ordered_plots_found = {}
        for action in action_order:
            if action in analyzed_actions:
                plot_filename = f"{action}_{patient_id}_AUs.png"
                if os.path.exists(os.path.join(patient_output_dir, plot_filename)):
                    ordered_plots_found[action] = plot_filename
                    try:
                        analyzed_actions.remove(action)
                    except ValueError:
                        pass
        for action in analyzed_actions:
            plot_filename = f"{action}_{patient_id}_AUs.png"
            if os.path.exists(os.path.join(patient_output_dir, plot_filename)):
                ordered_plots_found[action] = plot_filename

        severity_map = {'None': 0, 'Partial': 1, 'Complete': 2, 'Error': -1};
        paralysis_details = {'Left': {}, 'Right': {}};
        paralysis_found_overall = False
        first_action_key = next(
            (k for k, v in results.items() if k != 'patient_summary' and isinstance(v, dict) and 'paralysis' in v),
            None)
        if first_action_key:
            zones_data = results[first_action_key].get('paralysis', {}).get('zones', {})
            for side_key, side_zones in zones_data.items():
                side_cap = side_key.capitalize(); paralysis_details[side_cap] = {};
                for zone_key, severity in side_zones.items():
                    paralysis_details[side_cap][zone_key.capitalize()] = severity;
                    if severity not in ['None', 'Error', 'NA']: paralysis_found_overall = True
        else:
            logger.warning(
                f"({patient_id}) Dashboard: Could not find action with paralysis info."); paralysis_details = {
                'Left': {'Upper': 'Error', 'Mid': 'Error', 'Lower': 'Error'},
                'Right': {'Upper': 'Error', 'Mid': 'Error', 'Lower': 'Error'}}

        # Synkinesis detection removed - paralysis detection only

        paralysis_text_lines = [];
        p_finding_width = 7;
        p_side_width = 7;

        # Paralysis Text
        paralysis_text_lines.append(f"{'Zone':<{p_finding_width}} {'L':<{p_side_width}} {'R':<{p_side_width}}")
        paralysis_text_lines.append("-" * (p_finding_width + p_side_width * 2 + 1))
        for row_label, zone_key in [('Upper', 'Upper'), ('Mid', 'Mid'), ('Lower', 'Lower')]:
            line_parts = [f"{row_label:<{p_finding_width}}"];
            for side in ['Left', 'Right']:
                algo_key = f"{side} {zone_key} Face Paralysis";
                algo_severity = paralysis_details.get(side, {}).get(zone_key, 'Error')
                if algo_severity not in self.sev_abbr: algo_severity = 'Error';
                has_contradiction = False;
                expert_severity = contradictions.get(algo_key);
                if expert_severity is not None and algo_severity != expert_severity: has_contradiction = True
                cell_text = self.sev_abbr.get(algo_severity, 'E');
                if has_contradiction:
                    expert_abbr = self.sev_abbr.get(expert_severity, '?'); cell_text += f" [{expert_abbr}!]"
                else:
                    cell_text += "     ";
                line_parts.append(f"{cell_text:<{p_side_width}}")
            paralysis_text_lines.append(" ".join(line_parts));
        paralysis_summary_text = "\n".join(paralysis_text_lines)

        color_primary = self.color_scheme['right_raw'];
        color_secondary = self.color_scheme['left_raw'];
        color_text = self.color_scheme['text'];
        color_grey = self.color_scheme['threshold_line'];
        color_background = self.color_scheme['background'];
        color_textbox_bg = self.color_scheme['textbox_bg'];
        color_textbox_border = self.color_scheme['textbox_border'];
        color_header_text = self.color_scheme['header_text']
        # Dashboard for paralysis detection only
        html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Facial Analysis Dashboard - {patient_id}</title><style>body{{font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;margin:0;padding:0;background-color:#f8f9fa;color:{color_text}}} .container{{max-width:1400px;margin:20px auto;background-color:{color_background};padding:25px;border-radius:8px;box-shadow:0 4px 10px rgba(0,0,0,0.08)}} h1,h2{{color:{color_secondary};border-bottom:2px solid {color_primary};padding-bottom:10px;margin-top:30px;margin-bottom:15px}} h1{{text-align:center;font-size:2em;color:{color_primary}}} .findings-container {{ display: flex; justify-content: space-around; margin-bottom: 20px; gap: 15px; flex-wrap: wrap; }} .findings-box {{ background-color: {color_textbox_bg}; border: 1px solid {color_textbox_border}; border-radius: 4px; padding: 10px; flex: 1; min-width: 250px; }} .findings-box h3 {{ color:{color_header_text}; margin-top:0; margin-bottom:8px; font-size:1.1em; border-bottom: 1px solid #ddd; padding-bottom: 4px; text-align: center;}} pre {{ font-family: monospace; font-size: 0.85em; white-space: pre; margin: 0; line-height: 1.3; }} .plot-container{{text-align:center;margin-bottom:30px;padding:15px;border:1px solid #eee;border-radius:5px;background:#fdfdfd}} .plot-container img{{max-width:98%;height:auto;border:1px solid #ccc;border-radius:4px}} .grid-container{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:20px;margin-top:20px}} .grid-item{{background-color:{color_background};padding:15px;border-radius:5px;border:1px solid #eee;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,0.05)}} .grid-item img{{max-width:100%;height:auto;border:1px solid #ddd;border-radius:4px}} .grid-item h4{{color:{color_text};margin-top:15px;margin-bottom:10px;text-align:center;font-size:1.1em}}</style></head><body><div class="container"><h1>Facial Analysis Dashboard - {patient_id}</h1>
        <h2>Clinical Findings</h2><div class="findings-container"><div class="findings-box"><h3>Paralysis</h3><pre>{paralysis_summary_text}</pre></div></div>"""
        if symmetry_plot_path_rel: html += f'<h2>Symmetry Analysis</h2><div class="plot-container"><img src="{symmetry_plot_filename}" alt="Symmetry Plot"></div>'

        html += '<h2>Action Visualizations</h2><div class="grid-container">'
        if ordered_plots_found:
            for action, plot_filename in ordered_plots_found.items():
                action_desc = action_descriptions.get(action, action)
                html += f'<div class="grid-item"><h4>{action_desc}</h4><img src="{plot_filename}" alt="{action_desc} Plot"></div>'
        else:
            html += f"<div class=\"grid-item\" style='grid-column: 1 / -1;'><p style='color:{color_grey};margin-top:50px;'>No individual action plots generated or found.</p></div>"

        html += '</div></div></body></html>'
        try:
            with open(dashboard_path, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"Saved dashboard: {dashboard_path}");
            return dashboard_path
        except Exception as e:
            logger.error(f"Failed to save dashboard {dashboard_path}: {e}", exc_info=True); return None

    def _add_bar_labels(self, ax, bars):
        min_label_height = 0.01
        for bar in bars:
            height = bar.get_height()
            if pd.notna(height) and height > min_label_height:
                label_text = f'{height:.2f}'
                ax.annotate( label_text, xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=6, color=self.color_scheme.get('text', '#212121') )

    def _create_confidence_colormap(self):
        return LinearSegmentedColormap.from_list("conf_cmap", self.confidence_colormap_colors)

    def _add_confidence_scale(self, ax):
        cmap = self._create_confidence_colormap();
        norm = plt.Normalize(vmin=0, vmax=1);
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm);
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.02);
        cbar.set_label('Confidence Level', size=9, color=self.color_scheme['text']);
        cbar.ax.tick_params(labelsize=8, colors=self.color_scheme['text']);
        cbar.outline.set_edgecolor(self.color_scheme['text'])

    def _calculate_ratio_scalar(self, series1, series2):
        val1 = series1.iloc[0] if not series1.empty else np.nan
        val2 = series2.iloc[0] if not series2.empty else np.nan
        if pd.isna(val1) or pd.isna(val2): return np.nan
        min_val_config = self.color_scheme.get('min_value_threshold', 0.01)
        max_v = max(val1, val2)
        if max_v < min_val_config: return 1.0
        return min(val1, val2) / max_v if max_v > 0 else 1.0

    def _calculate_percent_diff_scalar(self, series1, series2):
        val1 = series1.iloc[0] if not series1.empty else np.nan
        val2 = series2.iloc[0] if not series2.empty else np.nan
        if pd.isna(val1) or pd.isna(val2): return np.nan
        min_val_config = self.color_scheme.get('min_value_threshold', 0.01)
        avg = (val1 + val2) / 2.0
        if avg < min_val_config: return 0.0
        return (abs(val1 - val2) / avg) * 100 if avg > 0 else 0.0
