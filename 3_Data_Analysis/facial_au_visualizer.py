"""
Visualization module for facial AU analysis.
Creates visualizations of AU values, paralysis, and synkinesis.
"""

import os
import cv2
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
import re
from facial_au_constants import (
    ALL_AU_COLUMNS,
    AU_NAMES, FACIAL_ZONES, ZONE_SPECIFIC_ACTIONS,
    SYNKINESIS_PATTERNS, SYNKINESIS_THRESHOLDS, SYNKINESIS_TYPES,
)

# Configure logging
logger = logging.getLogger(__name__) # Assuming configured by main

class FacialAUVisualizer:
    """
    Creates visualizations of facial AU analysis results.
    """

    def __init__(self):
        """Initialize the visualizer."""
        # ... (init remains the same) ...
        self.au_names = AU_NAMES
        self.facial_zones = FACIAL_ZONES
        self.zone_specific_actions = ZONE_SPECIFIC_ACTIONS
        self.synkinesis_patterns = SYNKINESIS_PATTERNS
        self.synkinesis_types = SYNKINESIS_TYPES
        self.color_scheme = {
            'left_raw': '#4286f4', 'right_raw': '#f44242', 'left_norm': '#1a53b3',
            'right_norm': '#b31a1a', 'key_au_highlight': '#ffd700', 'key_au_text': '#000000',
            'complete_threshold': '#d9534f', 'partial_threshold': '#f0ad4e',
            'background': '#f8f9fa', 'grid': '#dee2e6', 'text': '#212529',
            'detection_marker': '#ff5722', 'synkinesis_trigger': '#9c27b0',
            'synkinesis_response': '#009688', 'normal': '#81c784'
        }

    # --- create_au_visualization, _add_bar_labels, _add_detection_markers ---
    # --- _format_clinical_findings, create_symmetry_visualization ---
    # --- _get_confidence_color, _add_confidence_scale, _create_confidence_colormap ---
    # ... (remain the same as previous version) ...
    def create_au_visualization(self, analyzer, au_values_left, au_values_right, norm_au_values_left,
                                norm_au_values_right, action, frame_num, patient_output_dir,
                                frame_path, # Expect single path string now
                                action_descriptions, action_to_aus, results):
        """ Creates the per-action AU visualization plot. """
        logger.debug(f"Generating AU visualization for {analyzer.patient_id} - {action}")
        # Create figure
        fig = plt.figure(figsize=(22, 12))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 1])
        ax_raw = fig.add_subplot(gs[0, 0]); ax_norm = fig.add_subplot(gs[1, 0]); ax_img = fig.add_subplot(gs[:, 1])

        # --- Prepare data ---
        all_aus = sorted([au for au in ALL_AU_COLUMNS if au.endswith('_r')]) # Use constant
        # Filter AUs: Keep if *either* normalized value is > 0.01 (focus on activation)
        significant_aus = [
            au for au in all_aus
            if (norm_au_values_left.get(au, 0) > 0.01 or norm_au_values_right.get(au, 0) > 0.01)
            ]
        if not significant_aus:
             # Also check raw if normalized are all zero (e.g., due to high baseline)
             significant_aus = [
                 au for au in all_aus
                 if (au_values_left.get(au, 0) > 0.05 or au_values_right.get(au, 0) > 0.05) # Use slightly higher raw threshold
             ]
             if not significant_aus:
                  logger.warning(f"({analyzer.patient_id} - {action}) No significant AUs found (raw or normalized > 0.01). Skipping plot generation.")
                  plt.close(fig); return None # Return None if no data to plot

        # Group AUs by region (simple grouping)
        def get_au_region(au_name): # Simplified grouping
            match = re.search(r'\d+', au_name)
            if not match: return 'Unknown' # Handle AUs without numbers if any
            num = int(match.group())
            if num <= 5: return 'Upper'   # e.g., AU1, AU2, AU4, AU5
            elif num <= 10: return 'Mid'  # e.g., AU6, AU7, AU9, AU10
            elif num <= 20: return 'Lower' # e.g., AU12, AU14, AU15, AU17, AU20
            else: return 'Mouth/Other'      # e.g., AU23, AU25, AU26, AU45
        significant_aus = sorted(significant_aus, key=lambda au: (get_au_region(au), int(re.search(r'\d+', au).group())))


        x = np.arange(len(significant_aus))
        width = 0.38
        # Ensure values are floats, default to 0 if missing or NaN
        left_raw = [float(au_values_left.get(au, 0.0) if pd.notna(au_values_left.get(au)) else 0.0) for au in significant_aus]
        right_raw = [float(au_values_right.get(au, 0.0) if pd.notna(au_values_right.get(au)) else 0.0) for au in significant_aus]
        left_norm = [float(norm_au_values_left.get(au, 0.0) if pd.notna(norm_au_values_left.get(au)) else 0.0) for au in significant_aus]
        right_norm = [float(norm_au_values_right.get(au, 0.0) if pd.notna(norm_au_values_right.get(au)) else 0.0) for au in significant_aus]

        key_aus = action_to_aus.get(action, [])
        action_info = results.get(action, {}) # Get info for this action
        paralysis_info = action_info.get('paralysis', {})
        synkinesis_info = action_info.get('synkinesis', {})

        # --- Plot Raw Values ---
        plt.sca(ax_raw); ax_raw.set_facecolor(self.color_scheme['background'])
        bars1 = ax_raw.bar(x - width / 2, left_raw, width, label='Left Raw', color=self.color_scheme['left_raw'], edgecolor='black', linewidth=0.5)
        bars2 = ax_raw.bar(x + width / 2, right_raw, width, label='Right Raw', color=self.color_scheme['right_raw'], edgecolor='black', linewidth=0.5)
        self._add_bar_labels(ax_raw, bars1); self._add_bar_labels(ax_raw, bars2) # Helper for labels
        ax_raw.set_title(f"Raw AU Values - {action_descriptions.get(action, action)}", fontsize=14, fontweight='bold')
        ax_raw.set_ylabel("Raw Intensity", fontsize=12)
        tick_labels = [f"{au}\n{self.au_names.get(au, '').split('(')[0].strip()}" for au in significant_aus]
        ax_raw.set_xticks(x); ax_raw.set_xticklabels(tick_labels, rotation=45, fontsize=9, ha='right')
        # Highlight key AUs
        key_au_indices = [i for i, au in enumerate(significant_aus) if au in key_aus]
        for idx in key_au_indices:
             # Check index bounds
             if 0 <= idx < len(ax_raw.get_xticklabels()):
                 ax_raw.get_xticklabels()[idx].set_bbox(dict(facecolor=self.color_scheme['key_au_highlight'], alpha=0.4, boxstyle='round,pad=0.2'))
                 ax_raw.get_xticklabels()[idx].set_fontweight('bold')
             else:
                  logger.warning(f"Index {idx} out of bounds for xticklabels (len={len(ax_raw.get_xticklabels())}) in raw plot.")

        ax_raw.legend(fontsize=9, loc='upper right'); ax_raw.grid(axis='y', linestyle='--', alpha=0.3)
        # Add paralysis markers if data exists
        self._add_detection_markers(ax_raw, 'paralysis', paralysis_info, significant_aus, left_raw, right_raw, x, width)

        # --- Plot Normalized Values ---
        plt.sca(ax_norm); ax_norm.set_facecolor(self.color_scheme['background'])
        bars_ln = ax_norm.bar(x - width / 2, left_norm, width, label='Left Norm', color=self.color_scheme['left_norm'], alpha=0.8, edgecolor='black', linewidth=0.5)
        bars_rn = ax_norm.bar(x + width / 2, right_norm, width, label='Right Norm', color=self.color_scheme['right_norm'], alpha=0.8, edgecolor='black', linewidth=0.5)
        self._add_bar_labels(ax_norm, bars_ln); self._add_bar_labels(ax_norm, bars_rn)
        ax_norm.set_title("Normalized AU Values", fontsize=14, fontweight='bold')
        ax_norm.set_ylabel("Normalized Intensity", fontsize=12); ax_norm.set_xlabel("Action Units", fontsize=12)
        ax_norm.set_xticks(x); ax_norm.set_xticklabels(tick_labels, rotation=45, fontsize=9, ha='right')
        for idx in key_au_indices: # Highlight key AUs again
             # Check index bounds
             if 0 <= idx < len(ax_norm.get_xticklabels()):
                 ax_norm.get_xticklabels()[idx].set_bbox(dict(facecolor=self.color_scheme['key_au_highlight'], alpha=0.4, boxstyle='round,pad=0.2'))
                 ax_norm.get_xticklabels()[idx].set_fontweight('bold')
             else:
                  logger.warning(f"Index {idx} out of bounds for xticklabels (len={len(ax_norm.get_xticklabels())}) in norm plot.")

        ax_norm.legend(fontsize=9, loc='upper right'); ax_norm.grid(axis='y', linestyle='--', alpha=0.3)
        # Add synkinesis markers if data exists
        self._add_detection_markers(ax_norm, 'synkinesis', synkinesis_info, significant_aus, left_norm, right_norm, x, width)

        # --- Plot Frame Image ---
        plt.sca(ax_img)
        if frame_path and os.path.exists(frame_path):
            frame_img = cv2.imread(frame_path);
            if frame_img is None: # Check if image loading failed
                 ax_img.text(0.5, 0.5, "Frame Load Error", ha='center', va='center', fontsize=12, color='red')
                 logger.error(f"Failed to load frame image: {frame_path}")
            else:
                 img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                 ax_img.imshow(img_rgb); ax_img.set_title(f"Frame {frame_num}", fontsize=14)
                 # Add findings text
                 findings_text = self._format_clinical_findings(action_info, analyzer.patient_id, action, action_descriptions)
                 ax_img.text(0.5, -0.05, findings_text, ha='center', va='top', fontsize=9, family='monospace', wrap=True, transform=ax_img.transAxes, bbox=dict(boxstyle='round,pad=0.5', fc=self.color_scheme['background'], alpha=0.9, ec='#cccccc'))
        else:
             ax_img.text(0.5, 0.5, "Frame Unavailable", ha='center', va='center', fontsize=12, color='grey')
             logger.warning(f"Frame path not found or invalid for plotting: {frame_path}")
        ax_img.axis('off')

        # --- Final Touches ---
        fig.suptitle(f"{analyzer.patient_id} - {action_descriptions.get(action, action)}", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

        # Save figure
        output_filename = f"{action}_{analyzer.patient_id}_AUs.png"
        # Prefix with Key_ if it's a key diagnostic action
        if action in ["ES", "ET", "BS", "RE"]: output_filename = "Key_" + output_filename
        output_path = os.path.join(patient_output_dir, output_filename) # Use specific dir
        try:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved AU visualization: {output_path}")
        except Exception as e:
             logger.error(f"Failed to save AU visualization {output_path}: {e}", exc_info=True)
             output_path = None # Indicate failure
        finally:
            plt.close(fig) # Ensure figure is closed

        return output_path

    def _add_bar_labels(self, ax, bars):
        """Helper to add labels to bars."""
        # ... (remains the same) ...
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.05: # Threshold to add label
                 label = f"{height:.2f}"
                 y_pos = height + (0.02 * np.sign(height)) # Position above/below bar
                 ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label,
                         ha='center', va='bottom' if height > 0 else 'top', fontsize=7)

    def _add_detection_markers(self, ax, detection_type, info_dict, aus_plotted, left_values, right_values, x_coords, bar_width):
        """Adds markers for detected paralysis or synkinesis."""
        # ... (remains the same) ...
        if not info_dict or not info_dict.get('detected', False): return # Nothing to mark

        affected_aus = info_dict.get('affected_aus', {})
        contributing_aus_synk = info_dict.get('contributing_aus', {}) # For synkinesis

        if detection_type == 'paralysis':
            marker = 'v'; color = self.color_scheme['detection_marker']; offset = 0.08
            # Affected AUS are now stored as sets directly under left/right keys
            left_list = affected_aus.get('left', set())
            right_list = affected_aus.get('right', set())
            if not isinstance(left_list, (set, list)): left_list = set() # Ensure iterable
            if not isinstance(right_list, (set, list)): right_list = set() # Ensure iterable

        elif detection_type == 'synkinesis':
            marker = '*'; color = self.color_scheme['synkinesis_trigger']; offset = 0.1 # Slightly higher offset maybe
            # Collect all contributing AUs across types/sides
            left_list = set(); right_list = set()
            for synk_type_data in contributing_aus_synk.values():
                 left_list.update(synk_type_data.get('left',{}).get('trigger',[]))
                 left_list.update(synk_type_data.get('left',{}).get('response',[]))
                 right_list.update(synk_type_data.get('right',{}).get('trigger',[]))
                 right_list.update(synk_type_data.get('right',{}).get('response',[]))
        else:
            return

        for i, au in enumerate(aus_plotted):
            # Ensure index is valid before plotting
            if i >= len(left_values) or i >= len(right_values) or i >= len(x_coords):
                 logger.warning(f"Index {i} out of bounds when adding detection markers for AU {au}.")
                 continue

            x_base = x_coords[i]
            # Ensure values are numeric before adding offset
            y_left_base = left_values[i] if isinstance(left_values[i], (int, float)) else 0.0
            y_right_base = right_values[i] if isinstance(right_values[i], (int, float)) else 0.0

            y_left = y_left_base + offset
            y_right = y_right_base + offset

            # Check if AU is in the affected list for the side
            if au in left_list:
                 ax.plot(x_base - bar_width / 2, y_left, marker=marker, color=color, markersize=8, linestyle='', markeredgecolor='black')
            if au in right_list:
                 ax.plot(x_base + bar_width / 2, y_right, marker=marker, color=color, markersize=8, linestyle='', markeredgecolor='black')

    def _format_clinical_findings(self, info, patient_id, action, action_descriptions):
        """ Formats paralysis and synkinesis findings for display. """
        # ... (remains the same) ...
        if not info: return "Analysis data missing."

        action_desc = action_descriptions.get(action, action)
        lines = [f"FINDINGS - {patient_id} - {action_desc}", "="*30]

        # Paralysis
        paralysis_info = info.get('paralysis', {})
        lines.append("Paralysis:")
        if paralysis_info.get('detected'):
            for side in ['Left', 'Right']:
                side_findings = []
                for zone in ['upper', 'mid', 'lower']:
                    severity = paralysis_info.get('zones', {}).get(side.lower(), {}).get(zone, 'None')
                    if severity not in ['None', 'Error']: # Only show detected levels
                        side_findings.append(f"{zone.capitalize()}({severity})")
                if side_findings: lines.append(f"  {side}: {', '.join(side_findings)}")
                else: lines.append(f"  {side}: None Detected")

            # Safely access affected_aus (now sets)
            affected = paralysis_info.get('affected_aus', {}) # Use .get()
            left_affected = affected.get('left', set())
            right_affected = affected.get('right', set())
            # Convert sets to sorted lists for display
            if left_affected: lines.append(f"  L Affected AUs: {', '.join(sorted(list(left_affected)))}")
            if right_affected: lines.append(f"  R Affected AUs: {', '.join(sorted(list(right_affected)))}")
        else:
            lines.append("  None Detected")

        # Synkinesis
        synkinesis_info = info.get('synkinesis', {})
        lines.append("\nSynkinesis:")
        if synkinesis_info.get('detected'):
            synk_types_detected = []
            for synk_type in SYNKINESIS_TYPES:
                 sides_info = synkinesis_info.get('side_specific', {}).get(synk_type, {})
                 detected_sides = []
                 if sides_info.get('left', False): detected_sides.append("Left")
                 if sides_info.get('right', False): detected_sides.append("Right")
                 if detected_sides:
                      synk_types_detected.append(synk_type) # Track which types were detected
                      conf_str = ""
                      conf_dict = synkinesis_info.get('confidence', {}).get(synk_type, {})
                      left_conf = conf_dict.get('left', 0.0)
                      right_conf = conf_dict.get('right', 0.0)
                      if "Left" in detected_sides and "Right" in detected_sides: conf_str=f"(L:{left_conf:.2f}, R:{right_conf:.2f})"
                      elif "Left" in detected_sides: conf_str=f"(L:{left_conf:.2f})"
                      elif "Right" in detected_sides: conf_str=f"(R:{right_conf:.2f})"
                      lines.append(f"  - {synk_type}: {', '.join(detected_sides)} {conf_str}")
            if not synk_types_detected: # If detected=True but no specific types listed
                 lines.append("  Detected (Type details unavailable)")
        else:
             lines.append("  None Detected")

        return "\n".join(lines)

    def create_symmetry_visualization(self, analyzer, patient_output_dir, patient_id, results, action_descriptions):
        """ Creates the multi-panel symmetry plot. """
        # ... (remains the same) ...
        if not results: logger.error(f"({patient_id}) No analysis results for symmetry plot."); return None
        os.makedirs(patient_output_dir, exist_ok=True)
        fig = plt.figure(figsize=(18, 15)); gs = gridspec.GridSpec(4, 1, height_ratios=[3, 3, 3, 2])
        zone_titles = {'upper': 'Upper Face', 'mid': 'Mid Face', 'lower': 'Lower Face'}
        zone_data = {'upper': {'actions': [], 'left_raw': [], 'right_raw': [], 'left_norm': [], 'right_norm': []}, 'mid': {'actions': [], 'left_raw': [], 'right_raw': [], 'left_norm': [], 'right_norm': []}, 'lower': {'actions': [], 'left_raw': [], 'right_raw': [], 'left_norm': [], 'right_norm': []}}

        for action, info in results.items():
            if not isinstance(info, dict): # Skip if info isn't a dict (e.g., error marker)
                 logger.warning(f"({patient_id}) Invalid data structure for action '{action}' in results. Skipping for symmetry plot.")
                 continue

            action_desc = action_descriptions.get(action, action); primary_zone = None
            for zone, actions in ZONE_SPECIFIC_ACTIONS.items():
                if action in actions: primary_zone = zone; break
            if not primary_zone: continue
            zone_aus = FACIAL_ZONES.get(primary_zone) # Use .get()
            if not zone_aus: continue # Skip if zone has no AUs defined

            # Ensure required data structures exist
            left_info = info.get('left', {}); right_info = info.get('right', {})
            left_au_values = left_info.get('au_values', {}); right_au_values = right_info.get('au_values', {})
            left_norm_au_values = left_info.get('normalized_au_values', {}); right_norm_au_values = right_info.get('normalized_au_values', {})

            # Calculate averages safely
            left_raw_values = [left_au_values.get(au, 0) for au in zone_aus if pd.notna(left_au_values.get(au))];
            right_raw_values = [right_au_values.get(au, 0) for au in zone_aus if pd.notna(right_au_values.get(au))]
            left_norm_values = [left_norm_au_values.get(au, 0) for au in zone_aus if pd.notna(left_norm_au_values.get(au))];
            right_norm_values = [right_norm_au_values.get(au, 0) for au in zone_aus if pd.notna(right_norm_au_values.get(au))]

            zone_data[primary_zone]['actions'].append(action_desc)
            zone_data[primary_zone]['left_raw'].append(np.mean(left_raw_values) if left_raw_values else 0);
            zone_data[primary_zone]['right_raw'].append(np.mean(right_raw_values) if right_raw_values else 0)
            zone_data[primary_zone]['left_norm'].append(np.mean(left_norm_values) if left_norm_values else 0);
            zone_data[primary_zone]['right_norm'].append(np.mean(right_norm_values) if right_norm_values else 0)

        # Determine overall paralysis from the first action's results (assuming consistency)
        paralyzed_zones = {'left': {}, 'right': {}}
        first_action_info = next(iter(results.values()), None)
        if isinstance(first_action_info, dict) and 'paralysis' in first_action_info:
             paralysis_info = first_action_info['paralysis']
             if paralysis_info.get('detected'):
                 zones_dict = paralysis_info.get('zones', {})
                 for side in ['left', 'right']:
                     side_zones = zones_dict.get(side, {})
                     for zone, severity in side_zones.items():
                         if severity not in ['None', 'Error']:
                             paralyzed_zones[side][zone] = severity

        # Plotting loop
        for i, zone in enumerate(['upper', 'mid', 'lower']):
            ax = fig.add_subplot(gs[i])
            if not zone_data[zone]['actions']:
                 ax.text(0.5, 0.5, f"No data for {zone} zone actions", ha='center', va='center');
                 ax.set_title(zone_titles[zone]); ax.axis('off'); continue

            actions = zone_data[zone]['actions']; x_coords = np.arange(len(actions)); width = 0.2
            bars1 = ax.bar(x_coords - width*1.5, zone_data[zone]['left_raw'], width, label='Left Raw', color=self.color_scheme['left_raw'])
            bars2 = ax.bar(x_coords - width/2, zone_data[zone]['left_norm'], width, label='Left Norm', color=self.color_scheme['left_norm'], alpha=0.8)
            bars3 = ax.bar(x_coords + width/2, zone_data[zone]['right_raw'], width, label='Right Raw', color=self.color_scheme['right_raw'])
            bars4 = ax.bar(x_coords + width*1.5, zone_data[zone]['right_norm'], width, label='Right Norm', color=self.color_scheme['right_norm'], alpha=0.8)

            # Set title with paralysis info
            left_sev = paralyzed_zones.get('left', {}).get(zone, 'None');
            right_sev = paralyzed_zones.get('right', {}).get(zone, 'None')
            title = zone_titles[zone]; paralysis_title_info = []
            # Highlight bars if paralyzed
            color_map = {'Complete': self.color_scheme['complete_threshold'], 'Partial': self.color_scheme['partial_threshold']}
            alpha_color_map = {'Complete': '#b31a1a', 'Partial': '#e07000'} # Darker shades for normalized
            if left_sev != 'None':
                paralysis_title_info.append(f"Left: {left_sev}")
                for bar in bars1: bar.set_color(color_map.get(left_sev, self.color_scheme['left_raw']))
                for bar in bars2: bar.set_color(alpha_color_map.get(left_sev, self.color_scheme['left_norm']))
            if right_sev != 'None':
                paralysis_title_info.append(f"Right: {right_sev}")
                for bar in bars3: bar.set_color(color_map.get(right_sev, self.color_scheme['right_raw']))
                for bar in bars4: bar.set_color(alpha_color_map.get(right_sev, self.color_scheme['right_norm']))

            if paralysis_title_info: title += f" - (Paralysis: {'; '.join(paralysis_title_info)})"
            ax.set_title(title, fontsize=14); ax.set_ylabel('Avg AU Intensity'); ax.set_xticks(x_coords); ax.set_xticklabels(actions, rotation=45, ha='right')
            ax.legend(fontsize=9); ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Synkinesis Summary Panel
        synkinesis_info = {st: {'left': False, 'right': False, 'left_conf': 0, 'right_conf': 0} for st in SYNKINESIS_TYPES}
        synkinesis_detected = False
        if isinstance(first_action_info, dict) and 'synkinesis' in first_action_info:
             synk_res = first_action_info['synkinesis']
             if synk_res.get('detected'):
                  synkinesis_detected = True
                  side_spec = synk_res.get('side_specific', {})
                  conf_spec = synk_res.get('confidence', {})
                  for synk_type in SYNKINESIS_TYPES:
                      if side_spec.get(synk_type, {}).get('left', False):
                           synkinesis_info[synk_type]['left'] = True
                           synkinesis_info[synk_type]['left_conf'] = max(synkinesis_info[synk_type]['left_conf'], conf_spec.get(synk_type, {}).get('left', 0.0))
                      if side_spec.get(synk_type, {}).get('right', False):
                           synkinesis_info[synk_type]['right'] = True
                           synkinesis_info[synk_type]['right_conf'] = max(synkinesis_info[synk_type]['right_conf'], conf_spec.get(synk_type, {}).get('right', 0.0))

        ax_synk = fig.add_subplot(gs[3])
        if synkinesis_detected:
            synk_types = list(synkinesis_info.keys()); x_coords = np.arange(len(synk_types)); width = 0.35
            left_conf = [synkinesis_info[s]['left_conf'] if synkinesis_info[s]['left'] else 0 for s in synk_types];
            right_conf = [synkinesis_info[s]['right_conf'] if synkinesis_info[s]['right'] else 0 for s in synk_types]
            left_colors = [self._get_confidence_color(c) for c in left_conf]; right_colors = [self._get_confidence_color(c) for c in right_conf]
            bars1 = ax_synk.bar(x_coords - width/2, left_conf, width, label='Left Conf', color=left_colors);
            bars2 = ax_synk.bar(x_coords + width/2, right_conf, width, label='Right Conf', color=right_colors)
            self._add_bar_labels(ax_synk, bars1); self._add_bar_labels(ax_synk, bars2)
            ax_synk.set_title("Synkinesis Detection Confidence", fontsize=14); ax_synk.set_ylabel("Confidence"); ax_synk.set_xticks(x_coords); ax_synk.set_xticklabels(synk_types); ax_synk.legend(); ax_synk.set_ylim(0, 1.1); self._add_confidence_scale(ax_synk)
        else:
             ax_synk.text(0.5, 0.5, "No synkinesis detected", ha='center', va='center', fontsize=14); ax_synk.axis('off')

        # Overall Summary Text Box
        summary_text = f"SUMMARY - {patient_id}\n" + "="*(12 + len(patient_id)) + "\nParalysis:\n"
        paralysis_found = any(p != 'None' for side_dict in paralyzed_zones.values() for p in side_dict.values())
        if not paralysis_found: summary_text += "  None Detected\n"
        else:
            for side in ['left', 'right']:
                findings = [f"{z.upper()}:{s}" for z,s in paralyzed_zones.get(side, {}).items()]
                summary_text += f"  {side.capitalize()}: {', '.join(findings) if findings else 'None'}\n"
        summary_text += "\nSynkinesis:\n"
        if not synkinesis_detected: summary_text += "  None Detected\n"
        else:
            synk_lines = []
            for synk_type, info in synkinesis_info.items():
                 detected_sides = [s.capitalize() for s, detected in info.items() if s in ['left', 'right'] and detected]
                 if detected_sides:
                      conf_str = ""
                      if 'Left' in detected_sides and 'Right' in detected_sides: conf_str=f"(L:{info['left_conf']:.2f}, R:{info['right_conf']:.2f})"
                      elif 'Left' in detected_sides: conf_str=f"(L:{info['left_conf']:.2f})"
                      elif 'Right' in detected_sides: conf_str=f"(R:{info['right_conf']:.2f})"
                      synk_lines.append(f"  - {synk_type}: {', '.join(detected_sides)} {conf_str}")
            if synk_lines: summary_text += "\n".join(synk_lines) + "\n"
            else: summary_text += "  None Detected (inconsistent state?)\n" # Should not happen if synkinesis_detected is True

        fig.text(0.5, 0.02, summary_text, ha='center', va='bottom', fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc=self.color_scheme['background'], alpha=0.9), family='monospace')

        fig.suptitle(f'Facial Symmetry Analysis - {patient_id}', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.08, 1, 0.96]) # Adjust bottom/top margins
        output_path = os.path.join(patient_output_dir, f"symmetry_{patient_id}.png")
        try: plt.savefig(output_path, dpi=150); logger.info(f"({patient_id}) Saved symmetry visualization: {output_path}")
        except Exception as e: logger.error(f"({patient_id}) Failed save symmetry plot: {e}"); output_path = None
        finally: plt.close(fig)
        return output_path

    def _get_confidence_color(self, confidence, alpha=1.0):
        """ Generate color based on confidence. Red (low) -> Yellow -> Green (high). """
        # ... (Identical logic) ...
        if confidence is None or not pd.notna(confidence): return (0.8, 0.8, 0.8, alpha) # Gray for NaN/None
        conf = max(0, min(1, confidence))
        if conf < 0.5: r = 1.0; g = conf * 2; b = 0
        else: r = 2 - conf * 2; g = 1.0; b = 0
        return (r, g, b, alpha)

    def _add_confidence_scale(self, ax):
        """ Add confidence scale legend to axes. """
        # ... (Identical logic) ...
        gradient = np.linspace(0, 1, 100); gradient = np.vstack((gradient, gradient))
        scale_ax = ax.inset_axes([0.7, 0.05, 0.25, 0.05]) # Adjusted position/size
        scale_ax.imshow(gradient, aspect='auto', cmap=self._create_confidence_colormap())
        scale_ax.text(0, -0.5, "Low Conf", ha='left', va='top', fontsize=7)
        scale_ax.text(99, -0.5, "High Conf", ha='right', va='top', fontsize=7)
        scale_ax.set_axis_off()

    def _create_confidence_colormap(self):
        """ Create Red -> Yellow -> Green colormap. """
        # ... (Identical logic) ...
        colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]; cmap = LinearSegmentedColormap.from_list('confidence', colors, N=100); return cmap


    # --- MODIFIED: create_patient_dashboard ---
    def create_patient_dashboard(self, analyzer, patient_output_dir, patient_id, results, action_descriptions):
        """Generates an HTML dashboard summarizing results."""
        logger.info(f"Generating dashboard for patient: {patient_id}")
        # *** Ensure dashboard path uses the patient_output_dir ***
        dashboard_filename = f"dashboard_{patient_id}.html"
        dashboard_path = os.path.join(patient_output_dir, dashboard_filename)
        logger.debug(f"Dashboard path set to: {dashboard_path}")

        # Define paths for plots relative to the HTML file (they are in the same dir)
        symmetry_plot_filename = f"symmetry_{patient_id}.png"
        symmetry_plot_path_abs = os.path.join(patient_output_dir, symmetry_plot_filename)

        key_action_plots = {}
        for action in ["RE", "ES", "ET", "BS"]: # Key actions
             plot_filename_base = f"{action}_{patient_id}_AUs.png"
             plot_filename_key = f"Key_{plot_filename_base}"
             plot_path_abs_key = os.path.join(patient_output_dir, plot_filename_key)
             plot_path_abs_base = os.path.join(patient_output_dir, plot_filename_base)

             if os.path.exists(plot_path_abs_key):
                  key_action_plots[action] = plot_filename_key # Store relative path for HTML
             elif os.path.exists(plot_path_abs_base):
                   key_action_plots[action] = plot_filename_base # Store relative path for HTML
             else:
                  logger.warning(f"({patient_id}) AU plot for key action '{action}' not found at expected paths.")


        # --- Extract Summary Info (Remains the same) ---
        paralysis_summary = "None Detected"
        synkinesis_summary = "None Detected"
        paralysis_details = {} # {'Left': {'Upper': 'None', ...}, 'Right': ...}
        synkinesis_details = {} # {'Ocular-Oral': {'Left': False, 'Right': False, 'Conf_L': 0.0, 'Conf_R': 0.0}, ...}

        first_action_info = next(iter(results.values()), None)
        if isinstance(first_action_info, dict):
            # Paralysis Summary
            paralysis_info = first_action_info.get('paralysis', {})
            if paralysis_info.get('detected'):
                paralysis_summary = "Detected"
                zones_dict = paralysis_info.get('zones', {})
                for side in ['left', 'right']:
                    paralysis_details[side.capitalize()] = {}
                    for zone in ['upper', 'mid', 'lower']:
                        severity = zones_dict.get(side, {}).get(zone, 'None')
                        paralysis_details[side.capitalize()][zone.capitalize()] = severity

            # Synkinesis Summary
            synk_info = first_action_info.get('synkinesis', {})
            if synk_info.get('detected'):
                synkinesis_summary = "Detected"
                side_spec = synk_info.get('side_specific', {})
                conf_spec = synk_info.get('confidence', {})
                for synk_type in SYNKINESIS_TYPES:
                    synkinesis_details[synk_type] = {'Left': False, 'Right': False, 'Conf_L': 0.0, 'Conf_R': 0.0}
                    if side_spec.get(synk_type, {}).get('left', False):
                        synkinesis_details[synk_type]['Left'] = True
                        synkinesis_details[synk_type]['Conf_L'] = conf_spec.get(synk_type, {}).get('left', 0.0)
                    if side_spec.get(synk_type, {}).get('right', False):
                        synkinesis_details[synk_type]['Right'] = True
                        synkinesis_details[synk_type]['Conf_R'] = conf_spec.get(synk_type, {}).get('right', 0.0)


        # --- HTML Generation ---
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Facial Analysis Dashboard - {patient_id}</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; background-color: #f4f4f4; }}
                .container {{ max-width: 1200px; margin: auto; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }} /* Added max-width */
                h1, h2 {{ color: #333; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                h1 {{ text-align: center; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; table-layout: fixed; }} /* Added fixed layout */
                th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; word-wrap: break-word; }} /* Added word-wrap */
                th {{ background-color: #e9ecef; font-weight: bold; }} /* Bold header */
                .plot-container {{ text-align: center; margin-bottom: 30px; padding: 10px; border: 1px solid #eee; border-radius: 5px; background: #fdfdfd;}}
                .plot-container img {{ max-width: 95%; height: auto; border: 1px solid #ccc; }}
                .grid-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; }} /* Adjusted minmax */
                .grid-item {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #eee; text-align: center; }}
                .severity-Complete {{ color: red; font-weight: bold; }}
                .severity-Partial {{ color: orange; font-weight: bold; }}
                .severity-None {{ color: green; }}
                .severity-Error {{ color: grey; font-style: italic; }}
                .detected-Yes {{ color: purple; font-weight: bold; }}
                .detected-No {{ color: green; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Facial Analysis Dashboard - {patient_id}</h1>

                <h2>Summary Findings</h2>
                <table>
                    <colgroup> <col style="width: 30%;"> <col style="width: 70%;"> </colgroup>
                    <tr><th>Condition</th><th>Status</th></tr>
                    <tr><td>Facial Paralysis</td><td>{paralysis_summary}</td></tr>
                    <tr><td>Synkinesis</td><td>{synkinesis_summary}</td></tr>
                </table>

                <h2>Paralysis Details</h2>
        """
        # Paralysis Table
        if paralysis_details:
            html += """
                <table>
                    <colgroup> <col style="width: 25%;"> <col style="width: 25%;"> <col style="width: 25%;"> <col style="width: 25%;"> </colgroup>
                    <tr><th>Side</th><th>Upper</th><th>Mid</th><th>Lower</th></tr>
            """
            for side in ['Left', 'Right']:
                 html += f"<tr><td><b>{side}</b></td>"
                 for zone in ['Upper', 'Mid', 'Lower']:
                      severity = paralysis_details.get(side, {}).get(zone, 'None')
                      html += f"<td class='severity-{severity}'>{severity}</td>"
                 html += "</tr>\n"
            html += "</table>\n"
        else:
             html += "<p>No paralysis details available.</p>\n"

        # --- CORRECTED: Synkinesis Details HTML Generation ---
        html += "<h2>Synkinesis Details</h2>\n"
        # Check if there are any detected synkinesis details to display
        synkinesis_to_display = {k: v for k, v in synkinesis_details.items() if v['Left'] or v['Right']}
        if synkinesis_to_display:
             html += """
                 <table>
                     <colgroup> <col style="width: 30%;"> <col style="width: 20%;"> <col style="width: 20%;"> <col style="width: 30%;"> </colgroup>
                     <tr><th>Type</th><th>Left</th><th>Right</th><th>Confidence (L / R)</th></tr>
             """
             for synk_type, details in synkinesis_to_display.items():
                  html += f"<tr><td>{synk_type}</td>"
                  html += f"<td class='detected-{'Yes' if details['Left'] else 'No'}'>{'Yes' if details['Left'] else 'No'}</td>"
                  html += f"<td class='detected-{'Yes' if details['Right'] else 'No'}'>{'Yes' if details['Right'] else 'No'}</td>"
                  conf_l_str = f"{details['Conf_L']:.2f}" if details['Left'] else "-"
                  conf_r_str = f"{details['Conf_R']:.2f}" if details['Right'] else "-"
                  html += f"<td>{conf_l_str} / {conf_r_str}</td></tr>\n"
             html += "</table>\n"
        else:
             html += "<p>No synkinesis details available or detected.</p>\n"
        # --- END CORRECTION ---

        # --- Embed Plots (Remains the same) ---
        html += "<h2>Visualizations</h2>"
        # Symmetry Plot
        if os.path.exists(symmetry_plot_path_abs):
             html += f"""
             <div class="plot-container">
                 <h3>Symmetry Analysis</h3>
                 <img src="{symmetry_plot_filename}" alt="Symmetry Plot">
             </div>
             """
        else:
            html += "<p>Symmetry plot not found or not generated.</p>" # More informative message

        # Key Action Plots
        if key_action_plots:
            html += """
                <h3>Key Action AU Details</h3>
                <div class="grid-container">
            """
            for action, plot_file in key_action_plots.items():
                 action_desc = action_descriptions.get(action, action)
                 html += f"""
                     <div class="grid-item">
                         <h4>{action_desc} ({action})</h4>
                         <img src="{plot_file}" alt="AU Plot for {action}">
                     </div>
                 """
            html += "</div>"
        else:
             html += "<p>Key action AU plots not found or not generated.</p>" # More informative


        html += """
            </div>
        </body>
        </html>
        """

        # --- Save HTML ---
        try:
            with open(dashboard_path, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"({patient_id}) Dashboard generated successfully: {dashboard_path}")
            return dashboard_path
        except Exception as e:
             logger.error(f"({patient_id}) Failed to write dashboard HTML file: {e}", exc_info=True)
             return None
    # --- END MODIFIED create_patient_dashboard ---