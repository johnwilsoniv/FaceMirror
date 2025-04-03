"""
Visualization module for facial AU analysis.
Creates visualizations of AU values, paralysis, and synkinesis.
"""

import os
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from facial_au_constants import (
    AU_NAMES, FACIAL_ZONES, ZONE_SPECIFIC_ACTIONS, 
    SYNKINESIS_PATTERNS, SYNKINESIS_THRESHOLDS, SYNKINESIS_TYPES,
    PARALYSIS_THRESHOLDS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FacialAUVisualizer:
    """
    Creates visualizations of facial AU analysis results.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.au_names = AU_NAMES
        self.facial_zones = FACIAL_ZONES
        self.zone_specific_actions = ZONE_SPECIFIC_ACTIONS
        self.synkinesis_patterns = SYNKINESIS_PATTERNS
        self.synkinesis_thresholds = SYNKINESIS_THRESHOLDS
        self.synkinesis_types = SYNKINESIS_TYPES
        self.paralysis_thresholds = PARALYSIS_THRESHOLDS
        
        # Define standard colors for consistent visualization
        self.color_scheme = {
            'left_raw': '#4286f4',       # Blue for left side raw values
            'right_raw': '#f44242',      # Red for right side raw values
            'left_norm': '#1a53b3',      # Darker blue for left normalized
            'right_norm': '#b31a1a',     # Darker red for right normalized
            'key_au_highlight': '#ffd700',  # Gold for key AU highlighting
            'key_au_text': '#8B6914',    # Darker gold for key AU text
            'complete_threshold': '#d9534f',  # Red for complete paralysis threshold
            'partial_threshold': '#f0ad4e',   # Orange for partial paralysis threshold
            'background': '#f8f9fa',     # Light gray for plot background
            'grid': '#dee2e6',           # Lighter gray for grid lines
            'text': '#212529',           # Dark gray for text
            'detection_marker': '#ff5722',  # Bright orange for detection markers
            'detection_highlight': '#fff9c4',  # Light yellow for detection highlights
            'synkinesis_trigger': '#9c27b0',  # Purple for synkinesis trigger
            'synkinesis_response': '#009688',  # Teal for synkinesis response
            'normal': '#81c784'          # Green for normal function
        }

    def create_au_visualization(self, analyzer, au_values_left, au_values_right, norm_au_values_left,
                                norm_au_values_right, action, frame_num, output_dir,
                                frame_paths, action_descriptions, action_to_aus, results):
        # Create figure with better proportions
        fig = plt.figure(figsize=(22, 12))  # Increased width to accommodate external labels

        # Create a more flexible grid layout with enlarged frame image
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 1])

        # Create subplots
        ax_raw = fig.add_subplot(gs[0, 0])  # Raw AU values (top left)
        ax_norm = fig.add_subplot(gs[1, 0])  # Normalized AU values (bottom left)
        ax_img = fig.add_subplot(gs[:, 1])  # Enlarged frame image (entire right side)

        # Define descriptive titles for facial zones
        zone_titles = {
            'upper': 'Upper Face Zone (Eyebrows/Forehead)',
            'mid': 'Mid Face Zone (Eyes/Cheeks)',
            'lower': 'Lower Face Zone (Mouth/Jaw)'
        }

        # Define colors for synkinesis markers
        trigger_marker_color = '#FFD700'  # Yellow for trigger stars
        response_marker_color = '#C0C0C0'  # Silver for response circles

        # Prepare AU data - select AUs based on normalized values to focus on meaningful ones
        all_aus = set(au_values_left.keys()) | set(au_values_right.keys())
        all_aus = [au for au in all_aus if au.endswith('_r')]  # Filter out AU_c values

        # Filter out AUs with very low values on both sides
        significant_aus = [au for au in all_aus
                           if (norm_au_values_left.get(au, 0) > 0.01 or
                               norm_au_values_right.get(au, 0) > 0.01)]

        # Group AUs by facial region for more intuitive display
        def get_au_region(au):
            au_num = int(au[2:4])
            if au_num <= 2:
                return 0  # Brows
            elif au_num <= 7:
                return 1  # Eyes
            elif au_num <= 10:
                return 2  # Mid-face
            else:
                return 3  # Lower face

        significant_aus = sorted(significant_aus, key=lambda au: (get_au_region(au), au))

        if not significant_aus:
            logger.warning(f"No significant AU values for {action}")
            plt.close()
            return None

        # Prepare data for plotting raw values
        x = np.arange(len(significant_aus))
        width = 0.38  # Wider bars for better visibility

        # Get values for each side
        left_raw_values = [au_values_left.get(au, 0) for au in significant_aus]
        right_raw_values = [au_values_right.get(au, 0) for au in significant_aus]

        # Get normalized values
        left_norm_values = [norm_au_values_left.get(au, 0) for au in significant_aus]
        right_norm_values = [norm_au_values_right.get(au, 0) for au in significant_aus]

        # Get key AUs for this action for highlighting
        key_aus = action_to_aus[action]

        # Check if paralysis or synkinesis is detected in this action
        paralysis_detected = False
        synkinesis_detected = False

        # Get detailed paralysis and synkinesis information per zone
        paralysis_by_zone = {'upper': False, 'mid': False, 'lower': False}
        synkinesis_types = []

        if action in results:
            # Check for paralysis per zone
            info = results[action]
            for side in ['left', 'right']:
                for zone, severity in info['paralysis']['zones'][side].items():
                    if severity != 'None':
                        paralysis_detected = True
                        paralysis_by_zone[zone] = True

            # Check for synkinesis types
            if info['synkinesis']['detected']:
                synkinesis_detected = True
                synkinesis_types = info['synkinesis']['types']

        # Plot raw values (top left) - focus on paralysis thresholds
        plt.sca(ax_raw)

        # Set a light background color for consistency
        ax_raw.set_facecolor(self.color_scheme['background'])

        # Create bars for raw values
        bars1 = ax_raw.bar(x - width / 2, left_raw_values, width, label='Left Side',
                           color=self.color_scheme['left_raw'], edgecolor='black', linewidth=0.5)
        bars2 = ax_raw.bar(x + width / 2, right_raw_values, width, label='Right Side',
                           color=self.color_scheme['right_raw'], edgecolor='black', linewidth=0.5)

        # Add value labels to the bars - INSIDE the bars when tall enough
        def add_value_labels(bars, values):
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0.01:  # Only add label if value is significant
                    # If bar is tall enough, place label inside
                    if height > 0.2:  # Threshold for inside placement
                        ax_raw.text(bar.get_x() + bar.get_width() / 2, height * 0.5,
                                   f'{height:.2f}', ha='center', va='center', fontsize=8,
                                   fontweight='bold', color='white')
                    else:  # Otherwise place above
                        ax_raw.text(bar.get_x() + bar.get_width() / 2, height + 0.05,
                                   f'{height:.2f}', ha='center', va='bottom', fontsize=8,
                                   fontweight='bold')

        add_value_labels(bars1, left_raw_values)
        add_value_labels(bars2, right_raw_values)

        # Determine if this action is relevant for paralysis detection
        current_zone = None
        action_relevant_for_paralysis = False
        for zone, actions in self.zone_specific_actions.items():
            if action in actions:
                current_zone = zone
                action_relevant_for_paralysis = True
                break

        # Add title and labels - indicate if the action is relevant for paralysis detection
        action_desc = action_descriptions.get(action, action)
        if action_relevant_for_paralysis and paralysis_detected:
            ax_raw.set_title(
                f"Raw AU Values - Paralysis Assessment for {zone_titles.get(current_zone, current_zone)} Zone",
                fontsize=14, fontweight='bold')
        else:
            ax_raw.set_title(f"Raw AU Values", fontsize=14, fontweight='bold')

        ax_raw.set_ylabel("Intensity", fontsize=12, fontweight='bold')

        # Create AU labels - highlight key AUs with gold text
        tick_labels = []
        key_au_indices = []  # Track key AU indices for styling
        
        for i, au in enumerate(significant_aus):
            au_name = self.au_names.get(au, au)
            # Simplified label format
            au_code = au.split('_')[0]  # Just get AU01 instead of AU01_r
            tick_labels.append(f"{au_code}\n{au_name.split('(')[0]}")
            
            # Track key AU indices
            if au in key_aus:
                key_au_indices.append(i)

        ax_raw.set_xticks(x)
        ax_raw.set_xticklabels(tick_labels, rotation=45, fontsize=9, ha='right')
        
        # Style the key AU labels with gold background highlight but black text
        for label_idx in key_au_indices:
            labels = ax_raw.get_xticklabels()
            if label_idx < len(labels):
                # Create a background patch with gold color
                label = labels[label_idx]
                bbox = dict(facecolor=self.color_scheme['key_au_highlight'], 
                           edgecolor=None,
                           alpha=0.3,
                           boxstyle='round,pad=0.2')
                label.set_bbox(bbox)
                # Keep text black for better readability
                label.set_fontweight('bold')

        # Create a new custom legend only for paralysis thresholds
        legend_elements = []
        
        # Add paralysis threshold elements if this action is relevant for paralysis detection
        if action_relevant_for_paralysis and current_zone and paralysis_by_zone[current_zone]:
            # Create colored patch for complete paralysis threshold area
            legend_elements.append(
                patches.Patch(facecolor=self.color_scheme['complete_threshold'], 
                              alpha=0.1, 
                              label=f'Complete Paralysis Zone')
            )
            
            # Create colored patch for partial paralysis threshold area
            legend_elements.append(
                patches.Patch(facecolor=self.color_scheme['partial_threshold'], 
                              alpha=0.08, 
                              label=f'Partial Paralysis Zone')
            )

        # Add the custom legend at the upper left if it has elements
        if legend_elements:
            ax_raw.legend(handles=legend_elements, fontsize=9, loc='upper left')
        
        ax_raw.grid(axis='y', linestyle='--', alpha=0.3, color=self.color_scheme['grid'])

        # Mark AUs that contributed to paralysis detection with detection indicator
        paralysis_contributing_aus = {'left': set(), 'right': set()}

        if action in results and paralysis_detected:
            for side in ['left', 'right']:
                if 'paralysis' in results[action] and 'affected_aus' in results[action]['paralysis']:
                    paralysis_contributing_aus[side].update(results[action]['paralysis']['affected_aus'].get(side, []))

        # Add detection markers for AUs that contributed to paralysis detection
        for i, au in enumerate(significant_aus):
            # Check if this AU contributed to left side paralysis detection
            if au in paralysis_contributing_aus['left']:
                # Add indicator mark directly above the bar
                value = left_raw_values[i]
                ax_raw.plot(i - width / 2 + width/2, value + 0.08, marker='v',
                            color=self.color_scheme['detection_marker'],
                            markersize=8, markeredgecolor='black', zorder=10)

            # Check if this AU contributed to right side paralysis detection
            if au in paralysis_contributing_aus['right']:
                # Add indicator mark directly above the bar
                value = right_raw_values[i]
                ax_raw.plot(i + width / 2 + width/2 - width, value + 0.08, marker='v',
                            color=self.color_scheme['detection_marker'],
                            markersize=8, markeredgecolor='black', zorder=10)

        # Add threshold regions for better visualization of detection boundaries
        # Only add thresholds if this action is relevant for paralysis detection AND if paralysis was detected
        if action_relevant_for_paralysis and current_zone and paralysis_by_zone[current_zone]:
            # Get the detailed paralysis information from the results
            info = results[action]

            # Get standard thresholds for this zone
            complete_threshold = self.paralysis_thresholds[current_zone]['complete']['minimal_movement']
            partial_threshold = self.paralysis_thresholds[current_zone]['partial']['minimal_movement']

            # Draw threshold regions for easier understanding of detection boundaries
            ax_raw.axhspan(0, complete_threshold, alpha=0.1,
                           color=self.color_scheme['complete_threshold'], zorder=-5)
            ax_raw.axhspan(complete_threshold, partial_threshold, alpha=0.08,
                           color=self.color_scheme['partial_threshold'], zorder=-5)

            # Add a note about detection markers if there are AUs that contributed
            if paralysis_contributing_aus['left'] or paralysis_contributing_aus['right']:
                ax_raw.text(
                    0.98, 0.98,
                    "▼ Markers show AUs that\ntriggered detection",
                    transform=ax_raw.transAxes, fontsize=8,
                    ha='right', va='top', color=self.color_scheme['detection_marker'],
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=None,
                              boxstyle='round,pad=0.1')
                )
        else:
            # For actions not relevant to paralysis, add note if there was paralysis detected elsewhere
            any_paralysis_detected = False
            for zone_detected in paralysis_by_zone.values():
                if zone_detected:
                    any_paralysis_detected = True
                    break

            if any_paralysis_detected and not action_relevant_for_paralysis:
                ax_raw.text(0.5, 0.9, "Note: This action was not used for paralysis detection",
                            ha='center', va='center', transform=ax_raw.transAxes,
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', linewidth=0),
                            fontsize=10, color='gray')

        # Plot normalized values (bottom left) - focus on synkinesis thresholds
        plt.sca(ax_norm)

        # Set same background color for consistency
        ax_norm.set_facecolor(self.color_scheme['background'])

        # Create bars for normalized values and store references to bar objects
        left_bars = ax_norm.bar(x - width / 2, left_norm_values, width, label='Left Side (Normalized)',
                             color=self.color_scheme['left_norm'], alpha=0.8, edgecolor='black', linewidth=0.5)
        right_bars = ax_norm.bar(x + width / 2, right_norm_values, width, label='Right Side (Normalized)',
                              color=self.color_scheme['right_norm'], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Store references to bars in a dictionary for easy access when placing markers
        bar_dict = {
            'left': {},
            'right': {}
        }
        
        for i, au in enumerate(significant_aus):
            bar_dict['left'][au] = left_bars[i]
            bar_dict['right'][au] = right_bars[i]

        # Add value labels inside the bars when tall enough
        def add_norm_value_labels(bars, values):
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0.01:  # Only add label if value is significant
                    # If bar is tall enough, place label inside
                    if height > 0.2:  # Threshold for inside placement
                        ax_norm.text(bar.get_x() + bar.get_width() / 2, height * 0.5,
                                   f'{height:.2f}', ha='center', va='center', fontsize=8,
                                   fontweight='bold', color='white')
                    else:  # Otherwise place above
                        ax_norm.text(bar.get_x() + bar.get_width() / 2, height + 0.05,
                                   f'{height:.2f}', ha='center', va='bottom', fontsize=8,
                                   fontweight='bold')

        add_norm_value_labels(left_bars, left_norm_values)
        add_norm_value_labels(right_bars, right_norm_values)

        # Add title and labels - simple title without colored squares
        if synkinesis_detected:
            # Create basic title 
            synk_types_str = ", ".join(synkinesis_types)
            title_text = f"Normalized AU Values - Synkinesis Assessment ({synk_types_str})"
            ax_norm.set_title(title_text, fontsize=14, fontweight='bold')
        else:
            ax_norm.set_title("Normalized AU Values", fontsize=14, fontweight='bold')

        ax_norm.set_xlabel("Action Units", fontsize=12, fontweight='bold')
        ax_norm.set_ylabel("Normalized Intensity", fontsize=12, fontweight='bold')

        # Use same x-ticks as raw plot
        ax_norm.set_xticks(x)
        ax_norm.set_xticklabels(tick_labels, rotation=45, fontsize=9, ha='right')
        
        # Style the key AU labels with gold background highlight but black text
        for label_idx in key_au_indices:
            labels = ax_norm.get_xticklabels()
            if label_idx < len(labels):
                # Create a background patch with gold color
                label = labels[label_idx]
                bbox = dict(facecolor=self.color_scheme['key_au_highlight'], 
                           edgecolor=None,
                           alpha=0.3,
                           boxstyle='round,pad=0.2')
                label.set_bbox(bbox)
                # Keep text black for better readability
                label.set_fontweight('bold')

        # Add standard legend without Key AUs entry
        ax_norm.legend(fontsize=10, loc='upper right')
        ax_norm.grid(axis='y', linestyle='--', alpha=0.3, color=self.color_scheme['grid'])

        # IMPROVED SYNKINESIS VISUALIZATION
        if synkinesis_detected and action in results:
            info = results[action]['synkinesis']

            # Collect AU information for each synkinesis type
            synk_aus = {}
            for synk_type in synkinesis_types:
                synk_aus[synk_type] = {
                    'left': {'trigger': [], 'response': []},
                    'right': {'trigger': [], 'response': []}
                }

                # Check if we have contributing_aus information
                if 'contributing_aus' in info and synk_type in info['contributing_aus']:
                    for side in ['left', 'right']:
                        if side in info['contributing_aus'][synk_type]:
                            # Get trigger and response AUs
                            synk_aus[synk_type][side]['trigger'] = info['contributing_aus'][synk_type][side].get(
                                'trigger', [])
                            synk_aus[synk_type][side]['response'] = info['contributing_aus'][synk_type][side].get(
                                'response', [])

            # We need to add back the trigger/response symbol legend
            legend_elements = [
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=trigger_marker_color,
                           markersize=10, label='Trigger AU'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=response_marker_color,
                           markersize=8, label='Response AU')
            ]

            # Create synkinesis legend at the top left
            synk_legend = ax_norm.legend(handles=legend_elements, loc='upper left', fontsize=9)
            
            # Add the legend to the plot
            ax_norm.add_artist(synk_legend)

            # Process each synkinesis type
            for synk_type in synkinesis_types:
                if synk_type in self.synkinesis_thresholds:
                    trigger_threshold = self.synkinesis_thresholds[synk_type].get('trigger', 1.5)
                    response_threshold = self.synkinesis_thresholds[synk_type].get('coupled', 1.0)

                    # Track AUs that trigger multiple synkinesis types
                    trigger_au_synk_types = {}
                    
                    # First collect all trigger AUs and their synkinesis types
                    for current_synk_type in synkinesis_types:
                        if current_synk_type in info['contributing_aus']:
                            for side in ['left', 'right']:
                                if side in info['contributing_aus'][current_synk_type]:
                                    # Get trigger AUs
                                    triggers = info['contributing_aus'][current_synk_type][side].get('trigger', [])
                                    for au in triggers:
                                        if au not in trigger_au_synk_types:
                                            trigger_au_synk_types[au] = []
                                        if current_synk_type not in trigger_au_synk_types[au]:
                                            trigger_au_synk_types[au].append(current_synk_type)
                    
                    # Create separate legend for thresholds with semi-transparency
                    threshold_legend_elements = [
                        Line2D([0], [0], color=trigger_marker_color, lw=1.5, 
                               label=f"{synk_type} trigger threshold ({trigger_threshold:.1f})"),
                        Line2D([0], [0], color=response_marker_color, lw=1.5, linestyle='--',
                               label=f"{synk_type} response threshold ({response_threshold:.1f})")
                    ]
                    
                    # Add threshold legend box in the upper right with semi-transparency
                    threshold_legend = ax_norm.legend(handles=threshold_legend_elements, 
                                                     loc='upper right', 
                                                     fontsize=9,
                                                     title="Threshold Values",
                                                     framealpha=0.6)  # Make legend semi-transparent
                    
                    # Add back the trigger/response symbol legend
                    ax_norm.add_artist(threshold_legend)

                    # Define color mapping for synkinesis types - using lighter colors
                    synk_color_map = {
                        'Oral-Ocular': '#FF5555',  # Lighter red
                        'Ocular-Oral': '#5555FF',  # Lighter blue
                        'Snarl-Smile': '#55AA55'   # Lighter green
                    }
                    
                    # Define a function to create a split-colored star marker
                    def split_colored_star(x, y, size, colors, ax, zorder=20):
                        """Create a star with multiple colors (one for each synkinesis type)"""
                        from matplotlib.path import Path
                        import matplotlib.patches as mpatches
                        
                        # Star shape points (5-pointed star)
                        outer_radius = size / 72.0  # Convert points to data units
                        inner_radius = outer_radius * 0.4
                        angles = np.linspace(0, 2*np.pi, 11, endpoint=False)
                        angles = np.roll(angles, -1)  # Rotate to start at top
                        
                        # Create the star vertices
                        xs = np.zeros(10)
                        ys = np.zeros(10)
                        for i in range(10):
                            radius = outer_radius if i % 2 == 0 else inner_radius
                            xs[i] = x + radius * np.sin(angles[i])
                            ys[i] = y + radius * np.cos(angles[i])
                        
                        # Close the path
                        xs = np.append(xs, xs[0])
                        ys = np.append(ys, ys[0])
                        
                        # Number of colors to use
                        n_colors = len(colors)
                        
                        # Create a pie-like division of the star
                        for i in range(n_colors):
                            # Calculate the slice boundaries
                            start_angle = 2 * np.pi * i / n_colors
                            end_angle = 2 * np.pi * (i + 1) / n_colors
                            
                            # Create slice vertices
                            slice_vs = [(x, y)]  # Start at center
                            
                            # Find points within this sector
                            for j in range(len(xs)-1):
                                angle = np.arctan2(xs[j]-x, ys[j]-y)
                                if angle < 0:
                                    angle += 2 * np.pi
                                
                                # Add point if it's within this sector or near the boundaries
                                if start_angle <= angle <= end_angle or \
                                   abs(angle - start_angle) < 0.1 or \
                                   abs(angle - end_angle) < 0.1:
                                    slice_vs.append((xs[j], ys[j]))
                            
                            # If we have any points, draw this slice
                            if len(slice_vs) > 2:
                                poly = mpatches.Polygon(slice_vs, closed=True, 
                                                       facecolor=colors[i], 
                                                       edgecolor='black',
                                                       linewidth=0.5,
                                                       zorder=zorder)
                                ax.add_patch(poly)
                        
                        return
                    
            # We need to create two legends - one for synkinesis types and one for markers
            # First create the marker legend (star/circle) at top left - without title
            marker_legend_elements = [
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
                           markersize=10, label='Trigger AU'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                           markersize=8, label='Response AU')
            ]
            
            marker_legend = ax_norm.legend(handles=marker_legend_elements, 
                                         loc='upper left', fontsize=9)
            
            # Add the marker legend to the plot
            ax_norm.add_artist(marker_legend)
            
            # Now create a synkinesis type legend for the title - without title
            synk_legend_elements = []
            for synk_type in synkinesis_types:
                color = synk_color_map.get(synk_type, '#AA55AA')  # Default to purple
                synk_legend_elements.append(
                    patches.Patch(facecolor=color, edgecolor='black', label=synk_type)
                )
            
            # Create a small synkinesis type legend
            if synk_legend_elements:
                synk_type_legend = ax_norm.legend(handles=synk_legend_elements, 
                                                loc='upper right', 
                                                fontsize=9)
                ax_norm.add_artist(synk_type_legend)
                
            # Remove any other legends that might be in the plot
            # Get all child artists
            for child in ax_norm.get_children():
                # Check if it's a legend and not our legends
                if (isinstance(child, Legend) and 
                    child != marker_legend and 
                    child != synk_type_legend):
                    child.remove()
                    
                    # Process each synkinesis type
                    for synk_type in synkinesis_types:
                        if synk_type in self.synkinesis_thresholds:
                            trigger_threshold = self.synkinesis_thresholds[synk_type].get('trigger', 1.5)
                            response_threshold = self.synkinesis_thresholds[synk_type].get('coupled', 1.0)
                            
                            # Get color for this synkinesis type
                            synk_color = synk_color_map.get(synk_type, '#AA55AA')  # Default to lighter purple
                            
                            # Add horizontal lines for thresholds using synkinesis type color
                            trigger_line = ax_norm.axhline(y=trigger_threshold, color=synk_color,
                                           linestyle='-', alpha=0.7, linewidth=1.5)
                            
                            response_line = ax_norm.axhline(y=response_threshold, color=synk_color,
                                           linestyle='--', alpha=0.7, linewidth=1.5)
                    
                    # Get all trigger and response AUs across all synkinesis types
                    all_trigger_aus = set()
                    all_response_aus = set()
                    active_sides_by_synk = {}
                    
                    for synk_type in synkinesis_types:
                        active_sides_by_synk[synk_type] = []
                        for side in ['left', 'right']:
                            if info['side_specific'][synk_type][side]:
                                active_sides_by_synk[synk_type].append(side)
                                all_trigger_aus.update(synk_aus[synk_type][side]['trigger'])
                                all_response_aus.update(synk_aus[synk_type][side]['response'])
                                
                    # Mark response AUs with circles
                    for au in all_response_aus:
                        if au in significant_aus:
                            # Process each synkinesis type to see which ones have this AU as a response
                            for synk_type in synkinesis_types:
                                synk_color = synk_color_map.get(synk_type, '#AA55AA')
                                
                                # Check both sides
                                for side in active_sides_by_synk[synk_type]:
                                    if au in synk_aus[synk_type][side]['response']:
                                        bar = bar_dict[side][au]
                                        value = bar.get_height()
                                        response_threshold = self.synkinesis_thresholds[synk_type].get('coupled', 1.0)
                                        
                                        # Add circle marker using the exact center of the bar
                                        if value >= response_threshold:
                                            bar_center = bar.get_x() + bar.get_width() / 2
                                            ax_norm.plot(bar_center, value + 0.08, marker='o', markersize=10,
                                                      color=synk_color,
                                                      markeredgecolor='black', alpha=0.9, zorder=15)
                    
                    # Mark trigger AUs - using double-layered stars for multiple synkinesis types
                    for au in all_trigger_aus:
                        if au in significant_aus:
                            # Get all synkinesis types and sides for this AU
                            triggered_sides = {}  # Maps synk_type to list of sides where it's a trigger
                            
                            for synk_type in synkinesis_types:
                                for side in active_sides_by_synk[synk_type]:
                                    if au in synk_aus[synk_type][side]['trigger']:
                                        if synk_type not in triggered_sides:
                                            triggered_sides[synk_type] = []
                                        triggered_sides[synk_type].append(side)
                            
                            # If this AU is a trigger for multiple synkinesis types
                            if len(triggered_sides) > 1:
                                # Get the colors in a consistent order
                                triggered_types = sorted(triggered_sides.keys())
                                color1 = synk_color_map.get(triggered_types[0], '#AA55AA')
                                color2 = synk_color_map.get(triggered_types[1], '#55AA55')
                                
                                # Draw the stars for all relevant sides
                                for synk_type in triggered_types:
                                    for side in triggered_sides[synk_type]:
                                        bar = bar_dict[side][au]
                                        value = bar.get_height()
                                        trigger_threshold = self.synkinesis_thresholds[synk_type].get('trigger', 1.5)
                                        
                                        if value >= trigger_threshold:
                                            bar_center = bar.get_x() + bar.get_width() / 2
                                            
                                            # First, draw background star in first color (same size as regular)
                                            ax_norm.plot(bar_center, value + 0.08, marker='*', markersize=12,
                                                     color=color1, markeredgecolor=color1, 
                                                     alpha=0.9, zorder=19)
                                                     
                                            # Then draw smaller foreground star in second color
                                            ax_norm.plot(bar_center, value + 0.08, marker='*', markersize=8,
                                                     color=color2, markeredgecolor=color2, 
                                                     alpha=1.0, zorder=20)
                                            
                                            # Only need to draw once per bar
                                            break
                                        
                            else:
                                # For AU that only triggers one synkinesis type, use regular star
                                synk_type = list(triggered_sides.keys())[0]
                                synk_color = synk_color_map.get(synk_type, '#AA55AA')
                                
                                for side in triggered_sides[synk_type]:
                                    bar = bar_dict[side][au]
                                    value = bar.get_height()
                                    trigger_threshold = self.synkinesis_thresholds[synk_type].get('trigger', 1.5)
                                    
                                    # Add standard star
                                    if value >= trigger_threshold:
                                        bar_center = bar.get_x() + bar.get_width() / 2
                                        ax_norm.plot(bar_center, value + 0.08, marker='*', markersize=12,
                                                 color=synk_color, markeredgecolor='black', 
                                                 alpha=0.9, zorder=20)

        # Add enlarged frame image (entire right side) - KEEP THIS PART UNCHANGED
        plt.sca(ax_img)
        if action in frame_paths and os.path.exists(frame_paths[action]):
            # Read with OpenCV and convert from BGR to RGB
            frame_img = cv2.imread(frame_paths[action])
            if frame_img is not None:
                # Convert BGR to RGB for correct color representation in matplotlib
                frame_img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

                # Add frame around the image
                ax_img.imshow(frame_img_rgb)
                for spine in ax_img.spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(1.5)

                ax_img.set_title(f"Video Frame {frame_num}", fontsize=14, fontweight='bold')
                ax_img.axis('off')

                # Add clinical findings as text overlay at the bottom of the frame
                if action in results:
                    info = results[action]
                    findings_text = self._format_clinical_findings(info, analyzer.patient_id, action,
                                                                   action_descriptions)
                    ax_img.text(0.5, -0.05, findings_text,
                                ha='center', va='top', fontsize=10,
                                family='monospace', wrap=True,
                                transform=ax_img.transAxes,
                                bbox=dict(facecolor=self.color_scheme['background'],
                                          edgecolor='#dee2e6',
                                          alpha=0.9,
                                          boxstyle='round,pad=0.5',
                                          linewidth=1.5))
            else:
                ax_img.text(0.5, 0.5, "Error reading frame image",
                            ha='center', va='center', fontsize=14, color='red')
                ax_img.axis('off')
        else:
            # If no frame image available, display a message
            ax_img.text(0.5, 0.5, "Frame image not available",
                        ha='center', va='center', fontsize=14)
            ax_img.axis('off')

        # Add a main title
        main_title = f"{analyzer.patient_id} - {action_descriptions.get(action, action)} (Frame {frame_num})"
        fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)

        # Add a subtle version indicator
        fig.text(0.98, 0.01, "Facial AU Analyzer v2.3",
                 ha='right', va='bottom', fontsize=8, style='italic', alpha=0.5)

        # Adjust layout to accommodate external labels
        plt.tight_layout(rect=[0, 0, 0.9, 0.96])

        # Save the visualization
        if action in ["ES", "BS", "RE"]:
            output_path = os.path.join(output_dir, f"Key_{action}_{analyzer.patient_id}_AUs.png")
        else:
            output_path = os.path.join(output_dir, f"{action}_{analyzer.patient_id}_AUs.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved enhanced AU visualization: {output_path}")
        return output_path
    
    def _format_clinical_findings(self, info, patient_id, action, action_descriptions):
        """
        Format clinical findings in a clean, text-based format.
        
        Args:
            info (dict): Results information
            patient_id (str): Patient ID
            action (str): Action code
            action_descriptions (dict): Dictionary of action descriptions
            
        Returns:
            str: Formatted clinical findings text
        """
        action_desc = action_descriptions.get(action, action)
        
        # Create header
        findings_text = f"CLINICAL FINDINGS - {patient_id} - {action_desc}\n\n"
        
        # Paralysis reporting in clean text format
        if info['paralysis']['detected']:
            findings_text += "PARALYSIS ASSESSMENT:\n"
            
            # Left side information
            findings_text += "  LEFT SIDE:\n"
            for zone in ['upper', 'mid', 'lower']:
                severity = info['paralysis']['zones']['left'][zone]
                status = severity if severity != 'None' else 'Normal'
                findings_text += f"    {zone.upper()} ZONE: {status}\n"
            
            # Right side information
            findings_text += "  RIGHT SIDE:\n"
            for zone in ['upper', 'mid', 'lower']:
                severity = info['paralysis']['zones']['right'][zone]
                status = severity if severity != 'None' else 'Normal'
                findings_text += f"    {zone.upper()} ZONE: {status}\n"
            
            # Report affected AUs if any
            if info['paralysis']['affected_aus']['left']:
                findings_text += f"  Left affected AUs: {', '.join(info['paralysis']['affected_aus']['left'])}\n"
            if info['paralysis']['affected_aus']['right']:
                findings_text += f"  Right affected AUs: {', '.join(info['paralysis']['affected_aus']['right'])}\n"
        else:
            findings_text += "PARALYSIS ASSESSMENT: None detected\n"
        
        findings_text += "\n"
        
        # Synkinesis reporting in clean text format
        if info['synkinesis']['detected']:
            findings_text += "SYNKINESIS ASSESSMENT:\n"
            
            for synk_type in self.synkinesis_types:
                sides = info['synkinesis']['side_specific'].get(synk_type, {'left': False, 'right': False})
                
                detected_sides = []
                if sides["left"]:
                    detected_sides.append("LEFT")
                if sides["right"]:
                    detected_sides.append("RIGHT")
                
                if detected_sides:
                    findings_text += f"  {synk_type}: Detected on {' and '.join(detected_sides)} side(s)\n"
                else:
                    findings_text += f"  {synk_type}: Not detected\n"
        else:
            findings_text += "SYNKINESIS ASSESSMENT: None detected\n"
        
        # Add note about methods
        findings_text += "\nNOTE: Raw values used for paralysis detection,\n"
        findings_text += "normalized values used for synkinesis detection."
        
        return findings_text
        
    def create_symmetry_visualization(self, analyzer, output_dir, patient_id, results, action_descriptions):
        """
        Create visualization of facial symmetry. Delegates to visualizer component.
        
        Args:
            analyzer: The analyzer instance containing results
            output_dir (str): Output directory
            patient_id (str): Patient ID
            results (dict): Results dictionary
            action_descriptions (dict): Dictionary of action descriptions
            
        Returns:
            str: Path to saved visualization
        """
        if not results:
            logger.error("No analysis results. Please run analyze_maximal_intensity() first")
            return None
        
        patient_output_dir = os.path.join(output_dir, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # Create figure with enhanced layout
        fig = plt.figure(figsize=(18, 15))  # Increased width for external labels if needed
        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 3, 3, 2])
        
        # Define zone titles with more detail
        zone_titles = {
            'upper': 'Upper Face Zone (Eyebrows/Forehead)',
            'mid': 'Mid Face Zone (Eyes/Cheeks)',
            'lower': 'Lower Face Zone (Mouth/Jaw)'
        }
        
        # Prepare data for visualization by zone
        zone_data = {
            'upper': {'actions': [], 'left_raw': [], 'right_raw': [], 'left_norm': [], 'right_norm': []},
            'mid': {'actions': [], 'left_raw': [], 'right_raw': [], 'left_norm': [], 'right_norm': []},
            'lower': {'actions': [], 'left_raw': [], 'right_raw': [], 'left_norm': [], 'right_norm': []}
        }
        
        # Collect values by zone
        for action, info in results.items():
            action_desc = action_descriptions.get(action, action)
            
            # Find which zone this action primarily tests
            primary_zone = None
            for zone, actions in self.zone_specific_actions.items():
                if action in actions:
                    primary_zone = zone
                    break
            
            if not primary_zone:
                continue  # Skip if action doesn't clearly fit a zone
            
            # Calculate average of AUs in this zone
            zone_aus = self.facial_zones[primary_zone]
            
            # Raw values
            left_raw_values = [info['left']['au_values'].get(au, 0) for au in zone_aus]
            right_raw_values = [info['right']['au_values'].get(au, 0) for au in zone_aus]
            
            # Normalized values
            left_norm_values = [info['left']['normalized_au_values'].get(au, 0) for au in zone_aus]
            right_norm_values = [info['right']['normalized_au_values'].get(au, 0) for au in zone_aus]
            
            left_raw_avg = np.mean(left_raw_values) if left_raw_values else 0
            right_raw_avg = np.mean(right_raw_values) if right_raw_values else 0
            
            left_norm_avg = np.mean(left_norm_values) if left_norm_values else 0
            right_norm_avg = np.mean(right_norm_values) if right_norm_values else 0
            
            zone_data[primary_zone]['actions'].append(action_desc)
            zone_data[primary_zone]['left_raw'].append(left_raw_avg)
            zone_data[primary_zone]['right_raw'].append(right_raw_avg)
            zone_data[primary_zone]['left_norm'].append(left_norm_avg)
            zone_data[primary_zone]['right_norm'].append(right_norm_avg)
        
        # Determine which zones show paralysis on each side
        paralyzed_zones = {
            'left': {},
            'right': {}
        }
        
        for info in results.values():
            if info['paralysis']['detected']:
                for side in ['left', 'right']:
                    for zone, severity in info['paralysis']['zones'][side].items():
                        if severity != 'None':
                            paralyzed_zones[side][zone] = severity
        
        # Plot each zone with both raw and normalized data - side by side
        for i, zone in enumerate(['upper', 'mid', 'lower']):
            ax = fig.add_subplot(gs[i])
            
            if not zone_data[zone]['actions']:
                ax.text(0.5, 0.5, f"No {zone} zone data available", 
                       ha='center', va='center', fontsize=12)
                ax.set_title(zone_titles[zone])
                ax.axis('off')
                continue
            
            actions = zone_data[zone]['actions']
            x = np.arange(len(actions))
            width = 0.2  # Narrower bars to fit 4 series
            
            # Create bars for raw and normalized values
            bars1 = ax.bar(x - width*1.5, zone_data[zone]['left_raw'], width, 
                         label='Left (Raw)', color=self.color_scheme['left_raw'])
            bars2 = ax.bar(x - width/2, zone_data[zone]['left_norm'], width, 
                         label='Left (Normalized)', color=self.color_scheme['left_norm'], alpha=0.8)
            bars3 = ax.bar(x + width/2, zone_data[zone]['right_raw'], width, 
                         label='Right (Raw)', color=self.color_scheme['right_raw'])
            bars4 = ax.bar(x + width*1.5, zone_data[zone]['right_norm'], width, 
                         label='Right (Normalized)', color=self.color_scheme['right_norm'], alpha=0.8)
            
            # Highlight paralyzed sides if applicable for this zone
            left_zone_severity = paralyzed_zones['left'].get(zone, 'None')
            right_zone_severity = paralyzed_zones['right'].get(zone, 'None')
            
            # Generate title with paralysis information
            zone_title = zone_titles[zone]
            paralysis_info = []
            
            # Add paralysis info to title if detected
            if left_zone_severity != 'None':
                paralysis_info.append(f"Left: {left_zone_severity} paralysis")
                # Color bars based on severity
                color_raw = self.color_scheme['complete_threshold'] if left_zone_severity == 'Complete' else self.color_scheme['partial_threshold']
                color_norm = '#b31a1a' if left_zone_severity == 'Complete' else '#e07000'
                for bar in bars1:
                    bar.set_color(color_raw)
                for bar in bars2:
                    bar.set_color(color_norm)
                    
            if right_zone_severity != 'None':
                paralysis_info.append(f"Right: {right_zone_severity} paralysis")
                # Color bars based on severity
                color_raw = self.color_scheme['complete_threshold'] if right_zone_severity == 'Complete' else self.color_scheme['partial_threshold']
                color_norm = '#b31a1a' if right_zone_severity == 'Complete' else '#e07000'
                for bar in bars3:
                    bar.set_color(color_raw)
                for bar in bars4:
                    bar.set_color(color_norm)
            
            if paralysis_info:
                ax.set_title(f"{zone_title} - {' / '.join(paralysis_info)}", fontsize=14)
            else:
                ax.set_title(zone_title, fontsize=14)
            
            ax.set_ylabel('AU Intensity')
            ax.set_xticks(x)
            ax.set_xticklabels(actions, rotation=45, ha='right')
            
            # Add value labels inside the bars when tall enough
            def add_value_labels(bars, side):
                for bar in bars:
                    height = bar.get_height()
                    if height > 0.1:
                        if height > 0.4:  # Threshold for inside placement
                            ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                                  f'{side}: {height:.2f}', ha='center', va='center', 
                                  fontsize=7, color='white', fontweight='bold')
                        else:
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                                  f'{side}: {height:.2f}', ha='center', va='bottom', 
                                  fontsize=7, rotation=45)
            
            add_value_labels(bars1, 'L-raw')
            add_value_labels(bars3, 'R-raw')
            
            ax.legend(fontsize=9)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Gather comprehensive synkinesis information for the summary
        synkinesis_info = {synk_type: {'left': False, 'right': False, 'left_conf': 0, 'right_conf': 0} 
                          for synk_type in self.synkinesis_types}
        synkinesis_detected = False
        
        for info in results.values():
            if info['synkinesis']['detected']:
                synkinesis_detected = True
                for synk_type, sides in info['synkinesis']['side_specific'].items():
                    if sides['left']:
                        synkinesis_info[synk_type]['left'] = True
                        # Get confidence if available
                        if 'confidence' in info['synkinesis'] and synk_type in info['synkinesis']['confidence']:
                            if 'left' in info['synkinesis']['confidence'][synk_type]:
                                conf = info['synkinesis']['confidence'][synk_type]['left']
                                synkinesis_info[synk_type]['left_conf'] = max(
                                    synkinesis_info[synk_type]['left_conf'], conf)
                                
                    if sides['right']:
                        synkinesis_info[synk_type]['right'] = True
                        # Get confidence if available
                        if 'confidence' in info['synkinesis'] and synk_type in info['synkinesis']['confidence']:
                            if 'right' in info['synkinesis']['confidence'][synk_type]:
                                conf = info['synkinesis']['confidence'][synk_type]['right']
                                synkinesis_info[synk_type]['right_conf'] = max(
                                    synkinesis_info[synk_type]['right_conf'], conf)
        
        # Create synkinesis confidence visualization
        ax_synk = fig.add_subplot(gs[3])
        
        if synkinesis_detected:
            # Create bar chart for synkinesis confidence
            synk_types = list(synkinesis_info.keys())
            x = np.arange(len(synk_types))
            width = 0.35
            
            # Prepare confidence values
            left_conf = [synkinesis_info[s]['left_conf'] if synkinesis_info[s]['left'] else 0 for s in synk_types]
            right_conf = [synkinesis_info[s]['right_conf'] if synkinesis_info[s]['right'] else 0 for s in synk_types]
            
            # Create bars with color gradient based on confidence
            left_colors = [self._get_confidence_color(conf) for conf in left_conf]
            right_colors = [self._get_confidence_color(conf) for conf in right_conf]
            
            # Plot confidence bars
            bars1 = ax_synk.bar(x - width/2, left_conf, width, label='Left Side Confidence', color=left_colors)
            bars2 = ax_synk.bar(x + width/2, right_conf, width, label='Right Side Confidence', color=right_colors)
            
            # Add labels inside the bars when tall enough
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                if height > 0:
                    if height > 0.3:  # Threshold for inside placement
                        ax_synk.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                                  f'{height:.2f}', ha='center', va='center', 
                                  fontsize=9, color='white', fontweight='bold')
                    else:
                        ax_synk.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                  f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                if height > 0:
                    if height > 0.3:  # Threshold for inside placement
                        ax_synk.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                                  f'{height:.2f}', ha='center', va='center', 
                                  fontsize=9, color='white', fontweight='bold')
                    else:
                        ax_synk.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                  f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            
            ax_synk.set_title("Synkinesis Detection Confidence", fontsize=14)
            ax_synk.set_ylabel("Confidence Score (0-1)")
            ax_synk.set_xticks(x)
            ax_synk.set_xticklabels(synk_types)
            ax_synk.legend()
            ax_synk.set_ylim(0, 1.1)  # Set y-axis limit for confidence
            
            # Add confidence scale
            self._add_confidence_scale(ax_synk)
            
        else:
            # If no synkinesis detected, show a message
            ax_synk.text(0.5, 0.5, "No synkinesis detected", 
                      ha='center', va='center', fontsize=14, fontweight='bold')
            ax_synk.axis('off')
        
        # Create summary text
        summary_text = "SUMMARY OF FINDINGS:\n\n"
        
        # Add methodological note
        summary_text += "METHODOLOGY NOTE:\n"
        summary_text += "- Paralysis detection uses raw AU values\n"
        summary_text += "- Synkinesis detection uses normalized (baseline-subtracted) values\n\n"
        
        # Paralysis summary
        paralysis_detected = bool(paralyzed_zones['left'] or paralyzed_zones['right'])
        
        if paralysis_detected:
            summary_text += "PARALYSIS:\n"
            
            # Left side
            if paralyzed_zones['left']:
                summary_text += "LEFT SIDE:\n"
                for zone, severity in paralyzed_zones['left'].items():
                    summary_text += f"- {zone.upper()} ZONE: {severity}\n"
            else:
                summary_text += "LEFT SIDE: No paralysis detected\n"
                
            # Right side
            if paralyzed_zones['right']:
                summary_text += "RIGHT SIDE:\n"
                for zone, severity in paralyzed_zones['right'].items():
                    summary_text += f"- {zone.upper()} ZONE: {severity}\n"
            else:
                summary_text += "RIGHT SIDE: No paralysis detected\n"
        else:
            summary_text += "PARALYSIS: None detected on either side\n"
        
        # Synkinesis summary with confidence
        summary_text += "\nSYNKINESIS:\n"
        
        if synkinesis_detected:
            for synk_type, info in synkinesis_info.items():
                detected_sides = []
                confidence_info = []
                
                if info['left']:
                    detected_sides.append("LEFT")
                    confidence_info.append(f"left conf: {info['left_conf']:.2f}")
                        
                if info['right']:
                    detected_sides.append("RIGHT")
                    confidence_info.append(f"right conf: {info['right_conf']:.2f}")
                
                if detected_sides:
                    summary_text += f"- {synk_type}: Detected on {' and '.join(detected_sides)} side(s)"
                    
                    if confidence_info:
                        summary_text += f" ({', '.join(confidence_info)})"
                        
                    summary_text += "\n"
                else:
                    summary_text += f"- {synk_type}: Not detected\n"
        else:
            summary_text += "No synkinesis patterns detected\n"
        
        # Add the summary text at the bottom
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10, 
               bbox=dict(facecolor=self.color_scheme['background'], 
                         edgecolor='#dee2e6', alpha=0.95, boxstyle='round,pad=0.5'),
               family='monospace')
        
        fig.suptitle(f'Enhanced Facial Analysis - Patient {patient_id}', fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save the visualization
        output_path = os.path.join(patient_output_dir, f"symmetry_{patient_id}.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Saved enhanced symmetry visualization: {output_path}")
        return output_path
    
    def _get_confidence_color(self, confidence, alpha=1.0):
        """
        Generate a color based on confidence value.
        Low confidence: Red -> Yellow -> Green: High confidence
        
        Args:
            confidence (float): Confidence value 0-1
            alpha (float): Transparency 0-1
            
        Returns:
            str or tuple: Color representation
        """
        if confidence is None:
            return (0.8, 0.8, 0.8, alpha)  # Gray for no data
        
        # Clip confidence to 0-1 range
        conf = max(0, min(1, confidence))
        
        if conf < 0.5:
            # Red to Yellow gradient (0-0.5)
            r = 1.0
            g = conf * 2
            b = 0
        else:
            # Yellow to Green gradient (0.5-1.0)
            r = 2 - conf * 2
            g = 1.0
            b = 0
        
        return (r, g, b, alpha)
    
    def _add_confidence_scale(self, ax):
        """
        Add a confidence color scale to the given axes.
        
        Args:
            ax (matplotlib.axes.Axes): The axes to add the scale to
        """
        # Create a color gradient
        gradient = np.linspace(0, 1, 100)
        gradient = np.vstack((gradient, gradient))
        
        # Create a small axes for the gradient
        scale_ax = ax.inset_axes([0.7, 0.1, 0.25, 0.1])
        
        # Plot the gradient
        scale_ax.imshow(gradient, aspect='auto', cmap=self._create_confidence_colormap())
        
        # Add labels
        scale_ax.text(0, -1, "Low Confidence", ha='left', va='top', fontsize=8)
        scale_ax.text(99, -1, "High Confidence", ha='right', va='top', fontsize=8)
        
        # Hide axes
        scale_ax.set_xticks([])
        scale_ax.set_yticks([])
        scale_ax.spines['top'].set_visible(False)
        scale_ax.spines['right'].set_visible(False)
        scale_ax.spines['bottom'].set_visible(False)
        scale_ax.spines['left'].set_visible(False)
    
    def _create_confidence_colormap(self):
        """
        Create a custom colormap for confidence values.
        Red -> Yellow -> Green
        
        Returns:
            matplotlib.colors.LinearSegmentedColormap: The colormap
        """
        # Define colors for the gradient
        colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]  # Red -> Yellow -> Green
        
        # Create a custom colormap
        cmap = LinearSegmentedColormap.from_list('confidence', colors, N=100)
        return cmap
