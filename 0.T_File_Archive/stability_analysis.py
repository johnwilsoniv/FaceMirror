import numpy as np
from pathlib import Path

def analyze_midline_stability(self, log_path):
    """Analyze midline stability data from the log file"""
    if not Path(log_path).exists():
        print(f"Log file not found: {log_path}")
        return
    
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Load the data
        df = pd.read_csv(log_path)
        
        # Calculate frame-to-frame changes
        df['total_position_delta'] = np.sqrt(
            df['glabella_delta_x']**2 + df['glabella_delta_y']**2 +
            df['chin_delta_x']**2 + df['chin_delta_y']**2
        )
        
        # Calculate head movement metrics
        df['total_head_movement'] = np.sqrt(
            df['pitch_delta']**2 + df['yaw_delta']**2 + df['roll_delta']**2
        )
        
        # Calculate midline angle delta
        df['midline_angle_delta'] = df['midline_angle'].diff().abs()
        
        # Mark method changes
        df['method_change'] = df['method'] != df['method'].shift(1)
        
        # Create stability analysis plots
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot 1: Midline position changes
        axes[0].plot(df['frame'], df['total_position_delta'], 'b-', label='Position Change')
        axes[0].set_title('Midline Position Stability')
        axes[0].set_ylabel('Pixel Change')
        axes[0].grid(True)
        axes[0].legend()
        
        # Plot 2: Head pose and midline angle
        axes[1].plot(df['frame'], df['roll'], 'r-', label='Head Roll')
        axes[1].plot(df['frame'], df['midline_angle'], 'b-', label='Midline Angle')
        axes[1].set_title('Head Pose vs Midline Angle')
        axes[1].set_ylabel('Angle (degrees)')
        axes[1].grid(True)
        axes[1].legend()
        
        # Plot 3: Method changes and quality
        sc = axes[2].scatter(df['frame'], df['midline_angle_delta'], 
                          c=df['method'].map({'standard': 0, 'RANSAC': 1, 'fallback': 2}),
                          cmap='viridis', s=10, alpha=0.7)
        # Highlight method changes
        method_changes = df[df['method_change']]
        axes[2].scatter(method_changes['frame'], method_changes['midline_angle_delta'],
                     color='red', s=30, marker='x')
        
        axes[2].set_title('Midline Angle Changes and Method Transitions')
        axes[2].set_ylabel('Angle Delta (degrees)')
        axes[2].set_xlabel('Frame')
        axes[2].grid(True)
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=sc.cmap(0), markersize=8, label='Standard'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=sc.cmap(0.5), markersize=8, label='RANSAC'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=sc.cmap(1), markersize=8, label='Fallback'),
            plt.Line2D([0], [0], marker='x', color='red', markersize=8, label='Method Change')
        ]
        axes[2].legend(handles=legend_elements)
        
        # Add overall statistics
        stat_text = (
            f"Total Frames: {len(df)}\n"
            f"Avg Position Delta: {df['total_position_delta'].mean():.2f} px\n"
            f"Avg Angle Delta: {df['midline_angle_delta'].mean():.2f}°\n"
            f"Method Changes: {df['method_change'].sum()}\n"
            f"Standard Method: {(df['method'] == 'standard').mean()*100:.1f}%\n"
            f"RANSAC Method: {(df['method'] == 'RANSAC').mean()*100:.1f}%\n"
            f"Fallback Method: {(df['method'] == 'fallback').mean()*100:.1f}%"
        )
        
        plt.figtext(0.02, 0.02, stat_text, fontsize=10, 
                  bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the figure
        output_path = str(Path(log_path).with_suffix('.png'))
        plt.savefig(output_path)
        plt.close()
        
        print(f"Stability analysis saved to: {output_path}")
        
        # Create a detailed summary
        summary = {
            'total_frames': len(df),
            'mean_position_delta': df['total_position_delta'].mean(),
            'max_position_delta': df['total_position_delta'].max(),
            'method_changes': df['method_change'].sum(),
            'standard_method_percent': (df['method'] == 'standard').mean()*100,
            'ransac_method_percent': (df['method'] == 'RANSAC').mean()*100,
            'fallback_method_percent': (df['method'] == 'fallback').mean()*100,
            'mean_head_movement': df['total_head_movement'].mean(),
            'stability_rating': 'Unknown',
            'method_stability': 'Unknown'
        }
        
        # Add stability ratings
        if summary['mean_position_delta'] < 1:
            summary['stability_rating'] = 'Excellent'
        elif summary['mean_position_delta'] < 2:
            summary['stability_rating'] = 'Good'
        elif summary['mean_position_delta'] < 5:
            summary['stability_rating'] = 'Fair'
        else:
            summary['stability_rating'] = 'Poor'
            
        method_changes_per_frame = summary['method_changes'] / summary['total_frames']
        if method_changes_per_frame < 0.01:
            summary['method_stability'] = 'Excellent'
        elif method_changes_per_frame < 0.05:
            summary['method_stability'] = 'Good'
        elif method_changes_per_frame < 0.1:
            summary['method_stability'] = 'Fair'
        else:
            summary['method_stability'] = 'Poor'
        
        # Print summary
        print("\nStability Analysis Summary:")
        print(f"- Total Frames: {summary['total_frames']}")
        print(f"- Average Midline Movement: {summary['mean_position_delta']:.2f} pixels")
        print(f"- Maximum Midline Movement: {summary['max_position_delta']:.2f} pixels")
        print(f"- Method Changes: {summary['method_changes']} ({method_changes_per_frame*100:.1f}%)")
        print(f"- Method Usage: Standard {summary['standard_method_percent']:.1f}%, "
             f"RANSAC {summary['ransac_method_percent']:.1f}%, "
             f"Fallback {summary['fallback_method_percent']:.1f}%")
        print(f"- Overall Stability Rating: {summary['stability_rating']}")
        print(f"- Method Stability Rating: {summary['method_stability']}")
        
        return summary
        
    except Exception as e:
        print(f"Error analyzing stability data: {str(e)}")
        return None
