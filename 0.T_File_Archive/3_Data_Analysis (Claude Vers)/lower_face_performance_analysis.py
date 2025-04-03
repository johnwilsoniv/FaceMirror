"""
Analysis script to evaluate facial paralysis detection performance.
Compares model predictions against expert labels.
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Configure logging
os.makedirs('logs', exist_ok=True)
logger = logging.getLogger(__name__)

def analyze_performance(results_file='combined_results.csv', expert_file='FPRS FP Key.csv', 
                      output_dir='analysis_results'):
    """
    Analyze model performance against expert labels.
    
    Args:
        results_file (str): Path to combined results CSV
        expert_file (str): Path to expert labels CSV
        output_dir (str): Directory to save analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'analysis.log')),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Starting performance analysis")
    
    try:
        # Load data
        logger.info(f"Loading results from {results_file}")
        results_df = pd.read_csv(results_file)
        
        logger.info(f"Loading expert labels from {expert_file}")
        expert_df = pd.read_csv(expert_file)
        
        # Rename expert columns
        expert_df = expert_df.rename(columns={
            'Patient': 'Patient ID',
            'Paralysis - Left Lower Face': 'Expert_Left_Lower_Face',
            'Paralysis - Right Lower Face': 'Expert_Right_Lower_Face'
        })
        
        # Merge datasets
        merged_df = pd.merge(results_df, expert_df, on='Patient ID', how='inner')
        logger.info(f"Merged data contains {len(merged_df)} patients")
        
        # Analyze lower face paralysis
        analyze_zone(merged_df, 'Lower Face', 
                    'Left Lower Face Paralysis', 'Expert_Left_Lower_Face',
                    'Right Lower Face Paralysis', 'Expert_Right_Lower_Face',
                    output_dir)
        
        logger.info("Performance analysis complete")
        
    except Exception as e:
        logger.error(f"Error in performance analysis: {str(e)}", exc_info=True)

def analyze_zone(data, zone_name, left_col, left_expert_col, right_col, right_expert_col, output_dir):
    """
    Analyze model performance for a specific facial zone.
    
    Args:
        data (pandas.DataFrame): Merged data
        zone_name (str): Name of facial zone
        left_col (str): Column with left side predictions
        left_expert_col (str): Column with left side expert labels
        right_col (str): Column with right side predictions
        right_expert_col (str): Column with right side expert labels
        output_dir (str): Directory to save analysis results
    """
    logger.info(f"Analyzing {zone_name} performance")
    
    # Create analysis dataframe
    left_analysis = data[[left_col, left_expert_col]].copy()
    right_analysis = data[[right_col, right_expert_col]].copy()
    
    left_analysis['Side'] = 'Left'
    right_analysis['Side'] = 'Right'
    
    left_analysis.columns = ['Prediction', 'Expert', 'Side']
    right_analysis.columns = ['Prediction', 'Expert', 'Side']
    
    # Standardize categories and ensure string type
    categories = ['None', 'Partial', 'Complete']
    
    # Force all values to be strings and standardize labels
    def standardize_labels(val):
        if val is None or pd.isna(val):
            return 'None'
        val_str = str(val).strip()
        if val_str.lower() in ['none', 'no', 'n/a', '0', '0.0', 'normal']:
            return 'None'
        elif val_str.lower() in ['partial', 'mild', 'moderate', '1', '1.0']:
            return 'Partial'
        elif val_str.lower() in ['complete', 'severe', '2', '2.0']:
            return 'Complete'
        return 'None'  # Default 
    
    left_analysis['Prediction'] = left_analysis['Prediction'].apply(standardize_labels)
    left_analysis['Expert'] = left_analysis['Expert'].apply(standardize_labels)
    right_analysis['Prediction'] = right_analysis['Prediction'].apply(standardize_labels)
    right_analysis['Expert'] = right_analysis['Expert'].apply(standardize_labels)
    
    zone_df = pd.concat([left_analysis, right_analysis], ignore_index=True)
    
    # Print sample data for debugging
    logger.info(f"Sample data (first 5 rows):")
    logger.info(f"{zone_df.head().to_string()}")
    
    # Calculate confusion matrices
    # Left side
    try:
        left_cm = confusion_matrix(
            left_analysis['Expert'], 
            left_analysis['Prediction'],
            labels=categories
        )
        
        # Right side
        right_cm = confusion_matrix(
            right_analysis['Expert'], 
            right_analysis['Prediction'],
            labels=categories
        )
        
        # Combined
        combined_cm = confusion_matrix(
            zone_df['Expert'], 
            zone_df['Prediction'],
            labels=categories
        )
        
        # Generate classification reports
        left_report = classification_report(
            left_analysis['Expert'], 
            left_analysis['Prediction'],
            labels=categories,
            target_names=categories,
            output_dict=True,
            zero_division=0
        )
        
        right_report = classification_report(
            right_analysis['Expert'], 
            right_analysis['Prediction'],
            labels=categories,
            target_names=categories,
            output_dict=True,
            zero_division=0
        )
        
        combined_report = classification_report(
            zone_df['Expert'], 
            zone_df['Prediction'],
            labels=categories,
            target_names=categories,
            output_dict=True,
            zero_division=0
        )
        
        # Log results
        logger.info(f"Left {zone_name} accuracy: {left_report['accuracy']:.4f}")
        logger.info(f"Right {zone_name} accuracy: {right_report['accuracy']:.4f}")
        logger.info(f"Combined {zone_name} accuracy: {combined_report['accuracy']:.4f}")
        
        # Class-specific metrics for Partial paralysis (our focus)
        partial_metrics = {
            'Left': {
                'precision': left_report['Partial']['precision'],
                'recall': left_report['Partial']['recall'],
                'f1-score': left_report['Partial']['f1-score']
            },
            'Right': {
                'precision': right_report['Partial']['precision'],
                'recall': right_report['Partial']['recall'],
                'f1-score': right_report['Partial']['f1-score']
            },
            'Combined': {
                'precision': combined_report['Partial']['precision'],
                'recall': combined_report['Partial']['recall'],
                'f1-score': combined_report['Partial']['f1-score']
            }
        }
        
        logger.info(f"Partial paralysis metrics:")
        logger.info(f"Left - Precision: {partial_metrics['Left']['precision']:.4f}, " +
                    f"Recall: {partial_metrics['Left']['recall']:.4f}, " +
                    f"F1: {partial_metrics['Left']['f1-score']:.4f}")
        logger.info(f"Right - Precision: {partial_metrics['Right']['precision']:.4f}, " +
                    f"Recall: {partial_metrics['Right']['recall']:.4f}, " +
                    f"F1: {partial_metrics['Right']['f1-score']:.4f}")
        logger.info(f"Combined - Precision: {partial_metrics['Combined']['precision']:.4f}, " +
                    f"Recall: {partial_metrics['Combined']['recall']:.4f}, " +
                    f"F1: {partial_metrics['Combined']['f1-score']:.4f}")
        
        # Visualize confusion matrices
        visualize_confusion_matrix(left_cm, categories, f"Left {zone_name}", output_dir)
        visualize_confusion_matrix(right_cm, categories, f"Right {zone_name}", output_dir)
        visualize_confusion_matrix(combined_cm, categories, f"Combined {zone_name}", output_dir)
        
        # Error analysis
        perform_error_analysis(zone_df, output_dir, f"{zone_name}_errors")

        analyze_critical_errors(zone_df, output_dir)
        analyze_partial_errors(zone_df, output_dir)
        
    except Exception as e:
        logger.error(f"Error in performance metrics calculation: {str(e)}", exc_info=True)
        # Continue with basic error analysis even if metrics fail
        perform_error_analysis(zone_df, output_dir, f"{zone_name}_errors")

def visualize_confusion_matrix(cm, categories, title, output_dir):
    """
    Create and save confusion matrix visualization.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        categories (list): Category names
        title (str): Plot title
        output_dir (str): Directory to save visualization
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('Expert')
    plt.title(f"{title} Confusion Matrix")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}_confusion_matrix.png"))
    plt.close()

def perform_error_analysis(data, output_dir, filename):
    """
    Perform detailed error analysis.
    
    Args:
        data (pandas.DataFrame): Analysis data
        output_dir (str): Directory to save analysis results
        filename (str): Output filename
    """
    # Identify error cases
    error_cases = data[data['Prediction'] != data['Expert']].copy()
    
    # Calculate error patterns
    error_patterns = {}
    for _, row in error_cases.iterrows():
        pattern = f"{row['Expert']}_to_{row['Prediction']}"
        if pattern not in error_patterns:
            error_patterns[pattern] = 0
        error_patterns[pattern] += 1
    
    # Log error patterns
    logger.info(f"Error patterns:")
    for pattern, count in error_patterns.items():
        logger.info(f"{pattern}: {count} cases")
    
    # Save detailed error analysis to file
    with open(os.path.join(output_dir, f"{filename}.txt"), 'w') as f:
        f.write(f"Error Analysis\n")
        f.write(f"=============\n\n")
        f.write(f"Total errors: {len(error_cases)} out of {len(data)} ({len(error_cases)/len(data)*100:.2f}%)\n\n")
        
        f.write(f"Error patterns:\n")
        for pattern, count in error_patterns.items():
            f.write(f"{pattern}: {count} cases ({count/len(error_cases)*100:.2f}% of errors)\n")
        
        f.write("\nDetailed error cases:\n")
        for i, row in error_cases.iterrows():
            f.write(f"Case {i+1}: {row['Side']} side - Expert: {row['Expert']}, Predicted: {row['Prediction']}\n")


def analyze_critical_errors(data, output_dir):
    """
    Analyze critical error cases (None→Complete and Complete→None).

    Args:
        data (pandas.DataFrame): Analysis dataframe with predictions and expert labels
        output_dir (str): Directory to save analysis results
    """
    logger.info("Analyzing critical error cases...")

    # Identify the critical error cases
    critical_errors = data[
        ((data['Expert'] == 'None') & (data['Prediction'] == 'Complete')) |
        ((data['Expert'] == 'Complete') & (data['Prediction'] == 'None'))
        ]

    logger.info(f"Found {len(critical_errors)} critical errors")

    # Create detailed report
    with open(os.path.join(output_dir, "critical_errors_analysis.txt"), "w") as f:
        f.write("Critical Errors Analysis\n")
        f.write("=======================\n\n")

        for idx, error in critical_errors.iterrows():
            f.write(f"Error #{idx + 1}: {error['Expert']} misclassified as {error['Prediction']}\n")
            f.write(f"Side: {error['Side']}\n")

            # Analysis for this specific error
            f.write("\nPossible causes of misclassification:\n")

            if error['Expert'] == 'None' and error['Prediction'] == 'Complete':
                f.write("- Extreme asymmetry might trigger Complete classification\n")
                f.write("- Low overall activation may make comparisons unreliable\n")
                f.write("- Possible data outlier or anomaly in feature values\n")

            elif error['Expert'] == 'Complete' and error['Prediction'] == 'None':
                f.write("- Low asymmetry might lead to None classification\n")
                f.write("- Possible balanced paralysis on both sides\n")
                f.write("- Specialist classifier may have over-corrected\n")

            f.write("\n---\n\n")

        # Summary
        f.write("\nSummary of Critical Errors\n")
        f.write("======================\n\n")

        none_to_complete = len(critical_errors[(critical_errors['Expert'] == 'None') &
                                               (critical_errors['Prediction'] == 'Complete')])
        complete_to_none = len(critical_errors[(critical_errors['Expert'] == 'Complete') &
                                               (critical_errors['Prediction'] == 'None')])

        f.write(f"None→Complete errors: {none_to_complete}\n")
        f.write(f"Complete→None errors: {complete_to_none}\n")

    logger.info(f"Critical errors analysis saved to {os.path.join(output_dir, 'critical_errors_analysis.txt')}")


def analyze_partial_errors(data, output_dir):
    """
    Analyze partial error cases to ensure model tuning.

    Args:
        data (pandas.DataFrame): Analysis dataframe with predictions and expert labels
        output_dir (str): Directory to save analysis results
    """
    logger.info("Analyzing partial misclassification cases...")

    # Identify error cases involving Partial class
    partial_errors = data[
        ((data['Expert'] == 'Partial') & (data['Prediction'] != 'Partial')) |
        ((data['Prediction'] == 'Partial') & (data['Expert'] != 'Partial'))
        ]

    # Group by error type
    partial_to_complete = partial_errors[(partial_errors['Expert'] == 'Partial') &
                                         (partial_errors['Prediction'] == 'Complete')]

    partial_to_none = partial_errors[(partial_errors['Expert'] == 'Partial') &
                                     (partial_errors['Prediction'] == 'None')]

    none_to_partial = partial_errors[(partial_errors['Expert'] == 'None') &
                                     (partial_errors['Prediction'] == 'Partial')]

    complete_to_partial = partial_errors[(partial_errors['Expert'] == 'Complete') &
                                         (partial_errors['Prediction'] == 'Partial')]

    logger.info(f"Found {len(partial_errors)} partial misclassification cases:")
    logger.info(f"- Partial→Complete: {len(partial_to_complete)} cases")
    logger.info(f"- Partial→None: {len(partial_to_none)} cases")
    logger.info(f"- None→Partial: {len(none_to_partial)} cases")
    logger.info(f"- Complete→Partial: {len(complete_to_partial)} cases")

    # Create detailed report
    with open(os.path.join(output_dir, "partial_errors_analysis.txt"), "w") as f:
        f.write("Partial Misclassification Analysis\n")
        f.write("================================\n\n")

        # Analyze each type of error
        for error_type, error_cases in [
            ("Partial→Complete", partial_to_complete),
            ("Partial→None", partial_to_none),
            ("None→Partial", none_to_partial),
            ("Complete→Partial", complete_to_partial)
        ]:
            f.write(f"{error_type} Errors ({len(error_cases)} cases)\n")
            f.write(f"{'-' * (len(error_type) + 14)}\n\n")

            for idx, error in error_cases.iterrows():
                f.write(f"Error #{idx + 1}: Side: {error['Side']}\n")

                # Add specific insights based on error type
                if error_type == "Partial→Complete":
                    f.write("Likely causes: High asymmetry but less severe than expert assessment\n")
                    f.write("Potential solution: Increase confidence threshold for Complete classification\n")
                elif error_type == "Partial→None":
                    f.write("Likely causes: Low asymmetry detection, may need lower thresholds\n")
                    f.write("Potential solution: Decrease threshold for Partial classification\n")
                elif error_type == "None→Partial":
                    f.write("Likely causes: Asymmetry detected that expert considered normal variation\n")
                    f.write("Potential solution: Increase threshold for Partial classification\n")
                elif error_type == "Complete→Partial":
                    f.write("Likely causes: Asymmetry detected but not severe enough for Complete\n")
                    f.write("Potential solution: Decrease confidence threshold for Complete classification\n")

                f.write("\n")

            f.write("\n")

        # Summary of partial errors
        f.write("\nSummary of Partial Misclassifications\n")
        f.write("==================================\n\n")
        f.write(f"Total partial misclassifications: {len(partial_errors)}\n")
        f.write(f"Partial→Complete errors: {len(partial_to_complete)}\n")
        f.write(f"Partial→None errors: {len(partial_to_none)}\n")
        f.write(f"None→Partial errors: {len(none_to_partial)}\n")
        f.write(f"Complete→Partial errors: {len(complete_to_partial)}\n")

    logger.info(f"Partial errors analysis saved to {os.path.join(output_dir, 'partial_errors_analysis.txt')}")

if __name__ == "__main__":
    analyze_performance()