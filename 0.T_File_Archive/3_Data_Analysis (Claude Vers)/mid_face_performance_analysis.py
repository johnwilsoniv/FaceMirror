"""
Analysis script to evaluate mid face paralysis detection performance.
Compares model predictions against expert labels.
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from mid_face_config import LOG_DIR

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
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
            logging.FileHandler(os.path.join(output_dir, 'midface_analysis.log')),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Starting midface performance analysis")
    
    try:
        # Load data
        logger.info(f"Loading results from {results_file}")
        results_df = pd.read_csv(results_file)
        
        logger.info(f"Loading expert labels from {expert_file}")
        expert_df = pd.read_csv(expert_file)
        
        # Rename expert columns
        expert_df = expert_df.rename(columns={
            'Patient': 'Patient ID',
            'Paralysis - Left Mid Face': 'Expert_Left_Mid_Face',
            'Paralysis - Right Mid Face': 'Expert_Right_Mid_Face'
        })
        
        # Merge datasets
        merged_df = pd.merge(results_df, expert_df, on='Patient ID', how='inner')
        logger.info(f"Merged data contains {len(merged_df)} patients")
        
        # Analyze mid face paralysis
        analyze_zone(merged_df, 'Mid Face', 
                    'Left Mid Face Paralysis', 'Expert_Left_Mid_Face',
                    'Right Mid Face Paralysis', 'Expert_Right_Mid_Face',
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
        
        # Complete paralysis metrics
        complete_metrics = {
            'Left': {
                'precision': left_report['Complete']['precision'],
                'recall': left_report['Complete']['recall'],
                'f1-score': left_report['Complete']['f1-score']
            },
            'Right': {
                'precision': right_report['Complete']['precision'],
                'recall': right_report['Complete']['recall'],
                'f1-score': right_report['Complete']['f1-score']
            },
            'Combined': {
                'precision': combined_report['Complete']['precision'],
                'recall': combined_report['Complete']['recall'],
                'f1-score': combined_report['Complete']['f1-score']
            }
        }
        
        logger.info(f"Complete paralysis metrics:")
        logger.info(f"Left - Precision: {complete_metrics['Left']['precision']:.4f}, " +
                    f"Recall: {complete_metrics['Left']['recall']:.4f}, " +
                    f"F1: {complete_metrics['Left']['f1-score']:.4f}")
        logger.info(f"Right - Precision: {complete_metrics['Right']['precision']:.4f}, " +
                    f"Recall: {complete_metrics['Right']['recall']:.4f}, " +
                    f"F1: {complete_metrics['Right']['f1-score']:.4f}")
        logger.info(f"Combined - Precision: {complete_metrics['Combined']['precision']:.4f}, " +
                    f"Recall: {complete_metrics['Combined']['recall']:.4f}, " +
                    f"F1: {complete_metrics['Combined']['f1-score']:.4f}")
        
        # Visualize confusion matrices
        visualize_confusion_matrix(left_cm, categories, f"Left {zone_name}", output_dir)
        visualize_confusion_matrix(right_cm, categories, f"Right {zone_name}", output_dir)
        visualize_confusion_matrix(combined_cm, categories, f"Combined {zone_name}", output_dir)
        
        # Error analysis
        perform_error_analysis(zone_df, output_dir, f"{zone_name}_errors")

        analyze_critical_errors(zone_df, output_dir)
        analyze_subtle_paralysis_detection(zone_df, data, output_dir)
        
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
    with open(os.path.join(output_dir, "midface_critical_errors_analysis.txt"), "w") as f:
        f.write("Critical Errors Analysis - Midface\n")
        f.write("================================\n\n")

        for idx, error in critical_errors.iterrows():
            f.write(f"Error #{idx + 1}: {error['Expert']} misclassified as {error['Prediction']}\n")
            f.write(f"Side: {error['Side']}\n")

            # Analysis for this specific error
            f.write("\nPossible causes of misclassification:\n")

            if error['Expert'] == 'None' and error['Prediction'] == 'Complete':
                f.write("- AU45_r value might be abnormally low despite lack of paralysis\n")
                f.write("- Asymmetry between left and right sides might be coincidental\n")
                f.write("- Patient might have blinked unevenly during recording\n")
                f.write("- Rule-based detection might have triggered on asymmetry threshold\n")

            elif error['Expert'] == 'Complete' and error['Prediction'] == 'None':
                f.write("- AU45_r value might be deceptively normal despite paralysis\n")
                f.write("- Model may have missed subtle signs of incomplete eyelid closure\n")
                f.write("- ET/ES ratio may not have been distinctive enough\n")
                f.write("- May need to lower detection thresholds for increased sensitivity\n")

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
        
        # Recommendations
        f.write("\nRecommendations:\n")
        f.write("1. Adjust AU45_DETECTION_THRESHOLDS to better balance sensitivity and specificity\n")
        f.write("2. Consider incorporating more AU07_r features in the decision process\n")
        f.write("3. Add verification step for Complete classification using additional AUs\n")
        f.write("4. Improve subtle paralysis detection for Complete cases that appear normal\n")

    logger.info(f"Critical errors analysis saved to {os.path.join(output_dir, 'midface_critical_errors_analysis.txt')}")


def analyze_subtle_paralysis_detection(data, raw_data, output_dir):
    """
    Analyze how well the model detects subtle midface paralysis.
    Focus on cases where standard AU45 detection would fail.

    Args:
        data (pandas.DataFrame): Analysis dataframe with predictions and expert labels
        raw_data (pandas.DataFrame): Original data with AU values
        output_dir (str): Directory to save analysis results
    """
    logger.info("Analyzing subtle paralysis detection performance...")

    # We need to identify subtly paralyzed cases that have relatively normal AU45 values
    # Merge the information we need
    try:
        # Extract patient IDs and sides for identified paralysis cases
        paralysis_cases = data[(data['Expert'] == 'Partial') | (data['Expert'] == 'Complete')].copy()
        paralysis_cases['Patient_Side'] = paralysis_cases.apply(
            lambda row: f"{row.name}_{row['Side']}", axis=1
        )
        
        # Get AU45 values for the 'ET' (Close Eyes Tightly) action
        au_cols = [col for col in raw_data.columns if 'ET_' in col and 'AU45_r' in col]
        if not au_cols:
            logger.warning("Could not find ET AU45_r columns in raw data")
            return
            
        au_data = raw_data[['Patient ID'] + au_cols].copy()
        
        # Create a dataset of paralyzed sides with their AU values
        case_data = []
        for _, row in paralysis_cases.iterrows():
            patient_id = row.name
            side = row['Side']
            expert_label = row['Expert']
            predicted_label = row['Prediction']
            
            # Find AU values
            patient_row = raw_data[raw_data['Patient ID'] == patient_id]
            
            if len(patient_row) == 0:
                continue
                
            # Use the first matching row
            patient_row = patient_row.iloc[0]
            
            # Get AU values for this side
            side_col = f"ET_{side} AU45_r"
            opposite_side = 'Right' if side == 'Left' else 'Left'
            opposite_col = f"ET_{opposite_side} AU45_r"
            
            if side_col in patient_row and opposite_col in patient_row:
                au45_val = patient_row[side_col]
                au45_opp = patient_row[opposite_col]
                
                # Calculate asymmetry
                if au45_val == 0 and au45_opp == 0:
                    percent_diff = 0
                    ratio = 1.0
                else:
                    avg = (au45_val + au45_opp) / 2
                    percent_diff = abs(au45_val - au45_opp) / avg * 100 if avg > 0 else 0
                    ratio = min(au45_val, au45_opp) / max(au45_val, au45_opp, 0.001)
                
                # Record the case
                case_data.append({
                    'Patient ID': patient_id,
                    'Side': side,
                    'Expert Label': expert_label,
                    'Predicted Label': predicted_label,
                    'AU45_Value': au45_val,
                    'AU45_Opposite': au45_opp,
                    'Asymmetry_Percent': percent_diff,
                    'Ratio': ratio,
                    'Correctly_Detected': expert_label == predicted_label
                })
        
        # Create a DataFrame
        case_df = pd.DataFrame(case_data)
        
        if len(case_df) == 0:
            logger.warning("No paralysis cases with AU values found for analysis")
            return
            
        # Identify subtle cases (those with relatively normal AU45 values)
        subtle_cases = case_df[case_df['AU45_Value'] > 1.0].copy()  # AU45 > 1.0 is moderately active
        
        logger.info(f"Found {len(subtle_cases)} paralysis cases with normal-range AU45 values")
        
        # Analyze the detection rate for subtle cases
        subtle_correct = subtle_cases['Correctly_Detected'].mean() * 100
        logger.info(f"Detection rate for subtle paralysis: {subtle_correct:.2f}%")
        
        # Create report
        with open(os.path.join(output_dir, "midface_subtle_detection_analysis.txt"), "w") as f:
            f.write("Subtle Midface Paralysis Detection Analysis\n")
            f.write("=========================================\n\n")
            
            f.write(f"Total paralysis cases: {len(case_df)}\n")
            f.write(f"Paralysis cases with normal-range AU45 values (>1.0): {len(subtle_cases)}\n")
            f.write(f"Detection rate for subtle paralysis: {subtle_correct:.2f}%\n\n")
            
            f.write("Asymmetry statistics for subtle cases:\n")
            f.write(f"- Average asymmetry: {subtle_cases['Asymmetry_Percent'].mean():.2f}%\n")
            f.write(f"- Average ratio: {subtle_cases['Ratio'].mean():.4f}\n\n")
            
            f.write("Detection rate by severity:\n")
            for severity in ['Partial', 'Complete']:
                severity_cases = subtle_cases[subtle_cases['Expert Label'] == severity]
                if len(severity_cases) > 0:
                    detection_rate = severity_cases['Correctly_Detected'].mean() * 100
                    f.write(f"- {severity}: {detection_rate:.2f}% ({len(severity_cases)} cases)\n")
            
            f.write("\nDetailed case analysis:\n")
            for idx, case in subtle_cases.iterrows():
                f.write(f"\nCase {idx + 1}: {case['Patient ID']} ({case['Side']})\n")
                f.write(f"Expert: {case['Expert Label']}, Predicted: {case['Predicted Label']}\n")
                f.write(f"AU45 value: {case['AU45_Value']:.2f}, Opposite side: {case['AU45_Opposite']:.2f}\n")
                f.write(f"Asymmetry: {case['Asymmetry_Percent']:.2f}%, Ratio: {case['Ratio']:.4f}\n")
                f.write(f"Correctly detected: {'Yes' if case['Correctly_Detected'] else 'No'}\n")
                
                # Additional analysis for missed cases
                if not case['Correctly_Detected']:
                    f.write("Possible reasons for misclassification:\n")
                    if case['Expert Label'] == 'Partial' and case['Predicted Label'] == 'None':
                        f.write("- Partial paralysis with limited asymmetry\n")
                        f.write("- AU45 value appears normal but clinical evaluation shows partial impairment\n")
                    elif case['Expert Label'] == 'Complete' and case['Predicted Label'] in ['Partial', 'None']:
                        f.write("- Complete paralysis with unusually high AU45 value\n")
                        f.write("- Clinical evaluation shows full paralysis despite some movement detected\n")
            
            # Recommendations
            f.write("\n\nRecommendations for improving subtle paralysis detection:\n")
            f.write("1. Incorporate ES/ET action ratio more heavily in detection logic\n")
            f.write("2. Add secondary AU analysis (AU07_r) for borderline cases\n")
            f.write("3. Lower asymmetry thresholds for high AU45 value cases\n")
            f.write("4. Consider specific handling for cases with AU45 > 1.0 but clinical paralysis\n")
            f.write("5. Add temporal analysis to detect inconsistent eye closure patterns\n")
            
    except Exception as e:
        logger.error(f"Error in subtle paralysis analysis: {str(e)}", exc_info=True)

if __name__ == "__main__":
    analyze_performance()