import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr, ttest_ind

def analyze_midface_paralysis(fprs_key_path, combined_results_path):
    """
    Simple correlation analysis for facial Action Units and midface paralysis.
    Uses point-biserial correlation for binary outcomes.
    
    Parameters:
    -----------
    fprs_key_path : str
        Path to the FPRS FP Key.csv file
    combined_results_path : str
        Path to the combined_results.csv file
    """
    print("Loading data files...")
    
    # Load the datasets
    fprs_key = pd.read_csv(fprs_key_path)
    combined_results = pd.read_csv(combined_results_path)
    
    # Rename columns to match between datasets
    fprs_key = fprs_key.rename(columns={'Patient': 'Patient ID'})
    
    # Define the AUs and expressions of interest
    expressions = ['ES', 'ET', 'BS', 'SE']
    aus = ['06', '07', '09', '10', '45']
    
    # Generate feature column names
    feature_columns = []
    for expr in expressions:
        for au in aus:
            # Left side
            left_col = f"{expr}_Left AU{au}_r"
            left_col_norm = f"{expr}_Left AU{au}_r (Normalized)"
            
            if left_col in combined_results.columns:
                feature_columns.append(left_col)
            
            if left_col_norm in combined_results.columns:
                feature_columns.append(left_col_norm)
            
            # Right side
            right_col = f"{expr}_Right AU{au}_r"
            right_col_norm = f"{expr}_Right AU{au}_r (Normalized)"
            
            if right_col in combined_results.columns:
                feature_columns.append(right_col)
            
            if right_col_norm in combined_results.columns:
                feature_columns.append(right_col_norm)
    
    print(f"Found {len(feature_columns)} AU features.")
    
    # Merge datasets
    merged_data = pd.merge(combined_results, fprs_key, on='Patient ID')
    print(f"Merged dataset has {len(merged_data)} rows.")
    
    # Create asymmetry features (difference between left and right)
    asymmetry_columns = []
    for expr in expressions:
        for au in aus:
            left_col = f"{expr}_Left AU{au}_r"
            right_col = f"{expr}_Right AU{au}_r"
            
            if left_col in merged_data.columns and right_col in merged_data.columns:
                asym_col = f"{expr}_Asymmetry AU{au}_r"
                merged_data[asym_col] = merged_data[left_col] - merged_data[right_col]
                asymmetry_columns.append(asym_col)
    
    print(f"Created {len(asymmetry_columns)} asymmetry features.")
    
    # All feature columns
    all_features = feature_columns + asymmetry_columns
    
    # Handle missing values
    merged_data[all_features] = merged_data[all_features].fillna(merged_data[all_features].mean())
    
    # Print class distribution
    print("\nClass distribution:")
    print("Left Midface Paralysis:")
    print(merged_data['Paralysis - Left Mid Face'].value_counts())
    print("\nRight Midface Paralysis:")
    print(merged_data['Paralysis - Right Mid Face'].value_counts())
    
    # Create binary variables for correlation analysis
    merged_data['Left_Any_Paralysis'] = (merged_data['Paralysis - Left Mid Face'] != 'None').astype(int)
    merged_data['Right_Any_Paralysis'] = (merged_data['Paralysis - Right Mid Face'] != 'None').astype(int)
    
    # ------------------------------------
    # Correlation Analysis - LEFT Side
    # ------------------------------------
    print("\n=== LEFT MIDFACE PARALYSIS ANALYSIS ===")
    
    # Calculate correlations for each feature with left paralysis
    left_correlations = []
    
    for feature in all_features:
        # Skip if feature has no variance (constant)
        if merged_data[feature].std() == 0:
            continue
            
        # Calculate point-biserial correlation (correlation between continuous and binary variables)
        try:
            corr, p_value = pointbiserialr(merged_data['Left_Any_Paralysis'], merged_data[feature])
            left_correlations.append({
                'Feature': feature,
                'Correlation': corr,
                'P_Value': p_value
            })
        except Exception as e:
            print(f"Error calculating correlation for {feature}: {str(e)}")
    
    # Convert to DataFrame and sort by absolute correlation
    left_corr_df = pd.DataFrame(left_correlations)
    left_corr_df['Abs_Correlation'] = left_corr_df['Correlation'].abs()
    left_corr_df = left_corr_df.sort_values('Abs_Correlation', ascending=False)
    
    # Print top correlations for left side
    print("\nTop 10 features correlated with LEFT midface paralysis:")
    print(left_corr_df.head(10)[['Feature', 'Correlation', 'P_Value']])
    
    # ------------------------------------
    # Correlation Analysis - RIGHT Side
    # ------------------------------------
    print("\n=== RIGHT MIDFACE PARALYSIS ANALYSIS ===")
    
    # Calculate correlations for each feature with right paralysis
    right_correlations = []
    
    for feature in all_features:
        # Skip if feature has no variance
        if merged_data[feature].std() == 0:
            continue
            
        # Calculate point-biserial correlation
        try:
            corr, p_value = pointbiserialr(merged_data['Right_Any_Paralysis'], merged_data[feature])
            right_correlations.append({
                'Feature': feature,
                'Correlation': corr,
                'P_Value': p_value
            })
        except Exception as e:
            print(f"Error calculating correlation for {feature}: {str(e)}")
    
    # Convert to DataFrame and sort by absolute correlation
    right_corr_df = pd.DataFrame(right_correlations)
    right_corr_df['Abs_Correlation'] = right_corr_df['Correlation'].abs()
    right_corr_df = right_corr_df.sort_values('Abs_Correlation', ascending=False)
    
    # Print top correlations for right side
    print("\nTop 10 features correlated with RIGHT midface paralysis:")
    print(right_corr_df.head(10)[['Feature', 'Correlation', 'P_Value']])
    
    # ------------------------------------
    # Feature comparison between groups
    # ------------------------------------
    print("\n=== FEATURE COMPARISON BETWEEN GROUPS ===")
    
    # Get top 5 features for both sides
    top_left_features = left_corr_df.head(5)['Feature'].tolist()
    top_right_features = right_corr_df.head(5)['Feature'].tolist()
    
    # Calculate summary statistics for left side features
    print("\nLEFT Side Features:")
    for feature in top_left_features:
        # Normal group
        normal_values = merged_data[merged_data['Paralysis - Left Mid Face'] == 'None'][feature]
        # Partial group
        partial_values = merged_data[merged_data['Paralysis - Left Mid Face'] == 'Partial'][feature]
        # Complete group
        complete_values = merged_data[merged_data['Paralysis - Left Mid Face'] == 'Complete'][feature]
        
        print(f"\nFeature: {feature}")
        print(f"None (n={len(normal_values)}): mean={normal_values.mean():.4f}, median={normal_values.median():.4f}, std={normal_values.std():.4f}")
        print(f"Partial (n={len(partial_values)}): mean={partial_values.mean():.4f}, median={partial_values.median():.4f}, std={partial_values.std():.4f}")
        print(f"Complete (n={len(complete_values)}): mean={complete_values.mean():.4f}, median={complete_values.median():.4f}, std={complete_values.std():.4f}")
        
        # T-test between None and Any Paralysis
        paralysis_values = pd.concat([partial_values, complete_values])
        t_stat, p_val = ttest_ind(normal_values, paralysis_values, equal_var=False)
        print(f"T-test (None vs Any): t={t_stat:.4f}, p={p_val:.4f}")
    
    # Calculate summary statistics for right side features
    print("\nRIGHT Side Features:")
    for feature in top_right_features:
        # Normal group
        normal_values = merged_data[merged_data['Paralysis - Right Mid Face'] == 'None'][feature]
        # Partial group
        partial_values = merged_data[merged_data['Paralysis - Right Mid Face'] == 'Partial'][feature]
        # Complete group
        complete_values = merged_data[merged_data['Paralysis - Right Mid Face'] == 'Complete'][feature]
        
        print(f"\nFeature: {feature}")
        print(f"None (n={len(normal_values)}): mean={normal_values.mean():.4f}, median={normal_values.median():.4f}, std={normal_values.std():.4f}")
        print(f"Partial (n={len(partial_values)}): mean={partial_values.mean():.4f}, median={partial_values.median():.4f}, std={partial_values.std():.4f}")
        print(f"Complete (n={len(complete_values)}): mean={complete_values.mean():.4f}, median={complete_values.median():.4f}, std={complete_values.std():.4f}")
        
        # T-test between None and Any Paralysis
        paralysis_values = pd.concat([partial_values, complete_values])
        t_stat, p_val = ttest_ind(normal_values, paralysis_values, equal_var=False)
        print(f"T-test (None vs Any): t={t_stat:.4f}, p={p_val:.4f}")
    
    # ------------------------------------
    # Visualization
    # ------------------------------------
    try:
        # Create box plots for top features
        plt.figure(figsize=(15, 10))
        
        # Left side features
        for i, feature in enumerate(top_left_features[:3]):
            plt.subplot(2, 3, i+1)
            sns.boxplot(x='Paralysis - Left Mid Face', y=feature, data=merged_data)
            plt.title(f"{feature}")
            plt.xticks(rotation=45)
        
        # Right side features
        for i, feature in enumerate(top_right_features[:3]):
            plt.subplot(2, 3, i+4)
            sns.boxplot(x='Paralysis - Right Mid Face', y=feature, data=merged_data)
            plt.title(f"{feature}")
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('paralysis_boxplots.png')
        print("\nBox plots saved as 'paralysis_boxplots.png'")
        
        # Correlation heatmap for top features
        top_features = list(set(top_left_features + top_right_features))
        plt.figure(figsize=(12, 10))
        correlation_matrix = merged_data[top_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        print("Correlation heatmap saved as 'correlation_heatmap.png'")
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
    
    # Return results dictionary
    return {
        'left_top_features': top_left_features,
        'right_top_features': top_right_features,
        'left_correlations': left_corr_df,
        'right_correlations': right_corr_df
    }

# Run the analysis
if __name__ == "__main__":
    print("Running simple correlation analysis for midface paralysis...")
    
    # File paths - adjust if needed
    fprs_key_path = "FPRS FP Key.csv"
    combined_results_path = "combined_results.csv"
    
    results = analyze_midface_paralysis(fprs_key_path, combined_results_path)
    
    print("\n=== SUMMARY OF KEY FINDINGS ===")
    
    print("\nTop 5 predictive features for LEFT midface paralysis:")
    for i, row in results['left_correlations'].head(5).iterrows():
        sign = "+" if row['Correlation'] > 0 else "-"
        sig = "significant" if row['P_Value'] < 0.05 else "not significant"
        print(f"{i+1}. {row['Feature']}: r = {row['Correlation']:.4f} ({sign}) ({sig}, p = {row['P_Value']:.4f})")
    
    print("\nTop 5 predictive features for RIGHT midface paralysis:")
    for i, row in results['right_correlations'].head(5).iterrows():
        sign = "+" if row['Correlation'] > 0 else "-"
        sig = "significant" if row['P_Value'] < 0.05 else "not significant"
        print(f"{i+1}. {row['Feature']}: r = {row['Correlation']:.4f} ({sign}) ({sig}, p = {row['P_Value']:.4f})")
    
    print("\nRecommended features for paralysis detection model:")
    print("1. For LEFT midface paralysis:")
    for i, feature in enumerate(results['left_top_features'][:3], 1):
        print(f"   {i}. {feature}")
    
    print("2. For RIGHT midface paralysis:")
    for i, feature in enumerate(results['right_top_features'][:3], 1):
        print(f"   {i}. {feature}")
    
    print("\nAnalysis complete!")
