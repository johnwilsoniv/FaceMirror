import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

def load_and_merge_data(au_file, expert_file):
    """Load and merge the AU data with expert gradings."""
    # Load data
    au_data = pd.read_csv(au_file)
    expert_grades = pd.read_csv(expert_file)
    
    # Merge datasets
    merged_data = pd.merge(
        au_data, 
        expert_grades, 
        left_on='Patient ID', 
        right_on='Patient', 
        how='inner'
    )
    
    print(f"Merged data contains {merged_data.shape[0]} patients with {merged_data.shape[1]} columns")
    return merged_data

def create_features(df, expressions=['ES', 'ET', 'BS', 'SE'], 
                   aus=['AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14', 'AU45']):
    """Extract features for modeling, focusing on specified expressions and AUs."""
    
    # Create empty dataframe for features
    features_df = pd.DataFrame({
        'patient_id': df['Patient ID'],
        'left_midface_paralysis': df['Paralysis - Left Mid Face'],
        'right_midface_paralysis': df['Paralysis - Right Mid Face']
    })
    
    # Clean up categorical values and ensure consistent formatting
    features_df['left_midface_paralysis'] = features_df['left_midface_paralysis'].str.strip()
    features_df['right_midface_paralysis'] = features_df['right_midface_paralysis'].str.strip()
    
    # Map values to ensure consistent categories
    paralysis_map = {
        'None': 'None',
        'Partial': 'Partial',
        'Complete': 'Complete',
        # Add any other variations that might exist in the data
        'none': 'None',
        'partial': 'Partial',
        'complete': 'Complete'
    }
    
    features_df['left_midface_paralysis'] = features_df['left_midface_paralysis'].map(paralysis_map).fillna('None')
    features_df['right_midface_paralysis'] = features_df['right_midface_paralysis'].map(paralysis_map).fillna('None')
    
    # Convert to numeric categories
    features_df['left_midface_numeric'] = features_df['left_midface_paralysis'].map({
        'None': 0, 'Partial': 1, 'Complete': 2
    })
    
    features_df['right_midface_numeric'] = features_df['right_midface_paralysis'].map({
        'None': 0, 'Partial': 1, 'Complete': 2
    })
    
    # Print distribution of target classes
    print("\nTarget distribution (Left Midface):")
    print(features_df['left_midface_paralysis'].value_counts())
    print("\nTarget distribution (Right Midface):")
    print(features_df['right_midface_paralysis'].value_counts())
    
    # Extract both normalized and raw AU values for each expression
    for expr in expressions:
        for au in aus:
            # Raw values
            left_col = f"{expr}_Left {au}_r"
            right_col = f"{expr}_Right {au}_r"
            
            if left_col in df.columns and right_col in df.columns:
                # Convert to numeric, handling any non-numeric values
                df[left_col] = pd.to_numeric(df[left_col], errors='coerce')
                df[right_col] = pd.to_numeric(df[right_col], errors='coerce')
                
                # Raw left and right values
                features_df[f"{expr}_{au}_left"] = df[left_col]
                features_df[f"{expr}_{au}_right"] = df[right_col]
                
                # Asymmetry (left - right)
                features_df[f"{expr}_{au}_asymmetry"] = df[left_col] - df[right_col]
                
                # Absolute asymmetry
                features_df[f"{expr}_{au}_abs_asymmetry"] = abs(df[left_col] - df[right_col])
                
                # Ratio (left / right) - handle division by zero
                features_df[f"{expr}_{au}_ratio"] = df[left_col] / df[right_col].replace(0, np.nan)
            
            # Normalized values
            left_norm_col = f"{expr}_Left {au}_r (Normalized)"
            right_norm_col = f"{expr}_Right {au}_r (Normalized)"
            
            if left_norm_col in df.columns and right_norm_col in df.columns:
                # Convert to numeric, handling any non-numeric values
                df[left_norm_col] = pd.to_numeric(df[left_norm_col], errors='coerce')
                df[right_norm_col] = pd.to_numeric(df[right_norm_col], errors='coerce')
                
                # Normalized left and right values
                features_df[f"{expr}_{au}_left_norm"] = df[left_norm_col]
                features_df[f"{expr}_{au}_right_norm"] = df[right_norm_col]
                
                # Normalized asymmetry
                features_df[f"{expr}_{au}_asymmetry_norm"] = df[left_norm_col] - df[right_norm_col]
                
                # Normalized absolute asymmetry
                features_df[f"{expr}_{au}_abs_asymmetry_norm"] = abs(df[left_norm_col] - df[right_norm_col])
    
    # Fill NaN values with column means
    numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
    features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].mean())
    
    return features_df

def analyze_feature_importance(features_df, target_col='left_midface_numeric', n_top_features=20):
    """Analyze feature importance for predicting midface paralysis."""
    # Prepare X and y
    X = features_df.drop(['patient_id', 'left_midface_paralysis', 'right_midface_paralysis', 
                          'left_midface_numeric', 'right_midface_numeric'], axis=1)
    y = features_df[target_col].astype(int)  # Ensure target is integer
    
    # Check for empty dataframe or missing values
    if X.empty:
        print("Error: Feature dataframe is empty")
        return None
    
    # Check for any remaining NaN values
    if X.isnull().any().any():
        print("Warning: NaN values found in features, filling with 0")
        X = X.fillna(0)
    
    # Check unique values in target
    unique_vals = y.unique()
    print(f"Unique values in target: {unique_vals}")
    
    if len(unique_vals) < 2:
        print("Error: Target has fewer than 2 classes")
        return None
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Use a Random Forest to get initial feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_scaled, y)
    
    # Get feature importance
    importances = rf.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Perform permutation importance for more reliable estimates
    perm_importance = permutation_importance(rf, X_scaled, y, n_repeats=10, random_state=42)
    
    # Combine with feature_importance DataFrame
    feature_importance['Permutation_Importance'] = perm_importance.importances_mean
    feature_importance = feature_importance.sort_values('Permutation_Importance', ascending=False)
    
    # Print top features
    print(f"\nTop {n_top_features} features for predicting {target_col}:")
    for i, (idx, row) in enumerate(feature_importance.head(n_top_features).iterrows()):
        print(f"{i+1}. {row['Feature']}: {row['Permutation_Importance']:.4f}")
    
    # Calculate average importance by expression and AU
    expressions = ['ES', 'ET', 'BS', 'SE']
    aus = ['AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14', 'AU45']
    
    # Average importance by expression
    expr_importance = {}
    for expr in expressions:
        expr_features = feature_importance[feature_importance['Feature'].str.contains(expr)]
        if not expr_features.empty:
            expr_importance[expr] = expr_features['Permutation_Importance'].mean()
    
    print("\nAverage feature importance by expression:")
    for expr, importance in sorted(expr_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{expr}: {importance:.4f}")
    
    # Average importance by AU
    au_importance = {}
    for au in aus:
        au_features = feature_importance[feature_importance['Feature'].str.contains(au)]
        if not au_features.empty:
            au_importance[au] = au_features['Permutation_Importance'].mean()
    
    print("\nAverage feature importance by AU:")
    for au, importance in sorted(au_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{au}: {importance:.4f}")
    
    # Analyze asymmetry vs. raw features
    asymmetry_features = feature_importance[feature_importance['Feature'].str.contains('asymmetry')]
    raw_features = feature_importance[~feature_importance['Feature'].str.contains('asymmetry') & 
                                     (feature_importance['Feature'].str.contains('_left') | 
                                      feature_importance['Feature'].str.contains('_right'))]
    
    print("\nComparison of asymmetry vs. raw features:")
    print(f"Average importance of asymmetry features: {asymmetry_features['Permutation_Importance'].mean():.4f}")
    print(f"Average importance of raw features: {raw_features['Permutation_Importance'].mean():.4f}")
    
    # Normalized vs. non-normalized
    norm_features = feature_importance[feature_importance['Feature'].str.contains('_norm')]
    non_norm_features = feature_importance[~feature_importance['Feature'].str.contains('_norm')]
    
    print("\nComparison of normalized vs. non-normalized features:")
    print(f"Average importance of normalized features: {norm_features['Permutation_Importance'].mean():.4f}")
    print(f"Average importance of non-normalized features: {non_norm_features['Permutation_Importance'].mean():.4f}")
    
    return feature_importance

def train_model(features_df, target_col='left_midface_numeric', top_n_features=15, model_type='xgboost'):
    """Train and evaluate a model to predict midface paralysis."""
    # Prepare X and y
    X = features_df.drop(['patient_id', 'left_midface_paralysis', 'right_midface_paralysis', 
                          'left_midface_numeric', 'right_midface_numeric'], axis=1)
    y = features_df[target_col].astype(int)  # Ensure target is integer
    
    # Check for empty dataframe or missing values
    if X.empty:
        print("Error: Feature dataframe is empty")
        return None
    
    # Check for any remaining NaN values
    if X.isnull().any().any():
        print("Warning: NaN values found in features, filling with 0")
        X = X.fillna(0)
    
    # Check unique values in target
    unique_vals = np.sort(y.unique())
    print(f"Unique values in target: {unique_vals}")
    n_classes = len(unique_vals)
    
    if n_classes < 2:
        print("Error: Target has fewer than 2 classes")
        return None
    
    # Initial feature importance to select top features
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    perm_importance = permutation_importance(rf, X, y, n_repeats=5, random_state=42)
    
    # Select top features
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': perm_importance.importances_mean
    }).sort_values('Importance', ascending=False)
    
    top_features = feature_importance.head(top_n_features)['Feature'].tolist()
    print(f"Training model using top {top_n_features} features:")
    for i, feature in enumerate(top_features):
        print(f"{i+1}. {feature}")
    
    X_selected = X[top_features]
    
    # Define model based on number of classes
    if model_type == 'logistic':
        if n_classes == 2:
            model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        else:
            model = LogisticRegression(max_iter=1000, class_weight='balanced', 
                                      multi_class='multinomial', solver='lbfgs', random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    else:  # default to xgboost
        if n_classes == 2:
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=3,  # Adjust for class imbalance
                random_state=42
            )
        else:
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=n_classes,
                eval_metric='mlogloss',
                use_label_encoder=False,
                scale_pos_weight=3,  # Adjust for class imbalance
                random_state=42
            )
    
    # Create pipeline with SMOTE for class imbalance (only if more than one sample per class)
    pipeline_steps = [('scaler', StandardScaler())]
    
    # Only add SMOTE if we have at least 6 samples for each class
    class_counts = y.value_counts()
    min_samples = class_counts.min()
    
    if min_samples >= 6:
        pipeline_steps.append(('smote', SMOTE(random_state=42)))
    else:
        print(f"Warning: Insufficient samples for SMOTE (min class has {min_samples} samples)")
    
    pipeline_steps.append(('model', model))
    pipeline = ImbPipeline(pipeline_steps)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_selected, y, cv=cv, scoring='balanced_accuracy')
    
    print(f"\nCross-validation balanced accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train on full dataset for final evaluation
    pipeline.fit(X_selected, y)
    y_pred = pipeline.predict(X_selected)
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Define labels based on number of classes
    if n_classes == 2:
        labels = ['None', 'Affected']
    else:
        labels = ['None', 'Partial', 'Complete']
        
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels[:n_classes], 
                yticklabels=labels[:n_classes])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Return the trained pipeline and selected features
    return {
        'pipeline': pipeline,
        'selected_features': top_features,
        'cv_scores': cv_scores,
        'classification_report': classification_report(y, y_pred, output_dict=True)
    }

def analyze_specific_features(df, target_col='left_midface_numeric'):
    """Analyze specific features of interest (AU45 and AU07)."""
    expressions = ['ES', 'ET', 'BS', 'SE']
    
    # Verify target column exists and has expected values
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in dataframe")
        return
    
    # Ensure target is numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Warning: Converting {target_col} to numeric")
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    # Group by target
    target_values = sorted(df[target_col].dropna().unique())
    groups = {}
    for value in target_values:
        groups[value] = df[df[target_col] == value]
    
    # Map numeric values back to labels for clarity
    paralysis_map = {0: 'None', 1: 'Partial', 2: 'Complete'}
    
    # Analyze AU45 (Blink)
    print("\n--- Analysis of AU45 (Blink) Features ---")
    for expr in expressions:
        asymmetry_col = f"{expr}_AU45_asymmetry"
        if asymmetry_col in df.columns:
            print(f"\n{expr} expression:")
            for group_val, group_df in groups.items():
                if pd.api.types.is_numeric_dtype(group_df[asymmetry_col]):
                    mean_val = group_df[asymmetry_col].mean()
                    std_val = group_df[asymmetry_col].std()
                    print(f"{paralysis_map.get(group_val, group_val)}: {mean_val:.4f} ± {std_val:.4f}")
                else:
                    print(f"Warning: {asymmetry_col} is not numeric")
    
    # Analyze AU07 (Lid Tightener)
    print("\n--- Analysis of AU07 (Lid Tightener) Features ---")
    for expr in expressions:
        asymmetry_col = f"{expr}_AU07_asymmetry"
        if asymmetry_col in df.columns:
            print(f"\n{expr} expression:")
            for group_val, group_df in groups.items():
                if pd.api.types.is_numeric_dtype(group_df[asymmetry_col]):
                    mean_val = group_df[asymmetry_col].mean()
                    std_val = group_df[asymmetry_col].std()
                    print(f"{paralysis_map.get(group_val, group_val)}: {mean_val:.4f} ± {std_val:.4f}")
                else:
                    print(f"Warning: {asymmetry_col} is not numeric")

def create_asymmetry_plot(df, au='AU45', expressions=['ES', 'ET', 'BS', 'SE'], 
                        target_col='left_midface_numeric'):
    """Create a plot showing asymmetry by paralysis group for a specific AU."""
    # Ensure target is numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    # Map numeric values to labels 
    paralysis_map = {0: 'None', 1: 'Partial', 2: 'Complete'}
    df['paralysis_label'] = df[target_col].map(paralysis_map)
    
    plt.figure(figsize=(12, 8))
    
    for i, expr in enumerate(expressions):
        asymmetry_col = f"{expr}_{au}_asymmetry"
        if asymmetry_col in df.columns:
            plt.subplot(2, 2, i+1)
            sns.boxplot(x='paralysis_label', y=asymmetry_col, data=df, palette='Blues')
            plt.title(f"{expr} - {au} Asymmetry")
            plt.xlabel('Midface Paralysis')
            plt.ylabel('Left-Right Asymmetry')
    
    plt.tight_layout()
    plt.suptitle(f"{au} Asymmetry by Expression and Paralysis Group", y=1.02, fontsize=16)

def main():
    # File paths
    au_file = 'combined_results.csv'
    expert_file = 'FPRS FP Key.csv'
    
    # Load and merge data
    merged_data = load_and_merge_data(au_file, expert_file)
    
    # Create features
    features_df = create_features(merged_data)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(features_df, target_col='left_midface_numeric')
    
    # Analyze specific features of interest
    analyze_specific_features(features_df)
    
    # Train model
    model_results = train_model(features_df, target_col='left_midface_numeric', top_n_features=15)
    
    # Create plots for important AUs
    create_asymmetry_plot(features_df, au='AU45')
    create_asymmetry_plot(features_df, au='AU07')
    
    print("\nAnalysis complete!")
    return {
        'feature_importance': feature_importance,
        'model_results': model_results
    }

if __name__ == "__main__":
    results = main()
