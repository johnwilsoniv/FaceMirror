import pandas as pd
import numpy as np

# Load the datasets
fprs_key = pd.read_csv("FPRS FP Key.csv")
combined_results = pd.read_csv("combined_results.csv")

# Preprocessing Expert graded Key
fprs_key.rename(columns={"Patient": "Patient ID"}, inplace=True)


# Define a function to convert FPRS grading to numerical values
def grading_to_numerical(grading):
    if grading == 'None':
        return 0  # No paralysis
    elif grading == 'Partial':
        return 1  # Partial paralysis
    elif grading == 'Complete':
        return 2  # Complete paralysis
    else:
        return np.nan  # Handle unexpected values


# Apply the conversion to the relevant columns in the FPRS key
paralysis_cols = [
    "Paralysis - Left Upper Face",
    "Paralysis - Left Mid Face",
    "Paralysis - Left Lower Face",
    "Paralysis - Right Upper Face",
    "Paralysis - Right Mid Face",
    "Paralysis - Right Lower Face",
]
for col in paralysis_cols:
    fprs_key[col] = fprs_key[col].apply(grading_to_numerical)

# Create presence/absence of paralysis (1/0) rather than degree of paralysis
for col in paralysis_cols:
    fprs_key[col] = fprs_key[col].apply(lambda x: 0 if x == 0 else 1)

# Merge the datasets
merged_data = pd.merge(combined_results, fprs_key, on="Patient ID", how="inner")

# Define the face parts and sides
face_parts = ["Upper Face", "Mid Face", "Lower Face"]
sides = ["Left", "Right"]

# Create the predicted columns with correct names
for side in sides:
    for face_part in face_parts:
        predicted_col_name = f"predicted_{side.lower()}_{face_part.lower().replace(' ', '_')}"
        col_name = f"{side} {face_part} Paralysis"

        # Determine Paralysis in terms of True or False
        merged_data[predicted_col_name] = merged_data[col_name].apply(lambda x: 0 if x == 'None' else 1)


# Calculate accuracy, precision, and recall
def calculate_metrics(df, side, face_part):
    true_col = f"Paralysis - {side} {face_part}"
    predicted_col = f"predicted_{side.lower()}_{face_part.lower().replace(' ', '_')}"

    TP = df[(df[true_col] == 1) & (df[predicted_col] == 1)].shape[0]
    TN = df[(df[true_col] == 0) & (df[predicted_col] == 0)].shape[0]
    FP = df[(df[true_col] == 0) & (df[predicted_col] == 1)].shape[0]
    FN = df[(df[true_col] == 1) & (df[predicted_col] == 0)].shape[0]

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return accuracy, precision, recall


# Calculate and print results for each area
results = {}

for side in sides:
    for face_part in face_parts:
        accuracy, precision, recall = calculate_metrics(merged_data, side, face_part)
        results[f"{side} {face_part}"] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall
        }

# Print the results in a structured format
print("Paralysis Detection Accuracy:")
for area, metrics in results.items():
    print(f"\n{area}:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")