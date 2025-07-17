import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Motion analysis CSVs from different datasets
paths = {
    "Assembly": "../results/Assembly_motion_analysis_results.csv",
    "G-AI-HMS": "../results/G-AI-HMS_motion_analysis_results.csv",
    "Near Duplicate": "../results/Near Duplicate_motion_analysis_results.csv",
    "UCF Sports Action": "../results/UCF Sports Action_motion_analysis_results.csv",
    "UCF50": "../results/UCF50_motion_analysis_results.csv"
}

# Load and tag datasets with its source name
dataframes = []
for name, path in paths.items():
    df = pd.read_csv(path)
    df["source_dataset"] = name
    dataframes.append(df)

# Join all datasets into one df
all_data = pd.concat(dataframes, ignore_index=True)

# Motion-based features for classification
features = ['RMSD', 'mean_motion', 'std_motion', 'UCL']

# Get rid of rows with missing feature values
all_data.dropna(subset=features, inplace=True)

# Leave out Assembly dataset for training; use the rest for testing
assembly_df = all_data[all_data["source_dataset"] == "Assembly"].copy()
other_df = all_data[all_data["source_dataset"] != "Assembly"].copy()

# Binary warning label based on the median number of warnings in Assembly
assembly_threshold = assembly_df["warnings"].median()
assembly_df["warning_label"] = (assembly_df["warnings"] > assembly_threshold).astype(int)

# Split Assembly data into training and validation sets
X = assembly_df[features]
y = assembly_df["warning_label"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest classifier with balanced class weights
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)

print("== Adaptive Threshold Results ==")

# Model evaluation on each dataset using that dataset’s own median-based warning threshold
for dataset_name in all_data["source_dataset"].unique():
    df = all_data[all_data["source_dataset"] == dataset_name].copy()

    # Dataset-specific threshold to define warning labels
    dataset_threshold = df["warnings"].median()
    df["warning_label"] = (df["warnings"] > dataset_threshold).astype(int)

    # Prediction
    X_test = df[features]
    y_test = df["warning_label"]
    y_pred = rf_model.predict(X_test)

    print(f"\nResults on {dataset_name} (threshold = {dataset_threshold:.2f}):")
    print(classification_report(y_test, y_pred))
