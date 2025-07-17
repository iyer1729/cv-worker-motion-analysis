import os
import pandas as pd
import numpy as np
from scipy.stats import f
from numpy.linalg import inv

# Path to the root directory of datasets
ROOT_DIR = "../data/UCF Sports Action"  # Update this path with dataset needed

# Step sizes to compute motion between frames
STEP_SIZES = [1, 2, 4]

# Landmark point subsets for different motion analysis ablations
LANDMARK_SETS = {
    "full": list(range(33)),            # All 33 body landmarks
    "hands_only": list(range(15, 23)),  # Focused on hands
    "upper_body": list(range(11, 23)),  # Torso, arms, and head
    "lower_body": list(range(23, 33))   # Hips, legs, and feet
}

results = []  # To store analysis results for each file/condition

def extract_meta(filepath):
    """Extract dataset, participant, and task metadata from filepath structure."""
    parts = filepath.split("/")
    dataset = parts[-3]
    participant = parts[-2]
    task = parts[-1]
    return dataset, participant, task

def process_file(filepath, step_size, ablation_name, landmark_ids):
    """Process a single CSV file with given step size and landmark subset."""
    print(f"\nProcessing: {filepath}")
    print(f"Step size: {step_size}, Ablation: {ablation_name}, Landmarks: {landmark_ids}")

    df = pd.read_csv(filepath)

    # List of coordinate columns for picked landmarks
    coords = []
    for i in landmark_ids:
        coords.extend([f"x_{i}", f"y_{i}", f"z_{i}"])

    motion_vectors = []
    motion_magnitudes = []

    # Motion vectors and their magnitudes
    for k in range(step_size, len(df)):
        prev = df.loc[k - step_size, coords].values.astype(float)
        curr = df.loc[k, coords].values.astype(float)
        diff = curr - prev
        motion_vectors.append(diff)
        motion_magnitudes.append(np.linalg.norm(diff.reshape(-1, 3), axis=1).sum())

    motion_vectors = np.array(motion_vectors)
    mk_mean = np.mean(motion_vectors, axis=0)
    S = np.cov(motion_vectors.T)
    S_inv = inv(S)

    # Hotelling’s T² statistic for each motion vector
    T2 = [(mv - mk_mean) @ S_inv @ (mv - mk_mean).T for mv in motion_vectors]

    # Upper control limit (UCL) using F-distribution
    K = len(motion_vectors)
    p = motion_vectors.shape[1]
    alpha = 0.05
    F_val = f.ppf(1 - alpha, p, K - p)
    UCL = (p * (K + 1) * (K - 1)) / (K**2 - K * p) * F_val
    warnings = sum(np.array(T2) > UCL)

    # RMS deviation of motion magnitude
    RMSD = np.sqrt(np.mean((np.array(motion_magnitudes) - np.mean(motion_magnitudes)) ** 2))

    # Accumulate results
    dataset, participant, task = extract_meta(filepath)
    results.append({
        "dataset": dataset,
        "participant": participant,
        "task": task,
        "step_size": step_size,
        "ablation": ablation_name,
        "landmarks": landmark_ids,
        "RMSD": RMSD,
        "mean_motion": np.mean(motion_magnitudes),
        "std_motion": np.std(motion_magnitudes),
        "mean_T2": np.mean(T2),
        "std_T2": np.std(T2),
        "warnings": warnings,
        "UCL": UCL
    })

for subdir, _, files in os.walk(ROOT_DIR):
    for file in files:
        if file.endswith(".csv"):
            filepath = os.path.join(subdir, file)
            for step in STEP_SIZES:
                for ablation_name, landmark_ids in LANDMARK_SETS.items():
                    try:
                        process_file(filepath, step, ablation_name, landmark_ids)
                    except Exception as e:
                        print(f"Error processing {filepath} with step {step} and ablation {ablation_name}: {e}")

results_df = pd.DataFrame(results)
output_filename = f"{os.path.basename(ROOT_DIR)}_motion_analysis_results.csv"
results_df.to_csv(output_filename, index=False)

print(f"\nSaved results to '{output_filename}'")
print(results_df)
