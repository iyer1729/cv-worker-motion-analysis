import pandas as pd
from scipy.stats import ttest_ind, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# For debugging
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# merged data file path
merged_df = pd.read_csv("../results/Merged_NASA_TLX_and_assembly_motion_data.csv")
merged_df.columns = merged_df.columns.str.strip()  # Remove trailing spaces in column names

desc_stats = merged_df[[
    "RMSD", "mean_motion", "std_motion", "mean_T2", "std_T2", "warnings", "UCL",
    "Mental Demand", "Physical Demand", "Temporal Demand", "Performance", "Effort", "Frustration", "average"
]].describe()

correlation_matrix = merged_df[[
    "RMSD", "mean_motion", "std_motion", "mean_T2", "std_T2", "warnings",
    "Mental Demand", "Physical Demand", "Temporal Demand", "Performance", "Effort", "Frustration", "average"
]].corr(method="pearson")

ablation_values = merged_df["ablation"].unique()
t_test_results = []

# Pairwise t-tests for warnings across ablation types
for i in range(len(ablation_values)):
    for j in range(i+1, len(ablation_values)):
        group1 = merged_df[merged_df["ablation"] == ablation_values[i]]["warnings"]
        group2 = merged_df[merged_df["ablation"] == ablation_values[j]]["warnings"]
        t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
        t_test_results.append({
            "group1": ablation_values[i],
            "group2": ablation_values[j],
            "t_statistic": t_stat,
            "p_value": p_val
        })

t_test_df = pd.DataFrame(t_test_results)

X_lin = merged_df[[
    "RMSD", "mean_motion", "std_motion", "mean_T2", "std_T2",
    "Mental Demand", "Physical Demand", "Temporal Demand", "Performance", "Effort", "Frustration", "average"
]]
y_lin = merged_df["warnings"]

X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(X_lin, y_lin, test_size=0.2, random_state=42)

lin_model = LinearRegression()
lin_model.fit(X_train_lin, y_train_lin)
y_pred_lin = lin_model.predict(X_test_lin)

lin_rmse = mean_squared_error(y_test_lin, y_pred_lin, squared=False)
lin_r2 = r2_score(y_test_lin, y_pred_lin)

# Categorical ablation using one-hot encoding
X_rf = merged_df[[
    "RMSD", "mean_motion", "std_motion", "mean_T2", "std_T2", "ablation"
]]
y_rf = merged_df["warnings"]

preprocessor = make_column_transformer(
    (OneHotEncoder(), ["ablation"]),
    remainder="passthrough"
)

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

rf_model = make_pipeline(preprocessor, RandomForestRegressor(random_state=42))
rf_model.fit(X_train_rf, y_train_rf)
y_pred_rf = rf_model.predict(X_test_rf)

rf_rmse = mean_squared_error(y_test_rf, y_pred_rf, squared=False)
rf_r2 = r2_score(y_test_rf, y_pred_rf)

tlx_columns = ["Mental Demand", "Physical Demand", "Temporal Demand", "Performance", "Effort", "Frustration", "average"]
correlation_results = []

for col in tlx_columns:
    r, p = pearsonr(merged_df[col], merged_df["warnings"])
    correlation_results.append({
        "metric": col,
        "correlation": r,
        "p_value": p
    })

tlx_warning_corr_df = pd.DataFrame(correlation_results)

print("\n1. DESCRIPTIVE STATISTICS:\n")
print(desc_stats)

print("\n2. PEARSON CORRELATION MATRIX:\n")
print(correlation_matrix)

print("\n3. T-TESTS ACROSS ABLATION CONDITIONS (WARNINGS):\n")
for _, row in t_test_df.iterrows():
    print(f"{row['group1']} vs {row['group2']}: t = {row['t_statistic']:.3f}, p = {row['p_value']:.4f}")

print("\n4. LINEAR REGRESSION TO PREDICT WARNINGS:\n")
print("Coefficients:")
for name, coef in zip(X_lin.columns, lin_model.coef_):
    print(f"{name}: {coef:.4f}")
print(f"Intercept: {lin_model.intercept_:.4f}")
print(f"RMSE: {lin_rmse:.4f}")
print(f"R^2: {lin_r2:.4f}")

print("\n5. RANDOM FOREST REGRESSION TO PREDICT WARNINGS:\n")
print(f"RMSE: {rf_rmse:.4f}")
print(f"R^2: {rf_r2:.4f}")

print("\n6. TLX METRICS vs WARNINGS (PAIRWISE CORRELATIONS):\n")
for _, row in tlx_warning_corr_df.iterrows():
    print(f"{row['metric']}: r = {row['correlation']:.3f}, p = {row['p_value']:.4f}")
