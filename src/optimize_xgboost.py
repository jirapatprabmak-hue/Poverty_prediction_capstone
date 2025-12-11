import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, f1_score, r2_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ADVANCED XGBOOST OPTIMIZATION FOR POVERTY PREDICTION")
print("="*70)

# Load data
print("\n[1/6] Loading and preparing data...")
df = pd.read_csv('../dataset/train_data_cleaned_v1.csv')
if 'row_id' in df.columns:
    df = df.drop(columns=['row_id'])

# Calculate data statistics
poverty_prob = df['poverty_probability']
DATA_MIN = float(poverty_prob.min())
DATA_MAX = float(poverty_prob.max())
DATA_MEAN = float(poverty_prob.mean())
DATA_STD = float(poverty_prob.std())

print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Target statistics: Mean={DATA_MEAN:.4f}, Std={DATA_STD:.4f}")

# Comprehensive feature engineering
print("\n[2/6] Engineering comprehensive feature set...")

# Start with all meaningful features
exclude_cols = ['poverty_probability', 'row_id']
base_features = [col for col in df.columns if col not in exclude_cols]

df_features = df[base_features].copy()

# 1. Financial inclusion indicators
df_features['financial_inclusion_index'] = (
    df['reg_bank_acct'] + df['reg_mm_acct'] + df['active_bank_user'] + 
    df['active_mm_user'] + df['financially_included']
)

# 2. Digital capability score
df_features['digital_capability'] = (
    df['can_use_internet'] + df['can_make_transaction'] + df['can_text'] + 
    df['can_call'] + df['phone_technology']
)

# 3. Financial literacy composite
df_features['financial_literacy'] = (
    df['can_add'] + df['can_divide'] + df['can_calc_percents'] + 
    df['can_calc_compounding'] + df['literacy']
)

# 4. Income aggregations
income_cols = [col for col in df.columns if 'income_' in col and '_last_year' in col]
df_features['total_income'] = df[income_cols].sum(axis=1)
df_features['income_diversity'] = (df[income_cols] > 0).sum(axis=1)
df_features['income_per_source'] = df_features['total_income'] / (df_features['income_diversity'] + 1)

# 5. Savings behavior
df_features['total_savings'] = df['formal_savings'] + df['informal_savings'] + df['cash_property_savings']
df_features['savings_diversity'] = (df['formal_savings'] > 0).astype(int) + (df['informal_savings'] > 0).astype(int) + (df['cash_property_savings'] > 0).astype(int)

# 6. Financial stress indicators
df_features['borrowing_intensity'] = df['num_times_borrowed_last_year']
df_features['emergency_borrowing'] = df['borrowed_for_emergency_last_year']
df_features['daily_expense_borrowing'] = df['borrowed_for_daily_expenses_last_year']
df_features['financial_stress'] = df_features['borrowing_intensity'] + df_features['emergency_borrowing'] + df_features['daily_expense_borrowing']

# 7. Shock vulnerability
df_features['shock_exposure'] = df['num_shocks_last_year'] * df['avg_shock_strength_last_year']
df_features['shock_resilience'] = df_features['total_savings'] / (df_features['shock_exposure'] + 1)

# 8. Financial activity intensity
df_features['formal_institution_usage'] = df['num_formal_institutions_last_year']
df_features['informal_institution_usage'] = df['num_informal_institutions_last_year']
df_features['total_financial_activities'] = df['num_financial_activities_last_year']

# 9. Key interaction features
df_features['education_x_urban'] = df['education_level'] * df['is_urban']
df_features['education_x_employed'] = df['education_level'] * df['employed_last_year']
df_features['urban_x_phone_tech'] = df['is_urban'] * df['phone_technology']
df_features['urban_x_financial_inclusion'] = df['is_urban'] * df_features['financial_inclusion_index']
df_features['education_x_financial_literacy'] = df['education_level'] * df_features['financial_literacy']
df_features['education_x_income'] = df['education_level'] * df_features['total_income']

# 10. Polynomial features for key predictors
df_features['education_squared'] = df['education_level'] ** 2
df_features['education_cubed'] = df['education_level'] ** 3
df_features['age_squared'] = df['age'] ** 2
df_features['financial_activities_squared'] = df['num_financial_activities_last_year'] ** 2

# 11. Ratio features
df_features['savings_to_income'] = df_features['total_savings'] / (df_features['total_income'] + 1)
df_features['borrowing_to_income'] = df_features['borrowing_intensity'] / (df_features['total_income'] + 1)
df_features['formal_to_informal_institutions'] = df_features['formal_institution_usage'] / (df_features['informal_institution_usage'] + 1)

# 12. Country-specific interactions (for top countries)
for country in ['country_A', 'country_D', 'country_F', 'country_I']:
    if country in df.columns:
        df_features[f'{country}_x_education'] = df[country] * df['education_level']
        df_features[f'{country}_x_urban'] = df[country] * df['is_urban']
        df_features[f'{country}_x_income'] = df[country] * df_features['total_income']

# 13. Complex multi-way interactions
df_features['edu_urban_tech'] = df['education_level'] * df['is_urban'] * df['phone_technology']
df_features['urban_digital_financial'] = df['is_urban'] * df_features['digital_capability'] * df_features['financial_inclusion_index']
df_features['education_digital_employed'] = df['education_level'] * df_features['digital_capability'] * df['employed_last_year']

# 14. Log transformations for skewed features
df_features['log_age'] = np.log1p(df['age'])
df_features['log_income'] = np.log1p(df_features['total_income'])
df_features['log_financial_activities'] = np.log1p(df['num_financial_activities_last_year'])

X = df_features
y = df['poverty_probability']

print(f"Total features after engineering: {X.shape[1]}")

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Data split: Train={len(X_train)}, Valid={len(X_valid)}, Test={len(X_test)}")

# Evaluation function
def evaluate(model, X, y_true, name):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    y_pred_bin = (y_pred >= 0.5).astype(int)
    y_true_bin = (y_true >= 0.5).astype(int)
    f1 = f1_score(y_true_bin, y_pred_bin)
    print(f"  [{name:25s}] R2: {r2:.4f}, RMSE: {rmse:.4f}, F1: {f1:.4f}")
    return r2, rmse, f1

print("\n[3/6] Hyperparameter Optimization with RandomizedSearchCV...")

# Comprehensive hyperparameter space for XGBoost
param_distributions = {
    'n_estimators': [1500, 2000, 2500, 3000],
    'max_depth': [5, 6, 7, 8, 9],
    'learning_rate': [0.002, 0.003, 0.005, 0.007, 0.01],
    'subsample': [0.75, 0.8, 0.85, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'colsample_bylevel': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 2, 3, 4],
    'gamma': [0, 0.01, 0.05, 0.1],
    'reg_alpha': [0, 0.1, 0.2, 0.3],
    'reg_lambda': [0.5, 0.8, 1.0, 1.5],
    'max_delta_step': [0, 1, 2],
}

# Base XGBoost model
base_xgb = XGBRegressor(
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    objective='reg:squarederror'
)

# RandomizedSearchCV - balanced speed and thoroughness
random_search = RandomizedSearchCV(
    estimator=base_xgb,
    param_distributions=param_distributions,
    n_iter=30,  # Test 30 combinations
    cv=3,  # 3-fold cross-validation (faster)
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

print(f"\nSearching through 30 parameter combinations with 3-fold CV...")
print("This will take a few minutes...\n")

random_search.fit(X_train, y_train)

print(f"\n[4/6] Best Parameters Found:")
print("="*70)
for param, value in random_search.best_params_.items():
    print(f"  {param:25s}: {value}")

print(f"\nBest CV R2 Score: {random_search.best_score_:.4f}")

# Get best model
best_xgb = random_search.best_estimator_

print("\n[5/6] Evaluating Best Model...")
print("="*70)

# Validation performance
r2_val, rmse_val, f1_val = evaluate(best_xgb, X_valid, y_valid, "Validation Set")

# Test performance
r2_test, rmse_test, f1_test = evaluate(best_xgb, X_test, y_test, "Test Set")

print("\n[6/6] Top 10 Parameter Combinations:")
print("="*70)

results = pd.DataFrame(random_search.cv_results_)
top_results = results.nlargest(10, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]

for idx, (_, row) in enumerate(top_results.iterrows(), 1):
    print(f"\nRank {idx}: CV R2 = {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})")
    params = row['params']
    print(f"  n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, "
          f"learning_rate={params['learning_rate']}")
    print(f"  subsample={params['subsample']}, colsample_bytree={params['colsample_bytree']}")

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Validation R2: {r2_val:.4f}")
print(f"Test R2:       {r2_test:.4f}")
print(f"Test RMSE:     {rmse_test:.4f}")
print(f"Test F1:       {f1_test:.4f}")
print("="*70)

# Save optimized model
with open('xgboost_optimized.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)

with open('xgboost_optimized_features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

# Save scaling stats
scaling_stats = {
    'min': DATA_MIN,
    'max': DATA_MAX,
    'mean': DATA_MEAN,
    'std': DATA_STD
}
with open('xgboost_scaling_stats.pkl', 'wb') as f:
    pickle.dump(scaling_stats, f)

print("\nModel saved to 'xgboost_optimized.pkl'")
print("Features saved to 'xgboost_optimized_features.pkl'")
print("Scaling stats saved to 'xgboost_scaling_stats.pkl'")

# Feature importance analysis
print("\n" + "="*70)
print("TOP 20 MOST IMPORTANT FEATURES")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_xgb.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(20).iterrows():
    print(f"{row['feature']:45s}: {row['importance']:.4f}")
