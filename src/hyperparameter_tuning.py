import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("HYPERPARAMETER TUNING FOR POVERTY PREDICTION MODEL")
print("="*70)

# 1. Load Data
print("\n[1/6] Loading data...")
df = pd.read_csv('../dataset/train_data_cleaned_v1.csv')

if 'row_id' in df.columns:
    df = df.drop(columns=['row_id'])

print(f"  Data shape: {df.shape}")

# 2. Feature Engineering (same as train_model_improved.py)
print("\n[2/6] Engineering features...")

top_features = [
    'education_level', 'is_urban', 'phone_technology', 'can_use_internet',
    'can_text', 'num_financial_activities_last_year', 'formal_savings',
    'phone_ownership', 'advanced_phone_use', 'active_bank_user'
]

additional_features = [
    'country_D', 'reg_bank_acct', 'can_make_transaction', 'num_formal_institutions_last_year',
    'literacy', 'country_A', 'financially_included', 'active_mm_user', 'has_investment',
    'employment_type_last_year_salaried', 'can_call', 'income_private_sector_last_year',
    'employment_type_last_year_irregular_seasonal', 'country_F', 'num_shocks_last_year',
    'avg_shock_strength_last_year', 'income_friends_family_last_year', 'reg_mm_acct',
    'country_I', 'age', 'female', 'married', 'employed_last_year', 'share_hh_income_provided',
    'income_own_business_last_year', 'income_public_sector_last_year',
    'num_times_borrowed_last_year', 'borrowed_for_emergency_last_year',
    'borrowed_for_daily_expenses_last_year', 'informal_savings', 'cash_property_savings',
    'can_divide', 'can_calc_percents'
]

selected_features = list(set(top_features + additional_features))
df_features = df[selected_features].copy()

# Create interaction features
df_features['education_urban'] = df['education_level'] * df['is_urban']
df_features['financial_tech_score'] = (df['can_use_internet'] + df['phone_ownership'] + df['active_bank_user']) / 3
df_features['tech_access'] = (df['phone_technology'] + df['can_use_internet'] + df['can_make_transaction']) / 3
df_features['formal_account'] = (df['reg_bank_acct'] + df['reg_mm_acct']) / 2
df_features['digital_inclusion'] = df['can_use_internet'] * df['phone_ownership'] * df['can_make_transaction']
df_features['financial_capability'] = df['literacy'] * df['education_level'] * df['financially_included']
df_features['education_squared'] = df['education_level'] ** 2
df_features['age_squared'] = df['age'] ** 2
df_features['financial_stress'] = df['num_times_borrowed_last_year'] * df['num_shocks_last_year']
df_features['economic_stability'] = df['employed_last_year'] * df['share_hh_income_provided']

X = df_features
y = df['poverty_probability']

print(f"  Features: {len(X.columns)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# 3. Define Parameter Grids
print("\n[3/6] Defining parameter search spaces...")

# XGBoost parameters
xgb_params = {
    'n_estimators': [500, 1000, 1500, 2000],
    'max_depth': [5, 6, 7, 8, 9, 10],
    'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 2, 3, 5],
    'gamma': [0, 0.1, 0.2, 0.5],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0.5, 1.0, 1.5, 2.0]
}

# LightGBM parameters
lgbm_params = {
    'n_estimators': [500, 1000, 1500, 2000],
    'max_depth': [7, 8, 9, 10, 12, 15],
    'learning_rate': [0.005, 0.01, 0.02, 0.03],
    'num_leaves': [31, 50, 70, 90, 110],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_samples': [5, 10, 15, 20],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1.0, 1.5]
}

# Random Forest parameters
rf_params = {
    'n_estimators': [300, 500, 700, 1000],
    'max_depth': [15, 20, 25, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5, 0.7]
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': [500, 700, 1000],
    'max_depth': [5, 7, 9, 11],
    'learning_rate': [0.01, 0.02, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 4. Hyperparameter Tuning
print("\n[4/6] Starting hyperparameter search (this will take a while)...")

results = {}

# XGBoost
print("\n  [1/4] Tuning XGBoost...")
xgb_model = XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist')
xgb_search = RandomizedSearchCV(
    xgb_model, xgb_params, n_iter=50, cv=3, scoring='r2',
    n_jobs=-1, verbose=1, random_state=42
)
xgb_search.fit(X_train, y_train)
results['XGBoost'] = {
    'best_params': xgb_search.best_params_,
    'best_score': xgb_search.best_score_,
    'model': xgb_search.best_estimator_
}
print(f"    Best R2: {xgb_search.best_score_:.4f}")
print(f"    Best params: {xgb_search.best_params_}")

# LightGBM
print("\n  [2/4] Tuning LightGBM...")
lgbm_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
lgbm_search = RandomizedSearchCV(
    lgbm_model, lgbm_params, n_iter=50, cv=3, scoring='r2',
    n_jobs=-1, verbose=1, random_state=42
)
lgbm_search.fit(X_train, y_train)
results['LightGBM'] = {
    'best_params': lgbm_search.best_params_,
    'best_score': lgbm_search.best_score_,
    'model': lgbm_search.best_estimator_
}
print(f"    Best R2: {lgbm_search.best_score_:.4f}")
print(f"    Best params: {lgbm_search.best_params_}")

# Random Forest
print("\n  [3/4] Tuning Random Forest...")
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_search = RandomizedSearchCV(
    rf_model, rf_params, n_iter=30, cv=3, scoring='r2',
    n_jobs=-1, verbose=1, random_state=42
)
rf_search.fit(X_train, y_train)
results['Random Forest'] = {
    'best_params': rf_search.best_params_,
    'best_score': rf_search.best_score_,
    'model': rf_search.best_estimator_
}
print(f"    Best R2: {rf_search.best_score_:.4f}")
print(f"    Best params: {rf_search.best_params_}")

# Gradient Boosting
print("\n  [4/4] Tuning Gradient Boosting...")
gb_model = GradientBoostingRegressor(random_state=42)
gb_search = RandomizedSearchCV(
    gb_model, gb_params, n_iter=30, cv=3, scoring='r2',
    n_jobs=-1, verbose=1, random_state=42
)
gb_search.fit(X_train, y_train)
results['Gradient Boosting'] = {
    'best_params': gb_search.best_params_,
    'best_score': gb_search.best_score_,
    'model': gb_search.best_estimator_
}
print(f"    Best R2: {gb_search.best_score_:.4f}")
print(f"    Best params: {gb_search.best_params_}")

# 5. Test Set Evaluation
print("\n[5/6] Evaluating tuned models on test set...")
print("\n" + "="*70)

for name, result in results.items():
    model = result['model']
    y_pred = np.clip(model.predict(X_test), 0, 1)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    result['test_r2'] = r2
    result['test_rmse'] = rmse
    result['test_mae'] = mae
    
    print(f"{name:20s} - R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

# Find best model
best_model_name = max(results.items(), key=lambda x: x[1]['test_r2'])[0]
best_model = results[best_model_name]['model']
best_r2 = results[best_model_name]['test_r2']

print(f"\nBEST MODEL: {best_model_name} (Test R2: {best_r2:.4f})")

# 6. Save Results
print("\n[6/6] Saving tuned model and results...")

# Save best model
with open('best_model_tuned.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("  [OK] Best tuned model saved to 'best_model_tuned.pkl'")

# Save feature names
with open('model_features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("  [OK] Features saved")

# Save tuning results
with open('tuning_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("  [OK] All tuning results saved")

# Print summary report
print("\n" + "="*70)
print("HYPERPARAMETER TUNING COMPLETE!")
print("="*70)
print("\nOPTIMAL HYPERPARAMETERS:\n")

for name, result in results.items():
    print(f"{name}:")
    for param, value in result['best_params'].items():
        print(f"  {param}: {value}")
    print(f"  CV R2: {result['best_score']:.4f}")
    print(f"  Test R2: {result['test_r2']:.4f}")
    print()

print("="*70)
print(f"\nBest Model: {best_model_name}")
print(f"Test R2: {best_r2:.4f} (explains {best_r2*100:.1f}% of variance)")
print(f"Test RMSE: {results[best_model_name]['test_rmse']:.4f}")
print(f"Test MAE: {results[best_model_name]['test_mae']:.4f}")
print("\n" + "="*70)
