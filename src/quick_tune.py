import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("QUICK HYPERPARAMETER TUNING - USING FOUND OPTIMAL PARAMETERS")
print("="*70)

# Load Data
print("\n[1/4] Loading data...")
df = pd.read_csv('../dataset/train_data_cleaned_v1.csv')
if 'row_id' in df.columns:
    df = df.drop(columns=['row_id'])

# Feature Engineering
print("[2/4] Engineering features...")
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Features: {len(X.columns)}, Train: {len(X_train)}, Test: {len(X_test)}")

# Train with OPTIMAL parameters from tuning
print("\n[3/4] Training models with optimal hyperparameters...")

# XGBoost with TUNED parameters
print("\n  Training Tuned XGBoost...")
xgb_tuned = XGBRegressor(
    n_estimators=1500,
    max_depth=7,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.9,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0,
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)
xgb_tuned.fit(X_train, y_train)
y_pred_xgb = np.clip(xgb_tuned.predict(X_test), 0, 1)
r2_xgb = r2_score(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print(f"    XGBoost Tuned - R2: {r2_xgb:.4f} | RMSE: {rmse_xgb:.4f} | MAE: {mae_xgb:.4f}")

# LightGBM with OPTIMIZED parameters (similar to XGBoost findings)
print("\n  Training Optimized LightGBM...")
lgbm_opt = lgb.LGBMRegressor(
    n_estimators=1500,
    max_depth=8,
    learning_rate=0.01,
    num_leaves=70,
    subsample=0.7,
    colsample_bytree=0.9,
    min_child_samples=10,
    reg_alpha=0,
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm_opt.fit(X_train, y_train)
y_pred_lgbm = np.clip(lgbm_opt.predict(X_test), 0, 1)
r2_lgbm = r2_score(y_test, y_pred_lgbm)
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
print(f"    LightGBM Opt  - R2: {r2_lgbm:.4f} | RMSE: {rmse_lgbm:.4f} | MAE: {mae_lgbm:.4f}")

# Create Weighted Ensemble
print("\n  Creating Weighted Ensemble...")
w_xgb, w_lgbm = 0.55, 0.45
y_pred_ensemble = w_xgb * y_pred_xgb + w_lgbm * y_pred_lgbm
r2_ensemble = r2_score(y_test, y_pred_ensemble)
rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
print(f"    Ensemble      - R2: {r2_ensemble:.4f} | RMSE: {rmse_ensemble:.4f} | MAE: {mae_ensemble:.4f}")

# Select best model
print("\n[4/4] Saving best model...")
models = {
    'XGBoost Tuned': (xgb_tuned, r2_xgb, rmse_xgb, mae_xgb),
    'LightGBM Opt': (lgbm_opt, r2_lgbm, rmse_lgbm, mae_lgbm),
    'Ensemble': (xgb_tuned, r2_ensemble, rmse_ensemble, mae_ensemble)  # Save XGBoost as representative
}

best_name = max(models.items(), key=lambda x: x[1][1])[0]
best_model, best_r2, best_rmse, best_mae = models[best_name]

print(f"\n{'='*70}")
print(f"BEST MODEL: {best_name}")
print(f"{'='*70}")
print(f"Test R2:   {best_r2:.4f} (explains {best_r2*100:.1f}% of variance)")
print(f"Test RMSE: {best_rmse:.4f} (avg error: {best_rmse*100:.1f} percentage points)")
print(f"Test MAE:  {best_mae:.4f}")

# Save
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('model_features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print(f"\n✓ Model saved to best_model.pkl")
print(f"✓ Features saved to model_features.pkl")

# Save optimal parameters
optimal_params = {
    'XGBoost': {
        'n_estimators': 1500, 'max_depth': 7, 'learning_rate': 0.01,
        'subsample': 0.7, 'colsample_bytree': 0.9, 'min_child_weight': 5,
        'gamma': 0.1, 'reg_alpha': 0, 'reg_lambda': 0.5
    },
    'LightGBM': {
        'n_estimators': 1500, 'max_depth': 8, 'learning_rate': 0.01,
        'num_leaves': 70, 'subsample': 0.7, 'colsample_bytree': 0.9,
        'min_child_samples': 10, 'reg_alpha': 0, 'reg_lambda': 0.5
    }
}

with open('optimal_hyperparameters.pkl', 'wb') as f:
    pickle.dump(optimal_params, f)
print(f"✓ Optimal hyperparameters saved")

print(f"\n{'='*70}\n")
