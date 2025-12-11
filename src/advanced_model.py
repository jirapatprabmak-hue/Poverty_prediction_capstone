import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ADVANCED POVERTY PREDICTION MODEL - TARGET R² > 0.5")
print("="*70)

# Load data
print("\n[1/7] Loading data...")
df = pd.read_csv('../dataset/train_data_cleaned_v1.csv')
if 'row_id' in df.columns:
    df = df.drop(columns=['row_id'])

print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Use ALL meaningful features (not just top 10)
print("\n[2/7] Engineering comprehensive feature set...")

# Exclude only the target and non-predictive columns
exclude_cols = ['poverty_probability']
all_features = [col for col in df.columns if col not in exclude_cols]

print(f"Using ALL {len(all_features)} base features from dataset")

# Create extensive feature engineering
df_features = df[all_features].copy()

# 1. Financial inclusion index
df_features['financial_inclusion_score'] = (
    df['reg_bank_acct'] + df['reg_mm_acct'] + df['active_bank_user'] + 
    df['active_mm_user'] + df['financially_included']
) / 5

# 2. Digital literacy score
df_features['digital_literacy_score'] = (
    df['can_use_internet'] + df['can_make_transaction'] + df['can_text'] + 
    df['can_call'] + df['phone_technology'] + df['phone_ownership']
) / 6

# 3. Numeracy/literacy score
df_features['numeracy_score'] = (
    df['can_add'] + df['can_divide'] + df['can_calc_percents'] + 
    df['can_calc_compounding'] + df['literacy']
) / 5

# 4. Total income from all sources
income_cols = [col for col in df.columns if 'income_' in col and '_last_year' in col]
df_features['total_income'] = df[income_cols].sum(axis=1)
df_features['income_diversity'] = (df[income_cols] > 0).sum(axis=1)

# 5. Savings diversity
df_features['savings_diversity'] = (
    df['formal_savings'] + df['informal_savings'] + df['cash_property_savings']
)

# 6. Financial stress indicators
df_features['financial_stress'] = (
    df['num_times_borrowed_last_year'] + 
    df['borrowed_for_emergency_last_year'] + 
    df['borrowed_for_daily_expenses_last_year']
)

# 7. Shock vulnerability
df_features['shock_impact'] = df['num_shocks_last_year'] * df['avg_shock_strength_last_year']

# 8. Financial activity intensity
df_features['financial_activity_intensity'] = (
    df['num_financial_activities_last_year'] + 
    df['num_formal_institutions_last_year'] + 
    df['num_informal_institutions_last_year']
)

# 9. Education-Employment interaction
df_features['education_employed_interaction'] = df['education_level'] * df['employed_last_year']
df_features['education_urban_interaction'] = df['education_level'] * df['is_urban']

# 10. Age-based features
df_features['age_squared'] = df['age'] ** 2
df_features['age_cubed'] = df['age'] ** 3
df_features['working_age'] = ((df['age'] >= 18) & (df['age'] <= 65)).astype(int)

# 11. Urban advantage features
df_features['urban_tech_advantage'] = df['is_urban'] * df_features['digital_literacy_score']
df_features['urban_financial_access'] = df['is_urban'] * df_features['financial_inclusion_score']
df_features['urban_income'] = df['is_urban'] * df_features['total_income']

# 12. Polynomial features for key variables
df_features['education_squared'] = df['education_level'] ** 2
df_features['education_cubed'] = df['education_level'] ** 3
df_features['financial_activities_squared'] = df['num_financial_activities_last_year'] ** 2

# 13. Ratio features
df_features['savings_to_income'] = df_features['savings_diversity'] / (df_features['total_income'] + 1)
df_features['borrowing_ratio'] = df['num_times_borrowed_last_year'] / (df_features['total_income'] + 1)
df_features['income_per_activity'] = df_features['total_income'] / (df_features['financial_activity_intensity'] + 1)

# 14. Country-specific interactions
for country in ['country_A', 'country_C', 'country_D', 'country_F', 'country_G', 'country_I', 'country_J']:
    if country in df.columns:
        df_features[f'{country}_education'] = df[country] * df['education_level']
        df_features[f'{country}_urban'] = df[country] * df['is_urban']

# 15. Cross-feature interactions
df_features['edu_x_digital'] = df['education_level'] * df_features['digital_literacy_score']
df_features['edu_x_financial'] = df['education_level'] * df_features['financial_inclusion_score']
df_features['income_x_savings'] = df_features['total_income'] * df_features['savings_diversity']

X = df_features
y = df['poverty_probability']

print(f"Total engineered features: {X.shape[1]}")

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
    print(f"  [{name:25s}] R²: {r2:.4f}, RMSE: {rmse:.4f}, F1: {f1:.4f}")
    return r2, rmse, f1

print("\n[3/7] Training Individual Models...")

# 1. XGBoost with tuned params
print("\n  Training XGBoost (Tuned)...")
xgb_model = XGBRegressor(
    n_estimators=2000,
    max_depth=7,
    learning_rate=0.003,
    subsample=0.85,
    colsample_bytree=0.7,
    min_child_weight=2,
    gamma=0.01,
    reg_alpha=0.2,
    reg_lambda=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)
xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
r2_xgb, rmse_xgb, f1_xgb = evaluate(xgb_model, X_valid, y_valid, "XGBoost")

# 2. LightGBM
print("  Training LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.003,
    num_leaves=80,
    subsample=0.85,
    colsample_bytree=0.7,
    min_child_samples=10,
    reg_alpha=0.1,
    reg_lambda=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[lgb.early_stopping(100, verbose=False)])
r2_lgb, rmse_lgb, f1_lgb = evaluate(lgb_model, X_valid, y_valid, "LightGBM")

# 3. Gradient Boosting
print("  Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=1000,
    max_depth=7,
    learning_rate=0.01,
    subsample=0.8,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42
)
gb_model.fit(X_train, y_train)
r2_gb, rmse_gb, f1_gb = evaluate(gb_model, X_valid, y_valid, "Gradient Boosting")

# 4. Random Forest (Deep)
print("  Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
r2_rf, rmse_rf, f1_rf = evaluate(rf_model, X_valid, y_valid, "Random Forest")

print("\n[4/7] Creating Advanced Ensemble Models...")

# Weighted Voting Regressor
print("\n  Training Weighted Voting Ensemble...")
voting_model = VotingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('gb', gb_model),
        ('rf', rf_model)
    ],
    weights=[0.35, 0.35, 0.2, 0.1]  # Higher weights for best performers
)
voting_model.fit(X_train, y_train)
r2_voting, rmse_voting, f1_voting = evaluate(voting_model, X_valid, y_valid, "Voting Ensemble")

# Stacking with meta-learner
print("  Training Stacking Ensemble...")
stacking_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('gb', gb_model)
    ],
    final_estimator=Ridge(alpha=0.1),
    cv=5,
    n_jobs=-1
)
stacking_model.fit(X_train, y_train)
r2_stacking, rmse_stacking, f1_stacking = evaluate(stacking_model, X_valid, y_valid, "Stacking Ensemble")

# Custom weighted ensemble
print("  Creating Custom Weighted Ensemble...")
y_pred_xgb = xgb_model.predict(X_valid)
y_pred_lgb = lgb_model.predict(X_valid)
y_pred_gb = gb_model.predict(X_valid)
y_pred_rf = rf_model.predict(X_valid)

# Optimize weights based on validation performance
w1, w2, w3, w4 = 0.4, 0.35, 0.15, 0.1
y_pred_weighted = w1 * y_pred_xgb + w2 * y_pred_lgb + w3 * y_pred_gb + w4 * y_pred_rf
r2_weighted = r2_score(y_valid, y_pred_weighted)
rmse_weighted = np.sqrt(mean_squared_error(y_valid, y_pred_weighted))
y_pred_bin = (y_pred_weighted >= 0.5).astype(int)
y_true_bin = (y_valid >= 0.5).astype(int)
f1_weighted = f1_score(y_true_bin, y_pred_bin)
print(f"  [Custom Weighted          ] R²: {r2_weighted:.4f}, RMSE: {rmse_weighted:.4f}, F1: {f1_weighted:.4f}")

print("\n[5/7] Model Comparison on Validation Set:")
print("="*70)
results = [
    ("XGBoost", r2_xgb),
    ("LightGBM", r2_lgb),
    ("Gradient Boosting", r2_gb),
    ("Random Forest", r2_rf),
    ("Voting Ensemble", r2_voting),
    ("Stacking Ensemble", r2_stacking),
    ("Custom Weighted", r2_weighted)
]
results.sort(key=lambda x: x[1], reverse=True)

for name, r2 in results:
    print(f"  {name:25s}: R² = {r2:.4f}")

# Select best model
best_name, best_r2 = results[0]
model_map = {
    "XGBoost": xgb_model,
    "LightGBM": lgb_model,
    "Gradient Boosting": gb_model,
    "Random Forest": rf_model,
    "Voting Ensemble": voting_model,
    "Stacking Ensemble": stacking_model
}

if best_name == "Custom Weighted":
    print(f"\n[6/7] Best Model: Custom Weighted Ensemble (R² = {best_r2:.4f})")
    print("\n[7/7] Final Evaluation on Test Set:")
    print("="*70)
    
    y_pred_xgb_test = xgb_model.predict(X_test)
    y_pred_lgb_test = lgb_model.predict(X_test)
    y_pred_gb_test = gb_model.predict(X_test)
    y_pred_rf_test = rf_model.predict(X_test)
    y_pred_test = w1 * y_pred_xgb_test + w2 * y_pred_lgb_test + w3 * y_pred_gb_test + w4 * y_pred_rf_test
    
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    y_pred_bin = (y_pred_test >= 0.5).astype(int)
    y_true_bin = (y_test >= 0.5).astype(int)
    f1_test = f1_score(y_true_bin, y_pred_bin)
    
    print(f"\nFINAL TEST SET RESULTS:")
    print(f"  R²:   {r2_test:.4f}")
    print(f"  RMSE: {rmse_test:.4f}")
    print(f"  F1:   {f1_test:.4f}")
    
    # Save ensemble components
    best_model = xgb_model  # Save XGBoost as primary
else:
    best_model = model_map[best_name]
    print(f"\n[6/7] Best Model: {best_name} (R² = {best_r2:.4f})")
    print("\n[7/7] Final Evaluation on Test Set:")
    print("="*70)
    
    r2_test, rmse_test, f1_test = evaluate(best_model, X_test, y_test, best_name)
    
    print(f"\nFINAL TEST SET RESULTS:")
    print(f"  R²:   {r2_test:.4f}")
    print(f"  RMSE: {rmse_test:.4f}")
    print(f"  F1:   {f1_test:.4f}")

print("\n" + "="*70)
if r2_test >= 0.5:
    print(f"✓ SUCCESS! Achieved R² = {r2_test:.4f} (>0.5)")
else:
    print(f"Best achieved R² = {r2_test:.4f}")
    print("Note: R² > 0.5 is very challenging for poverty prediction.")
    print("This is a complex socioeconomic problem with inherent randomness.")
print("="*70)

# Save model
with open('best_advanced_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
    
with open('advanced_model_features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("\nModel saved to 'best_advanced_model.pkl'")
