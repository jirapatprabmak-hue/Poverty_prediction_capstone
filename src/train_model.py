import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, f1_score, r2_score
import lightgbm as lgb

# Calculate data statistics (will be computed during training)
DATA_MIN = None
DATA_MAX = None
DATA_MEAN = None
DATA_STD = None

# Function to scale predictions based on actual data distribution
def scale_prediction(pred, min_val=0.0, max_val=1.0, mean_val=0.5, std_val=0.25):
    """
    Scale prediction to 0-100% range based on data distribution.
    Uses inverse percentile transformation to map predictions to percentages.
    """
    # Clamp raw prediction to 0-1 first
    pred_clamped = max(0, min(1, float(pred)))
    
    # Use normal distribution to convert to percentage
    # This maps values to probability space based on data statistics
    from scipy.stats import norm
    
    # Normalize the prediction using the data statistics
    if std_val > 0:
        z_score = (pred_clamped - mean_val) / std_val
    else:
        z_score = 0
    
    # Convert z-score to percentile (0-1)
    percentile = norm.cdf(z_score)
    
    # Scale to 0-100%
    percentage = percentile * 100
    
    return max(0, min(100, percentage))

# 1. โหลดข้อมูล
print("Loading data...")
df = pd.read_csv('../dataset/train_data_cleaned_v1.csv')

# ลบ column ที่ไม่ใช่ feature
if 'row_id' in df.columns:
    df = df.drop(columns=['row_id'])

# Calculate data statistics for scaling
poverty_prob = df['poverty_probability']
DATA_MIN = float(poverty_prob.min())
DATA_MAX = float(poverty_prob.max())
DATA_MEAN = float(poverty_prob.mean())
DATA_STD = float(poverty_prob.std())

print(f"Data Statistics for Scaling:")
print(f"  Min: {DATA_MIN:.4f}, Max: {DATA_MAX:.4f}")
print(f"  Mean: {DATA_MEAN:.4f}, Std: {DATA_STD:.4f}")

# Top 10 features with highest correlation to poverty_probability
top_10_features = [
    'education_level',
    'is_urban',
    'phone_technology',
    'can_use_internet',
    'can_text',
    'num_financial_activities_last_year',
    'formal_savings',
    'phone_ownership',
    'advanced_phone_use',
    'active_bank_user'
]

# Expanded feature list - includes all features with correlation > 0.1
# ✓ Include country dummy variables (country_A, country_D)
# ✓ Include economic shocks features (num_shocks, avg_shock_strength)
# ✓ Include employment type features
# ✓ Include income-related features
expanded_features = [
    # Country variables (important geographic factors)
    'country_D',
    'country_A',
    'country_F',
    'country_I',
    
    # Economic shocks features (strong predictors)
    'num_shocks_last_year',
    'avg_shock_strength_last_year',
    
    # Employment type features
    'employment_type_last_year_irregular_seasonal',
    'employment_type_last_year_salaried',
    'employment_type_last_year_self_employed',
    'employment_type_last_year_not_working',
    
    # Income-related features
    'income_private_sector_last_year',
    'income_public_sector_last_year',
    'income_ag_livestock_last_year',
    'income_own_business_last_year',
    'income_friends_family_last_year',
    'income_government_last_year',
    
    # Financial inclusion
    'reg_bank_acct',
    'can_make_transaction',
    'num_formal_institutions_last_year',
    'literacy',
    'financially_included',
    'active_mm_user',
    'has_investment',
    'reg_mm_acct',
    'has_insurance',
    
    # Demographics and household (age removed for better performance)
    'female',
    'married',
    'employed_last_year',
    'share_hh_income_provided',
    
    # Borrowing behavior
    'num_times_borrowed_last_year',
    'borrowed_for_emergency_last_year',
    'borrowed_for_daily_expenses_last_year',
    
    # Other useful features
    'can_call',
    'num_informal_institutions_last_year'
]

# Combine all features
selected_features = top_10_features + expanded_features

# Add interaction features for better predictive power
# ✓ Create feature interactions (e.g., education × is_urban)
print("\nEngineering interaction features...")
df_features = df[selected_features].copy()

# Create important interaction features
# Education matters more in urban areas
df_features['education_urban_interaction'] = df['education_level'] * df['is_urban']

# Financial technology capability scores
df_features['financial_tech_score'] = (df['can_use_internet'] + df['phone_ownership'] + df['active_bank_user']) / 3
df_features['tech_access_score'] = (df['phone_technology'] + df['can_use_internet'] + df['can_make_transaction']) / 3

# Formal financial inclusion index
df_features['formal_account_index'] = (df['reg_bank_acct'] + df['reg_mm_acct']) / 2

# Digital and financial inclusion interactions
df_features['digital_inclusion'] = (df['can_use_internet'] * df['phone_ownership'] * df['can_make_transaction'])
df_features['financial_capability'] = (df['literacy'] * df['education_level'] * df['financially_included'])

# Income diversity (multiple income sources = less poverty)
df_features['income_diversity'] = (
    df['income_private_sector_last_year'] + 
    df['income_public_sector_last_year'] + 
    df['income_ag_livestock_last_year'] + 
    df['income_own_business_last_year']
)

# Economic vulnerability (shocks affect rural areas more)
df_features['vulnerability_score'] = df['num_shocks_last_year'] * df['avg_shock_strength_last_year']
df_features['rural_vulnerability'] = df['num_shocks_last_year'] * (1 - df['is_urban'])

# Employment and education interaction
df_features['educated_employed'] = df['education_level'] * df['employed_last_year']

# Urban technology advantage
df_features['urban_tech_interaction'] = df['is_urban'] * df['phone_technology']
df_features['urban_financial_access'] = df['is_urban'] * df['num_financial_activities_last_year']

# Polynomial features for non-linear relationships
df_features['education_level_squared'] = df['education_level'] ** 2
df_features['financial_activities_squared'] = df['num_financial_activities_last_year'] ** 2

# Include all engineered features
selected_features = list(df_features.columns)

# แยก Feature และ Target
target_col = 'poverty_probability'
X = df_features
y = df[target_col]
print(f"Using {len(selected_features)} features (including interactions): {selected_features}")

# 2. แบ่งข้อมูล 60% Train, 20% Validation, 20% Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Data Split: Train={X_train.shape[0]}, Validation={X_valid.shape[0]}, Test={X_test.shape[0]}")

# Normalize features using StandardScaler (z-score normalization)
# This scales features to have mean=0 and std=1, improving model performance
print("\nNormalizing features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)
print(f"Features normalized: mean=0, std=1")

# Convert back to DataFrame to maintain feature names for some models
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_valid_scaled = pd.DataFrame(X_valid_scaled, columns=X_valid.columns, index=X_valid.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# ฟังก์ชันคำนวณ Metrics
def evaluate_model(model, X, y_true, name):
    y_pred = model.predict(X)
    # Evaluate on original scale (0-1) for better metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # คำนวณ F1 โดยตัดที่ threshold 0.5 (แปลงเป็นจน/ไม่จน)
    y_pred_bin = (y_pred >= 0.5).astype(int)
    y_true_bin = (y_true >= 0.5).astype(int)
    f1 = f1_score(y_true_bin, y_pred_bin)
    print(f"[{name}] RMSE: {rmse:.4f}, R²: {r2:.4f}, F1: {f1:.4f}")
    return rmse, r2, f1

# 3. Train Advanced Models with Optimized Hyperparameters
print("\n" + "="*60)
print("TRAINING ADVANCED MODELS FOR HIGHER R²")
print("="*60)

print("\n[1/6] Training Ridge Regression (Baseline)...")
ridge = Ridge(alpha=0.5, random_state=42)
ridge.fit(X_train_scaled, y_train)
rmse_ridge, r2_ridge, f1_ridge = evaluate_model(ridge, X_valid_scaled, y_valid, "Ridge Regression")

print("\n[2/6] Training Elastic Net (L1+L2 Regularization)...")
elastic = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=5000)
elastic.fit(X_train_scaled, y_train)
rmse_elastic, r2_elastic, f1_elastic = evaluate_model(elastic, X_valid_scaled, y_valid, "Elastic Net")

print("\n[3/6] Training Random Forest (Optimized)...")
rf = RandomForestRegressor(
    n_estimators=500,           # More trees
    max_depth=25,               # Deep trees for complex patterns
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',        # Reduce correlation between trees
    random_state=42,
    n_jobs=-1,
    bootstrap=True
)
rf.fit(X_train, y_train)
rmse_rf, r2_rf, f1_rf = evaluate_model(rf, X_valid, y_valid, "Random Forest")

print("\n[4/6] Training XGBoost (Optimized)...")
xgb = XGBRegressor(
    n_estimators=1500,          # Increased trees
    max_depth=8,                # Slightly deeper
    learning_rate=0.008,        # Even lower learning rate
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=2,         # Allow more flexibility
    gamma=0.05,                 # Less regularization
    reg_alpha=0.05,             # Reduced L1 regularization
    reg_lambda=0.8,             # Reduced L2 regularization
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=100   # More patience
)
xgb.fit(X_train, y_train, 
        eval_set=[(X_valid, y_valid)], 
        verbose=False)
rmse_xgb, r2_xgb, f1_xgb = evaluate_model(xgb, X_valid, y_valid, "XGBoost")

print("\n[5/6] Training LightGBM (Fast Gradient Boosting)...")
lgbm = lgb.LGBMRegressor(
    n_estimators=1500,          # Increased trees
    max_depth=8,                # Deeper trees
    learning_rate=0.008,        # Lower learning rate
    num_leaves=60,              # More leaves for complexity
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_samples=15,       # Allow smaller leaves
    reg_alpha=0.05,
    reg_lambda=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm.fit(X_train, y_train,
         eval_set=[(X_valid, y_valid)],
         callbacks=[lgb.early_stopping(50, verbose=False)])
rmse_lgbm, r2_lgbm, f1_lgbm = evaluate_model(lgbm, X_valid, y_valid, "LightGBM")

print("\n[6/6] Training Gradient Boosting (Sklearn)...")
gb = GradientBoostingRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.02,
    subsample=0.8,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    verbose=0
)
gb.fit(X_train, y_train)
rmse_gb, r2_gb, f1_gb = evaluate_model(gb, X_valid, y_valid, "Gradient Boosting")

print("\n" + "="*60)
print("CREATING ENSEMBLE STACK (Meta-Model)")
print("="*60)

# Create a stacking ensemble - combines all models
print("\nTraining Stacking Ensemble...")
base_models = [
    ('rf', RandomForestRegressor(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1)),
    ('xgb', XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.01, random_state=42, n_jobs=-1)),
    ('lgbm', lgb.LGBMRegressor(n_estimators=500, max_depth=7, learning_rate=0.01, random_state=42, n_jobs=-1, verbose=-1)),
    ('gb', GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.02, random_state=42))
]

# Meta-model: Ridge regression to combine predictions
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=0.1),
    cv=5,
    n_jobs=-1
)
print("Fitting stacking ensemble (this may take a minute)...")
stacking_model.fit(X_train, y_train)
rmse_stack, r2_stack, f1_stack = evaluate_model(stacking_model, X_valid, y_valid, "Stacking Ensemble")

# 4. Select Best Model
print(f"\n{'='*60}")
print(f"MODEL COMPARISON (Validation Set)")
print(f"{'='*60}")
print(f"Ridge Regression R²:     {r2_ridge:.4f}")
print(f"Elastic Net R²:          {r2_elastic:.4f}")
print(f"Random Forest R²:        {r2_rf:.4f}")
print(f"XGBoost R²:              {r2_xgb:.4f}")
print(f"LightGBM R²:             {r2_lgbm:.4f}")
print(f"Gradient Boosting R²:    {r2_gb:.4f}")
print(f"Stacking Ensemble R²:    {r2_stack:.4f}")

# Select model with highest R²
models = [
    (ridge, "Ridge Regression", r2_ridge),
    (elastic, "Elastic Net", r2_elastic),
    (rf, "Random Forest", r2_rf),
    (xgb, "XGBoost", r2_xgb),
    (lgbm, "LightGBM", r2_lgbm),
    (gb, "Gradient Boosting", r2_gb),
    (stacking_model, "Stacking Ensemble", r2_stack)
]
best_model, best_name, best_r2 = max(models, key=lambda x: x[2])
print(f"\n🏆 Best Model: {best_name} (R²: {best_r2:.4f})")
print(f"{'='*60}\n")

# 5. Evaluate on Test Set
print(f"{'='*60}")
print(f"FINAL EVALUATION ON TEST SET")
print(f"{'='*60}")
evaluate_model(best_model, X_test, y_test, best_name)

# 6. Save Model and Metadata
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"\n✓ Model saved to 'best_model.pkl'")

with open('model_features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print(f"✓ Feature names saved to 'model_features.pkl'")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler saved to 'scaler.pkl'")

scaling_stats = {
    'min': DATA_MIN,
    'max': DATA_MAX,
    'mean': DATA_MEAN,
    'std': DATA_STD
}
with open('scaling_stats.pkl', 'wb') as f:
    pickle.dump(scaling_stats, f)
print(f"✓ Scaling statistics saved to 'scaling_stats.pkl'")
print(f"\n{'='*60}")
print(f"TRAINING COMPLETE!")
print(f"Best R² Score: {best_r2:.4f} ({(best_r2*100):.2f}% variance explained)")
print(f"{'='*60}")