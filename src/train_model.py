import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, f1_score, r2_score
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available, using GradientBoosting instead")

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
print(f"  Skewness: {poverty_prob.skew():.4f}")

# Use ALL available features (excluding target)
print("\nEngineering comprehensive feature set...")
exclude_cols = ['poverty_probability', 'row_id']
base_features = [col for col in df.columns if col not in exclude_cols]
df_features = df[base_features].copy()

# === COMPREHENSIVE FEATURE ENGINEERING ===

# 1. Financial Inclusion Composite Indices
df_features['financial_inclusion_index'] = (df['reg_bank_acct'] + df['reg_mm_acct'] + df['active_bank_user'] + df['active_mm_user'] + df['financially_included'])
df_features['digital_capability'] = (df['can_use_internet'] + df['can_make_transaction'] + df['can_text'] + df['can_call'] + df['phone_technology'])
df_features['financial_literacy'] = (df['can_add'] + df['can_divide'] + df['can_calc_percents'] + df['can_calc_compounding'] + df['literacy'])

# 2. Income Features
income_cols = [col for col in df.columns if 'income_' in col and '_last_year' in col]
df_features['total_income'] = df[income_cols].sum(axis=1)
df_features['income_diversity'] = (df[income_cols] > 0).sum(axis=1)
df_features['income_per_source'] = df_features['total_income'] / (df_features['income_diversity'] + 1)

# 3. Savings Features
df_features['total_savings'] = df['formal_savings'] + df['informal_savings'] + df['cash_property_savings']
df_features['savings_diversity'] = (df['formal_savings'] > 0).astype(int) + (df['informal_savings'] > 0).astype(int) + (df['cash_property_savings'] > 0).astype(int)

# 4. Financial Stress Indicators
df_features['borrowing_intensity'] = df['num_times_borrowed_last_year']
df_features['financial_stress'] = df['num_times_borrowed_last_year'] + df['borrowed_for_emergency_last_year'] + df['borrowed_for_daily_expenses_last_year']
df_features['emergency_borrowing_ratio'] = df['borrowed_for_emergency_last_year'] / (df['num_times_borrowed_last_year'] + 1)

# 5. Shock Vulnerability
df_features['shock_exposure'] = df['num_shocks_last_year'] * df['avg_shock_strength_last_year']
df_features['shock_resilience'] = df_features['total_savings'] / (df_features['shock_exposure'] + 1)

# 6. Institutional Usage
df_features['formal_institution_usage'] = df['num_formal_institutions_last_year']
df_features['informal_institution_usage'] = df['num_informal_institutions_last_year']
df_features['institution_preference'] = df_features['formal_institution_usage'] / (df_features['informal_institution_usage'] + 1)

# 7. Key Two-Way Interactions
df_features['education_x_urban'] = df['education_level'] * df['is_urban']
df_features['education_x_employed'] = df['education_level'] * df['employed_last_year']
df_features['urban_x_phone_tech'] = df['is_urban'] * df['phone_technology']
df_features['urban_x_financial_inclusion'] = df['is_urban'] * df_features['financial_inclusion_index']
df_features['education_x_financial_literacy'] = df['education_level'] * df_features['financial_literacy']
df_features['education_x_income'] = df['education_level'] * df_features['total_income']
df_features['literacy_x_income'] = df_features['financial_literacy'] * df_features['total_income']

# 8. Polynomial Features (capture non-linear relationships)
df_features['education_squared'] = df['education_level'] ** 2
df_features['education_cubed'] = df['education_level'] ** 3
df_features['age_squared'] = df['age'] ** 2
df_features['financial_activities_squared'] = df['num_financial_activities_last_year'] ** 2

# 9. Important Ratios
df_features['savings_to_income'] = df_features['total_savings'] / (df_features['total_income'] + 1)
df_features['borrowing_to_income'] = df_features['borrowing_intensity'] / (df_features['total_income'] + 1)
df_features['formal_to_informal_institutions'] = df_features['formal_institution_usage'] / (df_features['informal_institution_usage'] + 1)
df_features['financial_activity_intensity'] = df['num_financial_activities_last_year'] / (df['age'] + 1)

# 10. Country-specific interactions (capture regional differences)
for country in ['country_A', 'country_D', 'country_F', 'country_I']:
    if country in df.columns:
        df_features[f'{country}_x_education'] = df[country] * df['education_level']
        df_features[f'{country}_x_urban'] = df[country] * df['is_urban']
        df_features[f'{country}_x_income'] = df[country] * df_features['total_income']

# 11. Three-way interactions (capture complex relationships)
df_features['edu_urban_tech'] = df['education_level'] * df['is_urban'] * df['phone_technology']
df_features['urban_digital_financial'] = df['is_urban'] * df_features['digital_capability'] * df_features['financial_inclusion_index']
df_features['education_digital_employed'] = df['education_level'] * df_features['digital_capability'] * df['employed_last_year']

# 12. Log transformations (handle skewed distributions)
df_features['log_age'] = np.log1p(df['age'])
df_features['log_income'] = np.log1p(df_features['total_income'])
df_features['log_financial_activities'] = np.log1p(df['num_financial_activities_last_year'])
df_features['log_borrowing'] = np.log1p(df['num_times_borrowed_last_year'])

# 13. Advanced ratios and indicators
df_features['digital_literacy_score'] = df_features['digital_capability'] * df_features['financial_literacy']
df_features['economic_stability'] = df_features['total_income'] / (df_features['financial_stress'] + 1)
df_features['financial_health'] = (df_features['total_savings'] + df_features['total_income']) / (df_features['borrowing_intensity'] + 1)
df_features['resource_utilization'] = df['num_financial_activities_last_year'] * df_features['financial_inclusion_index']
df_features['vulnerability_index'] = df_features['shock_exposure'] / (df_features['total_savings'] + 1)

# 14. Interaction with household size if available
if 'hh_size' in df.columns:
    df_features['income_per_capita'] = df_features['total_income'] / (df['hh_size'] + 1)
    df_features['savings_per_capita'] = df_features['total_savings'] / (df['hh_size'] + 1)
    df_features['education_hh_interaction'] = df['education_level'] * df['hh_size']

# 15. Square root transformations for heavily skewed features
df_features['sqrt_income'] = np.sqrt(df_features['total_income'])
df_features['sqrt_savings'] = np.sqrt(df_features['total_savings'])
df_features['sqrt_borrowing'] = np.sqrt(df_features['borrowing_intensity'])

# Include all engineered features
selected_features = list(df_features.columns)

# แยก Feature และ Target
target_col = 'poverty_probability'
X = df_features
y = df[target_col]
print(f"Using {len(selected_features)} features (including interactions): {selected_features}")

# Manual Z-Score Normalization on entire dataset BEFORE splitting
print("\nApplying Manual Z-Score Normalization on entire dataset...")
data_mean = X.mean()
data_std = X.std()

# Normalize features: (X - mean) / std
X_scaled = (X - data_mean) / data_std
X_scaled = X_scaled.fillna(0).replace([np.inf, -np.inf], 0)

# Normalize target variable as well
target_mean = y.mean()
target_std = y.std()
y_scaled = (y - target_mean) / target_std

print(f"Normalized features - Mean: {X_scaled.mean().mean():.6f}, Std: {X_scaled.std().mean():.6f}")
print(f"Normalized target - Mean: {y_scaled.mean():.6f}, Std: {y_scaled.std():.6f}")

# Store normalization parameters for later use
train_mean = data_mean
train_std = data_std


y_bins = pd.qcut(y, q=4, labels=False, duplicates='drop')
X_train_scaled, X_temp, y_train_scaled, y_temp, bins_train, bins_temp = train_test_split(
    X_scaled, y_scaled, y_bins, test_size=0.3, random_state=42, stratify=y_bins
)
X_valid_scaled, X_test_scaled, y_valid_scaled, y_test_scaled = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=bins_temp
)

print(f"Data Split: Train={X_train_scaled.shape[0]}, Validation={X_valid_scaled.shape[0]}, Test={X_test_scaled.shape[0]}")
print(f"Normalized target distribution - Train mean: {y_train_scaled.mean():.6f}, Valid mean: {y_valid_scaled.mean():.6f}, Test mean: {y_test_scaled.mean():.6f}")

print(f"Z-Score Normalization completed. Features and target normalized with entire dataset statistics")

# ฟังก์ชันคำนวณ Metrics
def evaluate_model(model, X, y_true_scaled, name, target_mean, target_std):
    y_pred_scaled = model.predict(X)
    
    # Denormalize predictions and true values to original scale
    y_pred = y_pred_scaled * target_std + target_mean
    y_true = y_true_scaled * target_std + target_mean
    
    # Evaluate on original scale (0-1) for better metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # คำนวณ F1 โดยตัดที่ threshold 0.5 (แปลงเป็นจน/ไม่จน)
    y_pred_bin = (y_pred >= 0.5).astype(int)
    y_true_bin = (y_true >= 0.5).astype(int)
    f1 = f1_score(y_true_bin, y_pred_bin)
    print(f"[{name}] RMSE: {rmse:.4f}, R²: {r2:.4f}, F1: {f1:.4f}")
    return rmse, r2, f1

# 3. เทรนโมเดล
print("\nTraining Ridge Regression (Linear Model)...")
# Test multiple alpha values for Ridge
ridge = Ridge(alpha=0.0001, random_state=42)  # Very low regularization
ridge.fit(X_train_scaled, y_train_scaled)
rmse_ridge, r2_ridge, f1_ridge = evaluate_model(ridge, X_valid_scaled, y_valid_scaled, "Ridge Regression", target_mean, target_std)

print("\nTraining Random Forest...")
# Improved hyperparameters for Random Forest
rf = RandomForestRegressor(
    n_estimators=500,        # More trees for stability
    random_state=42, 
    n_jobs=-1, 
    max_depth=25,            # Deeper trees
    min_samples_split=2,     # Allow smaller splits
    min_samples_leaf=1,      # Smaller leaf size
    max_features='sqrt',     # Feature sampling strategy
    bootstrap=True,          # Bootstrap sampling
    oob_score=True          # Out-of-bag score for validation
)
rf.fit(X_train_scaled, y_train_scaled)  # Using z-score normalized data
rmse_rf, r2_rf, f1_rf = evaluate_model(rf, X_valid_scaled, y_valid_scaled, "Random Forest", target_mean, target_std)
print(f"Random Forest OOB Score: {rf.oob_score_:.4f}")

print("\nTraining XGBoost (Optimized)...")
# Highly optimized XGBoost configuration for improved R²
xgb = XGBRegressor(
    n_estimators=2000,        # More boosting rounds
    random_state=42, 
    n_jobs=-1, 
    max_depth=8,              # Deeper trees for complex patterns
    learning_rate=0.008,      # Lower learning rate for better convergence
    subsample=0.85,           # Slightly higher subsample
    colsample_bytree=0.85,    # Feature sampling
    colsample_bylevel=0.9,    # Additional feature sampling
    min_child_weight=2,       # Slightly higher for regularization
    gamma=0.01,               # Lower gamma for more splits
    reg_alpha=0.05,           # L1 regularization
    reg_lambda=2.0,           # Stronger L2 regularization
    tree_method='hist',       # Fast histogram-based training
    max_bin=512,              # More bins for finer splits
    early_stopping_rounds=100 # More patience before stopping
)
# Train with early stopping using z-score normalized data
xgb.fit(X_train_scaled, y_train_scaled, eval_set=[(X_valid_scaled, y_valid_scaled)], verbose=False)
rmse_xgb, r2_xgb, f1_xgb = evaluate_model(xgb, X_valid_scaled, y_valid_scaled, "XGBoost Optimized", target_mean, target_std)

# Train additional models for stronger ensemble
if LIGHTGBM_AVAILABLE:
    print("\nTraining LightGBM...")
    lgbm = LGBMRegressor(
        n_estimators=2000,
        random_state=42,
        n_jobs=-1,
        max_depth=8,
        learning_rate=0.008,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_alpha=0.05,
        reg_lambda=2.0,
        num_leaves=64,          # More leaves for complex patterns
        min_data_in_leaf=15,    # Minimum data per leaf
        feature_fraction=0.85,  # Feature sampling per tree
        bagging_fraction=0.85,  # Row sampling
        bagging_freq=5,         # Bagging frequency
        verbose=-1
    )
    lgbm.fit(X_train_scaled, y_train_scaled, eval_set=[(X_valid_scaled, y_valid_scaled)], callbacks=[])
    rmse_lgbm, r2_lgbm, f1_lgbm = evaluate_model(lgbm, X_valid_scaled, y_valid_scaled, "LightGBM", target_mean, target_std)
else:
    print("\nTraining Gradient Boosting...")
    lgbm = GradientBoostingRegressor(
        n_estimators=500,
        random_state=42,
        max_depth=7,
        learning_rate=0.01,
        subsample=0.8,
        min_samples_split=5,
        min_samples_leaf=2
    )
    lgbm.fit(X_train_scaled, y_train_scaled)
    rmse_lgbm, r2_lgbm, f1_lgbm = evaluate_model(lgbm, X_valid_scaled, y_valid_scaled, "Gradient Boosting", target_mean, target_std)

# 4. Create Ensemble Stacking (combining predictions for better performance)
print("\n" + "="*50)
print("Creating Ensemble Model (Stacking)...")
print("="*50)

# Get predictions from all models on validation set (all using z-score normalized data)
pred_ridge_valid = ridge.predict(X_valid_scaled)
pred_rf_valid = rf.predict(X_valid_scaled)
pred_xgb_valid = xgb.predict(X_valid_scaled)
pred_lgbm_valid = lgbm.predict(X_valid_scaled)

# Stack predictions as new features
X_valid_stacked = np.column_stack([pred_ridge_valid, pred_rf_valid, pred_xgb_valid, pred_lgbm_valid])

# Use weighted average based on individual model performance (inverse of RMSE)
from sklearn.linear_model import Ridge as MetaRidge

# Calculate weights based on validation R² scores (better models get higher weights)
weights = np.array([r2_ridge, r2_rf, r2_xgb, r2_lgbm])
weights = np.maximum(weights, 0)  # Ensure non-negative
weights = weights / weights.sum()  # Normalize to sum to 1
print(f"Model weights - Ridge: {weights[0]:.3f}, RF: {weights[1]:.3f}, XGB: {weights[2]:.3f}, LGBM: {weights[3]:.3f}")

# Weighted ensemble prediction
pred_ensemble_valid_weighted = (pred_ridge_valid * weights[0] + 
                                 pred_rf_valid * weights[1] + 
                                 pred_xgb_valid * weights[2] + 
                                 pred_lgbm_valid * weights[3])

# Also train meta-model for comparison
meta_model = MetaRidge(alpha=0.1, random_state=42)
meta_model.fit(X_valid_stacked, y_valid_scaled)
pred_ensemble_valid_meta = meta_model.predict(X_valid_stacked)

# Denormalize predictions for evaluation
y_valid_original = y_valid_scaled * target_std + target_mean
pred_weighted_original = pred_ensemble_valid_weighted * target_std + target_mean
pred_meta_original = pred_ensemble_valid_meta * target_std + target_mean

# Compare weighted vs meta-model
rmse_weighted = np.sqrt(mean_squared_error(y_valid_original, pred_weighted_original))
r2_weighted = r2_score(y_valid_original, pred_weighted_original)
rmse_meta = np.sqrt(mean_squared_error(y_valid_original, pred_meta_original))
r2_meta = r2_score(y_valid_original, pred_meta_original)

print(f"Weighted Ensemble R²: {r2_weighted:.4f}")
print(f"Meta-Model Ensemble R²: {r2_meta:.4f}")

# Use the better ensemble method
if r2_weighted > r2_meta:
    print("Using Weighted Ensemble")
    use_weighted = True
    pred_ensemble_valid = pred_ensemble_valid_weighted
else:
    print("Using Meta-Model Ensemble")
    use_weighted = False
    pred_ensemble_valid = pred_ensemble_valid_meta

# Get stacked predictions for test set (all using z-score normalized data)
pred_ridge_test = ridge.predict(X_test_scaled)
pred_rf_test = rf.predict(X_test_scaled)
pred_xgb_test = xgb.predict(X_test_scaled)
pred_lgbm_test = lgbm.predict(X_test_scaled)

if use_weighted:
    pred_ensemble_test = (pred_ridge_test * weights[0] + 
                          pred_rf_test * weights[1] + 
                          pred_xgb_test * weights[2] + 
                          pred_lgbm_test * weights[3])
else:
    X_test_stacked = np.column_stack([pred_ridge_test, pred_rf_test, pred_xgb_test, pred_lgbm_test])
    pred_ensemble_test = meta_model.predict(X_test_stacked)

# Denormalize test predictions and true values
y_test_original = y_test_scaled * target_std + target_mean
pred_ensemble_test_original = pred_ensemble_test * target_std + target_mean

# Evaluate ensemble on test set
rmse_ensemble = np.sqrt(mean_squared_error(y_test_original, pred_ensemble_test_original))
r2_ensemble = r2_score(y_test_original, pred_ensemble_test_original)
y_pred_bin = (pred_ensemble_test_original >= 0.5).astype(int)
y_test_bin = (y_test_original >= 0.5).astype(int)
f1_ensemble = f1_score(y_test_bin, y_pred_bin)

# Use the validation R² from the chosen ensemble method
r2_ensemble_valid = r2_weighted if use_weighted else r2_meta

print(f"\n{'='*50}")
print(f"Model Comparison on Validation Set:")
print(f"Ridge R²: {r2_ridge:.4f}")
print(f"Random Forest R²: {r2_rf:.4f}")
print(f"XGBoost R²: {r2_xgb:.4f}")
print(f"{'LightGBM' if LIGHTGBM_AVAILABLE else 'GradientBoosting'} R²: {r2_lgbm:.4f}")
print(f"Ensemble R²: {r2_ensemble_valid:.4f}")
print(f"{'='*50}")
print(f"Test Set Results:")
print(f"Ensemble Test R²: {r2_ensemble:.4f}")

# Force XGBoost as the best model
best_model = xgb
best_name = "XGBoost"
best_r2 = r2_xgb

print(f"\nUsing XGBoost as selected model (Validation R²: {r2_xgb:.4f})")
print(f"{'='*50}\n")

# 5. ทดสอบกับ Test Set
print("\nEvaluating on Test Set...")
evaluate_model(best_model, X_test_scaled, y_test_scaled, best_name, target_mean, target_std)

# 6. บันทึกโมเดล (XGBoost)
model_package = {
    'type': 'individual',
    'model': best_model,
    'model_name': best_name,
    'train_mean': train_mean.to_dict(),
    'train_std': train_std.to_dict(),
    'target_mean': float(target_mean),
    'target_std': float(target_std)
}
with open('best_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)
print(f"{best_name} saved to 'best_model.pkl'")

# บันทึกรายชื่อ Feature ไว้ใช้กับเว็บ
with open('model_features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("Feature names saved to 'model_features.pkl'")

# บันทึกสถิติข้อมูลสำหรับการสเกลผลการทำนาย
scaling_stats = {
    'min': DATA_MIN,
    'max': DATA_MAX,
    'mean': DATA_MEAN,
    'std': DATA_STD
}
with open('scaling_stats.pkl', 'wb') as f:
    pickle.dump(scaling_stats, f)
print(f"Scaling statistics saved to 'scaling_stats.pkl': {scaling_stats}")

