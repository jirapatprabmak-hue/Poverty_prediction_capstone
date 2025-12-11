import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, f1_score, r2_score
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available")

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
df_features['urban_x_literacy'] = df['is_urban'] * df_features['financial_literacy']

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
        df_features[f'{country}_x_financial_lit'] = df[country] * df_features['financial_literacy']

# 11. Three-way interactions (capture complex relationships)
df_features['edu_urban_tech'] = df['education_level'] * df['is_urban'] * df['phone_technology']
df_features['urban_digital_financial'] = df['is_urban'] * df_features['digital_capability'] * df_features['financial_inclusion_index']
df_features['education_digital_employed'] = df['education_level'] * df_features['digital_capability'] * df['employed_last_year']

# 12. Log transformations (handle skewed distributions)
df_features['log_age'] = np.log1p(df['age'])
df_features['log_income'] = np.log1p(df_features['total_income'])
df_features['log_financial_activities'] = np.log1p(df['num_financial_activities_last_year'])
df_features['log_borrowing'] = np.log1p(df['num_times_borrowed_last_year'])

# Include all engineered features
selected_features = list(df_features.columns)

# แยก Feature และ Target
target_col = 'poverty_probability'
X = df_features
y = df[target_col]
print(f"Using {len(selected_features)} features (including interactions)")

# 2. แบ่งข้อมูล 60% Train, 20% Validation, 20% Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Data Split: Train={X_train.shape[0]}, Validation={X_valid.shape[0]}, Test={X_test.shape[0]}")

# Standardize features - helps with model convergence and performance
scaler = RobustScaler()  # More robust to outliers
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

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
    print(f"[{name}] RMSE: {rmse:.4f}, R2: {r2:.4f}, F1: {f1:.4f}")
    return rmse, r2, f1

# 3. Train multiple XGBoost models with different configurations
print("\n" + "="*60)
print("Training Multiple XGBoost Configurations")
print("="*60)

xgb_configs = [
    {
        'name': 'XGB_Deep',
        'params': {
            'n_estimators': 2000,
            'max_depth': 9,
            'learning_rate': 0.005,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'colsample_bylevel': 0.75,
            'min_child_weight': 0.5,
            'gamma': 0.01,
            'reg_alpha': 0.05,
            'reg_lambda': 2.0,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
    },
    {
        'name': 'XGB_Balanced',
        'params': {
            'n_estimators': 2000,
            'max_depth': 7,
            'learning_rate': 0.008,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'min_child_weight': 1,
            'gamma': 0.05,
            'reg_alpha': 0.1,
            'reg_lambda': 1.5,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
    },
    {
        'name': 'XGB_Conservative',
        'params': {
            'n_estimators': 2000,
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 2,
            'gamma': 0.1,
            'reg_alpha': 0.2,
            'reg_lambda': 2.5,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
    }
]

xgb_models = []
for config in xgb_configs:
    print(f"\nTraining {config['name']}...")
    model = XGBRegressor(**config['params'])
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    rmse, r2, f1 = evaluate_model(model, X_valid, y_valid, config['name'])
    xgb_models.append((model, config['name'], r2))

# Train other models
print("\n" + "="*60)
print("Training Additional Models")
print("="*60)

print("\nTraining Random Forest...")
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt'
)
rf.fit(X_train, y_train)
rmse_rf, r2_rf, f1_rf = evaluate_model(rf, X_valid, y_valid, "Random Forest")

print("\nTraining Extra Trees...")
et = ExtraTreesRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1
)
et.fit(X_train, y_train)
rmse_et, r2_et, f1_et = evaluate_model(et, X_valid, y_valid, "Extra Trees")

if LIGHTGBM_AVAILABLE:
    print("\nTraining LightGBM...")
    lgbm = LGBMRegressor(
        n_estimators=2000,
        random_state=42,
        n_jobs=-1,
        max_depth=8,
        learning_rate=0.008,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_alpha=0.1,
        reg_lambda=1.5,
        verbose=-1
    )
    lgbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    rmse_lgbm, r2_lgbm, f1_lgbm = evaluate_model(lgbm, X_valid, y_valid, "LightGBM")

# 4. Create Super Ensemble
print("\n" + "="*60)
print("Creating Super Ensemble (Stacking)")
print("="*60)

# Collect all predictions
all_preds_valid = []
all_preds_test = []
all_models = []

for model, name, r2 in xgb_models:
    all_preds_valid.append(model.predict(X_valid))
    all_preds_test.append(model.predict(X_test))
    all_models.append((model, name, r2))

all_preds_valid.append(rf.predict(X_valid))
all_preds_test.append(rf.predict(X_test))
all_models.append((rf, "Random Forest", r2_rf))

all_preds_valid.append(et.predict(X_valid))
all_preds_test.append(et.predict(X_test))
all_models.append((et, "Extra Trees", r2_et))

if LIGHTGBM_AVAILABLE:
    all_preds_valid.append(lgbm.predict(X_valid))
    all_preds_test.append(lgbm.predict(X_test))
    all_models.append((lgbm, "LightGBM", r2_lgbm))

# Stack all predictions
X_valid_stacked = np.column_stack(all_preds_valid)
X_test_stacked = np.column_stack(all_preds_test)

# Train meta-model
meta_model = Ridge(alpha=0.1, random_state=42)
meta_model.fit(X_valid_stacked, y_valid)

# Predict on test set
pred_ensemble_test = meta_model.predict(X_test_stacked)

# Evaluate ensemble
rmse_ensemble = np.sqrt(mean_squared_error(y_test, pred_ensemble_test))
r2_ensemble = r2_score(y_test, pred_ensemble_test)
y_pred_bin = (pred_ensemble_test >= 0.5).astype(int)
y_test_bin = (y_test >= 0.5).astype(int)
f1_ensemble = f1_score(y_test_bin, y_pred_bin)

print(f"\n{'='*60}")
print("Final Model Comparison:")
print(f"{'='*60}")
for model, name, r2 in sorted(all_models, key=lambda x: x[2], reverse=True):
    print(f"{name:25s} - Validation R2: {r2:.4f}")
print(f"{'Super Ensemble':25s} - Test R2: {r2_ensemble:.4f}")
print(f"{'='*60}")

# Select best model
best_individual = max(all_models, key=lambda x: x[2])
if r2_ensemble > best_individual[2]:
    best_model = "ensemble"
    best_name = "Super Ensemble"
    best_r2 = r2_ensemble
    print(f"\nBest Model: {best_name} (Test R2: {best_r2:.4f})")
else:
    best_model = best_individual[0]
    best_name = best_individual[1]
    best_r2 = best_individual[2]
    print(f"\nBest Model: {best_name} (Validation R2: {best_r2:.4f})")
    # Evaluate on test
    y_pred_test = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    print(f"  Test R2: {test_r2:.4f}")

# 5. Save model
print("\nSaving model...")
if best_model == "ensemble":
    ensemble_package = {
        'type': 'super_ensemble',
        'models': [m for m, _, _ in all_models],
        'model_names': [n for _, n, _ in all_models],
        'meta_model': meta_model,
        'scaler': scaler
    }
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(ensemble_package, f)
    print("Super Ensemble saved to 'best_model.pkl'")
else:
    model_package = {
        'type': 'individual',
        'model': best_model,
        'model_name': best_name,
        'scaler': scaler
    }
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    print(f"{best_name} saved to 'best_model.pkl'")

# บันทึกรายชื่อ Feature
with open('model_features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("Feature names saved to 'model_features.pkl'")

# บันทึกสถิติข้อมูล
scaling_stats = {
    'min': DATA_MIN,
    'max': DATA_MAX,
    'mean': DATA_MEAN,
    'std': DATA_STD
}
with open('scaling_stats.pkl', 'wb') as f:
    pickle.dump(scaling_stats, f)
print(f"Scaling statistics saved: {scaling_stats}")

print("\n" + "="*60)
print(f"Training Complete! Best Test R2: {r2_ensemble if best_model == 'ensemble' else test_r2:.4f}")
print("="*60)
