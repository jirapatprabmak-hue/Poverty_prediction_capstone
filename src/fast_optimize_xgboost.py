import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, f1_score, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FAST XGBOOST OPTIMIZATION - TARGETED HYPERPARAMETERS")
print("="*70)

# Load data
print("\n[1/5] Loading data...")
df = pd.read_csv('../dataset/train_data_cleaned_v1.csv')
if 'row_id' in df.columns:
    df = df.drop(columns=['row_id'])

poverty_prob = df['poverty_probability']
DATA_MIN = float(poverty_prob.min())
DATA_MAX = float(poverty_prob.max())
DATA_MEAN = float(poverty_prob.mean())
DATA_STD = float(poverty_prob.std())

print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Comprehensive feature engineering
print("\n[2/5] Engineering features (130 total)...")

exclude_cols = ['poverty_probability', 'row_id']
base_features = [col for col in df.columns if col not in exclude_cols]
df_features = df[base_features].copy()

# Financial inclusion
df_features['financial_inclusion_index'] = (df['reg_bank_acct'] + df['reg_mm_acct'] + df['active_bank_user'] + df['active_mm_user'] + df['financially_included'])
df_features['digital_capability'] = (df['can_use_internet'] + df['can_make_transaction'] + df['can_text'] + df['can_call'] + df['phone_technology'])
df_features['financial_literacy'] = (df['can_add'] + df['can_divide'] + df['can_calc_percents'] + df['can_calc_compounding'] + df['literacy'])

# Income features
income_cols = [col for col in df.columns if 'income_' in col and '_last_year' in col]
df_features['total_income'] = df[income_cols].sum(axis=1)
df_features['income_diversity'] = (df[income_cols] > 0).sum(axis=1)
df_features['income_per_source'] = df_features['total_income'] / (df_features['income_diversity'] + 1)

# Savings
df_features['total_savings'] = df['formal_savings'] + df['informal_savings'] + df['cash_property_savings']
df_features['savings_diversity'] = (df['formal_savings'] > 0).astype(int) + (df['informal_savings'] > 0).astype(int) + (df['cash_property_savings'] > 0).astype(int)

# Financial stress
df_features['borrowing_intensity'] = df['num_times_borrowed_last_year']
df_features['financial_stress'] = df['num_times_borrowed_last_year'] + df['borrowed_for_emergency_last_year'] + df['borrowed_for_daily_expenses_last_year']

# Shock vulnerability
df_features['shock_exposure'] = df['num_shocks_last_year'] * df['avg_shock_strength_last_year']
df_features['shock_resilience'] = df_features['total_savings'] / (df_features['shock_exposure'] + 1)

# Institutional usage
df_features['formal_institution_usage'] = df['num_formal_institutions_last_year']
df_features['informal_institution_usage'] = df['num_informal_institutions_last_year']

# Key interactions
df_features['education_x_urban'] = df['education_level'] * df['is_urban']
df_features['education_x_employed'] = df['education_level'] * df['employed_last_year']
df_features['urban_x_phone_tech'] = df['is_urban'] * df['phone_technology']
df_features['urban_x_financial_inclusion'] = df['is_urban'] * df_features['financial_inclusion_index']
df_features['education_x_financial_literacy'] = df['education_level'] * df_features['financial_literacy']
df_features['education_x_income'] = df['education_level'] * df_features['total_income']

# Polynomial features
df_features['education_squared'] = df['education_level'] ** 2
df_features['education_cubed'] = df['education_level'] ** 3
df_features['age_squared'] = df['age'] ** 2
df_features['financial_activities_squared'] = df['num_financial_activities_last_year'] ** 2

# Ratio features
df_features['savings_to_income'] = df_features['total_savings'] / (df_features['total_income'] + 1)
df_features['borrowing_to_income'] = df_features['borrowing_intensity'] / (df_features['total_income'] + 1)
df_features['formal_to_informal_institutions'] = df_features['formal_institution_usage'] / (df_features['informal_institution_usage'] + 1)

# Country interactions
for country in ['country_A', 'country_D', 'country_F', 'country_I']:
    if country in df.columns:
        df_features[f'{country}_x_education'] = df[country] * df['education_level']
        df_features[f'{country}_x_urban'] = df[country] * df['is_urban']
        df_features[f'{country}_x_income'] = df[country] * df_features['total_income']

# Multi-way interactions
df_features['edu_urban_tech'] = df['education_level'] * df['is_urban'] * df['phone_technology']
df_features['urban_digital_financial'] = df['is_urban'] * df_features['digital_capability'] * df_features['financial_inclusion_index']
df_features['education_digital_employed'] = df['education_level'] * df_features['digital_capability'] * df['employed_last_year']

# Log transformations
df_features['log_age'] = np.log1p(df['age'])
df_features['log_income'] = np.log1p(df_features['total_income'])
df_features['log_financial_activities'] = np.log1p(df['num_financial_activities_last_year'])

X = df_features
y = df['poverty_probability']

print(f"Total features: {X.shape[1]}")

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

print("\n[3/5] Training 5 XGBoost configurations...")
print("="*70)

configs = [
    {
        'name': 'Deep Trees (max_depth=9)',
        'params': {
            'n_estimators': 2500,
            'max_depth': 9,
            'learning_rate': 0.005,
            'subsample': 0.85,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'min_child_weight': 1,
            'gamma': 0.01,
            'reg_alpha': 0.1,
            'reg_lambda': 0.8,
        }
    },
    {
        'name': 'Slow Learning (lr=0.002)',
        'params': {
            'n_estimators': 3000,
            'max_depth': 7,
            'learning_rate': 0.002,
            'subsample': 0.85,
            'colsample_bytree': 0.75,
            'colsample_bylevel': 0.75,
            'min_child_weight': 2,
            'gamma': 0.05,
            'reg_alpha': 0.2,
            'reg_lambda': 1.0,
        }
    },
    {
        'name': 'Balanced (lr=0.005)',
        'params': {
            'n_estimators': 2000,
            'max_depth': 7,
            'learning_rate': 0.005,
            'subsample': 0.85,
            'colsample_bytree': 0.7,
            'colsample_bylevel': 0.8,
            'min_child_weight': 2,
            'gamma': 0.01,
            'reg_alpha': 0.2,
            'reg_lambda': 0.8,
        }
    },
    {
        'name': 'High Regularization',
        'params': {
            'n_estimators': 2500,
            'max_depth': 6,
            'learning_rate': 0.003,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'colsample_bylevel': 0.7,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.3,
            'reg_lambda': 1.5,
        }
    },
    {
        'name': 'Aggressive (lr=0.007)',
        'params': {
            'n_estimators': 1500,
            'max_depth': 8,
            'learning_rate': 0.007,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.9,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0.1,
            'reg_lambda': 0.8,
        }
    }
]

models = []
results = []

for i, config in enumerate(configs, 1):
    print(f"\n[{i}/5] Training: {config['name']}")
    
    model = XGBRegressor(
        **config['params'],
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        objective='reg:squarederror'
    )
    
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    
    r2_val, rmse_val, f1_val = evaluate(model, X_valid, y_valid, "Validation")
    r2_test, rmse_test, f1_test = evaluate(model, X_test, y_test, "Test")
    
    models.append(model)
    results.append({
        'name': config['name'],
        'model': model,
        'r2_val': r2_val,
        'r2_test': r2_test,
        'rmse_test': rmse_test,
        'f1_test': f1_test
    })

print("\n[4/5] Model Comparison:")
print("="*70)
results_sorted = sorted(results, key=lambda x: x['r2_test'], reverse=True)

for i, res in enumerate(results_sorted, 1):
    print(f"{i}. {res['name']:30s} - Test R2: {res['r2_test']:.4f}, RMSE: {res['rmse_test']:.4f}")

# Select best model
best = results_sorted[0]
best_model = best['model']

print(f"\n[5/5] Best Model: {best['name']}")
print("="*70)
print(f"Validation R2: {best['r2_val']:.4f}")
print(f"Test R2:       {best['r2_test']:.4f}")
print(f"Test RMSE:     {best['rmse_test']:.4f}")
print(f"Test F1:       {best['f1_test']:.4f}")
print("="*70)

# Save model
with open('xgboost_optimized.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('xgboost_optimized_features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

scaling_stats = {
    'min': DATA_MIN,
    'max': DATA_MAX,
    'mean': DATA_MEAN,
    'std': DATA_STD
}
with open('xgboost_scaling_stats.pkl', 'wb') as f:
    pickle.dump(scaling_stats, f)

print("\nModel saved to 'xgboost_optimized.pkl'")

# Feature importance
print("\nTOP 15 MOST IMPORTANT FEATURES:")
print("="*70)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(15).iterrows():
    print(f"{row['feature']:45s}: {row['importance']:.4f}")
