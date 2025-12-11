import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, r2_score
import warnings
import os
warnings.filterwarnings('ignore')

print("="*60)
print("IMPROVED POVERTY PREDICTION MODEL TRAINING")
print("="*60)

# Calculate data statistics
DATA_MIN = None
DATA_MAX = None
DATA_MEAN = None
DATA_STD = None

# 1. Load Data
print("\n[Step 1/10] Loading data...")
df = pd.read_csv('../dataset/train_data_cleaned_v1.csv')

if 'row_id' in df.columns:
    df = df.drop(columns=['row_id'])

# Calculate stats
poverty_prob = df['poverty_probability']
DATA_MIN = float(poverty_prob.min())
DATA_MAX = float(poverty_prob.max())
DATA_MEAN = float(poverty_prob.mean())
DATA_STD = float(poverty_prob.std())

print(f"  Data shape: {df.shape}")
print(f"  Statistics - Min: {DATA_MIN:.4f}, Max: {DATA_MAX:.4f}, Mean: {DATA_MEAN:.4f}, Std: {DATA_STD:.4f}")

# 2. Feature Engineering
print("\n[Step 2/10] Engineering features...")

# Top features
top_features = [
    'education_level', 'is_urban', 'phone_technology', 'can_use_internet',
    'can_text', 'num_financial_activities_last_year', 'formal_savings',
    'phone_ownership', 'advanced_phone_use', 'active_bank_user'
]

# Additional features
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

print(f"  Total features: {len(df_features.columns)}")

# Split data
X = df_features
y = df['poverty_probability']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"  Train: {X_train.shape[0]}, Valid: {X_valid.shape[0]}, Test: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_valid_scaled = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns, index=X_valid.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# 3. Evaluation Function
def evaluate_model(model, X, y_true, name):
    y_pred = np.clip(model.predict(X), 0, 1)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    y_pred_bin = (y_pred >= 0.5).astype(int)
    y_true_bin = (y_true >= 0.5).astype(int)
    f1 = f1_score(y_true_bin, y_pred_bin)
    accuracy = np.mean(y_pred_bin == y_true_bin)
    
    print(f"  [{name}]")
    print(f"    RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
    print(f"    F1: {f1:.4f} | Accuracy: {accuracy:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'f1': f1, 'accuracy': accuracy, 'predictions': y_pred}

# 4. Train Models
print("\n[Step 3/10] Training Ridge Regression...")
ridge = Ridge(alpha=0.01, random_state=42)
ridge.fit(X_train_scaled, y_train)
ridge_metrics = evaluate_model(ridge, X_valid_scaled, y_valid, "Ridge")

print("\n[Step 4/10] Training Elastic Net...")
elastic = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=10000)
elastic.fit(X_train_scaled, y_train)
elastic_metrics = evaluate_model(elastic, X_valid_scaled, y_valid, "Elastic Net")

print("\n[Step 5/10] Training Random Forest...")
rf = RandomForestRegressor(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_metrics = evaluate_model(rf, X_valid, y_valid, "Random Forest")

print("\n[Step 6/10] Training XGBoost...")
xgb = XGBRegressor(n_estimators=2000, max_depth=7, learning_rate=0.01, 
                   subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
xgb_metrics = evaluate_model(xgb, X_valid, y_valid, "XGBoost")

print("\n[Step 7/10] Training LightGBM...")
lgbm = lgb.LGBMRegressor(n_estimators=2000, max_depth=10, learning_rate=0.01,
                         num_leaves=80, random_state=42, n_jobs=-1, verbose=-1)
lgbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
         callbacks=[lgb.early_stopping(100, verbose=False)])
lgbm_metrics = evaluate_model(lgbm, X_valid, y_valid, "LightGBM")

print("\n[Step 8/10] Training Gradient Boosting...")
gb = GradientBoostingRegressor(n_estimators=800, max_depth=7, learning_rate=0.02,
                               subsample=0.8, random_state=42)
gb.fit(X_train, y_train)
gb_metrics = evaluate_model(gb, X_valid, y_valid, "Gradient Boosting")

print("\n[Step 9/10] Creating Weighted Ensemble...")
y_pred_xgb = np.clip(xgb.predict(X_valid), 0, 1)
y_pred_lgbm = np.clip(lgbm.predict(X_valid), 0, 1)
y_pred_gb = np.clip(gb.predict(X_valid), 0, 1)
y_pred_rf = np.clip(rf.predict(X_valid), 0, 1)

w_xgb, w_lgbm, w_gb, w_rf = 0.35, 0.35, 0.20, 0.10
y_pred_weighted = w_xgb * y_pred_xgb + w_lgbm * y_pred_lgbm + w_gb * y_pred_gb + w_rf * y_pred_rf

weighted_metrics = {
    'rmse': np.sqrt(mean_squared_error(y_valid, y_pred_weighted)),
    'mae': mean_absolute_error(y_valid, y_pred_weighted),
    'r2': r2_score(y_valid, y_pred_weighted),
    'f1': f1_score((y_valid >= 0.5).astype(int), (y_pred_weighted >= 0.5).astype(int)),
    'accuracy': np.mean((y_valid >= 0.5) == (y_pred_weighted >= 0.5)),
    'predictions': y_pred_weighted
}

print(f"  [Weighted Ensemble]")
print(f"    RMSE: {weighted_metrics['rmse']:.4f} | MAE: {weighted_metrics['mae']:.4f} | R2: {weighted_metrics['r2']:.4f}")
print(f"    F1: {weighted_metrics['f1']:.4f} | Accuracy: {weighted_metrics['accuracy']:.4f}")

# 5. Model Comparison
print("\n" + "="*60)
print("MODEL COMPARISON (Validation Set)")
print("="*60)

models_dict = {
    'Ridge': (ridge, ridge_metrics, X_test_scaled),
    'Elastic Net': (elastic, elastic_metrics, X_test_scaled),
    'Random Forest': (rf, rf_metrics, X_test),
    'XGBoost': (xgb, xgb_metrics, X_test),
    'LightGBM': (lgbm, lgbm_metrics, X_test),
    'Gradient Boosting': (gb, gb_metrics, X_test),
    'Weighted Ensemble': (None, weighted_metrics, X_test)
}

for name, (model, metrics, _) in models_dict.items():
    print(f"{name:20s} - R2: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f}")

best_name = max(models_dict.items(), key=lambda x: x[1][1]['r2'])[0]
best_model, best_metrics, best_test_data = models_dict[best_name]

print(f"\nBEST MODEL: {best_name} (R2: {best_metrics['r2']:.4f})")

# 6. Test Set Evaluation
print("\n" + "="*60)
print("[Step 10/10] FINAL TEST SET EVALUATION")
print("="*60)

if best_name == "Weighted Ensemble":
    y_pred_test = w_xgb * np.clip(xgb.predict(X_test), 0, 1) + \
                  w_lgbm * np.clip(lgbm.predict(X_test), 0, 1) + \
                  w_gb * np.clip(gb.predict(X_test), 0, 1) + \
                  w_rf * np.clip(rf.predict(X_test), 0, 1)
    
    test_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'mae': mean_absolute_error(y_test, y_pred_test),
        'r2': r2_score(y_test, y_pred_test),
        'f1': f1_score((y_test >= 0.5).astype(int), (y_pred_test >= 0.5).astype(int)),
        'accuracy': np.mean((y_test >= 0.5) == (y_pred_test >= 0.5))
    }
    best_model = xgb  # Save XGBoost as representative
else:
    test_metrics = evaluate_model(best_model, best_test_data, y_test, best_name)
    y_pred_test = test_metrics['predictions']

print(f"\nTest Set Results:")
print(f"  R2: {test_metrics['r2']:.4f}")
print(f"  RMSE: {test_metrics['rmse']:.4f}")
print(f"  MAE: {test_metrics['mae']:.4f}")
print(f"  F1: {test_metrics['f1']:.4f}")
print(f"  Accuracy: {test_metrics['accuracy']:.4f}")

# 7. Cross-Validation
print("\n" + "="*60)
print("CROSS-VALIDATION ANALYSIS")
print("="*60)

if best_name != "Weighted Ensemble":
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    print(f"5-Fold CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
else:
    cv_xgb = cross_val_score(xgb, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    print(f"XGBoost CV R2: {cv_xgb.mean():.4f} (+/- {cv_xgb.std() * 2:.4f})")

# 8. Save Models
print("\n" + "="*60)
print("SAVING MODELS AND ARTIFACTS")
print("="*60)

with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("  [OK] Model saved")

with open('model_features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("  [OK] Features saved")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("  [OK] Scaler saved")

scaling_stats = {'min': DATA_MIN, 'max': DATA_MAX, 'mean': DATA_MEAN, 'std': DATA_STD}
with open('scaling_stats.pkl', 'wb') as f:
    pickle.dump(scaling_stats, f)
print("  [OK] Scaling stats saved")

# 9. Create Visualizations
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

os.makedirs('../result/Graph', exist_ok=True)

# Plot 1: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, s=10)
plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect')
plt.xlabel('Actual Poverty Probability')
plt.ylabel('Predicted Poverty Probability')
plt.title(f'Actual vs Predicted - {best_name}\nTest R2 = {test_metrics["r2"]:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../result/Graph/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] actual_vs_predicted.png")

# Plot 2: Residuals
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred_test
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Frequency')
plt.title(f'Residuals Distribution\nMean: {residuals.mean():.4f}, Std: {residuals.std():.4f}')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.grid(True, alpha=0.3)
plt.savefig('../result/Graph/residuals.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] residuals.png")

# Plot 3: Model Comparison
plt.figure(figsize=(12, 6))
names = list(models_dict.keys())
r2_scores = [models_dict[n][1]['r2'] for n in names]
colors = ['#FF6B6B' if r2 < 0.5 else '#4ECDC4' if r2 < 0.7 else '#95E1D3' for r2 in r2_scores]
bars = plt.bar(names, r2_scores, color=colors, edgecolor='black')
plt.ylabel('R2 Score')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, max(r2_scores) * 1.1)
plt.grid(True, alpha=0.3, axis='y')
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., h, f'{h:.4f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('../result/Graph/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [OK] model_comparison.png")

# Plot 4: Feature Importance
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(12, 8))
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-20:]
    plt.barh(range(len(indices)), importances[indices], color='skyblue', edgecolor='black')
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(f'Top 20 Features - {best_name}')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('../result/Graph/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] feature_importance.png")

# 10. Final Summary
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nBest Model: {best_name}")
print(f"Test R2: {test_metrics['r2']:.4f} (explains {test_metrics['r2']*100:.1f}% of variance)")
print(f"Test RMSE: {test_metrics['rmse']:.4f} (avg error: {test_metrics['rmse']*100:.1f} percentage points)")
print(f"Test MAE: {test_metrics['mae']:.4f}")
print(f"F1 Score: {test_metrics['f1']:.4f}")
print(f"Accuracy: {test_metrics['accuracy']:.4f}")

print("\nModel Assessment:")
if test_metrics['r2'] >= 0.70:
    print("  Status: GOOD - Model explains most variance")
elif test_metrics['r2'] >= 0.50:
    print("  Status: MODERATE - Model has decent predictive power")
else:
    print("  Status: NEEDS IMPROVEMENT - Consider more features or different approach")

print("\n" + "="*60 + "\n")
