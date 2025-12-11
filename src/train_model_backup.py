import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, f1_score, r2_score

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

# 1. เนเธซเธฅเธ”เธเนเธญเธกเธนเธฅ
print("Loading data...")
df = pd.read_csv('../dataset/train_data_cleaned_v1.csv')

# เธฅเธ column เธ—เธตเนเนเธกเนเนเธเน feature
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
expanded_features = [
    'country_D',
    'reg_bank_acct',
    'can_make_transaction',
    'num_formal_institutions_last_year',
    'literacy',
    'country_A',
    'financially_included',
    'active_mm_user',
    'has_investment',
    'employment_type_last_year_salaried',
    'can_call',
    'income_private_sector_last_year',
    'employment_type_last_year_irregular_seasonal',
    'country_F',
    'num_shocks_last_year',
    'avg_shock_strength_last_year',
    'income_friends_family_last_year',
    'reg_mm_acct',
    'country_I',
    'age',
    'female',
    'married'
]

# Combine all features
selected_features = top_10_features + expanded_features

# Add interaction features for better predictive power
print("\nEngineering interaction features...")
df_features = df[selected_features].copy()

# Create important interaction features
df_features['education_urban_interaction'] = df['education_level'] * df['is_urban']
df_features['financial_tech_score'] = (df['can_use_internet'] + df['phone_ownership'] + df['active_bank_user']) / 3
df_features['tech_access_score'] = (df['phone_technology'] + df['can_use_internet'] + df['can_make_transaction']) / 3
df_features['formal_account_index'] = (df['reg_bank_acct'] + df['reg_mm_acct']) / 2
df_features['digital_inclusion'] = (df['can_use_internet'] * df['phone_ownership'] * df['can_make_transaction'])
df_features['financial_capability'] = (df['literacy'] * df['education_level'] * df['financially_included'])
df_features['income_diversity'] = df['income_private_sector_last_year'] + df['income_friends_family_last_year']

# Square important features for polynomial relationship
df_features['education_level_squared'] = df['education_level'] ** 2
df_features['urban_tech_interaction'] = df['is_urban'] * df['phone_technology']

# Include all engineered features
selected_features = list(df_features.columns)

# เนเธขเธ Feature เนเธฅเธฐ Target
target_col = 'poverty_probability'
X = df_features
y = df[target_col]
print(f"Using {len(selected_features)} features (including interactions): {selected_features}")

# 2. เนเธเนเธเธเนเธญเธกเธนเธฅ 60% Train, 20% Validation, 20% Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Data Split: Train={X_train.shape[0]}, Validation={X_valid.shape[0]}, Test={X_test.shape[0]}")

# Standardize features - helps with model convergence and performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to maintain feature names for some models
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_valid_scaled = pd.DataFrame(X_valid_scaled, columns=X_valid.columns, index=X_valid.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# เธเธฑเธเธเนเธเธฑเธเธเธณเธเธงเธ“ Metrics
def evaluate_model(model, X, y_true, name):
    y_pred = model.predict(X)
    # Evaluate on original scale (0-1) for better metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # เธเธณเธเธงเธ“ F1 เนเธ”เธขเธ•เธฑเธ”เธ—เธตเน threshold 0.5 (เนเธเธฅเธเน€เธเนเธเธเธ/เนเธกเนเธเธ)
    y_pred_bin = (y_pred >= 0.5).astype(int)
    y_true_bin = (y_true >= 0.5).astype(int)
    f1 = f1_score(y_true_bin, y_pred_bin)
    print(f"[{name}] RMSE: {rmse:.4f}, Rยฒ: {r2:.4f}, F1: {f1:.4f}")
    return rmse, r2, f1

# 3. เน€เธ—เธฃเธเนเธกเน€เธ”เธฅ
print("\nTraining Ridge Regression (Linear Model)...")
# Test multiple alpha values for Ridge
ridge = Ridge(alpha=0.0001, random_state=42)  # Very low regularization
ridge.fit(X_train_scaled, y_train)
rmse_ridge, r2_ridge, f1_ridge = evaluate_model(ridge, X_valid_scaled, y_valid, "Ridge Regression")

print("\nTraining Random Forest...")
# Improved hyperparameters for Random Forest
rf = RandomForestRegressor(
    n_estimators=300,        # More trees
    random_state=42, 
    n_jobs=-1, 
    max_depth=20,            # Deeper trees
    min_samples_split=3,     # Allow smaller splits
    min_samples_leaf=1       # Smaller leaf size
)
rf.fit(X_train, y_train)  # RF doesn't need scaling
rmse_rf, r2_rf, f1_rf = evaluate_model(rf, X_valid, y_valid, "Random Forest")

print("\nTraining XGBoost...")
# Very aggressive XGBoost configuration for maximum Rยฒ
xgb = XGBRegressor(
    n_estimators=500,        # Many boosting rounds
    random_state=42, 
    n_jobs=-1, 
    max_depth=8,             # Deeper trees to capture complex patterns
    learning_rate=0.005,     # Very low learning rate for fine-tuning
    subsample=0.85,          # Use 85% of samples per tree
    colsample_bytree=0.85,   # Use 85% of features per tree
    min_child_weight=0.5,    # Allow very small splits
    gamma=0.1,               # Penalize complex trees slightly
    reg_alpha=0.01,          # L1 regularization
    reg_lambda=1.0,          # L2 regularization
    tree_method='hist',      # Fast training
    early_stopping_rounds=50 # Stop if no improvement
)
# Train with early stopping
xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
rmse_xgb, r2_xgb, f1_xgb = evaluate_model(xgb, X_valid, y_valid, "XGBoost")

# 4. เน€เธฅเธทเธญเธเนเธกเน€เธ”เธฅเธ—เธตเนเธ”เธตเธ—เธตเนเธชเธธเธ”
# Compare Rยฒ scores and select the best one
print(f"\n{'='*50}")
print(f"Model Comparison:")
print(f"Ridge Rยฒ: {r2_ridge:.4f}")
print(f"Random Forest Rยฒ: {r2_rf:.4f}")
print(f"XGBoost Rยฒ: {r2_xgb:.4f}")

# Select model with highest Rยฒ
models = [
    (rf, "Random Forest", r2_rf),
    (xgb, "XGBoost", r2_xgb),
    (ridge, "Ridge Regression", r2_ridge)
]
best_model, best_name, best_r2 = max(models, key=lambda x: x[2])
print(f"\nBest Model selected: {best_name} (Rยฒ: {best_r2:.4f})")
print(f"{'='*50}\n")

# 5. เธ—เธ”เธชเธญเธเธเธฑเธ Test Set
print("\nEvaluating on Test Set...")
evaluate_model(best_model, X_test, y_test, best_name)

# 6. เธเธฑเธเธ—เธถเธเนเธกเน€เธ”เธฅ
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("Model saved to 'best_model.pkl'")

# เธเธฑเธเธ—เธถเธเธฃเธฒเธขเธเธทเนเธญ Feature เนเธงเนเนเธเนเธเธฑเธเน€เธงเนเธ
with open('model_features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("Feature names saved to 'model_features.pkl'")

# เธเธฑเธเธ—เธถเธเธชเธ–เธดเธ•เธดเธเนเธญเธกเธนเธฅเธชเธณเธซเธฃเธฑเธเธเธฒเธฃเธชเน€เธเธฅเธเธฅเธเธฒเธฃเธ—เธณเธเธฒเธข
scaling_stats = {
    'min': DATA_MIN,
    'max': DATA_MAX,
    'mean': DATA_MEAN,
    'std': DATA_STD
}
with open('scaling_stats.pkl', 'wb') as f:
    pickle.dump(scaling_stats, f)
print(f"Scaling statistics saved to 'scaling_stats.pkl': {scaling_stats}")
