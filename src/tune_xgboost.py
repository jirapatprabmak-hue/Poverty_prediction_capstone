import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

print("Loading data and features...")
# Load the same data and features as train_model.py
df = pd.read_csv('../dataset/train_data_cleaned_v1.csv')

if 'row_id' in df.columns:
    df = df.drop(columns=['row_id'])

# Same features as train_model.py
top_10_features = [
    'education_level', 'is_urban', 'phone_technology', 'can_use_internet', 'can_text',
    'num_financial_activities_last_year', 'formal_savings', 'phone_ownership',
    'advanced_phone_use', 'active_bank_user'
]

expanded_features = [
    'country_D', 'reg_bank_acct', 'can_make_transaction', 'num_formal_institutions_last_year',
    'literacy', 'country_A', 'financially_included', 'active_mm_user', 'has_investment',
    'employment_type_last_year_salaried', 'can_call', 'income_private_sector_last_year',
    'employment_type_last_year_irregular_seasonal', 'country_F', 'num_shocks_last_year',
    'avg_shock_strength_last_year', 'income_friends_family_last_year', 'reg_mm_acct',
    'country_I', 'age', 'female', 'married'
]

selected_features = top_10_features + expanded_features

print("Engineering features...")
df_features = df[selected_features].copy()

# Same feature engineering
df_features['education_urban_interaction'] = df['education_level'] * df['is_urban']
df_features['financial_tech_score'] = (df['can_use_internet'] + df['phone_ownership'] + df['active_bank_user']) / 3
df_features['tech_access_score'] = (df['phone_technology'] + df['can_use_internet'] + df['can_make_transaction']) / 3
df_features['formal_account_index'] = (df['reg_bank_acct'] + df['reg_mm_acct']) / 2
df_features['digital_inclusion'] = (df['can_use_internet'] * df['phone_ownership'] * df['can_make_transaction'])
df_features['financial_capability'] = (df['literacy'] * df['education_level'] * df['financially_included'])
df_features['income_diversity'] = df['income_private_sector_last_year'] + df['income_friends_family_last_year']
df_features['education_level_squared'] = df['education_level'] ** 2
df_features['urban_tech_interaction'] = df['is_urban'] * df['phone_technology']

X = df_features
y = df['poverty_probability']

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"\nData split: Train={len(X_train)}, Valid={len(X_valid)}, Test={len(X_test)}")
print(f"Features: {X.shape[1]}")

# Define hyperparameter search space
print("\n" + "="*70)
print("XGBOOST HYPERPARAMETER TUNING")
print("="*70)

param_distributions = {
    'n_estimators': [500, 800, 1000, 1500, 2000],
    'max_depth': [4, 5, 6, 7, 8],
    'learning_rate': [0.005, 0.008, 0.01, 0.015, 0.02],
    'subsample': [0.7, 0.8, 0.85, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.85, 0.9],
    'min_child_weight': [1, 2, 3, 4],
    'gamma': [0, 0.01, 0.02, 0.05, 0.1],
    'reg_alpha': [0, 0.01, 0.05, 0.1, 0.2],
    'reg_lambda': [0.5, 0.8, 1.0, 1.2, 1.5]
}

# Create base model
base_model = XGBRegressor(
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

# Create scorer
r2_scorer = make_scorer(r2_score)

# Randomized search (faster than grid search)
print(f"\nSearching over {np.prod([len(v) for v in param_distributions.values()])} possible combinations...")
print("Using RandomizedSearchCV with 50 iterations and 3-fold CV\n")

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=50,  # Try 50 random combinations
    scoring=r2_scorer,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit
print("Starting hyperparameter search (this will take several minutes)...\n")
random_search.fit(X_train, y_train)

# Best parameters
print("\n" + "="*70)
print("BEST PARAMETERS FOUND:")
print("="*70)
for param, value in random_search.best_params_.items():
    print(f"{param:20s}: {value}")

print(f"\nBest CV R² Score: {random_search.best_score_:.4f}")

# Train final model with best params
print("\n" + "="*70)
print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
print("="*70)

best_model = random_search.best_estimator_

# Evaluate on validation set
y_pred_valid = best_model.predict(X_valid)
r2_valid = r2_score(y_valid, y_pred_valid)
rmse_valid = np.sqrt(mean_squared_error(y_valid, y_pred_valid))

print(f"\nValidation Set Performance:")
print(f"  R²:   {r2_valid:.4f}")
print(f"  RMSE: {rmse_valid:.4f}")

# Evaluate on test set
y_pred_test = best_model.predict(X_test)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\nTest Set Performance:")
print(f"  R²:   {r2_test:.4f}")
print(f"  RMSE: {rmse_test:.4f}")

# Show top 10 parameter combinations
print("\n" + "="*70)
print("TOP 10 PARAMETER COMBINATIONS:")
print("="*70)

results_df = pd.DataFrame(random_search.cv_results_)
results_df = results_df.sort_values('rank_test_score')
top_10 = results_df.head(10)

for idx, row in top_10.iterrows():
    print(f"\nRank {int(row['rank_test_score'])}: R² = {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})")
    params = row['params']
    for param, value in params.items():
        print(f"  {param:20s}: {value}")

# Save best model
with open('xgboost_tuned.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("\n" + "="*70)
print("Best model saved to 'xgboost_tuned.pkl'")
print("="*70)

# Print code snippet for train_model.py
print("\n" + "="*70)
print("COPY THIS TO train_model.py:")
print("="*70)
print(f"""
xgb = XGBRegressor(
    n_estimators={random_search.best_params_['n_estimators']},
    max_depth={random_search.best_params_['max_depth']},
    learning_rate={random_search.best_params_['learning_rate']},
    subsample={random_search.best_params_['subsample']},
    colsample_bytree={random_search.best_params_['colsample_bytree']},
    min_child_weight={random_search.best_params_['min_child_weight']},
    gamma={random_search.best_params_['gamma']},
    reg_alpha={random_search.best_params_['reg_alpha']},
    reg_lambda={random_search.best_params_['reg_lambda']},
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)
""")
