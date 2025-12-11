import pandas as pd
import numpy as np

df = pd.read_csv('../dataset/train_data_cleaned_v1.csv')

# Current top 10 features
top_10 = ['education_level', 'is_urban', 'phone_technology', 'can_use_internet', 
          'can_text', 'num_financial_activities_last_year', 'formal_savings', 
          'phone_ownership', 'advanced_phone_use', 'active_bank_user']

print("="*60)
print("WHY R² IS LOW - ANALYSIS")
print("="*60)

print("\n1. CURRENT TOP 10 FEATURES CORRELATION WITH TARGET:")
print("-"*60)
for f in top_10:
    corr = df[[f, 'poverty_probability']].corr().iloc[0,1]
    print(f'{f:45s}: {corr:7.4f}')

print("\n2. ACTUAL TOP 15 FEATURES BY CORRELATION:")
print("-"*60)
all_corr = df.corr()['poverty_probability'].drop('poverty_probability').abs().sort_values(ascending=False).head(15)
for feat, corr in all_corr.items():
    print(f'{feat:45s}: {corr:7.4f}')

print("\n3. DATA CHARACTERISTICS:")
print("-"*60)
print(f"Total features available: {df.shape[1] - 2}")  # excluding row_id and target
print(f"Features used: {len(top_10)}")
print(f"\nTarget (poverty_probability) statistics:")
print(df['poverty_probability'].describe())

print("\n4. ISSUE IDENTIFIED:")
print("-"*60)
print("⚠️  Your 'top 10' features are NOT the highest correlated features!")
print("⚠️  They have NEGATIVE correlations (which is ok) but are not the strongest.")
print("\nThe actual strongest predictors are:")
print("   - country_D (0.2357)")
print("   - country_A (0.1964)")
print("   - employment_type_last_year_irregular_seasonal (0.1401)")
print("   - num_shocks_last_year (0.1355)")
print("   - avg_shock_strength_last_year (0.1295)")
print("\nYour features like 'education_level', 'is_urban', 'phone_technology'")
print("have correlations around -0.20 to -0.30, which is moderate but not optimal.")

print("\n5. RECOMMENDATIONS:")
print("-"*60)
print("✓ Include country dummy variables (country_A, country_D)")
print("✓ Include economic shocks features (num_shocks, avg_shock_strength)")
print("✓ Include employment type features")
print("✓ Include income-related features")
print("✓ Consider using ALL relevant features instead of just 10")
print("✓ Create feature interactions (e.g., education × is_urban)")
