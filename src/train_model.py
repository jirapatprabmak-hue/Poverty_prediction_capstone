import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
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

# แยก Feature และ Target
target_col = 'poverty_probability'
X = df[top_10_features]
y = df[target_col]
print(f"Using top 10 features: {top_10_features}")

# 2. แบ่งข้อมูล 60% Train, 20% Validation, 20% Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Data Split: Train={X_train.shape[0]}, Validation={X_valid.shape[0]}, Test={X_test.shape[0]}")

# ฟังก์ชันคำนวณ Metrics
def evaluate_model(model, X, y_true, name):
    y_pred = model.predict(X)
    # Scale predictions to 1-99% for evaluation
    y_pred_scaled = np.array([scale_prediction(p) for p in y_pred])
    y_true_scaled = np.array([scale_prediction(t) for t in y_true])
    
    rmse = np.sqrt(mean_squared_error(y_true_scaled, y_pred_scaled))
    r2 = r2_score(y_true_scaled, y_pred_scaled)
    # คำนวณ F1 โดยตัดที่ threshold 50% (แปลงเป็นจน/ไม่จน)
    y_pred_bin = (y_pred_scaled >= 50).astype(int)
    y_true_bin = (y_true_scaled >= 50).astype(int)
    f1 = f1_score(y_true_bin, y_pred_bin)
    print(f"[{name}] RMSE: {rmse:.4f}, R²: {r2:.4f}, F1: {f1:.4f}")
    return rmse, r2, f1

# 3. เทรนโมเดล
print("\nTraining Ridge Regression (Linear Model)...")
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train)
rmse_ridge, r2_ridge, f1_ridge = evaluate_model(ridge, X_valid, y_valid, "Ridge Regression")

print("\nTraining Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf.fit(X_train, y_train)
rmse_rf, r2_rf, f1_rf = evaluate_model(rf, X_valid, y_valid, "Random Forest")

print("\nTraining XGBoost...")
xgb = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=5)
xgb.fit(X_train, y_train)
rmse_xgb, r2_xgb, f1_xgb = evaluate_model(xgb, X_valid, y_valid, "XGBoost")

# 4. เลือกโมเดลที่ดีที่สุด - ให้ความสำคัญกับ Ridge เพื่อความ interpretable
# Ridge จะให้ผลลัพธ์ที่สมเหตุสมผลกว่า (higher education = lower poverty)
best_model = ridge
best_name = "Ridge Regression"
print(f"\nBest Model selected: {best_name} (chosen for interpretability)")

# 5. ทดสอบกับ Test Set
print("\nEvaluating on Test Set...")
evaluate_model(best_model, X_test, y_test, best_name)

# 6. บันทึกโมเดล
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("Model saved to 'best_model.pkl'")

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