import pandas as pd
import sys
from pathlib import Path

# --- 1. CONFIGURATION & PATH SETUP ---
# ตั้งค่า Path ให้ฉลาด ไม่ว่าจะรันจากโฟลเดอร์ไหนก็หาไฟล์เจอ
# Logic: หาที่อยู่ของไฟล์นี้ (src/) แล้วถอยหลัง 1 ขั้นไปหา Project Root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'Dataset'

# ชื่อไฟล์
INPUT_VALUES_STD = 'train_values.csv'
INPUT_VALUES_RAW = 'train_values_wJZrCmI.csv' # เผื่อกรณีชื่อไฟล์ยังไม่เปลี่ยน
INPUT_LABELS = 'train_labels.csv'
OUTPUT_FILE = 'train_data_cleaned_v1.csv'

def load_and_merge_data():
    """โหลดข้อมูลดิบและรวมร่าง (Merge)"""
    print(" [1/4] Loading Data...")
    
    # 1.1 พยายามโหลด Train Values
    path_std = DATA_DIR / INPUT_VALUES_STD
    path_raw = DATA_DIR / INPUT_VALUES_RAW
    
    if path_std.exists():
        df_values = pd.read_csv(path_std)
    elif path_raw.exists():
        print(f" Reading from raw filename: {INPUT_VALUES_RAW}")
        df_values = pd.read_csv(path_raw)
    else:
        raise FileNotFoundError(f" Critical Error: หาไฟล์ {INPUT_VALUES_STD} ไม่เจอใน {DATA_DIR}")

    # 1.2 โหลด Labels
    path_labels = DATA_DIR / INPUT_LABELS
    if not path_labels.exists():
        raise FileNotFoundError(f" Critical Error: หาไฟล์ {INPUT_LABELS} ไม่เจอ")
    df_labels = pd.read_csv(path_labels)

    # 1.3 Merge Data
    df_merged = df_values.merge(df_labels, on='row_id')
    print(f" Merge Success! Shape: {df_merged.shape}")
    
    return df_merged

def clean_data(df):
    """จัดการค่าว่าง (Missing Values) และการตั้งค่า Index"""
    print(" [2/4] Cleaning Missing Values...")
    df_clean = df.copy()

    # 2.1 Set Index (Prevent Leakage)
    if 'row_id' in df_clean.columns:
        df_clean.set_index('row_id', inplace=True)

    # 2.2 Fill Interest Rates with 0 (No Account = No Interest)
    interest_cols = ['bank_interest_rate', 'mm_interest_rate', 'mfi_interest_rate', 'other_fsp_interest_rate']
    df_clean[interest_cols] = df_clean[interest_cols].fillna(0)

    # 2.3 Fill Income Share with Median
    if 'share_hh_income_provided' in df_clean.columns:
        median_val = df_clean['share_hh_income_provided'].median()
        df_clean['share_hh_income_provided'] = df_clean['share_hh_income_provided'].fillna(median_val)

    # 2.4 Fill Education with Mode
    if 'education_level' in df_clean.columns:
        mode_val = df_clean['education_level'].mode()[0]
        df_clean['education_level'] = df_clean['education_level'].fillna(mode_val)

    missing_count = df_clean.isnull().sum().sum()
    print(f" Cleaning Done. Remaining Missing Values: {missing_count}")
    return df_clean

def transform_data(df):
    """แปลงข้อมูล (Boolean -> Int, Object -> One-Hot)"""
    print(" [3/4] Transforming Data (Encoding)...")
    df_trans = df.copy()

    # 3.1 Convert Boolean to Int (1/0)
    bool_cols = df_trans.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        df_trans[bool_cols] = df_trans[bool_cols].astype(int)
        print(f"   - Converted {len(bool_cols)} boolean columns.")

    # 3.2 One-Hot Encoding for Objects
    obj_cols = df_trans.select_dtypes(include=['object']).columns
    if len(obj_cols) > 0:
        df_trans = pd.get_dummies(df_trans, columns=obj_cols, prefix=obj_cols, dtype=int)
        print(f"   - One-Hot encoded {len(obj_cols)} object columns.")
    
    return df_trans

def save_data(df):
    """บันทึกไฟล์ CSV"""
    print(f" [4/4] Saving Data...")
    
    output_path = DATA_DIR / OUTPUT_FILE
    
    # ตรวจสอบความสมบูรณ์ครั้งสุดท้าย
    if 'poverty_probability' not in df.columns:
        print(" Warning: Target variable is missing!")
    
    # Save
    df.to_csv(output_path, index=True)
    print(f" Saved to: {output_path}")
    print(f" Final Shape: {df.shape}")

def main():
    try:
        # Pipeline Execution
        df = load_and_merge_data()
        df = clean_data(df)
        df = transform_data(df)
        save_data(df)
        print("\n Preprocessing Pipeline Completed Successfully!")
        
    except Exception as e:
        print(f"\n Pipeline Failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()