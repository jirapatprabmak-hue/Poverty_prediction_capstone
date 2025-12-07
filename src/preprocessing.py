# แก้ Path ให้ตรงกับที่อยู่ไฟล์จริง
# ถ้าไฟล์ train_values.csv อยู่ในโฟลเดอร์ Dataset โดยตรง (ไม่มีโฟลเดอร์ raw) ให้แก้เป็น:
RAW_DATA_PATH = 'Dataset/train_values.csv'
LABEL_PATH = 'Dataset/train_labels.csv'

# ส่วน Output จะให้ไปสร้างไว้ใน Dataset เหมือนเดิมก็ได้
OUTPUT_PATH = 'Dataset/train_data_cleaned_v1.csv'