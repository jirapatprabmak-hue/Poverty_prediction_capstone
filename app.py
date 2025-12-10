from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# โหลดโมเดลและรายชื่อฟีเจอร์
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except Exception as e:
    print("Error loading model:", e)
    model = None
    feature_names = []

@app.route('/')
def home():
    # ส่งรายชื่อฟีเจอร์ไปให้หน้าเว็บสร้างฟอร์ม
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    print(f"Received data: {data}")
    # เรียงข้อมูลให้ตรงกับตอนเทรน
    try:
        input_data = [float(data.get(feat, 0)) for feat in feature_names]
        print(f"Feature order: {feature_names}")
        print(f"Input data: {input_data}")
        # Convert to DataFrame to avoid warning
        input_df = pd.DataFrame([input_data], columns=feature_names)
        prediction = model.predict(input_df)[0]
        print(f"Prediction: {prediction}")
        return jsonify({'poverty_probability': float(prediction)})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)