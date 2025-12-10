from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Function to scale predictions using sigmoid curve (moves faster near extremes)
def scale_prediction(pred):
    """
    Scale prediction using sigmoid function for non-linear scaling.
    Numbers move slower in the middle (50%) and faster near 0% or 100%.
    Range: 0-100% with slower movement at center and faster at extremes
    """
    # Clamp raw prediction to 0-1 first
    pred_clamped = max(0, min(1, float(pred)))
    
    # Apply sigmoid function: converts 0-1 to smoother S-curve
    # This makes movement slower in middle and faster at extremes
    x = (pred_clamped - 0.5) * 12  # Increased steepness to reach extremes better
    sigmoid_val = 1 / (1 + np.exp(-x))  # Sigmoid function
    
    # Scale from [0,1] to [0,100]
    scaled = sigmoid_val * 100
    return max(0, min(100, scaled))

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
        input_data = []
        for feat in feature_names:
            value = data.get(feat, 0)
            # Handle empty strings and convert to float
            if value == '' or value is None:
                input_data.append(0)
            else:
                input_data.append(float(value))
        print(f"Feature order: {feature_names}")
        print(f"Input data: {input_data}")
        # Convert to DataFrame to avoid warning
        input_df = pd.DataFrame([input_data], columns=feature_names)
        prediction = model.predict(input_df)[0]
        # Scale prediction to 1-99% range
        poverty_percentage = scale_prediction(prediction)
        print(f"Prediction: {poverty_percentage}%")
        return jsonify({'poverty_probability': poverty_percentage})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)