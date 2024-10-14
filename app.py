import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
from flask import Flask, request, render_template
from io import BytesIO

app = Flask(__name__)

# โหลดโมเดลของคุณ
model = tf.keras.models.load_model('C:/Users/user/Desktop/MEDUCATION/final_model.h5')

# คำอธิบายของคลาสที่โมเดลทำนายได้
class_labels = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TB']

@app.route('/')
def web():
    return render_template('web.html')

@app.route('/upload')
def upload():
    return render_template('Upload.html')

@app.route('/feed')
def feed():
    return render_template('feed.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "Error: No image file uploaded"

    file = request.files['image']

    if file.filename == '':
        return "Error: No selected file"

    # โหลดภาพจากไฟล์อัปโหลด โดยใช้ stream เพื่อแปลงเป็น BytesIO
    img = load_img(BytesIO(file.read()), target_size=(400, 400))  # ปรับขนาดภาพให้ตรงกับขนาดที่โมเดลต้องการ
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)  # เพิ่ม dimension ให้ตรงกับ input ของโมเดล
    x = x / 255.0  # Normalization

    # ทำนายผลลัพธ์
    predictions = model.predict(x)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100  # ค่าความมั่นใจ

    result_label = class_labels[predicted_class]

    return render_template('result.html', prediction=result_label, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
