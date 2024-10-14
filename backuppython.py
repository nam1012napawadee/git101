import numpy as np
import tensorflow as tf
import keras.utils as image
import matplotlib
matplotlib.use('Agg')  # ใช้ Agg backend สำหรับ matplotlib
import matplotlib.pyplot as plt
import io
import os
from flask import Flask, request, render_template, send_file

app = Flask(__name__)

# ตรวจสอบว่าไฟล์โมเดลมีอยู่จริงหรือไม่
model_path = 'final_model.h5'
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found!")
    exit()

# โหลดโมเดลที่ฝึกแล้ว
model = tf.keras.models.load_model(model_path)

# คลาสที่โมเดลจะทำนาย
class_labels = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TB']

@app.route('/')
def web():
    return render_template('web.html')  # ส่งฟอร์ม HTML
@app.route('/upload')
def upload():
    return render_template('upload.html')  # ส่งฟอร์ม HTML
@app.route('/result')
def result():
    return render_template('result.html')  # ส่งฟอร์ม HTML
@app.route('/feed')
def feed():
    return render_template('feed.html')  # ส่งฟอร์ม HTML


@app.route('/predict', methods=['POST'])
def predict():
    # ตรวจสอบว่ามีไฟล์ภาพหรือไม่
    if 'image' not in request.files:
        return "Error: No image file uploaded!"

    img_file = request.files['image']

    if img_file.filename == '':
        return "Error: No image file selected!"

    # โหลดภาพจากไฟล์ที่อัปโหลด
    img = image.load_img(img_file, target_size=(400, 400))

    # แปลงภาพเป็น array และ normalize
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # ทำนายผลลัพธ์
    preds = model.predict(x)

    # หาคลาสที่มีค่าความมั่นใจสูงสุด
    pred_class = np.argmax(preds)
    confidence = np.max(preds) * 100
    

    # สร้างภาพ
    plt.imshow(img)
    plt.axis('off')  # ปิดการแสดงแกน
    plt.title(f"Predicted: {class_labels[pred_class]} ({confidence:.2f}%)")

    # สร้าง buffer เพื่อบันทึกภาพ
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()  # ปิด figure เพื่อให้แน่ใจว่าไม่มีการทำงานใน main loop

    # ส่งภาพกลับไปยังเบราว์เซอร์
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
