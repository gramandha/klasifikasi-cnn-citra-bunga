from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import expand_dims
from werkzeug.utils import secure_filename
import numpy as np
import os
from uuid import uuid4

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

try:
    model = load_model(os.path.join(BASE_DIR, 'model_cnn_bunga.h5'), compile=False)
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

class_dict = {0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'}

def predict_label(img_path):
    loaded_img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(loaded_img).astype('float32') / 255.0
    img_array = expand_dims(img_array, axis=0)
    probabilities = model.predict(img_array, verbose=0)[0]
    return class_dict[int(np.argmax(probabilities))]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files.get('image')
        if image is None or not image.filename:
            return render_template('index.html', model_ready=model is not None,
                                   error_message='Pilih gambar bunga terlebih dahulu.')

        safe_name = secure_filename(image.filename)
        extension = safe_name.rsplit('.', 1)[-1].lower() if '.' in safe_name else ''
        if not safe_name or extension not in ALLOWED_EXTENSIONS:
            return render_template('index.html', model_ready=model is not None,
                                   error_message='Format gambar harus JPG, PNG, atau WEBP.')

        if model is None:
            return render_template('index.html', model_ready=False,
                                   error_message='Model prediksi belum berhasil dimuat. Periksa file model dan versi TensorFlow/Keras.')

        stored_name = f'{uuid4().hex}_{safe_name}'
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_name)
        image.save(img_path)
        try:
            load_img(img_path)
        except (OSError, ValueError):
            os.remove(img_path)
            return render_template('index.html', model_ready=True,
                                   error_message='File tidak dapat dibaca sebagai gambar. Coba pilih file gambar lain.')

        prediction = predict_label(img_path)
        return render_template('index.html', model_ready=True,
                               uploaded_image=stored_name, prediction=prediction)

    return render_template('index.html', model_ready=model is not None)


@app.errorhandler(413)
def request_entity_too_large(_error):
    return render_template('index.html', model_ready=model is not None,
                           error_message='Ukuran gambar melebihi batas 10 MB.'), 413

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)