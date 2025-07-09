import traceback
import numpy as np
import cv2
import io
from flask import Flask, request, send_file, render_template

app = Flask(__name__)

def dark_channel(image, window_size):
    min_channel = np.min(image, axis=2)
    return cv2.erode(min_channel, np.ones((window_size, window_size), dtype=np.uint8))

def atmospheric_light(image, dark_channel, percentile=1):
    flat_image = image.reshape((-1, 3))
    flat_dark = dark_channel.flatten()
    index = np.argsort(flat_dark)[::-1][:int(flat_dark.size * (percentile / 100))]
    return np.percentile(flat_image[index], 99, axis=0)

def transmission(image, atmospheric_light, omega=0.95, window_size=15):
    normalized_image = image.astype(np.float32) / atmospheric_light.astype(np.float32)
    dark = dark_channel(normalized_image, window_size)
    return 1 - omega * dark

def guided_filter(I, p, r, eps):
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))
    q = mean_a * I + mean_b
    return q

def refine_transmission(image, t, r=60, eps=0.0001):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
    refined_t = guided_filter(gray_image, t, r, eps)
    return refined_t

def dehaze(image, atmospheric_light, t, t0=0.1):
    t = np.maximum(t, t0)
    result = np.empty_like(image, dtype=np.float32)
    for i in range(3):
        result[..., i] = ((image[..., i].astype(np.float32) - atmospheric_light[i]) / t) + atmospheric_light[i]
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def haze_level(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    haze_percentage = np.sum(hist[:80]) / np.sum(hist) * 100
    return haze_percentage

def dehaze_image(image, omega=0.95, t0=0.1):
    haze_percentage = haze_level(image)
    if haze_percentage < 20:
        print("Haze Percentage:", haze_percentage, "Category: Low Haze")
        percentile = 1
    elif haze_percentage < 45:
        print("Haze Percentage:", haze_percentage, "Category: Medium Haze")
        percentile = 5
    else:
        print("Haze Percentage:", haze_percentage, "Category: High Haze")
        percentile = 10

    dark = dark_channel(image, window_size=15)
    atmospheric = atmospheric_light(image, dark, percentile)
    t = transmission(image, atmospheric, omega, window_size=15)
    t = refine_transmission(image, t)
    dehazed = dehaze(image, atmospheric, t, t0)
    dehazed = apply_clahe(dehazed)
    return dehazed

def adaptive_parameter_tuning(image):
    haze_percentage = haze_level(image)
    print("Haze Percentage:", haze_percentage)

    if haze_percentage < 20:
        print("Category: Low Haze")
        percentile = 1
        omega = 0.85
        t0 = 0.05
    elif haze_percentage < 45:
        print("Category: Medium Haze")
        percentile = 5
        omega = 0.9
        t0 = 0.08
    else:
        print("Category: High Haze")
        percentile = 10
        omega = 0.95
        t0 = 0.1

    print("Chosen Parameters - Percentile:", percentile, "Omega:", omega, "T0:", t0)
    return percentile, omega, t0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dehaze', methods=['POST'])
def dehaze_endpoint():
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        try:
            npimg = np.fromfile(file, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            percentile, omega, t0 = adaptive_parameter_tuning(img)
            dehazed_img = dehaze_image(img, omega=omega, t0=t0)
            _, buffer = cv2.imencode('.png', dehazed_img)
            buffer = io.BytesIO(buffer)
            buffer.seek(0)
            return send_file(
                buffer,
                mimetype='image/png',
                as_attachment=True,
                download_name='dehazed_image.png'
            )
        except Exception as e:
            traceback.print_exc()
            return f"Error processing the image: {str(e)}", 500
    return "Invalid file type", 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True)
