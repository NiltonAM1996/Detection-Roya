from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)
            if file:
                image = Image.open(file.stream)

                # Guardar la imagen original
                image.save("static/uploaded_image.png")

                # Procesar la imagen para eliminar el fondo y calcular el porcentaje de infestación
                leaf_only, result, percentage_infestation = process_image(image)

                # Guardar la imagen sin fondo
                leaf_only_image = Image.fromarray(leaf_only)
                leaf_only_image.save("static/leaf_only.png")

                # Guardar la imagen procesada
                result_image = Image.fromarray(result)
                result_image.save("static/result.png")

                return render_template('result.html', percentage=f"{percentage_infestation:.1f}")
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred", 500

    return render_template('index.html')

def process_image(image):
    img = np.array(image)
    if img.shape[-1] == 4:  # Si tiene 4 canales (RGBA), convertir a RGB
        img = img[:, :, :3]

    # Convertir a espacio de color HSV para eliminar el fondo blanco
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Definir los rangos de color para segmentar y eliminar el fondo blanco
    lower_white = np.array([0, 0, 200])  # Límite inferior para blanco
    upper_white = np.array([180, 40, 255])  # Límite superior para blanco

    # Crear una máscara para el fondo blanco
    white_mask = cv2.inRange(hsv_img, lower_white, upper_white)

    # Invertir la máscara para conservar solo la hoja
    leaf_mask = cv2.bitwise_not(white_mask)

    # Encontrar los contornos de la hoja
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos en la imagen original (opcional para visualización)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)  # Dibuja en verde con un grosor de 2 píxeles

    # Aplicar la máscara de la hoja para eliminar el fondo
    leaf_only = cv2.bitwise_and(img, img, mask=leaf_mask)

    # Calcular el área total de la hoja (en píxeles)
    leaf_pixels = np.sum(leaf_mask > 0)

    # Aplicar un filtro de suavizado para reducir los brillos en la imagen
    img_smoothed = cv2.GaussianBlur(leaf_only, (7, 7), 0)

    # Convertir a espacio de color HSV para detectar la roya
    hsv_leaf_img = cv2.cvtColor(img_smoothed, cv2.COLOR_RGB2HSV)

    # Definir los rangos de color para detectar la roya
    lower_rust_dark = np.array([10, 100, 20])  # Para tonos oscuros de roya
    upper_rust_dark = np.array([30, 255, 255])

    lower_rust_light = np.array([0, 30, 160])  # Ajustar rango para excluir brillos más claros
    upper_rust_light = np.array([30, 255, 255])

    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([20, 150, 200])

    # Crear máscaras para las áreas oscuras, claras de roya y marrones
    mask_dark = cv2.inRange(hsv_leaf_img, lower_rust_dark, upper_rust_dark)
    mask_light = cv2.inRange(hsv_leaf_img, lower_rust_light, upper_rust_light)
    mask_brown = cv2.inRange(hsv_leaf_img, lower_brown, upper_brown)

    # Combinar las máscaras de roya
    rust_mask = cv2.bitwise_or(cv2.bitwise_or(mask_dark, mask_light), mask_brown)

    # Contar los píxeles de roya dentro de la hoja
    rust_pixels = np.sum(cv2.bitwise_and(rust_mask, rust_mask, mask=leaf_mask) > 0)

    # Crear una imagen de fondo negro para la imagen procesada
    black_background = np.full((img.shape[0], img.shape[1], 3), [0, 0, 0], dtype=np.uint8)

    # Aplicar la máscara combinada sobre la imagen de la hoja y reemplazar el fondo con negro
    result = np.where(rust_mask[:, :, np.newaxis] > 0, img, black_background)

    # Calcular el porcentaje de infestación basado en los píxeles de la hoja
    percentage_infestation = (rust_pixels / leaf_pixels) * 100

    return leaf_only, result, percentage_infestation

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=81)
