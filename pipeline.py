import os
import cv2
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from collections import Counter
from scipy.stats import skew
from skimage.filters import sobel, gaussian, roberts, prewitt
from skimage.segmentation import active_contour, watershed
from skimage.draw import polygon, ellipse
from skimage.morphology import skeletonize
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ---------------------------------------
# CONFIGURACIÓN GENERAL Y RUTAS DE SALIDA
# ---------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR,"videos", "video_dilatacion_1.mp4")  # MODIFICAR CON EL NOMBRE DEL VIDEO
OUTPUT_DIR = os.path.join(BASE_DIR, "resultados_pipeline")    # CARPETA CON RESULTADOS

MOD_DIRS = {
    "0": os.path.join(OUTPUT_DIR, "Modulo_0"),
    "1": os.path.join(OUTPUT_DIR, "Modulo_1"),
    "2": os.path.join(OUTPUT_DIR, "Modulo_2"),
    "3": os.path.join(OUTPUT_DIR, "Modulo_3"),
    "4": os.path.join(OUTPUT_DIR, "Modulo_4"),
}

for folder in MOD_DIRS.values():
    os.makedirs(folder, exist_ok=True)

def print_header(title):
    print("\n*************************************************************************")
    print(f" {title.upper()} ")
    print("*************************************************************************\n")

# ----------------------------------------------------------------------
# MÓDULO 0: INSPECCIÓN INICIAL, FRAME REPRESENTATIVO Y DEFINICIÓN DE ROI
# ----------------------------------------------------------------------
print_header("MÓDULO 0: Inspección Inicial del Video y Selección Interactiva de ROI")

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"No se encontró el archivo de video en: {VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duracion_s = total_frames / fps if fps > 0 else 0.0

print(f"Video cargado exitosamente: {os.path.basename(VIDEO_PATH)}")
print(f"Resolución   : {width}x{height} px")
print(f"FPS          : {fps:.2f}")
print(f"Total Frames : {total_frames}")
print(f"Duración     : {duracion_s:.2f} segundos")

# 1. Input del frame representativo
print()
seg_repr_input = input("Ingrese el segundo del video del cual quiere extraer el frame representativo: ")
print()
seg_repr = float(seg_repr_input)
frame_idx_repr = min(int(seg_repr * fps), total_frames - 1)

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_repr)
ret, frame_bgr = cap.read()
if not ret:
    raise RuntimeError("No se pudo leer el frame representativo seleccionado.")

imagen_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

# Guardamos y mostramos frame con ejes para inspección visual
plt.figure(figsize=(10, 8))
plt.imshow(imagen_rgb)
plt.axis("on")
plt.grid(True, color="white", linewidth=0.5, alpha=0.5)
plt.title(f"Frame Representativo (t = {seg_repr:.2f}s | Frame {frame_idx_repr})", fontweight="bold")
plt.xlabel("Coordenada X [px]")
plt.ylabel("Coordenada Y [px]")
plt.xticks(np.arange(0, imagen_rgb.shape[1], 100), fontsize = 8)
plt.yticks(np.arange(0, imagen_rgb.shape[0], 50), fontsize = 8)
path_frame_repr = os.path.join(MOD_DIRS["0"], "frame_representativo.png")
plt.savefig(path_frame_repr, bbox_inches="tight")
plt.close()

print("\nIMAGEN frame_representativo: examine el gráfico para definir los límites de la ROI del ojo.\n")

# 2. Input límites para la ROI
print("<<<<<<<<<<<<<<<< Límites de la ROI >>>>>>>>>>>>>>>>>>>")
x_inicial = int(input("Ingrese X inferior (x_inicial): "))
x_final = int(input("Ingrese X superior (x_final): "))
y_inicial = int(input("Ingrese Y inferior (y_inicial [arriba]): "))
y_final = int(input("Ingrese Y superior (y_final): "))

roi = imagen_rgb[y_inicial:y_final, x_inicial:x_final].copy() #definimos ROI por slicing
roi_gris = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) #ROI en escala de grises

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# ROI a color (RGB)
axs[0].imshow(roi)
axs[0].set_title("ROI a Color (RGB)", fontweight="bold")
axs[0].axis("on")
axs[0].grid(True, color="white", linewidth=0.5, alpha=0.5)

# ROI en Escala de Grises
axs[1].imshow(roi_gris, cmap="gray")
axs[1].set_title("ROI en Escala de Grises", fontweight="bold")
axs[1].axis("on")
axs[1].grid(True, color="white", linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["0"], "roi_seleccionada.png"), bbox_inches="tight")
plt.close()
print("\n IMAGEN roi_seleccionada: ROI resultante del slicing\n")

cap.release()

# --------------------------------------------------------------
# MÓDULO 1: FUNDAMENTOS, ESPACIOS DE COLOR Y MEJORAMIENTO DIGITAL
# --------------------------------------------------------------
print_header("MÓDULO 1: Espacios de Color, Transformaciones y Filtrado Espacial")

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("1.1 Carga, Muestreo y Espacios de Color")
print("-----------------------------------------")
# 0. Submuestreo Espacial +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
roi_50 = cv2.resize(roi, None, fx=0.50, fy=0.50, interpolation=cv2.INTER_NEAREST)
roi_25 = cv2.resize(roi, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
roi_05 = cv2.resize(roi, None, fx=0.05, fy=0.05, interpolation=cv2.INTER_NEAREST)

fig, axs = plt.subplots(1, 4, figsize=(15, 5))

# ROI Original
axs[0].imshow(roi)
axs[0].set_title(f"ROI Original\n{roi.shape[1]}x{roi.shape[0]} px", fontweight="bold")
axs[0].axis("off")

# Submuestreo 50%
axs[1].imshow(roi_50)
axs[1].set_title(f"Submuestreo 50%\n{roi_50.shape[1]}x{roi_50.shape[0]} px", fontweight="bold")
axs[1].axis("off")

# Submuestreo 25%
axs[2].imshow(roi_25)
axs[2].set_title(f"Submuestreo 25%\n{roi_25.shape[1]}x{roi_25.shape[0]} px", fontweight="bold")
axs[2].axis("off")

# Submuestreo 5%
axs[3].imshow(roi_05)
axs[3].set_title(f"Submuestreo 5%\n{roi_05.shape[1]}x{roi_05.shape[0]} px", fontweight="bold")
axs[3].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["1"], "00_submuestreo_espacial.png"), bbox_inches="tight")
plt.close()
print("\n IMAGEN 00_submuestreo_espacial: muestra la ROI submuestreada al 50, 25 y 5%\n")
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 1. Cuantización ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cuantizar_imagen(imagen, bits):
    niveles = 2 ** bits
    factor = 256 // niveles
    return ((imagen // factor) * factor).astype(np.uint8)

# Cuantización en Color
roi_4bits = cuantizar_imagen(roi, 4)
roi_2bits = cuantizar_imagen(roi, 2)

# Cuantización en Escala de Grises
roi_gris_4bits = cuantizar_imagen(roi_gris, 4)
roi_gris_2bits = cuantizar_imagen(roi_gris, 2)

# Figura 2x3 (Fila 1: Color | Fila 2: Escala de Grises)
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

# Fila 1: Color
axs[0, 0].imshow(roi)
axs[0, 0].set_title("Original (8 bits - 256 niveles)", fontweight="bold")
axs[0, 0].axis("off")

axs[0, 1].imshow(roi_4bits)
axs[0, 1].set_title("Cuantizada (4 bits - 16 niveles)", fontweight="bold")
axs[0, 1].axis("off")

axs[0, 2].imshow(roi_2bits)
axs[0, 2].set_title("Cuantizada (2 bits - 4 niveles)", fontweight="bold")
axs[0, 2].axis("off")

# Fila 2: Escala de Grises
axs[1, 0].imshow(roi_gris, cmap="gray")
axs[1, 0].set_title("Gris Original (8 bits - 256 niveles)", fontweight="bold")
axs[1, 0].axis("off")

axs[1, 1].imshow(roi_gris_4bits, cmap="gray")
axs[1, 1].set_title("Gris Cuantizada (4 bits - 16 niveles)", fontweight="bold")
axs[1, 1].axis("off")

axs[1, 2].imshow(roi_gris_2bits, cmap="gray")
axs[1, 2].set_title("Gris Cuantizada (2 bits - 4 niveles)", fontweight="bold")
axs[1, 2].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["1"], "01_cuantizacion.png"), bbox_inches="tight")
plt.close()
print("\n IMAGEN 01_cuantizacion: muestra la ROI cuantizada a 4 bits y 2 bits \n")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 2. Separación en Espacios de Color (RGB, HSV) ++++++++++++++++++++++++++++++++++++++++++++++++
# Separación en canales RGB
R, G, B = cv2.split(roi)

# Conversión a HSV y separación de canales
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
H, S, V = cv2.split(roi_hsv)

# Figura unificada 2x4 (Fila 1: RGB | Fila 2: HSV)
fig, axs = plt.subplots(2, 4, figsize=(16, 8))

# --- Fila 1: RGB ---
axs[0, 0].imshow(roi)
axs[0, 0].set_title("Imagen RGB Original")
axs[0, 0].axis("off")

axs[0, 1].imshow(R, cmap="gray")
axs[0, 1].set_title("Canal R (rojo)")
axs[0, 1].axis("off")

axs[0, 2].imshow(G, cmap="gray")
axs[0, 2].set_title("Canal G (verde)")
axs[0, 2].axis("off")

axs[0, 3].imshow(B, cmap="gray")
axs[0, 3].set_title("Canal B (azul)")
axs[0, 3].axis("off")

# --- Fila 2: HSV ---
axs[1, 0].imshow(roi)
axs[1, 0].set_title("Imagen RGB Original")
axs[1, 0].axis("off")

axs[1, 1].imshow(H, cmap="gray")
axs[1, 1].set_title("Canal H (Matiz / Hue)")
axs[1, 1].axis("off")

axs[1, 2].imshow(S, cmap="gray")
axs[1, 2].set_title("Canal S (Saturación)")
axs[1, 2].axis("off")

axs[1, 3].imshow(V, cmap="gray")
axs[1, 3].set_title("Canal V (Valor / Brillo)")
axs[1, 3].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["1"], "02_espacios_de_color.png"), bbox_inches="tight")
plt.close()

print("\n IMAGEN 02_espacios_de_color: muestra la descomposición en canales RGB y HSV\n")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("1.2 Calidad, Transformaciones de Intensidad y Filtrado Espacial")
print("---------------------------------------------------------------")
# 3. Modificación de Brillo, Contraste e Histogramas +++++++++++++++++++++++++++++++++++++++++++++++++++
frame_contraste = cv2.convertScaleAbs(roi_gris, alpha=1.5, beta=0)
frame_brillo = cv2.convertScaleAbs(roi_gris, alpha=1.0, beta=50)

fig, axs = plt.subplots(2, 3, figsize=(15, 8))

# En la primera fila las imágenes
axs[0, 0].imshow(roi_gris, cmap="gray")
axs[0, 0].set_title("Original en Grises")
axs[0, 0].axis("off")

axs[0, 1].imshow(frame_brillo, cmap="gray")
axs[0, 1].set_title("Brillo (+50)")
axs[0, 1].axis("off")

axs[0, 2].imshow(frame_contraste, cmap="gray")
axs[0, 2].set_title("Contraste (x1.5)")
axs[0, 2].axis("off")

# En la segunda fila los histogramas
axs[1, 0].hist(roi_gris.ravel(), bins=256, range=[0, 256], color="black")
axs[1, 0].set_title("Histograma Original")
axs[1, 0].set_xlabel("Nivel de intensidad")
axs[1, 0].set_ylabel("Frecuencia de aparición")
axs[1, 0].set_xticks(np.arange(10, 256, 20))

axs[1, 1].hist(frame_brillo.ravel(), bins=256, range=[0, 256], color="black")
axs[1, 1].set_title("Desplazamiento (Brillo)")
axs[1, 1].set_xlabel("Nivel de intensidad")
axs[1, 1].set_ylabel("Frecuencia de aparición")
axs[1, 1].set_xticks(np.arange(10, 256, 20))

axs[1, 2].hist(frame_contraste.ravel(), bins=256, range=[0, 256], color="black")
axs[1, 2].set_title("Hiperextensión (Contraste)")
axs[1, 2].set_xlabel("Nivel de intensidad")
axs[1, 2].set_ylabel("Frecuencia de aparición")
axs[1, 2].set_xticks(np.arange(10, 256, 20))

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["1"], "03_histogramas_brillo_contraste.png"), bbox_inches="tight")
plt.close()
print("\nIMAGEN 03_histogramas_brillo_contraste: muestra el histograma y la modificación del brillo y el contraste en él\n")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 4. Escalado Lineal y Ecualización ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
f1 = 20
f2 = 120

# 1. Escalado por Rango de Interés (Transformación Lineal)
roi_escalada = np.clip((roi_gris.astype(float) - f1) * (255 / (f2 - f1)), 0, 255).astype(np.uint8)

# 2. Ecualización Global
roi_ecualizada = cv2.equalizeHist(roi_gris)

# 3. Ecualización Local (CLAHE - Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
roi_clahe = clahe.apply(roi_gris)

# --- Gráficos ---
fig, axs = plt.subplots(2, 4, figsize=(20, 8))
imagenes = [roi_gris, roi_escalada, roi_ecualizada, roi_clahe]
titulos = ["Original", f"Escalado Lineal [{f1}, {f2}]", "Ecualización Global", "CLAHE (Local)"]

for i in range(4):
    # Fila de imágenes
    axs[0, i].imshow(imagenes[i], cmap="gray")
    axs[0, i].set_title(titulos[i])
    axs[0, i].axis("off")

    # Fila de histogramas
    axs[1, i].hist(imagenes[i].ravel(), bins=256, range=[0, 256], color="black")

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["1"], "04_escalado_y_ecualizacion.png"), bbox_inches="tight")
plt.close()
print("\nIMAGEN 04_escalado_y_ecualizacion: muestra el histograma escalado en el rango de interés y ecualizado globalmente y localmente (CLAHE)\n")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 5. Generación de Ruido y Filtrado Espacial +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 1. Funciones de Ruido
def agregar_ruido_gaussiano(imagen, desvio=25):
    media = 0
    desvio = 25
    ruido = np.random.normal(media, desvio, imagen.shape)
    imagen_ruidosa = imagen.astype(float) + ruido #ruido aditivo
    return np.clip(imagen_ruidosa, 0, 255).astype(np.uint8)

def agregar_ruido_sal_pimienta(imagen, prob =0.06):
    ruido = np.copy(imagen)
    aleatorio = np.random.rand(*ruido.shape)
    ruido[aleatorio < prob] = 0  #Pimienta
    ruido[aleatorio > (1.0 - prob)] = 255 #Sal
    return ruido

def agregar_ruido_speckle(imagen, factor = 0.2):
    ruido = np.random.randn(*imagen.shape)
    imagen_ruidosa = imagen.astype(float) + imagen.astype(float) * ruido * factor #Ruido multiplicativo
    #Se multiplica por 0,2 para que no destruya la imagen completa
    return np.clip(imagen_ruidosa, 0, 255).astype(np.uint8)

# 2. Generamos imágenes ruidosas
img_gauss = agregar_ruido_gaussiano(roi_gris)
img_sp = agregar_ruido_sal_pimienta(roi_gris)
img_speckle = agregar_ruido_speckle(roi_gris)

# 3. Aplicamos filtros espaciales
gauss_gauss = cv2.GaussianBlur(img_gauss, (5, 5), 0)
gauss_media = cv2.blur(img_gauss, (5, 5))
gauss_mediana = cv2.medianBlur(img_gauss, 5)

speckle_gauss = cv2.GaussianBlur(img_speckle, (5, 5), 0)
speckle_media = cv2.blur(img_speckle, (5, 5))
speckle_mediana = cv2.medianBlur(img_speckle, 5)

pimienta_gauss = cv2.GaussianBlur(img_sp, (5, 5), 0)
pimienta_media = cv2.blur(img_sp, (5, 5))
pimienta_mediana = cv2.medianBlur(img_sp, 5)

# 4. Graficar Resultados (3 filas x 4 columnas)
fig1, axs1 = plt.subplots(3, 4, figsize=(16, 12))

experimentos = [
    {
        "titulo_ruido": "RUIDO GAUSSIANO",
        "imgs": [img_gauss, gauss_gauss, gauss_media, gauss_mediana],
        "titulos": ["Ruidosa", "Filtro Gaussiano", "Filtro Media", "Filtro Mediana"]
    },
    {
        "titulo_ruido": "RUIDO SAL Y PIMIENTA",
        "imgs": [img_sp, pimienta_gauss, pimienta_media, pimienta_mediana],
        "titulos": ["Ruidosa", "Filtro Gaussiano", "Filtro Media", "Filtro Mediana"]
    },
    {
        "titulo_ruido": "RUIDO SPECKLE",
        "imgs": [img_speckle, speckle_gauss, speckle_media, speckle_mediana],
        "titulos": ["Ruidosa", "Filtro Gaussiano", "Filtro Media", "Filtro Mediana"]
    }
]

for fila, exp in enumerate(experimentos):
    for col in range(4):
        ax = axs1[fila, col]
        ax.imshow(exp["imgs"][col], cmap="gray")
        ax.set_title(exp["titulos"][col], fontsize=10, fontweight="bold")
        ax.axis("off")

    axs1[fila, 0].text(-0.05, 0.5, exp["titulo_ruido"], transform=axs1[fila, 0].transAxes,
                       fontsize=11, fontweight="bold", va="center", ha="right", rotation=90)

plt.subplots_adjust(hspace=-0.7, wspace=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["1"], "05_ruido_y_filtrado_espacial.png"), bbox_inches="tight")
plt.close()
print("\nIMAGEN 05_ruido_y_filtrado_espacial: se simulan tres tipos de ruido y se los filtra con 3 suavizadores distintos\n")

# 5. Histogramas de las imágenes ruidosas (1 fila x 3 columnas)

fig2, axs2 = plt.subplots(1, 3, figsize=(15, 4))

ruidosos = [
    ("Histograma - Ruido Gaussiano", img_gauss),
    ("Histograma - Ruido Sal y Pimienta", img_sp),
    ("Histograma - Ruido Speckle", img_speckle)
]

for col, (titulo, img) in enumerate(ruidosos):
    ax = axs2[col]
    ax.hist(img.ravel(), bins=256, range=[0, 256], color="gray", alpha=0.8)
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.set_xlim([-1, 258])
    ax.set_xlabel("Intensidad (px)", fontsize=9)
    if col == 0:
        ax.set_ylabel("Frecuencia (px)", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["1"], "06_histogramas_ruido.png"), bbox_inches="tight")
plt.close()
print("\nIMAGEN 06_histogramas_ruido: muestra el histograma de las imagenes con los ruidos simulados\n")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("1.3 Detección de Bordes y Dominio Frecuencial")
print("---------------------------------------------")

# 6. Detección de Bordes  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# --- DEFINICIÓN DE KERNELS MANUALES ---

# Operador de Roberts
kernel_roberts_x = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
], dtype=np.float32)

kernel_roberts_y = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, -1, 0]
], dtype=np.float32)

# Operador de Prewitt
kernel_prewitt_y = (1 / 3) * np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float32)

kernel_prewitt_x = (1 / 3) * np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

# Operador de Sobel
kernel_sobel_y = (1 / 4) * np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

kernel_sobel_x = (1 / 4) * np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# Operador Laplaciano
kernel_laplaciano = (1 / 4) * np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=np.float32)

# Definimos una función para aplicar los filtros
def aplicar_detectores(img):
    """Aplica las máscaras manuales con cv2.filter2D y combina componentes X/Y."""
    """Las definiciones de los kernels son los vistos en clase"""

    # Roberts
    rx = cv2.filter2D(img, cv2.CV_64F, kernel_roberts_x)
    ry = cv2.filter2D(img, cv2.CV_64F, kernel_roberts_y)
    roberts = cv2.addWeighted(cv2.convertScaleAbs(rx), 0.5, cv2.convertScaleAbs(ry), 0.5, 0)

    # Prewitt
    p_x = cv2.filter2D(img, cv2.CV_64F, kernel_prewitt_x)
    p_y = cv2.filter2D(img, cv2.CV_64F, kernel_prewitt_y)
    prewitt = cv2.addWeighted(cv2.convertScaleAbs(p_x), 0.5, cv2.convertScaleAbs(p_y), 0.5, 0)

    # Sobel
    s_x = cv2.filter2D(img, cv2.CV_64F, kernel_sobel_x)
    s_y = cv2.filter2D(img, cv2.CV_64F, kernel_sobel_y)
    sobel = cv2.addWeighted(cv2.convertScaleAbs(s_x), 0.5, cv2.convertScaleAbs(s_y), 0.5, 0)

    # Laplaciano
    lap = cv2.filter2D(img, cv2.CV_64F, kernel_laplaciano)
    laplaciano = cv2.convertScaleAbs(lap)

    # Canny (Algoritmo)
    canny = cv2.Canny(img, 50, 150)

    return [img, roberts, prewitt, sobel, laplaciano, canny]

# --- APLICAMOS SOBRE LA IMAGEN ORIGINAL Y FILTRADA ---
roi_filtrada = cv2.GaussianBlur(roi_gris, (5, 5), 0)

resultados_original = aplicar_detectores(roi_gris)
resultados_filtrada = aplicar_detectores(roi_filtrada)

# --- GRÁFICOS ---
titulos = ["Imagen Base", "Roberts", "Prewitt", "Sobel", "Laplaciano", "Canny"]
fig, axs = plt.subplots(2, 6, figsize=(20, 5))

for col in range(6):
    # Fila 1, sin filtrado previo
    axs[0, col].imshow(resultados_original[col], cmap="gray")
    axs[0, col].set_title(titulos[col], fontsize=11, fontweight="bold")
    axs[0, col].axis("off")

    # Fila 2, con filtrado previo
    axs[1, col].imshow(resultados_filtrada[col], cmap="gray")
    axs[1, col].axis("off")

axs[0, 0].text(-0.05, 0.5, "SIN FILTRAR", transform=axs[0, 0].transAxes,
               fontsize=12, fontweight="bold", va="center", ha="right", rotation=90)
axs[1, 0].text(-0.05, 0.5, "SUAVIZADA CON GAUSS", transform=axs[1, 0].transAxes,
               fontsize=12, fontweight="bold", va="center", ha="right", rotation=90)

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["1"], "07_deteccion_de_bordes.png"), bbox_inches="tight")
plt.close()
print("\nIMAGEN 07_deteccion_de_bordes: muestra el resultado los distintos filtros pasa altos\n")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 7. Transformada de Fourier 2D ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Extraemos la ROI secundaria del video (t_representativo + 0.5s)
cap_fourier = cv2.VideoCapture(VIDEO_PATH)
frame_idx_sec = min(int((seg_repr + 0.5) * fps), total_frames - 1)
cap_fourier.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_sec)
ret_sec, frame_sec_bgr = cap_fourier.read()
cap_fourier.release()

if ret_sec:
    roi_sec_rgb = cv2.cvtColor(frame_sec_bgr, cv2.COLOR_BGR2RGB)[y_inicial:y_final, x_inicial:x_final]
    img2 = cv2.cvtColor(roi_sec_rgb, cv2.COLOR_RGB2GRAY)
else:
    img2 = np.copy(roi_gris)

img2 = cv2.resize(img2, (roi_gris.shape[1], roi_gris.shape[0]))

# CALCULAMOS TRANSFORMADA (TFD-2D)
# Imagen 1 (Original)
f1 = np.fft.fft2(roi_gris)
f1_shift = np.fft.fftshift(f1)
mag1 = np.abs(f1_shift)
fase1 = np.angle(f1_shift)

# Imagen 2 (Secundaria: t + 0.5s)
f2 = np.fft.fft2(img2)
f2_shift = np.fft.fftshift(f2)
mag2 = np.abs(f2_shift)
fase2 = np.angle(f2_shift)

# RECONSTRUCCIONES INVERSAS
# A. Solo Magnitud (Fase = 0)
solo_mag_complejo = mag1 * np.exp(1j * 0)
recon_solo_mag = np.abs(np.fft.ifft2(np.fft.ifftshift(solo_mag_complejo)))

# B. Solo Fase (Magnitud = 1)
solo_fase_complejo = 1 * np.exp(1j * fase1)
recon_solo_fase = np.abs(np.fft.ifft2(np.fft.ifftshift(solo_fase_complejo)))

# C. Cruzadas
cruzada_mag1_fase2 = mag1 * np.exp(1j * fase2)
recon_cruzada_12 = np.abs(np.fft.ifft2(np.fft.ifftshift(cruzada_mag1_fase2)))

cruzada_mag2_fase1 = mag2 * np.exp(1j * fase1)
recon_cruzada_21 = np.abs(np.fft.ifft2(np.fft.ifftshift(cruzada_mag2_fase1)))

# GRAFICAMOS RESULTADOS (Magnitud y Fase)
fig, axs = plt.subplots(2, 3, figsize=(20, 10))

# Fila 1: Análisis de la imagen original
axs[0, 0].imshow(roi_gris, cmap="gray")
axs[0, 0].set_title("ROI Original")

axs[0, 1].imshow(20 * np.log(mag1 + 1), cmap="gray")
axs[0, 1].set_title("Magnitud [dB]")

axs[0, 2].imshow(fase1, cmap="gray")
axs[0, 2].set_title("Fase")

# Fila 2: Análisis de la imagen secundaria
axs[1, 0].imshow(img2, cmap="gray")
axs[1, 0].set_title("ROI Secundaria")

axs[1, 1].imshow(20 * np.log(mag2 + 1), cmap="gray")
axs[1, 1].set_title("Magnitud [dB]")

axs[1, 2].imshow(fase2, cmap="gray")
axs[1, 2].set_title("Fase")

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["1"], "08_fourier_magnitud_fase.png"), bbox_inches="tight")
plt.close()
print("\nIMAGEN 08_fourier_magnitud_fase: muestra la magnitud y fase de la TF2D para la ROI seleccionada, y el frame obtenido medio segundo después\n")

# RECONSTRUCCIONES CON TRANSFORMADA INVERSA

fig2, ax2 = plt.subplots(2, 2, figsize=(10.5, 6.5))

ax2[0, 0].imshow(recon_solo_mag, cmap="gray")
ax2[0, 0].set_title("Solo Magnitud")

ax2[0, 1].imshow(recon_solo_fase, cmap="gray")
ax2[0, 1].set_title("Solo Fase")

ax2[1, 0].imshow(recon_cruzada_12, cmap="gray")
ax2[1, 0].set_title("Mag Original + Fase Secundaria")

ax2[1, 1].imshow(recon_cruzada_21, cmap="gray")
ax2[1, 1].set_title("Mag Secundaria + Fase Original")

for ax in ax2.ravel():
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["1"], "09_fourier_reconstruccion.png"), bbox_inches="tight")
plt.close()
print("\nIMAGEN 09_fourier_reconstruccion: muestra la reconstrucción de la imagen usando ITF con distintas combinaciones\n")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 8. Densidad Espectral de Potencia (PSD) 2D y Perfiles 1D +++++++++++++++++++++++++++++++++++++
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Usamos un filtro Gaussiano como ejemplo de filtrado espacial
roi_filtrada = cv2.GaussianBlur(roi_gris, (5, 5), 0)

# Dimensiones de la imagen
N, M = roi_gris.shape
num_pixeles = N * M

# CÁLCULO DE LA PSD (Periodograma 2D)
# Imagen Original
f_orig = np.fft.fftshift(np.fft.fft2(roi_gris))
mag_orig = np.abs(f_orig)
psd_orig = (mag_orig ** 2) / num_pixeles

# Imagen Filtrada
f_filt = np.fft.fftshift(np.fft.fft2(roi_filtrada))
mag_filt = np.abs(f_filt)
psd_filt = (mag_filt ** 2) / num_pixeles

# Escala logarítmica para visualización 2D
psd_orig_log = 10 * np.log10(1 + psd_orig)
psd_filt_log = 10 * np.log10(1 + psd_filt)

# PERFIL 1D CORTANDO EN EL CENTRO DE FRECUENCIAS (Fila central)
centro_y = N // 2
corte_psd_orig = psd_orig_log[centro_y, :]
corte_psd_filt = psd_filt_log[centro_y, :]

# 1. FIGURA 1: COMPARACIÓN 2D (Imágenes y PSDs 2D) - Grilla 2x2
fig1, axs1 = plt.subplots(2, 2, figsize=(10, 7.3))

# --- FILA 1: ORIGINAL ---
# Imagen Original
axs1[0, 0].imshow(roi_gris, cmap="gray")
axs1[0, 0].set_title("Imagen Original", fontsize=11, fontweight="bold")

# PSD 2D Original
im1 = axs1[0, 1].imshow(psd_orig_log, cmap="jet")
axs1[0, 1].set_title("PSD 2D Original", fontsize=11, fontweight="bold")

divider1 = make_axes_locatable(axs1[0, 1])
cax1 = divider1.append_axes("right", size="5%", pad=0.08)
fig1.colorbar(im1, cax=cax1, label="Potencia [dB]")

# --- FILA 2: FILTRADA ---
# Imagen Filtrada
axs1[1, 0].imshow(roi_filtrada, cmap="gray", aspect="equal")
axs1[1, 0].set_title("Imagen Filtrada (Gauss 5x5)", fontsize=11, fontweight="bold")

# PSD 2D Filtrada
im2 = axs1[1, 1].imshow(psd_filt_log, cmap="jet", aspect="equal")
axs1[1, 1].set_title("PSD 2D Filtrada", fontsize=11, fontweight="bold")

divider2 = make_axes_locatable(axs1[1, 1])
cax2 = divider2.append_axes("right", size="5%", pad=0.08)
fig1.colorbar(im2, cax=cax2, label="Potencia [dB]")

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["1"], "10_psd_2d_comparacion.png"), bbox_inches="tight")
plt.close()
print("\nIMAGEN 10_psd_comparacion: muestra la PSD de la ROI original y de la ROI suavizada con Gauss\n")


# 2. FIGURA 2: ANÁLISIS PERFILES 1D - Grilla 1x2

fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))

# Perfil 1D Original
axs2[0].plot(corte_psd_orig, color="blue", label="Original")
axs2[0].set_title("Corte 1D de PSD Original (Centro)", fontsize=11, fontweight="bold")
axs2[0].set_xlabel("Frecuencia espacial (px)")
axs2[0].set_ylabel("Potencia [dB]")
axs2[0].grid(True, linestyle="--", alpha=0.6)

# Perfiles 1D Comparativa Superpuesta
axs2[1].plot(corte_psd_orig, color="blue", alpha=0.5, label="Original")
axs2[1].plot(corte_psd_filt, color="red", label="Filtrada")
axs2[1].set_title("Comparativa 1D PSD (Original vs Filtrada)", fontsize=11, fontweight="bold")
axs2[1].set_xlabel("Frecuencia espacial (px)")
axs2[1].set_ylabel("Potencia [dB]")
axs2[1].legend(loc="upper right")
axs2[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["1"], "11_psd_1d_perfil_central.png"), bbox_inches="tight")
plt.close()
print("\nIMAGEN 11_psd_1d_perfil_central: muestra los perfiles centrales de cada una graficados en una dimensión\n")
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 9. Filtrado en Frecuencia (Pasa-Bajos y Pasa-Altos) y Métricas de Calidad +++++++++++++++++++++++++

# 1. TFD-2D Y CENTRADO DE LA IMAGEN
filas, columnas = roi_gris.shape
centro_r, centro_c = filas // 2, columnas // 2

f_shift = np.fft.fftshift(np.fft.fft2(roi_gris))

# 2. CREAMOS LAS MÁSCARAS DE DISTANCIA AL CENTRO
y, x = np.ogrid[:filas, :columnas]
distancia = np.sqrt((y - centro_r)**2 + (x - centro_c)**2)

# Radio de corte de frecuencia (frecuencia de corte D0)
D0 = 30

# Máscaras de filtros (Usamos filtro Gaussiano frecuencial)
mascara_pasabajos = np.exp(-(distancia**2) / (2 * (D0**2)))
mascara_pasaaltos = 1.0 - mascara_pasabajos

# 3. APLICAMOS FILTROS EN FRECUENCIA
f_shift_pasabajos = f_shift * mascara_pasabajos
f_shift_pasaaltos = f_shift * mascara_pasaaltos

# 4. RECONSTRUCCIÓN MEDIANTE TRANSFORMADA INVERSA (IFFT)
img_pasabajos_frec = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift_pasabajos)))
img_pasaaltos_frec = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift_pasaaltos)))

# 5. FILTRADOS ESPACIALES DE REFERENCIA PARA COMPARAR
img_pasabajos_esp = cv2.GaussianBlur(roi_gris, (9, 9), 0)
img_pasaaltos_esp = cv2.Laplacian(roi_gris, cv2.CV_64F)
img_pasaaltos_esp = np.abs(img_pasaaltos_esp)

# 6. GRAFICAMOS RESULTADOS
fig, axs = plt.subplots(2, 3, figsize=(15, 7.3))

# Fila 1: Pasa-Bajos (Frecuencia vs Espacial)
axs[0, 0].imshow(mascara_pasabajos, cmap="gray")
axs[0, 0].set_title("Máscara Pasabajos Frecuencial", fontsize=10, fontweight="bold")

axs[0, 1].imshow(img_pasabajos_frec, cmap="gray")
axs[0, 1].set_title("Reconstrucción Pasabajos", fontsize=10, fontweight="bold")

axs[0, 2].imshow(img_pasabajos_esp, cmap="gray")
axs[0, 2].set_title("Filtro Espacial Gaussiano (Referencia)", fontsize=10, fontweight="bold")

# Fila 2: Pasa-Altos (Frecuencia vs Espacial)
axs[1, 0].imshow(mascara_pasaaltos, cmap="gray")
axs[1, 0].set_title("Máscara Pasaaltos Frecuencial", fontsize=10, fontweight="bold")

axs[1, 1].imshow(img_pasaaltos_frec, cmap="gray")
axs[1, 1].set_title("Reconstrucción Pasaaltos", fontsize=10, fontweight="bold")

axs[1, 2].imshow(img_pasaaltos_esp, cmap="gray")
axs[1, 2].set_title("Filtro Espacial Laplaciano (Referencia)", fontsize=10, fontweight="bold")

for ax in axs.ravel():
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["1"], "12_filtrado_frecuencial_pasabajos_pasaaltos.png"), bbox_inches="tight")
plt.close()
print("\nIMAGEN 12_filtrado_frecuencial_pasabajos_pasaaltos: muestra las máscaras de filtrado frecuencial, la reconstrucción con inversa y los filtrados de referencia\n")


# Asegurar que las imágenes reconstruidas estén en formato uint8 (0-255)
img_pasabajos_frec_u8 = np.clip(img_pasabajos_frec, 0, 255).astype(np.uint8)
img_pasabajos_esp_u8 = np.clip(img_pasabajos_esp, 0, 255).astype(np.uint8)

# 1. Cálculo de métricas para Pasa-Bajos Frecuencial
psnr_frec = psnr(roi_gris, img_pasabajos_frec_u8, data_range=255)
ssim_frec = ssim(roi_gris, img_pasabajos_frec_u8, data_range=255)

# 2. Cálculo de métricas para Pasa-Bajos Espacial (Gaussiano)
psnr_esp = psnr(roi_gris, img_pasabajos_esp_u8, data_range=255)
ssim_esp = ssim(roi_gris, img_pasabajos_esp_u8, data_range=255)

# Mostrar resultados
print(" ---- RECONTRUCCIÓN FRECUENCIAL: MÉTRICAS DE CALIDAD FRENTE A LA ROI ORIGINAL ----")
print(f"Reconstrucción Pasa-Bajos (Frecuencia)   -> PSNR: {psnr_frec:.2f} dB | SSIM: {ssim_frec:.4f}")
print(f"Filtrado Espacial Gaussiano (Referencia) -> PSNR: {psnr_esp:.2f} dB | SSIM: {ssim_esp:.4f}")
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------
# MÓDULO 2: SEGMENTACIÓN Y MORFOLOGÍA MATEMÁTICA DE LA PUPILA
# -----------------------------------------------------------
print_header("MÓDULO 2: Algoritmos de Segmentación y Limpieza Morfológica")

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("2.1 Algoritmos de Segmentación")
print("------------------------------")
# 1. Localización automática del centro y extracción de Sub-ROI ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Localización automática del centro y extracción de Sub-ROI AMPLIADA (150x150 px) ~~~~~~~~~~~~~~~~~~~~~~
# Extracción exclusiva del Canal Rojo (Canal 0) para máximo contraste pupilar
canal_rojo = roi[:, :, 0] if len(roi.shape) == 3 else roi.copy()

# Filtro de mediana inicial para atenuar ruido y textura fina
roi_filtrada = cv2.medianBlur(canal_rojo, 5)

# Localización gruesa robusta: Un suavizado de mediana amplio (15x15) disuelve
# pestañas y pequeños reflejos especulares, garantizando que minMaxLoc encuentre el centro pupilar real
rojo_suave = cv2.medianBlur(canal_rojo, 15)
_, _, (cx, cy), _ = cv2.minMaxLoc(rojo_suave)   # minMaxLoc devuelve (min, max, pos_min, pos_max)

# Sub-ROI ampliada centrada en el centroide estimado (150x150 px de trabajo)
semilado = 100
y0, y1 = max(cy - semilado, 0), min(cy + semilado, canal_rojo.shape[0])
x0, x1 = max(cx - semilado, 0), min(cx + semilado, canal_rojo.shape[1])

iris_gris = roi_filtrada[y0:y1, x0:x1]
iris_rgb = roi[y0:y1, x0:x1]
cx_l, cy_l = cx - x0, cy - y0   # Centro de la pupila en coordenadas locales de la sub-ROI ampliada


# --- GRÁFICO 1: INSPECCIÓN DE SUB-ROI ---
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(roi)
axs[0].plot(cx, cy, "c+", markersize=15, markeredgewidth=2)
axs[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="cyan", linewidth=1.5))
axs[0].set_title("ROI con centro estimado y sub-ROI", fontweight="bold")

axs[1].imshow(iris_rgb)
axs[1].set_title("Sub-ROI (RGB)", fontweight="bold")
axs[1].axis("off")

axs[2].imshow(iris_gris, cmap="gray")
axs[2].set_title("Sub-ROI Canal Rojo + Mediana 5x5", fontweight="bold")
axs[2].axis("off")

plt.tight_layout()
path_sub_roi = os.path.join(MOD_DIRS["2"], "01_sub_roi_estimacion_centro.png")
plt.savefig(path_sub_roi, bbox_inches="tight")
plt.close()

print("\nIMAGEN 01_sub_roi_estimacion_centro: muestra la sub-roi de trabajo extraída automáticamente para la segmentación")
print(f" Centro aproximado de la pupila en la ROI: (x={cx}, y={cy})")
print(f" Tamaño de la sub-ROI de trabajo: {iris_gris.shape[1]}x{iris_gris.shape[0]} px")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 2. Segmentación por Umbralización de Otsu (Global vs Jerárquico) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Función auxiliar: nos quedamos con la componente conexa que contiene (o está más cerca de) el centro de la pupila
def componente_central(mascara, centro):
    n, etiquetas = cv2.connectedComponents(mascara.astype(np.uint8))
    if n <= 1:
        return mascara.astype(bool)
    x, y = centro
    etiqueta = etiquetas[y, x]
    if etiqueta == 0:  # Si el centro cae en el fondo, tomamos la componente más cercana
        dist = ndi.distance_transform_edt(etiquetas == 0, return_indices=True)[1]
        etiqueta = etiquetas[dist[0][y, x], dist[1][y, x]]
    return etiquetas == etiqueta

# Solidez = Área / Área de la envolvente convexa. Para un disco (pupila) es ~1;
# si la región oscura "se escapa" hacia sombras o pestañas, la forma se vuelve irregular y la solidez baja
def solidez(mascara):
    contornos, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contornos:
        return 0
    c = max(contornos, key=cv2.contourArea)
    area_hull = cv2.contourArea(cv2.convexHull(c))
    return cv2.contourArea(c) / area_hull if area_hull > 0 else 0

# Otsu jerárquico: en cada pasada se recalcula el umbral de Otsu usando solo los píxeles
# que quedaron por debajo del umbral anterior (la clase oscura), hasta que la región central sea compacta
def otsu_jerarquico(imagen, centro, solidez_min=0.9, max_pasadas=4):
    umbrales = []
    pixeles = imagen.ravel()
    for _ in range(max_pasadas):
        u, _ = cv2.threshold(pixeles.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        umbrales.append(u)
        mascara = componente_central(imagen <= u, centro)
        if len(umbrales) >= 2 and solidez(mascara) >= solidez_min:
            break
        pixeles = imagen[imagen <= u]
    return mascara, umbrales

# Otsu clásico (una pasada): elige el umbral que maximiza la varianza entre clases del histograma
umbral_otsu, _ = cv2.threshold(iris_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
mascara_otsu_cruda = iris_gris <= umbral_otsu    # La pupila (oscura) queda en 1

mascara_otsu, umbrales_otsu = otsu_jerarquico(iris_gris, (cx_l, cy_l))
umbral_otsu_final = umbrales_otsu[-1]

# --- GRÁFICO 2: SEGMENTACIÓN OTSU ---
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].hist(iris_gris.ravel(), bins=256, range=[0, 256], color="black")
for k, u in enumerate(umbrales_otsu):
    axs[0].axvline(u, color=plt.cm.autumn(k / max(len(umbrales_otsu) - 1, 1)), label=f"Pasada {k + 1}: {u:.0f}")
axs[0].set_title("Histograma de la sub-ROI")
axs[0].set_xlabel("Nivel de intensidad")
axs[0].set_ylabel("Frecuencia de aparición")
axs[0].legend()

axs[1].imshow(mascara_otsu_cruda, cmap="gray")
axs[1].set_title("Otsu global (1 pasada)")
axs[1].axis("off")

axs[2].imshow(mascara_otsu, cmap="gray")
axs[2].set_title(f"Otsu jerárquico ({len(umbrales_otsu)} pasadas) + comp. central")
axs[2].axis("off")

axs[3].imshow(iris_rgb)
axs[3].contour(mascara_otsu, levels=[0.5], colors="cyan")
axs[3].set_title("Contorno Otsu superpuesto")
axs[3].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["2"], "02_segmentacion_otsu.png"), bbox_inches="tight")
plt.close()
print("\nIMAGEN 02_segmentacion_otsu: muestra los umbrales establecidos por otsu en el histograma y las máscaras de la primera y la segunda pasada")
print(f"  Umbral de Otsu global: {umbral_otsu:.0f}")
print(f"  Umbrales en cada pasada: {[int(u) for u in umbrales_otsu]}")
print(f"  Solidez de la máscara final: {solidez(mascara_otsu):.3f}")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 3. Segmentación por Agrupamiento K-Means ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Espacio croma/intensidad: usamos CIE L*a*b* (L = luminancia, a y b = cromaticidad)
iris_lab = cv2.cvtColor(cv2.medianBlur(iris_rgb, 5), cv2.COLOR_RGB2LAB)
caracteristicas = iris_lab.reshape(-1, 3).astype(float)

mascaras_kmeans = {}
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

for i, K in enumerate(range(2, 6)):
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(caracteristicas)
    etiquetas = km.labels_.reshape(iris_gris.shape)

    # El cluster de la pupila es el de menor luminancia media (L)
    cluster_pupila = np.argmin(km.cluster_centers_[:, 0])
    mascaras_kmeans[K] = componente_central(etiquetas == cluster_pupila, (cx_l, cy_l))

    axs[0, i].imshow(etiquetas, cmap="tab10")
    axs[0, i].set_title(f"K-Means K={K} (clusters)")
    axs[0, i].axis("off")

    axs[1, i].imshow(iris_rgb)
    axs[1, i].contour(mascaras_kmeans[K], levels=[0.5], colors="cyan")
    axs[1, i].set_title(f"Cluster pupila K={K}")
    axs[1, i].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["2"], "03_segmentacion_kmeans.png"), bbox_inches="tight")
plt.close()
print("\nIMAGEN 03_segmentacion_kmeans: clustering por k-means para distintos valores de K\n")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 4. Segmentación por Contornos Activos (Snake) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Radio inicial ajustable manualmente según el tamaño de la pupila en el video
radio_inicial = 46 

# Contorno inicial: círculo centrado en la pupila (cy_l, cx_l)
t = np.linspace(0, 2 * np.pi, 200)
snake_inicial = np.array([
    cy_l + radio_inicial * np.sin(t),
    cx_l + radio_inicial * np.cos(t)
]).T  # skimage usa coordenadas (fila, columna) -> (y, x)

# Imagen suavizada para que el gradiente atraiga al contorno
imagen_snake = gaussian(iris_gris, sigma=2, preserve_range=False)

# Parámetros originales del notebook
# alpha: elasticidad, beta: rigidez, w_line: atracción a intensidad, w_edge: atracción a bordes
snake = active_contour(
    imagen_snake,
    snake_inicial,
    alpha=0.015,
    beta=10,
    gamma=0.001,
    w_line=0,
    w_edge=1,
    max_num_iter=2500
)

# Conversión del contorno ajustado a máscara binaria rellena
mascara_snake = np.zeros(iris_gris.shape, dtype=bool)
rr, cc = polygon(snake[:, 0], snake[:, 1], iris_gris.shape)
mascara_snake[rr, cc] = True

# --- GRÁFICO 4: SEGMENTACIÓN SNAKE ---
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(iris_gris, cmap="gray")
axs[0].plot(snake_inicial[:, 1], snake_inicial[:, 0], "--r", label="Contorno inicial")
axs[0].plot(snake[:, 1], snake[:, 0], "-c", linewidth=2, label="Contorno final")
axs[0].set_title("Contorno activo (Snake)")
axs[0].legend(loc="lower right")
axs[0].axis("off")

axs[1].imshow(mascara_snake, cmap="gray")
axs[1].set_title("Máscara resultante")
axs[1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["2"], "04_segmentacion_snake.png"), bbox_inches="tight")
plt.close()

print("\nIMAGEN 04_segmentacion_snake: muestra el contorno inicial y la máscara final\n")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 5. Segmentación por Watershed ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Imagen de relieve: magnitud del gradiente (Sobel). Los bordes son las "crestas" entre cuencas
gradiente = sobel(iris_gris.astype(float) / 255.0)

# Marcadores (semillas de inundación)
marcadores = np.zeros(iris_gris.shape, dtype=np.int32)

# 1: Fondo -> todo lo claramente más claro que la pupila (iris, esclerótica, piel)
zona_borde = cv2.dilate(mascara_otsu.astype(np.uint8), np.ones((7, 7))) > 0   # Franja sin marcar alrededor de la pupila
umbral_ref_fondo = umbrales_otsu[1] if len(umbrales_otsu) > 1 else umbrales_otsu[0]
marcadores[(iris_gris > umbral_ref_fondo) & ~zona_borde] = 1

# 2: Núcleo de la pupila -> máscara de Otsu erosionada (seguro dentro de la pupila)
marcadores[cv2.erode(mascara_otsu.astype(np.uint8), np.ones((5, 5))) > 0] = 2

# 3: Reflejo corneal -> top-hat blanco sobre la sub-ROI SIN filtrar
# (Resalta objetos chicos y brillantes respecto de su entorno), restringido a la zona de la pupila
iris_sin_filtrar = roi_gris[y0:y1, x0:x1]
top_hat = cv2.morphologyEx(iris_sin_filtrar, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
cerca_pupila = cv2.dilate(mascara_otsu.astype(np.uint8), np.ones((9, 9))) > 0
reflejo = (top_hat > 40) & cerca_pupila
marcadores[reflejo] = 3

etiquetas_ws = watershed(gradiente, marcadores)
mascara_ws_sin_reflejo = etiquetas_ws == 2
cuenca_reflejo = etiquetas_ws == 3

# Pupila final: se reincorpora el reflejo (que está "dentro" de la pupila) y se rellenan huecos
mascara_ws = ndi.binary_fill_holes(mascara_ws_sin_reflejo | cuenca_reflejo)

# --- GRÁFICO 5: SEGMENTACIÓN WATERSHED ---
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(gradiente, cmap="magma")
axs[0].set_title("Magnitud del gradiente (relieve)")
axs[0].axis("off")

axs[1].imshow(marcadores, cmap="tab10", vmin=0, vmax=9)
axs[1].set_title("Marcadores: sin marcar(0), fondo(1),\npupila(2), reflejo(3)")
axs[1].axis("off")

axs[2].imshow(etiquetas_ws, cmap="tab10", vmin=0, vmax=9)
axs[2].set_title("Cuencas resultantes")
axs[2].axis("off")

axs[3].imshow(iris_rgb)
axs[3].contour(mascara_ws, levels=[0.5], colors="cyan")
if cuenca_reflejo.any():
    axs[3].contour(cuenca_reflejo, levels=[0.5], colors="red")
axs[3].set_title("Pupila (cian) y reflejo (rojo) - Watershed")
axs[3].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["2"], "05_segmentacion_watershed.png"), bbox_inches="tight")
plt.close()
print("\nIMAGEN 05_segmentacion_watershed: muestra la máscara de watershed resultante, los marcadores y la magnitud del gradiente (crestas)\n")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 6. Comparativa Final de Métodos de Segmentación ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
K_elegido = 5   

mascaras = {
    "Otsu": mascara_otsu,
    f"K-Means (K={K_elegido})": mascaras_kmeans[K_elegido],
    "Snake": mascara_snake,
    "Watershed": mascara_ws,
}

# --- GRÁFICO 6: COMPARATIVA DE MÉTODOS ---
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

for ax, (nombre, m) in zip(axs, mascaras.items()):
    ax.imshow(iris_rgb)
    ax.contour(m, levels=[0.5], colors="cyan")
    
    # Cálculo del área y diámetro equivalente
    area_px = m.sum()
    diametro_eq = 2 * np.sqrt(area_px / np.pi)
    
    ax.set_title(f"{nombre}\nÁrea = {area_px} px | D_eq = {diametro_eq:.1f} px)")
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["2"], "06_comparativa_metodos_segmentacion.png"), bbox_inches="tight")
plt.close()
print("\nIMAGEN 06_comparativa_metodos_segmentacion: compara las cuatro máscaras obtenidas\n")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("2.2 Morfología matemática para limpieza de máscara")
print("--------------------------------------------------")

# 7. Operaciones Morfológicas Matemáticas (Erosión, Dilatación, Apertura, Cierre y Combinada) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Máscara de partida: la del mejor método de segmentación 
mascara_base = mascaras["Otsu"].astype(np.uint8)

# Elemento estructurante elíptico (acompaña la forma circular de la pupila)
ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

m_erosion = cv2.erode(mascara_base, ee, iterations=1)
m_dilatacion = cv2.dilate(mascara_base, ee, iterations=1)
m_apertura = cv2.morphologyEx(mascara_base, cv2.MORPH_OPEN, ee)     # erosión -> dilatación
m_cierre = cv2.morphologyEx(mascara_base, cv2.MORPH_CLOSE, ee)      # dilatación -> erosión

# Combinación propuesta: cierre (tapa el hueco del reflejo) -> apertura (quita pestañas/protuberancias)
# -> relleno de huecos -> componente conexa central
ee_grande = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
m_comb = cv2.morphologyEx(mascara_base, cv2.MORPH_CLOSE, ee_grande)
m_comb = cv2.morphologyEx(m_comb, cv2.MORPH_OPEN, ee)
m_comb = ndi.binary_fill_holes(m_comb)
m_comb = componente_central(m_comb, (cx_l, cy_l))

resultados_morf = {
    "Máscara base": mascara_base,
    "Erosión": m_erosion,
    "Dilatación": m_dilatacion,
    "Apertura": m_apertura,
    "Cierre": m_cierre,
    "Cierre + Apertura + Relleno": m_comb,
}

# --- GRÁFICO 7: OPERACIONES MORFOLÓGICAS ---
fig, axs = plt.subplots(2, 6, figsize=(24, 8))

for i, (nombre, m) in enumerate(resultados_morf.items()):
    axs[0, i].imshow(m, cmap="gray")
    axs[0, i].set_title(nombre)
    axs[0, i].axis("off")

    axs[1, i].imshow(iris_rgb)
    axs[1, i].contour(m, levels=[0.5], colors="cyan")
    axs[1, i].set_title(f"Área = {int(m.sum())} px")
    axs[1, i].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["2"], "07_operaciones_morfologicas.png"), bbox_inches="tight")
plt.close()
print("\nIMAGEN 07_operaciones_morfologicas: muestra varias operaciones morfológicas aplicadas a la máscara de Otsu\n")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 8. Esqueletización (Eje Medio y Contorno Pupilar) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Esqueleto de la máscara (eje medio): para un disco tiende a colapsar en un punto/segmento cerca del centro
esqueleto_mascara = skeletonize(m_comb)

# Contorno pupilar: gradiente morfológico (máscara - erosión) y luego esqueletización para afinarlo a 1 px
borde = m_comb & ~cv2.erode(m_comb.astype(np.uint8), np.ones((3, 3))).astype(bool)
esqueleto_contorno = skeletonize(ndi.binary_dilation(borde))

# --- GRÁFICO 8: ESQUELETIZACIÓN ---
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(m_comb, cmap="gray")
axs[0].imshow(np.ma.masked_where(~esqueleto_mascara, esqueleto_mascara), cmap="autumn")
axs[0].set_title("Esqueleto de la máscara")
axs[0].axis("off")

axs[1].imshow(esqueleto_contorno, cmap="gray")
axs[1].set_title("Esqueleto del contorno pupilar")
axs[1].axis("off")

axs[2].imshow(iris_rgb)
yy, xx = np.nonzero(esqueleto_contorno)
axs[2].plot(xx, yy, "c.", markersize=2)
axs[2].set_title("Contorno esqueletizado superpuesto")
axs[2].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(MOD_DIRS["2"], "08_esqueletizacion.png"), bbox_inches="tight")
plt.close()
print("\nIMAGEN 08_esqueletizacion: muestra el contorno del contorno pupilar obtenido por esqueletización")
print(f"  Perímetro aproximado (píxeles del contorno esqueletizado): {esqueleto_contorno.sum()} px")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 9. Validación multiframe y métricas cuantitativas ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Extracción de frames de validación desde el video (t + 0.5s, t + 1.0s, t + 1.5s)
tiempos_val = [seg_repr + 0.5, seg_repr + 1.0, seg_repr + 1.5]
frames_val = {}

cap_val = cv2.VideoCapture(VIDEO_PATH)
for t_val in tiempos_val:
    idx_val = int(t_val * fps)
    cap_val.set(cv2.CAP_PROP_POS_FRAMES, idx_val)
    ret_val, frame_bgr_val = cap_val.read()
    if ret_val:
        nombre_frame = f"t + {t_val - seg_repr:.1f}s"
        frames_val[nombre_frame] = cv2.cvtColor(frame_bgr_val, cv2.COLOR_BGR2RGB)
cap_val.release()

# Helper para estimar centro en ROI usando Canal Rojo
def centro_pupila(frame_rgb):
    canal_r = frame_rgb[y_inicial:y_final, x_inicial:x_final, 0]
    _, _, (cx, cy), _ = cv2.minMaxLoc(cv2.medianBlur(canal_r, 15))
    return cx, cy

# 2. Visualización ampliada con grilla para inspección manual (ROI Completa)
fig, axs = plt.subplots(1, len(frames_val), figsize=(18, 6))
if len(frames_val) == 1:
    axs = [axs]

for ax, (nombre, frame) in zip(axs, frames_val.items()):
    roi_f = frame[y_inicial:y_final, x_inicial:x_final]
    alto_f, ancho_f = roi_f.shape[:2]

    ax.imshow(roi_f)

    # Grilla sobre la ROI completa del ojo (sin encajonar la pupila)
    ax.set_xticks(np.arange(0, ancho_f, 25))
    ax.set_yticks(np.arange(0, alto_f, 25))
    ax.set_xticks(np.arange(0, ancho_f, 5), minor=True)
    ax.set_yticks(np.arange(0, alto_f, 5), minor=True)

    ax.grid(which="major", color="white", linewidth=0.6, alpha=0.8)
    ax.grid(which="minor", color="white", linewidth=0.2, alpha=0.4)
    ax.set_title(f"Anotación: {nombre}", fontweight="bold")

plt.tight_layout()
path_anotacion_fig = os.path.join(MOD_DIRS["2"], "09_inspeccion_anotacion_manual.png")
plt.savefig(path_anotacion_fig, bbox_inches="tight")
plt.close()

print("\nIMAGEN 09_inspeccion_anotaciones_manual: muestra los frames de validación con la ROI completa")

# 3. Solicitud opcional de límites manuales por terminal (con navegación 'v' para volver)
opcion_val = input("\n¿Desea obtener las métricas de error respecto de una referencia manual? (si/no): ").strip().lower()
hacer_referencia = opcion_val in ["si", "s", "sí"]

anotaciones = {}

if hacer_referencia:
    print("\n||||||||||||||||||||||||||||||||||||||||||||")
    print("  SELECCIÓN MANUAL DE LÍMITES DE LA PUPILA")
    print("||||||||||||||||||||||||||||||||||||||||||||\n")
    print("  Consulte la figura 09 para identificar las coordenadas exactas de la pupila.")
    print("  Nota: en caso de error de tipeo, ingrese 'v' en cualquier momento para volver a la coordenada anterior.\n")

    pasos = []
    for nombre in frames_val.keys():
        for c_key, c_desc in [("x_izq", "x_izq (límite izquierdo)"), 
                             ("x_der", "x_der (límite derecho)"), 
                             ("y_sup", "y_sup (límite superior)"), 
                             ("y_inf", "y_inf (límite inferior)")]:
            pasos.append((nombre, c_key, c_desc))

    valores_anotacion = {}
    i = 0
    while i < len(pasos):
        nombre, c_key, c_desc = pasos[i]
        entrada = input(f"  Frame {nombre} -> {c_desc} [px]: ").strip().lower()
        if entrada == "v":
            if i > 0:
                i -= 1
                print("  << Volviendo a la coordenada anterior...\n")
            else:
                print("  Ya se encuentra en la primera coordenada.\n")
        else:
            try:
                valores_anotacion[(nombre, c_key)] = int(entrada)
                i += 1
            except ValueError:
                print("  Entrada inválida. Ingrese un número entero o 'v' para volver.")

    for nombre in frames_val.keys():
        anotaciones[nombre] = (
            valores_anotacion[(nombre, "x_izq")],
            valores_anotacion[(nombre, "x_der")],
            valores_anotacion[(nombre, "y_sup")],
            valores_anotacion[(nombre, "y_inf")]
        )

def mascara_referencia(anotacion, forma):
    x_izq, x_der, y_sup, y_inf = anotacion
    m = np.zeros(forma, dtype=bool)
    rr, cc = ellipse((y_sup + y_inf) / 2.0, (x_izq + x_der) / 2.0,
                     (y_inf - y_sup) / 2.0, (x_der - x_izq) / 2.0, shape=forma)
    m[rr, cc] = True
    return m

# 4. Segmentación en batch para frames de validación
def segmentar_todos(frame_rgb, semilado=45):
    roi_f = frame_rgb[y_inicial:y_final, x_inicial:x_final]
    gris = cv2.cvtColor(roi_f, cv2.COLOR_RGB2GRAY)
    filtrada = cv2.medianBlur(gris, 5)
    cx, cy = centro_pupila(frame_rgb)
    y0, y1 = max(cy - semilado, 0), min(cy + semilado, gris.shape[0])
    x0, x1 = max(cx - semilado, 0), min(cx + semilado, gris.shape[1])
    sub, sub_sf, sub_rgb = filtrada[y0:y1, x0:x1], gris[y0:y1, x0:x1], roi_f[y0:y1, x0:x1]
    c = (cx - x0, cy - y0)
    m = {}

    # Otsu jerárquico
    m["Otsu"], umbrales = otsu_jerarquico(sub, c)

    # K-Means (K=5)
    lab = cv2.cvtColor(cv2.medianBlur(sub_rgb, 5), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
    km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(lab)
    m["K-Means (K=5)"] = componente_central(km.labels_.reshape(sub.shape) == np.argmin(km.cluster_centers_[:, 0]), c)

    # Snake
    r0 = 1.5 * np.sqrt(m["Otsu"].sum() / np.pi) + 5
    t = np.linspace(0, 2 * np.pi, 200)
    ini = np.array([c[1] + r0 * np.sin(t), c[0] + r0 * np.cos(t)]).T
    sn = active_contour(gaussian(sub, sigma=2), ini, alpha=0.015, beta=10, gamma=0.001,
                        w_line=0, w_edge=1, max_num_iter=2500)
    m["Snake"] = np.zeros(sub.shape, dtype=bool)
    rr, cc = polygon(sn[:, 0], sn[:, 1], sub.shape)
    m["Snake"][rr, cc] = True

    # Watershed
    grad = sobel(sub.astype(float) / 255.0)
    marc = np.zeros(sub.shape, dtype=np.int32)
    u_ref = umbrales[1] if len(umbrales) > 1 else umbrales[0]
    marc[(sub > u_ref) & ~(cv2.dilate(m["Otsu"].astype(np.uint8), np.ones((7, 7))) > 0)] = 1
    marc[cv2.erode(m["Otsu"].astype(np.uint8), np.ones((5, 5))) > 0] = 2
    th = cv2.morphologyEx(sub_sf, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    marc[(th > 40) & (cv2.dilate(m["Otsu"].astype(np.uint8), np.ones((9, 9))) > 0)] = 3
    et = watershed(grad, marc)
    m["Watershed"] = ndi.binary_fill_holes((et == 2) | (et == 3))

    # Operaciones morfológicas
    base = m["Otsu"].astype(np.uint8)
    ee5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ee7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m["Otsu + Erosión"] = cv2.erode(base, ee5)
    m["Otsu + Dilatación"] = cv2.dilate(base, ee5)
    m["Otsu + Apertura"] = cv2.morphologyEx(base, cv2.MORPH_OPEN, ee5)
    m["Otsu + Cierre"] = cv2.morphologyEx(base, cv2.MORPH_CLOSE, ee5)
    comb = cv2.morphologyEx(cv2.morphologyEx(base, cv2.MORPH_CLOSE, ee7), cv2.MORPH_OPEN, ee5)
    m["Otsu + Cierre + Apertura + Relleno"] = componente_central(ndi.binary_fill_holes(comb), c)

    salida = {}
    for k, v in m.items():
        full = np.zeros(gris.shape, dtype=bool)
        full[y0:y1, x0:x1] = v.astype(bool)
        salida[k] = full
    return salida, (cx, cy)

# 5. Cálculo de métricas cuantitativas (si se seleccionó referencia)
def dice(a, b):
    denom = a.sum() + b.sum()
    return (2.0 * np.logical_and(a, b).sum() / denom) if denom > 0 else 0.0

def jaccard(a, b):
    denom = np.logical_or(a, b).sum()
    return (np.logical_and(a, b).sum() / denom) if denom > 0 else 0.0

filas = []
resultados_val = {}

for nombre in frames_val.keys():
    mascaras_f, centro_f = segmentar_todos(frames_val[nombre])
    
    if hacer_referencia and nombre in anotaciones:
        ref = mascara_referencia(anotaciones[nombre], mascaras_f["Otsu"].shape)
        resultados_val[nombre] = (mascaras_f, ref, centro_f)

        for metodo, mf in mascaras_f.items():
            error_area = (100.0 * (mf.sum() - ref.sum()) / ref.sum()) if ref.sum() > 0 else 0.0
            filas.append({
                "Frame": nombre,
                "Método": metodo,
                "Dice": dice(mf, ref),
                "Jaccard / IoU": jaccard(mf, ref),
                "Error de área [%]": error_area
            })
    else:
        resultados_val[nombre] = (mascaras_f, None, centro_f)

if hacer_referencia and len(filas) > 0:
    df_val = pd.DataFrame(filas)

    print("\n--------------------------------------------------")
    print("Métricas de validación + comparación de resultados")
    print("--------------------------------------------------\n")

    for nombre, (_, ref, _) in resultados_val.items():
        if ref is not None:
            d_ref = 2.0 * np.sqrt(ref.sum() / np.pi)
            print(f"{nombre}: Diámetro equivalente de referencia = {d_ref:.1f} px")

    # Guardado en CSV
    path_csv_val = os.path.join(MOD_DIRS["2"], "metricas_validacion.csv")
    df_val.to_csv(path_csv_val, index=False)
    print(f"\nTABLA metricas_validacion.csv: métricas Dice, Jaccard, Error de área respecto de las referencias\n")

    # Resumen numérico agrupado
    resumen = df_val.groupby("Método", sort=False)[["Dice", "Jaccard / IoU", "Error de área [%]"]].agg(["mean", "std"]).round(3)
    print("======================= RESUMEN DE MÉTRICAS (PROMEDIO Y DESVÍO ESTÁNDAR) =======================")
    print(resumen.to_string(col_space=12))
else:
    print("\nValidación con referencia manual omitida. Generando gráfico de superposición directo...")

# 6. Superposición de contornos sobre los frames de validación
metodos_graf = ["Otsu", "K-Means (K=5)", "Snake", "Watershed", "Otsu + Cierre + Apertura + Relleno"]
colores = ["cyan", "magenta", "yellow", "red", "orange"]

fig, axs = plt.subplots(1, len(resultados_val), figsize=(5 * len(resultados_val), 5), squeeze=False)

for ax, (nombre, (mascaras_f, ref, (cx_f, cy_f))) in zip(axs[0], resultados_val.items()):
    ax.imshow(frames_val[nombre][y_inicial:y_final, x_inicial:x_final])
    if ref is not None:
        ax.contour(ref, levels=[0.5], colors="lime", linewidths=2)
    for met, col in zip(metodos_graf, colores):
        ax.contour(mascaras_f[met], levels=[0.5], colors=col, linewidths=1)
    ax.set_xlim(cx_f - semilado, cx_f + semilado)
    ax.set_ylim(cy_f + semilado, cy_f - semilado)
    ax.set_title(nombre, fontweight="bold")
    ax.axis("off")

if hacer_referencia:
    handles = [plt.Line2D([], [], color=c_) for c_ in ["lime"] + colores]
    labels_legend = ["Referencia manual"] + metodos_graf
else:
    handles = [plt.Line2D([], [], color=c_) for c_ in colores]
    labels_legend = metodos_graf

fig.legend(handles, labels_legend, loc="lower center", ncol=len(labels_legend))
plt.tight_layout(rect=[0, 0.08, 1, 1])

path_fig_superposicion = os.path.join(MOD_DIRS["2"], "10_superposicion_contornos_validacion.png")
plt.savefig(path_fig_superposicion, bbox_inches="tight")
plt.close()

print("\nIMAGEN 10_superposicion_contornos_validacion: muestra la superposición de los contornos de cada método de segmentación\n")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ----------------------------------------------------
# MÓDULO 3: EXTRACTION DE CARACTERÍSTICAS Y VALIDACIÓN
# ----------------------------------------------------
borde_afinado = esqueleto_contorno
print_header("MÓDULO 3: Caracterización Morfométrica, Textura y Validación Manual")

print("¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨")
print("3.1 Descriptores Morfométricos y de Textura en el Frame Segmentado")
print("------------------------------------------------------------------")

# 1. Elementos estructurantes y posprocesamiento morfológico |||||||||||||||||||||||||||||||||||||||||||||||||||||||
ee5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
ee7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Cierre (7x7) -> Apertura (5x5) -> Relleno de huecos
mascara_cierre = cv2.morphologyEx(
    mascara_ws.astype(np.uint8), cv2.MORPH_CLOSE, ee7
)
mascara_apertura = cv2.morphologyEx(mascara_cierre, cv2.MORPH_OPEN, ee5)
mascara_ws = ndi.binary_fill_holes(mascara_apertura)

# Conversión correcta a uint8 (0 a 255)
mascara_uint8 = (mascara_ws.astype(np.uint8)) * 255

# 2. Área (A)
area = float(np.sum(mascara_ws))

# 3. Perímetro, Centroide y Descriptores
contornos, _ = cv2.findContours(
    mascara_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

if len(contornos) > 0:
    contorno_pupila = max(contornos, key=cv2.contourArea)
    perimetro = cv2.arcLength(contorno_pupila, True)

    momentos = cv2.moments(contorno_pupila)
    if momentos["m00"] != 0:
        xc_local = momentos["m10"] / momentos["m00"]
        yc_local = momentos["m01"] / momentos["m00"]
    else:
        xc_local, yc_local = 0.0, 0.0

    xc = xc_local + x0
    yc = yc_local + y0

    # Circularidad (C)
    circularidad = (
        (4.0 * np.pi * area) / (perimetro**2) if perimetro > 0 else 0.0
    )

    # Excentricidad (e) mediante ajuste elíptico
    if len(contorno_pupila) >= 5:
        (x_e_local, y_e_local), (eje_menor, eje_mayor), angulo = cv2.fitEllipse(
            contorno_pupila
        )
        a = eje_mayor / 2.0
        b = eje_menor / 2.0
        excentricidad = (
            np.sqrt(1.0 - (b**2) / (a**2)) if (a > 0 and a >= b) else 0.0
        )

        x_e = x_e_local + x0
        y_e = y_e_local + y0
    else:
        excentricidad = 0.0
        x_e, y_e = xc, yc
        eje_menor, eje_mayor, angulo = 0.0, 0.0, 0.0
else:
    perimetro = 0.0
    xc, yc = float(x0), float(y0)
    circularidad, excentricidad = 0.0, 0.0
    x_e, y_e = xc, yc
    eje_menor, eje_mayor, angulo = 0.0, 0.0, 0.0

# -----------------------------------------------------------------------------
# RECONSTRUCCIÓN DE MÁSCARAS GLOBALES (Antes de graficar)
# -----------------------------------------------------------------------------
h_sub, w_sub = mascara_ws.shape
mascara_ws_global = np.zeros(roi_gris.shape, dtype=bool)
mascara_ws_global[y0 : y0 + h_sub, x0 : x0 + w_sub] = mascara_ws

# Obtención segura de marcadores 
marcador_ref = locals().get("mascara_marcador", locals().get("marcador_ws", None))

# -----------------------------------------------------------------------------
# VISUALIZACIÓN Y LEYENDA
# -----------------------------------------------------------------------------
elipse_img = cv2.cvtColor(roi_gris.copy(), cv2.COLOR_GRAY2RGB)
if len(contornos) > 0 and len(contorno_pupila) >= 5:
    cv2.ellipse(
        elipse_img,
        ((int(x_e), int(y_e)), (int(eje_menor), int(eje_mayor)), angulo),
        (0, 255, 0),
        1,
    )

fig, ax = plt.subplots(figsize=(9, 6))
ax.imshow(elipse_img)

# Contorno final de Watershed posprocesado (Cian)
ax.contour(mascara_ws_global, levels=[0.5], colors="cyan", linewidths=1.5)

# Centroide geométrico (Rojo)
ax.plot(xc, yc, "ro", markersize=6)

# Marcadores de semilla inicial (Amarillo punteado)
if marcador_ref is not None:
    h_m, w_m = marcador_ref.shape
    marcador_ws_global = np.zeros(roi_gris.shape, dtype=bool)
    marcador_ws_global[y0 : y0 + h_m, x0 : x0 + w_m] = marcador_ref
    ax.contour(
        marcador_ws_global,
        levels=[0.5],
        colors="yellow",
        linewidths=1.2,
        linestyles="--",
    )

# Construcción de la leyenda
handles = [
    plt.Line2D(
        [],
        [],
        color="cyan",
        linewidth=1.5,
        label="Watershed Posproc. (Cierre+Apertura+Relleno)",
    ),
    plt.Line2D(
        [],
        [],
        color="lime",
        linewidth=1.5,
        label="Ajuste Elíptico (cv2.fitEllipse)",
    ),
    plt.Line2D(
        [],
        [],
        color="red",
        marker="o",
        linestyle="None",
        label=f"Centroide ({xc:.1f}, {yc:.1f}) px",
    ),
]

if marcador_ref is not None:
    handles.append(
        plt.Line2D(
            [],
            [],
            color="yellow",
            linestyle="--",
            linewidth=1.2,
            label="Marcador Inicial WS (Núcleo)",
        )
    )

ax.set_title(
    "Segmentación Watershed Posprocesada con Centroide y Elipse Ajustada",
    fontweight="bold",
    fontsize=11,
)
ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)
ax.axis("off")
plt.tight_layout()

# Guardado modular en la subcarpeta del Módulo 3
path_fig_morf = os.path.join(MOD_DIRS["3"], "01_descriptores_morfometricos_frame.png")
plt.savefig(path_fig_morf, bbox_inches="tight")
plt.close()
print("\nIMAGEN 01_descriptores_morfometricos_frame: se muestra la ROI original segmentada con watershed y la ubicacion del centroide\n")
# Imprimir resultados
print("DESCRIPTORES MORFOMÉTRICOS EN EL FRAME SEGMENTADO")
print(f"Área (A): {area:.2f} px")
print(f"Perímetro (P): {perimetro:.2f} px")
print(f"Centroide (Xc, Yc): ({xc:.2f}, {yc:.2f}) px")
print(f"Circularidad (C): {circularidad:.4f}")
print(f"Excentricidad (e): {excentricidad:.4f}")
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# 2. Descriptores de Textura de Primer Orden (Canales RGB) ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
canales_nombres = ["Rojo (R)", "Verde (G)", "Azul (B)"]
resultados_textura = []

for i, nombre_c in enumerate(canales_nombres):
    # Extracción de los píxeles dentro de la máscara para el canal correspondiente
    canal = roi[:, :, i]
    pixeles = canal[mascara_ws_global]

    if len(pixeles) > 0:
        # 1. Media
        media = float(np.mean(pixeles))

        # 2. Desvío Estándar
        desvio = float(np.std(pixeles))

        # 3. Asimetría (Skewness)
        asimetria = float(skew(pixeles))

        # 4. Entropía de 1er orden (histograma a 256 niveles de intensidad)
        hist, _ = np.histogram(pixeles, bins=256, range=(0, 256), density=True)
        hist_pos = hist[hist > 0]
        entropia_val = -float(np.sum(hist_pos * np.log2(hist_pos)))
    else:
        media, desvio, asimetria, entropia_val = 0.0, 0.0, 0.0, 0.0

    resultados_textura.append({
        "Canal": nombre_c,
        "Media": round(media, 2),
        "Desvío Estándar": round(desvio, 2),
        "Asimetría (Skewness)": round(asimetria, 4),
        "Entropía [bits]": round(entropia_val, 4)
    })

# Tabla resumen
df_textura_1er = pd.DataFrame(resultados_textura)

print("\nTABLA descriptores_tetura_1er_orden.csv: textura de primer orden\n")
print("       >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("       DESCRIPTORES DE TEXTURA DE 1ER ORDEN (CANALES RGB)")
print("       <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print(df_textura_1er.to_string(index=False))

# Guardado en CSV dentro de la subcarpeta del Módulo 3
path_csv_textura = os.path.join(MOD_DIRS["3"], "descriptores_textura_1er_orden.csv")
df_textura_1er.to_csv(path_csv_textura, index=False)
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# 3. Descriptores de Textura de 2do Orden (GLCM / Haralick en Canales RGB) |||||||||||||||||||||||||||||||||||||||||||||||||||
# Parametrización
N_NIVELES = 32
factor_cuant = 256 // N_NIVELES
distancias = [1]
angulos = [0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]

iris_cuant_rgb = (iris_rgb // factor_cuant).astype(np.uint8)
canales_nombres = ["Rojo (R)", "Verde (G)", "Azul (B)"]
resultados_glcm = []

fig = plt.figure(figsize=(15, 4))

for i, nombre_c in enumerate(canales_nombres):
    canal_cuant = iris_cuant_rgb[:, :, i]

    glcm = graycomatrix(
        canal_cuant,
        distances=distancias,
        angles=angulos,
        levels=N_NIVELES,
        symmetric=True,
        normed=True,
    )

    # 1. Métricas globales
    contraste = float(np.mean(graycoprops(glcm, "contrast")))
    homogeneidad = float(np.mean(graycoprops(glcm, "homogeneity")))
    energia = float(np.mean(graycoprops(glcm, "energy")))
    correlacion = float(np.mean(graycoprops(glcm, "correlation")))

    resultados_glcm.append({
        "Canal": nombre_c,
        "Contraste": round(contraste, 4),
        "Homogeneidad": round(homogeneidad, 4),
        "Energía": round(energia, 4),
        "Correlación": round(correlacion, 4),
    })

    # 2. Promediamos la GLCM sobre las 4 direcciones para visualizarla en 2D
    glcm_promedio = glcm[:, :, 0, :].mean(axis=2)

    # Graficar el mapa de calor de la matriz 32x32
    plt.subplot(1, 3, i + 1)
    plt.imshow(glcm_promedio, cmap="viridis")
    plt.colorbar(label="Probabilidad")
    plt.title(f"GLCM - Canal {nombre_c}")
    plt.xlabel("Nivel j")
    plt.ylabel("Nivel i")

plt.tight_layout()

# Guardado modular del mapa de calor de la GLCM
path_fig_glcm = os.path.join(MOD_DIRS["3"], "02_matrices_glcm_rgb.png")
plt.savefig(path_fig_glcm, bbox_inches="tight")
plt.close()
print("\nIMAGEN 02_matrices_glcm_rgb: muestra las matrices glcm en los canales R, G y B\n")

# Tabla resumen de descriptores de segundo orden
df_glcm = pd.DataFrame(resultados_glcm)

print("\nTABLA descriptores_textura_2do_orden_glcm.csv: textura de segundo orden para cada canal\n")

print("       ////////////////////////////////////////////")
print("       ATRIBUTOS DE SEGUNDO ORDEN (HARALICK / GLCM)")
print("       ////////////////////////////////////////////")
print(df_glcm.to_string(index=False))

# Guardado en CSV
path_csv_glcm = os.path.join(MOD_DIRS["3"], "descriptores_textura_2do_orden_glcm.csv")
df_glcm.to_csv(path_csv_glcm, index=False)
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

print("\n¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨")
print("3.2 Procesamiento en Video y Curva Temporal Fisiológica")
print("------------------------------------------------------------------")

# Anotación del Ground Truth ____________________________________________________________________________________________
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"No se encontró el archivo de video en: {VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duracion_s = total_frames / fps if fps > 0 else 0.0

# Coordenadas de la ROI del ojo (extendida con 5 px de margen sobre la elegida en Módulo 0)
y0_ojo = max(0, y_inicial - 5)
y1_ojo = min(height, y_final + 5)
x0_ojo = max(0, x_inicial - 5)
x1_ojo = min(width, x_final + 5)

# 1. Generación automática equispaciada de 6 keyframes
n_keyframes = 6
inicio_f = int(fps * 0.5) if total_frames > fps else 0
fin_f = int(total_frames - fps * 0.5) if total_frames > fps else total_frames - 1
frames_objetivo = np.linspace(inicio_f, fin_f, n_keyframes, dtype=int).tolist()

# Extracción de la ROI para cada keyframe y exportación de la grilla de inspección
cols = 3
rows = (len(frames_objetivo) + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(18, 5 * rows))
axs = np.array(axs).reshape(-1)

frames_extraidos = {}

for i, idx_frame in enumerate(frames_objetivo):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx_frame)
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi_ojo = frame_rgb[y0_ojo:y1_ojo, x0_ojo:x1_ojo].copy()
    tiempo_s = idx_frame / fps if fps > 0 else 0.0
    frames_extraidos[idx_frame] = (roi_ojo, tiempo_s)

    ax = axs[i]
    ax.imshow(roi_ojo)

    # Grilla de referencia espacial
    alto, ancho = roi_ojo.shape[:2]
    ax.set_xticks(np.arange(0, ancho, 25))
    ax.set_yticks(np.arange(0, alto, 25))
    ax.set_xticks(np.arange(0, ancho, 5), minor=True)
    ax.set_yticks(np.arange(0, alto, 5), minor=True)

    ax.grid(which="major", color="white", linewidth=0.6, alpha=0.7)
    ax.grid(which="minor", color="white", linewidth=0.2, alpha=0.3)

    ax.set_title(
        f"Frame {idx_frame} (t = {tiempo_s:.2f}s)",
        fontweight="bold",
        fontsize=10,
    )
    ax.tick_params(labelsize=7)

for j in range(len(frames_objetivo), len(axs)):
    axs[j].axis("off")

plt.tight_layout()
path_inspeccion_key = os.path.join(MOD_DIRS["3"], "03_inspeccion_keyframes_video.png")
plt.savefig(path_inspeccion_key, bbox_inches="tight")
plt.close()

print("\nIMAGEN 03_inspeccion_keyframes_video: muestra los frames comparativos\n")

# 2. Entrada interactiva o salteo de validación
opcion = input("¿Desea ingresar los límites para la validación Ground Truth? (si/no): ").strip().lower()

if opcion in ["no", "n"]:
    print("\nValidación omitida por el usuario. Se proseguirá únicamente con el procesamiento automático.\n")
    df_gt = None
else:
    print("\n||||||||||||||||||||||||||||||||||||||||||||")
    print("  SELECCIÓN MANUAL DE LÍMITES GROUND TRUTH")
    print("||||||||||||||||||||||||||||||||||||||||||||\n")
    print("Consulte '03_inspeccion_keyframes_video.png' para identificar las coordenadas.")
    print("Nota: Ingrese 'v' en cualquier momento para volver a la coordenada anterior.\n")

    # Construcción de la secuencia de pasos para navegación revertible
    pasos_gt = []
    for idx_frame in frames_objetivo:
        tiempo_s = frames_extraidos[idx_frame][1]
        for c_key, c_desc in [("x1", "x1"), ("x2", "x2"), ("y1", "y1"), ("y2", "y2")]:
            pasos_gt.append((idx_frame, tiempo_s, c_key, c_desc))

    valores_gt = {}
    k = 0
    while k < len(pasos_gt):
        idx_frame, tiempo_s, c_key, c_desc = pasos_gt[k]
        entrada = input(f"  Frame {idx_frame} (t = {tiempo_s:.2f}s) -> {c_desc}: ").strip().lower()

        if entrada == "v":
            if k > 0:
                k -= 1
                print("  << Volviendo a la coordenada anterior...\n")
            else:
                print("  Ya se encuentra en la primera coordenada.\n")
        else:
            try:
                valores_gt[(idx_frame, c_key)] = int(entrada)
                k += 1
            except ValueError:
                print("  Entrada inválida. Ingrese un entero o 'v' para volver.")

    anotaciones_video = {}
    for idx_frame in frames_objetivo:
        anotaciones_video[idx_frame] = (
            valores_gt[(idx_frame, "x1")],
            valores_gt[(idx_frame, "x2")],
            valores_gt[(idx_frame, "y1")],
            valores_gt[(idx_frame, "y2")],
        )

    # 3. Construcción visual y geométrica
    fig, axs = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axs = np.array(axs).reshape(-1)

    filas_gt = []

    for i, idx_frame in enumerate(frames_objetivo):
        roi_ojo, tiempo_s = frames_extraidos[idx_frame]
        roi_ojo = roi_ojo.copy()

        # Ordenamiento de límites para evitar inconsistencias de dirección
        x1, x2, y1, y2 = anotaciones_video[idx_frame]
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # Cálculo de semiejes y centroide local en coordenadas de la ROI
        a = (x_max - x_min) / 2.0  # Semieje horizontal
        b = (y_max - y_min) / 2.0  # Semieje vertical
        xc_loc = x_min + a
        yc_loc = y_min + b

        # Geometría de referencia Ground Truth (GT)
        area_gt = np.pi * a * b
        diam_gt = 2.0 * np.sqrt(a * b)

        filas_gt.append({
            "Frame": idx_frame,
            "Tiempo [s]": tiempo_s,
            "Área [px]": area_gt,
            "Diámetro [px]": diam_gt,
            "Xc_loc": xc_loc,
            "Yc_loc": yc_loc,
        })

        # Dibujo fino de elipse (verde, 1 px) y centroide (rojo, radio 3 px)
        cv2.ellipse(
            roi_ojo,
            (int(xc_loc), int(yc_loc)),
            (int(a), int(b)),
            0,
            0,
            360,
            (0, 255, 0),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            roi_ojo,
            (int(xc_loc), int(yc_loc)),
            3,
            (255, 0, 0),
            -1,
            lineType=cv2.LINE_AA,
        )

        ax = axs[i]
        ax.imshow(roi_ojo)

        # Grilla de referencia espacial
        alto, ancho = roi_ojo.shape[:2]
        ax.set_xticks(np.arange(0, ancho, 25))
        ax.set_yticks(np.arange(0, alto, 25))
        ax.set_xticks(np.arange(0, ancho, 5), minor=True)
        ax.set_yticks(np.arange(0, alto, 5), minor=True)

        ax.grid(which="major", color="white", linewidth=0.6, alpha=0.7)
        ax.grid(which="minor", color="white", linewidth=0.2, alpha=0.3)

        ax.set_title(
            f"Frame {idx_frame} (t = {tiempo_s:.2f}s)\nGT Manual: A = {area_gt:.1f} px"
            f" | D = {diam_gt:.1f} px",
            fontweight="bold",
            fontsize=10,
        )
        ax.tick_params(labelsize=7)

    # Apagar ejes vacíos en el subplot
    for j in range(len(frames_objetivo), len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    path_gt_fig = os.path.join(MOD_DIRS["3"], "04_keyframes_ground_truth.png")
    plt.savefig(path_gt_fig, bbox_inches="tight")
    plt.close()
    print("\nIMAGEN 04_keyframes_ground_truth: se ven las circunferencias ajustadas por el usuario\n")

    # 4. Exportación de la tabla Ground Truth (df_gt)
    df_gt = pd.DataFrame(filas_gt)
    path_csv_gt = os.path.join(MOD_DIRS["3"], "ground_truth_keyframes.csv")
    df_gt.to_csv(path_csv_gt, index=False)

    print("TABLA ground_truth_keyframes.csv: muestra los valores de referencia del área y el diámetro extraidos, que serán comparados con la segmentación automática\n")

    print("---------------------------------------------------")
    print("TABLA GROUND TRUTH (GT) CONSTRUIDA PARA COMPARACIÓN")
    print("---------------------------------------------------")
    print(
        df_gt[["Frame", "Tiempo [s]", "Área [px]", "Diámetro [px]"]].to_string(
            index=False
        )
    )

cap.release()
# ______________________________________________________________________________________________________________________________________________________________________________

# SEGMENTACIÓN DEL VIDEO FRAME A FRAME _________________________________________________________________________________________________________________________________________
# Definición de límites globales de ROI con holgura
Y_INICIAL, Y_FINAL = y0_ojo, y1_ojo
X_INICIAL, X_FINAL = x0_ojo, x1_ojo

# -----------------------------------------------------------------------------
# 1. Segmentación Robusta Frame por Frame
# -----------------------------------------------------------------------------
SEMILADO_SUBROI = 45 # Sub-ventana más compacta (70x70 px) para centrarse en la pupila

def segmentar_pupila_frame(frame_rgb, cx_prev=None, cy_prev=None):
    roi_rgb = frame_rgb[Y_INICIAL:Y_FINAL, X_INICIAL:X_FINAL]

    # Canal Rojo (Canal 0) + Filtro de Mediana
    canal_rojo = roi_rgb[:, :, 0]
    canal_rojo_filtrado = cv2.medianBlur(canal_rojo, 5)

    if cx_prev is None or cy_prev is None:
        cy_prev = canal_rojo.shape[0] // 2
        cx_prev = canal_rojo.shape[1] // 2

    # Sub-ROI centrada alrededor del centroide previo
    y0 = max(cy_prev - SEMILADO_SUBROI, 0)
    y1 = min(cy_prev + SEMILADO_SUBROI, canal_rojo.shape[0])
    x0 = max(cx_prev - SEMILADO_SUBROI, 0)
    x1 = min(cx_prev + SEMILADO_SUBROI, canal_rojo.shape[1])

    sub_rojo = canal_rojo_filtrado[y0:y1, x0:x1]
    c_local = (cx_prev - x0, cy_prev - y0)

    # 1. Otsu Nivel 1: Separa tejido claro (esclerótica/piel) del tejido oscuro (iris + pupila)
    u_global, _ = cv2.threshold(sub_rojo, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pixeles_oscuros = sub_rojo[sub_rojo < u_global]

    # 2. Otsu Nivel 2: Separa Pupila (muy oscura) de Iris (gris intermedio)
    if len(pixeles_oscuros) > 10:
        u_pupila, _ = cv2.threshold(pixeles_oscuros, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        u_pupila = u_global * 0.6  # Respaldo proporcional si hay pocos píxeles

    # Máscara binaria exclusiva de pupila
    b1 = (sub_rojo < u_pupila).astype(np.uint8)

    # 3. Marcadores de Watershed restringidos
    marc = np.zeros(sub_rojo.shape, dtype=np.int32)
    # Fondo (1): todo lo que sea más claro que el umbral de pupila y fuera de un margen dilatado
    marc[(sub_rojo >= u_pupila) & ~(cv2.dilate(b1, np.ones((5, 5))) > 0)] = 1
    # Núcleo de Pupila (2): máscara binarizada erosionada
    marc[cv2.erode(b1, np.ones((3, 3))) > 0] = 2

    grad = sobel(sub_rojo.astype(float) / 255.0)
    et = watershed(grad, marc)
    ws_raw = (et == 2).astype(np.uint8)

    # Posprocesamiento morfológico (Cierre + Apertura + Relleno de huecos)
    ee3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ee5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    ws_cierre = cv2.morphologyEx(ws_raw, cv2.MORPH_CLOSE, ee5)
    ws_apertura = cv2.morphologyEx(ws_cierre, cv2.MORPH_OPEN, ee3)
    ws_posproc = ndi.binary_fill_holes(ws_apertura)

    # Extracción de contornos
    contornos, _ = cv2.findContours(
        (ws_posproc.astype(np.uint8)) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    mask_pupila = np.zeros(ws_posproc.shape, dtype=np.uint8)
    xc_loc, yc_loc = c_local

    if len(contornos) > 0:
        cnt_pupila = max(contornos, key=cv2.contourArea)

        if cv2.contourArea(cnt_pupila) > 50:  # Umbral de área mínima ajustado para pupila chica
            cv2.drawContours(mask_pupila, [cnt_pupila], -1, 255, -1)

            M = cv2.moments(cnt_pupila)
            if M["m00"] != 0:
                xc_cand = M["m10"] / M["m00"]
                yc_cand = M["m01"] / M["m00"]

                # Limitar desplazamiento del centroide (máx 4 px por cuadro)
                dx = np.clip(xc_cand - c_local[0], -4.0, 4.0)
                dy = np.clip(yc_cand - c_local[1], -4.0, 4.0)

                xc_loc = c_local[0] + dx
                yc_loc = c_local[1] + dy

    area = float(np.sum(mask_pupila > 0))
    diametro = 2.0 * np.sqrt(area / np.pi) if area > 0 else 0.0

    xc_global = xc_loc + x0
    yc_global = yc_loc + y0

    return (
        area,
        diametro,
        (xc_global, yc_global),
        mask_pupila,
        (x0, x1, y0, y1),
    )

# -----------------------------------------------------------------------------
# 2. Procesamiento Completo de Video
# -----------------------------------------------------------------------------
def procesar_video_pupilometria(path_video_input):
    cap = cv2.VideoCapture(path_video_input)

    if not cap.isOpened():
        print(f"Error al abrir el video: {path_video_input}")
        return None, 0.0

    fps_v = cap.get(cv2.CAP_PROP_FPS)
    ancho_v = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    nombre_base = os.path.basename(path_video_input)
    out_path = os.path.join(MOD_DIRS["3"], f"anotado_{nombre_base}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_v, (ancho_v, alto_v))

    registros = []
    cx_p, cy_p = None, None
    n_frame = 0

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tiempo_seg = n_frame / fps_v if fps_v > 0 else 0.0

        # Segmentación frame por frame
        area, diam, (xc_roi, yc_roi), mask_sub, (x0, x1, y0, y1) = (
            segmentar_pupila_frame(frame_rgb, cx_p, cy_p)
        )

        cx_p, cy_p = int(xc_roi), int(yc_roi)

        xc_frame = int(xc_roi + X_INICIAL)
        yc_frame = int(yc_roi + Y_INICIAL)

        registros.append({
            "Frame": n_frame,
            "Tiempo [s]": tiempo_seg,
            "Área [px]": area,
            "Diámetro [px]": diam,
            "Xc_frame [px]": xc_frame,
            "Yc_frame [px]": yc_frame,
        })

        # Visualización sobre el frame exportado
        frame_vis = frame_bgr.copy()

        cnts_sub, _ = cv2.findContours(
            (mask_sub.astype(np.uint8)) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(cnts_sub) > 0:
            cnt_max = max(cnts_sub, key=cv2.contourArea)
            cnt_global = cnt_max + np.array([X_INICIAL + x0, Y_INICIAL + y0])
            cv2.drawContours(frame_vis, [cnt_global], -1, (0, 255, 128), 1, lineType=cv2.LINE_AA)

        cv2.circle(frame_vis, (xc_frame, yc_frame), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        txt_l1 = f"t = {tiempo_seg:.2f}s | D = {diam:.1f}px"
        cv2.putText(
            frame_vis,
            txt_l1,
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            6,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_vis,
            txt_l1,
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (220, 245, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame_vis)
        n_frame += 1

    cap.release()
    writer.release()

    df_resultados = pd.DataFrame(registros)
    print(
        f"\nVideo {nombre_base} procesado exitosamente ({n_frame} frames guardados).\n"
    )
    return df_resultados, fps_v


# -----------------------------------------------------------------------------
# 3. Evaluación de Error 
# -----------------------------------------------------------------------------
def evaluar_metricas_error(df_auto, df_gt):
    if df_gt is None or df_gt.empty:
        print("\nGround Truth no disponible. Se omite la evaluación de error cuantitativa.")
        return None, None

    df_merge = pd.merge(
        df_gt, df_auto, on="Frame", suffixes=("_GT", "_AUTO")
    )

    err_area = df_merge["Área [px]_AUTO"] - df_merge["Área [px]_GT"]
    mae_area = float(np.mean(np.abs(err_area)))
    rmse_area = float(np.sqrt(np.mean(err_area**2)))

    err_diam = df_merge["Diámetro [px]_AUTO"] - df_merge["Diámetro [px]_GT"]
    mae_diam = float(np.mean(np.abs(err_diam)))
    rmse_diam = float(np.sqrt(np.mean(err_diam**2)))

    print("\n---------------------------------------------")
    print("EVALUACIÓN DE ERROR AUTOMÁTICO VS MANUAL (GT)")
    print("---------------------------------------------")
    print(f"MAE  Área    : {mae_area:.2f} px")
    print(f"RMSE Área    : {rmse_area:.2f} px")
    print(f"MAE  Diámetro: {mae_diam:.2f} px")
    print(f"RMSE Diámetro: {rmse_diam:.2f} px\n")

    return df_merge, {
        "MAE_Area": mae_area,
        "RMSE_Area": rmse_area,
        "MAE_Diam": mae_diam,
        "RMSE_Diam": rmse_diam,
    }

# -----------------------------------------------------------------------------
# 4. Ejecución Principal y Filtrado
# -----------------------------------------------------------------------------
df_auto, fps_video = procesar_video_pupilometria(VIDEO_PATH)

# Exportación del CSV de resultados
path_csv_auto = os.path.join(MOD_DIRS["3"], "seguimiento_temporal_pupilometria.csv")
df_auto.to_csv(path_csv_auto, index=False)

# Filtrado de señal (Mediana móvil de 5 muestras para mitigar artefactos)
df_auto["Diámetro_Filtrado [px]"] = (
    df_auto["Diámetro [px]"]
    .rolling(window=5, center=True, min_periods=1)
    .median()
)
df_auto["Área_Filtrada [px]"] = (
    np.pi * (df_auto["Diámetro_Filtrado [px]"] / 2.0) ** 2
)

# Evaluación cuantitativa si Ground Truth existe
if "df_gt" in locals() and df_gt is not None and not df_gt.empty:
    df_evaluacion, metricas_error = evaluar_metricas_error(df_auto, df_gt)
else:
    df_evaluacion, metricas_error = None, None


# -----------------------------------------------------------------------------
# 5. Gráficos de Resultados
# -----------------------------------------------------------------------------
# Curva de Diámetro
plt.figure(figsize=(12, 5))
plt.plot(
    df_auto["Tiempo [s]"],
    df_auto["Diámetro [px]"],
    color="gray",
    alpha=0.4,
    linewidth=1.0,
    label="Medición cruda",
)
plt.plot(
    df_auto["Tiempo [s]"],
    df_auto["Diámetro_Filtrado [px]"],
    color="red",
    linewidth=2.0,
    label="Seguimiento automático (Filtrado Mediana)",
)

if "df_gt" in locals() and df_gt is not None and not df_gt.empty:
    plt.scatter(
        df_gt["Tiempo [s]"],
        df_gt["Diámetro [px]"],
        color="lime",
        edgecolors="black",
        s=70,
        zorder=5,
        label="Mediciones manuales (Ground Truth)",
    )

plt.title(
    f"Curva de Respuesta Dinámica Pupilar - {os.path.basename(VIDEO_PATH)}", fontweight="bold"
)
plt.xlabel("Tiempo [s]")
plt.ylabel("Diámetro equivalente [px]")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="lower right")
plt.tight_layout()

path_curva_din = os.path.join(MOD_DIRS["3"], "05_curva_dinamica_pupilar.png")
plt.savefig(path_curva_din, bbox_inches="tight")
plt.close()
print("\nFIGURA 05_curva_dinamica_pupilar: muestra el gráfico del diámetro en el tiempo\n")
print("\nTABLA seguimiento_temporal_pupilometria.csv: contiene los valores del diámetro y el área en el tiempo\n")

# Trayectoria Real del Centroide en el Marco del Video Completo
plt.figure(figsize=(12, 4))
plt.plot(
    df_auto["Tiempo [s]"],
    df_auto["Xc_frame [px]"],
    label="Centroide Xc (Video Global)",
    color="cyan",
)
plt.plot(
    df_auto["Tiempo [s]"],
    df_auto["Yc_frame [px]"],
    label="Centroide Yc (Video Global)",
    color="magenta",
)
plt.title(
    "Estabilidad y Trayectoria del Centroide Pupilar (Coordenadas de Escena)",
    fontweight="bold",
)
plt.xlabel("Tiempo [s]")
plt.ylabel("Coordenada Real [px]")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()

path_tray_cent = os.path.join(MOD_DIRS["3"], "06_trayectoria_centroide.png")
plt.savefig(path_tray_cent, bbox_inches="tight")
plt.close()
print("\nIMAGEN 06_trayectoria_centroide: se ve el movimiento del centroide en sus coordenadas x,y del video.\n")

# -----------------------------------------------------------------------------
# 6. Extracción y Exportación de Parámetros Fisiológicos
# -----------------------------------------------------------------------------
t_max_v = df_auto["Tiempo [s]"].max()
t_flash = 9.0 if t_max_v >= 9.0 else (t_max_v * 0.5)

# Estado Basal (Promedio antes del estímulo)
mask_basal = (df_auto["Tiempo [s]"] >= (t_flash * 0.1)) & (df_auto["Tiempo [s]"] <= t_flash)
if not mask_basal.any():
    mask_basal = df_auto["Tiempo [s]"] <= t_flash

d_basal = float(df_auto.loc[mask_basal, "Diámetro_Filtrado [px]"].mean())
a_basal = float(df_auto.loc[mask_basal, "Área_Filtrada [px]"].mean())

# Respuesta Mínima (Ventana de constricción post-estímulo)
mask_resp = (df_auto["Tiempo [s]"] >= t_flash) & (df_auto["Tiempo [s]"] <= min(t_flash + 4.0, t_max_v))
if not mask_resp.any():
    mask_resp = df_auto["Tiempo [s]"] >= t_flash

idx_min = df_auto.loc[mask_resp, "Diámetro_Filtrado [px]"].idxmin()

d_min = float(df_auto.loc[idx_min, "Diámetro_Filtrado [px]"])
a_min = float(df_auto.loc[idx_min, "Área_Filtrada [px]"])
t_min = float(df_auto.loc[idx_min, "Tiempo [s]"])

# Inicio de la constricción
delta_caida = 1.0 if d_basal < 10 else 2.0
mask_caida = (df_auto["Tiempo [s]"] >= t_flash) & (
    df_auto["Diámetro_Filtrado [px]"] < (d_basal - delta_caida)
)
t_inicio_const = (
    float(df_auto.loc[mask_caida, "Tiempo [s]"].iloc[0])
    if mask_caida.any()
    else t_flash + 0.2
)

# Cálculos dinámicos
latencia = t_inicio_const - t_flash
amplitud_diam = d_basal - d_min
amplitud_area_pct = ((a_basal - a_min) / a_basal) * 100.0 if a_basal > 0 else 0.0
dt_const = t_min - t_inicio_const
v_constriccion = (amplitud_diam / dt_const) if dt_const > 0 else 0.0

# Redilatación
mask_redil = (df_auto["Tiempo [s]"] > t_min) & (
    df_auto["Tiempo [s]"] <= min(t_min + 1.0, t_max_v)
)
if mask_redil.any():
    d_post = float(df_auto.loc[mask_redil, "Diámetro_Filtrado [px]"].iloc[-1])
    dt_redil = float(df_auto.loc[mask_redil, "Tiempo [s]"].iloc[-1] - t_min)
    v_redilatacion = ((d_post - d_min) / dt_redil) if dt_redil > 0 else 0.0
else:
    v_redilatacion = 0.0

# Construcción de la tabla resumen y exportación a CSV
datos_fisiologicos = [{
    "Diámetro Basal [px]": round(d_basal, 2),
    "Diámetro Mínimo [px]": round(d_min, 2),
    "Tiempo Latencia [s]": round(latencia, 2),
    "Amplitud Resistencia (D) [px]": round(amplitud_diam, 2),
    "Amplitud Área [%]": round(amplitud_area_pct, 1),
    "Velocidad Constricción [px/s]": round(v_constriccion, 2),
    "Velocidad Redilatación [px/s]": round(v_redilatacion, 2)
}]

df_fisiologico = pd.DataFrame(datos_fisiologicos)
path_csv_fisio = os.path.join(MOD_DIRS["3"], "parametros_fisiologicos.csv")
df_fisiologico.to_csv(path_csv_fisio, index=False)

# -----------------------------------------------------------------------------
# 6. Extracción y Exportación de Parámetros Fisiológicos (Soporta Contracción y Dilatación)
# -----------------------------------------------------------------------------
t_max_v = float(df_auto["Tiempo [s]"].max())

# 1. Solicitar por consola el momento del estímulo/flash
print("\n==================================================")
print(" EXTRACCIÓN DE PARÁMETROS FISIOLÓGICOS DINÁMICOS")
print("==================================================")
t_flash_def = round(t_max_v * 0.5, 2)
t_flash_in = input(f"Ingrese el segundo del estímulo/flash (t_flash) [Enter para {t_flash_def}s]: ").strip()

try:
    t_flash = float(t_flash_in) if t_flash_in != "" else t_flash_def
except ValueError:
    print(f"Entrada inválida. Asignando t_flash por defecto = {t_flash_def}s.")
    t_flash = t_flash_def

# 2. Estado Basal (Promedio de las mediciones previas al estímulo)
mask_basal = (df_auto["Tiempo [s]"] >= (t_flash * 0.1)) & (df_auto["Tiempo [s]"] <= t_flash)
if not mask_basal.any():
    mask_basal = df_auto["Tiempo [s]"] <= t_flash

d_basal = float(df_auto.loc[mask_basal, "Diámetro_Filtrado [px]"].mean())
a_basal = float(df_auto.loc[mask_basal, "Área_Filtrada [px]"].mean())

# 3. Ventana de Respuesta Post-Estímulo (4 segundos posteriores)
mask_resp = (df_auto["Tiempo [s]"] >= t_flash) & (df_auto["Tiempo [s]"] <= min(t_flash + 4.0, t_max_v))
if not mask_resp.any():
    mask_resp = df_auto["Tiempo [s]"] >= t_flash

df_post = df_auto.loc[mask_resp]

idx_min = df_post["Diámetro_Filtrado [px]"].idxmin()
idx_max = df_post["Diámetro_Filtrado [px]"].idxmax()

d_min = float(df_post.loc[idx_min, "Diámetro_Filtrado [px]"])
d_max = float(df_post.loc[idx_max, "Diámetro_Filtrado [px]"])

caida_max = d_basal - d_min
subida_max = d_max - d_basal

# Determinación automática del tipo de dinámica (Contracción vs Dilatación)
es_contraccion = caida_max >= subida_max

if es_contraccion:
    tipo_respuesta = "Contracción (Miosis)"
    d_extremo = d_min
    t_extremo = float(df_post.loc[idx_min, "Tiempo [s]"])
    a_extremo = float(df_post.loc[idx_min, "Área_Filtrada [px]"])
    
    # Inicio de la caída
    delta_caida = 1.0 if d_basal < 10 else 2.0
    mask_inicio = (df_auto["Tiempo [s]"] >= t_flash) & (df_auto["Diámetro_Filtrado [px]"] < (d_basal - delta_caida))
    t_inicio = float(df_auto.loc[mask_inicio, "Tiempo [s]"].iloc[0]) if mask_inicio.any() else t_flash + 0.2
    
    latencia = max(0.0, t_inicio - t_flash)
    amplitud_diam = caida_max
    amplitud_area_pct = ((a_basal - a_extremo) / a_basal) * 100.0 if a_basal > 0 else 0.0
    
    dt_const = t_extremo - t_inicio
    v_constriccion = (amplitud_diam / dt_const) if dt_const > 0 else 0.0
    
    # Redilatación posterior al mínimo
    mask_redil = (df_auto["Tiempo [s]"] > t_extremo) & (df_auto["Tiempo [s]"] <= min(t_extremo + 1.5, t_max_v))
    if mask_redil.any():
        d_post_redil = float(df_auto.loc[mask_redil, "Diámetro_Filtrado [px]"].iloc[-1])
        dt_redil = float(df_auto.loc[mask_redil, "Tiempo [s]"].iloc[-1] - t_extremo)
        v_dilatacion = ((d_post_redil - d_extremo) / dt_redil) if dt_redil > 0 else 0.0
    else:
        v_dilatacion = 0.0

else:
    tipo_respuesta = "Dilatación (Midriasis)"
    d_extremo = d_max
    t_extremo = float(df_post.loc[idx_max, "Tiempo [s]"])
    a_extremo = float(df_post.loc[idx_max, "Área_Filtrada [px]"])
    
    # Inicio del aumento de tamaño
    delta_subida = 1.0 if d_basal < 10 else 2.0
    mask_inicio = (df_auto["Tiempo [s]"] >= t_flash) & (df_auto["Diámetro_Filtrado [px]"] > (d_basal + delta_subida))
    t_inicio = float(df_auto.loc[mask_inicio, "Tiempo [s]"].iloc[0]) if mask_inicio.any() else t_flash + 0.2
    
    latencia = max(0.0, t_inicio - t_flash)
    amplitud_diam = subida_max
    amplitud_area_pct = ((a_extremo - a_basal) / a_basal) * 100.0 if a_basal > 0 else 0.0
    
    v_constriccion = 0.0
    dt_dil = t_extremo - t_inicio
    v_dilatacion = (amplitud_diam / dt_dil) if dt_dil > 0 else 0.0

# 4. Construcción del diccionario estructurado
datos_fisiologicos = {
    "Tipo de Respuesta": tipo_respuesta,
    "Instante Estímulo [s]": round(t_flash, 2),
    "Diámetro Basal [px]": round(d_basal, 2),
    "Diámetro Extremo [px]": round(d_extremo, 2),
    "Tiempo Latencia [s]": round(latencia, 2),
    "Amplitud Respuesta (D) [px]": round(amplitud_diam, 2),
    "Amplitud Área [%]": round(amplitud_area_pct, 1),
    "Velocidad Constricción [px/s]": round(v_constriccion, 2),
    "Velocidad Dilatación [px/s]": round(v_dilatacion, 2),
}

# Exportación CSV (formato dataset de 1 fila)
df_fisiologico = pd.DataFrame([datos_fisiologicos])
path_csv_fisio = os.path.join(MOD_DIRS["3"], "parametros_fisiologicos.csv")
df_fisiologico.to_csv(path_csv_fisio, index=False)

# DataFrame traspuesto (vertical) para impresión prolija en consola
df_fisio_transp = pd.DataFrame(
    list(datos_fisiologicos.items()), 
    columns=["Parámetro Fisiológico", "Valor"]
)

print("\nTABLA parametros_fisiologicos.csv: se ven los parámetros extraídos de la curva\n")
print("//////////////////////////////////////////////")
print(" PARÁMETROS FISIOLÓGICOS EXTRAÍDOS DE LA CURVA")
print("//////////////////////////////////////////////")
print(df_fisio_transp.to_string(index=False, col_space=32, justify="left"))
print(f"\nVIDEO anotado_{os.path.basename(VIDEO_PATH)}: muestra la segmentación de la pupila frame a frame y la ubicación del centroide.\n  Además se calcula el diámetro y se muestra el tiempo\n")
# _______________________________________________________________________________________________________________________________________________________________


# MÓDULO 4: COMPRESIÓN DE IMÁGENES Y EVALUACIÓN DE DEGRADACIÓN ###########################################################################################################

print_header("MÓDULO 4: COMPRESIÓN DE LA INFORMACIÓN DE LA IMAGEN")
# -----------------------------------------------------------------------------
# 1. Clases y Funciones de Compresión (Huffman, RLE, JPEG)
# -----------------------------------------------------------------------------
class NodoHuffman:
    def __init__(self, simbolo, frecuencia):
        self.simbolo = simbolo
        self.frecuencia = frecuencia
        self.izq = None
        self.der = None

    def __lt__(self, otro):
        return self.frecuencia < otro.frecuencia


def evaluar_huffman(img):
    flat = img.flatten()
    n_pixels = flat.size
    frecuencias = Counter(flat)

    heap = [NodoHuffman(sym, freq) for sym, freq in frecuencias.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        tabla_codigos = {heap[0].simbolo: "0"}
    else:
        while len(heap) > 1:
            n1, n2 = heapq.heappop(heap), heapq.heappop(heap)
            padre = NodoHuffman(None, n1.frecuencia + n2.frecuencia)
            padre.izq, padre.der = n1, n2
            heapq.heappush(heap, padre)

        raiz = heap[0]
        tabla_codigos = {}

        def _gen_cods(nodo, c=""):
            if nodo:
                if nodo.simbolo is not None:
                    tabla_codigos[nodo.simbolo] = c
                _gen_cods(nodo.izq, c + "0")
                _gen_cods(nodo.der, c + "1")

        _gen_cods(raiz)

    bits_datos = sum(len(tabla_codigos[val]) for val in flat)
    bits_overhead = len(tabla_codigos) * 16  # Overhead de la tabla de frecuencias
    bits_comprimidos = bits_datos + bits_overhead
    bytes_comprimidos = bits_comprimidos / 8.0
    bytes_originales = img.nbytes

    cr = bytes_originales / bytes_comprimidos if bytes_comprimidos > 0 else 0.0
    bpp = bits_comprimidos / n_pixels

    return {
        "Método": "Huffman",
        "Bytes Orig.": bytes_originales,
        "Bytes Comp.": int(bytes_comprimidos),
        "CR": cr,
        "bpp": bpp,
        "MSE": 0.0,
        "PSNR [dB]": float("inf"),
        "SSIM": 1.0,
    }


def evaluar_rle(img, direccion="horizontal"):
    flat = img.T.flatten() if direccion == "vertical" else img.flatten()
    if len(flat) == 0:
        return {}

    secuencia = []
    val_act, len_act = flat[0], 1
    for val in flat[1:]:
        if val == val_act and len_act < 255:
            len_act += 1
        else:
            secuencia.append((val_act, len_act))
            val_act, len_act = val, 1
    secuencia.append((val_act, len_act))

    bytes_comprimidos = len(secuencia) * 2  # 2 bytes por par (valor + longitud)
    bytes_originales = img.nbytes
    cr = bytes_originales / bytes_comprimidos if bytes_comprimidos > 0 else 0.0
    bpp = (bytes_comprimidos * 8) / flat.size

    return {
        "Método": f"RLE ({direccion.capitalize()})",
        "Bytes Orig.": bytes_originales,
        "Bytes Comp.": bytes_comprimidos,
        "CR": cr,
        "bpp": bpp,
        "MSE": 0.0,
        "PSNR [dB]": float("inf"),
        "SSIM": 1.0,
    }


def evaluar_jpeg(img, Q):
    bytes_originales = img.nbytes
    es_gris = len(img.shape) == 2 or img.shape[2] == 1

    param_encode = [int(cv2.IMWRITE_JPEG_QUALITY), Q]
    _, buffer_jpg = cv2.imencode(".jpg", img, param_encode)

    bytes_comprimidos = len(buffer_jpg)
    cr = bytes_originales / bytes_comprimidos if bytes_comprimidos > 0 else 0.0
    bpp = (bytes_comprimidos * 8) / img.size

    img_dec = (
        cv2.imdecode(buffer_jpg, cv2.IMREAD_GRAYSCALE)
        if es_gris
        else cv2.imdecode(buffer_jpg, cv2.IMREAD_COLOR)
    )

    mse = float(np.mean((img.astype(float) - img_dec.astype(float)) ** 2))
    psnr_val = (
        float("inf") if mse == 0 else 20.0 * np.log10(255.0 / np.sqrt(mse))
    )
    ssim_val = float(ssim(img, img_dec, data_range=255.0))

    return {
        "Método": f"JPEG (Q={Q})",
        "Bytes Orig.": bytes_originales,
        "Bytes Comp.": bytes_comprimidos,
        "CR": cr,
        "bpp": bpp,
        "MSE": mse,
        "PSNR [dB]": psnr_val,
        "SSIM": ssim_val,
        "Imagen Decodificada": img_dec,
    }


# -----------------------------------------------------------------------------
# 2. Carga Segura de Variables desde la Memoria del Script
# -----------------------------------------------------------------------------
if "roi_gris" in locals():
    img_roi_gris = roi_gris.copy()
else:
    # Respaldo de extracción directa si no existe en la memoria local
    cap_tmp = cv2.VideoCapture(VIDEO_PATH)
    ret_tmp, frame_tmp = cap_tmp.read()
    cap_tmp.release()
    img_roi_gris = cv2.cvtColor(
        frame_tmp[y_inicial:y_final, x_inicial:x_final], cv2.COLOR_BGR2GRAY
    )

mascara_ref = locals().get(
    "mascara_ws",
    locals().get(
        "mask_sub",
        locals().get("img_4", (img_roi_gris < 50).astype(np.uint8)),
    ),
)
img_mascara_bin = (
    (mascara_ref.astype(np.uint8)) * 255
    if mascara_ref.max() <= 1
    else mascara_ref.astype(np.uint8)
)


# -----------------------------------------------------------------------------
# 3. Generación de Tablas de Evaluación Comparativa
# -----------------------------------------------------------------------------
def construir_tabla_comparativa(img_target, es_binaria=False):
    lista_res = []

    # 1. Huffman
    lista_res.append(evaluar_huffman(img_target))

    # 2. RLE Horizontal y Vertical
    lista_res.append(evaluar_rle(img_target, "horizontal"))
    lista_res.append(evaluar_rle(img_target, "vertical"))

    # 3. JPEG
    dict_jpeg = {}
    for Q in [90, 50, 10, 5]:
        res_j = evaluar_jpeg(img_target, Q)
        dict_jpeg[Q] = res_j.pop("Imagen Decodificada")
        lista_res.append(res_j)

    df = pd.DataFrame(lista_res)
    df["CR_str"] = df["CR"].apply(lambda x: f"{x:.2f}:1")
    df["bpp_str"] = df["bpp"].apply(lambda x: f"{x:.2f}")
    df["PSNR_str"] = df["PSNR [dB]"].apply(
        lambda x: "Infinito" if np.isinf(x) else f"{x:.2f}"
    )
    df["SSIM_str"] = df["SSIM"].apply(lambda x: f"{x:.4f}")
    df["MSE_str"] = df["MSE"].apply(lambda x: f"{x:.2f}")

    return (
        df[
            [
                "Método",
                "Bytes Orig.",
                "Bytes Comp.",
                "CR_str",
                "bpp_str",
                "MSE_str",
                "PSNR_str",
                "SSIM_str",
            ]
        ].rename(
            columns={
                "CR_str": "CR (Ratio)",
                "bpp_str": "bpp",
                "MSE_str": "MSE",
                "PSNR_str": "PSNR [dB]",
                "SSIM_str": "SSIM",
            }
        ),
        dict_jpeg,
    )


df_metricas_roi, imgs_jpeg_roi = construir_tabla_comparativa(
    img_roi_gris, es_binaria=False
)
df_metricas_mask, imgs_jpeg_mask = construir_tabla_comparativa(
    img_mascara_bin, es_binaria=True
)

# Exportación a CSV
df_metricas_roi.to_csv(os.path.join(MOD_DIRS["4"], "metricas_compresion_roi.csv"), index=False)
df_metricas_mask.to_csv(os.path.join(MOD_DIRS["4"], "metricas_compresion_mascara.csv"), index=False)


print("\nTABLA metricas_compresion_roi.csv: compara los métodos de compresión sobre la ROI\n")
print("\n                      000000000000000000000000000000000000000000000000000")
print("                       COMPRESIÓN SOBRE ROI EN ESCALA DE GRISES (CONTINUA)")
print("                      000000000000000000000000000000000000000000000000000\n")
print(df_metricas_roi.to_string(index=False, col_space=12, justify="left"))

print("\nTABLA metricas_compresion_mascara.csv: compresion de la mascara pupilar")
print("\n                      /////////////////////////////////////////////////////")
print("                       COMPRESIÓN SOBRE MÁSCARA BINARIA SEGMENTADA (PUPILA)")
print("                      /////////////////////////////////////////////////////\n")
print(df_metricas_mask.to_string(index=False, col_space=12, justify="left"))

# -----------------------------------------------------------------------------
# 4. Reevaluación del Diámetro Pupilar según Nivel de Compresión JPEG
# -----------------------------------------------------------------------------
def estimar_diametro_de_imagen(img_input):
    u, binaria = cv2.threshold(
        img_input, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    proc = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, ee)
    proc = cv2.morphologyEx(proc, cv2.MORPH_OPEN, ee)
    proc = ndi.binary_fill_holes(proc > 0)

    cnts, _ = cv2.findContours(
        (proc.astype(np.uint8)) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(cnts) == 0:
        return 0.0

    cnt_max = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt_max)
    diametro = 2.0 * np.sqrt(area / np.pi)
    return diametro


diam_orig = estimar_diametro_de_imagen(img_roi_gris)
res_diametros = [
    {
        "Calidad (Q)": "Original (Sin comp.)",
        "Diámetro [px]": f"{diam_orig:.2f}",
        "Error Absoluto [px]": "0.00",
        "Diferencia [%]": "0.00%",
    }
]

for Q in [90, 50, 10, 5]:
    d_q = estimar_diametro_de_imagen(imgs_jpeg_roi[Q])
    err_abs = abs(d_q - diam_orig)
    err_pct = (err_abs / diam_orig) * 100.0 if diam_orig > 0 else 0.0
    res_diametros.append({
        "Calidad (Q)": f"JPEG Q={Q}",
        "Diámetro [px]": f"{d_q:.2f}",
        "Error Absoluto [px]": f"{err_abs:.2f}",
        "Diferencia [%]": f"{err_pct:.2f}%",
    })

df_impacto_diametro = pd.DataFrame(res_diametros)
df_impacto_diametro.to_csv(os.path.join(MOD_DIRS["4"], "impacto_compresion_diametro.csv"), index=False)

print("\nTABLA impacto_compresion_diametro.csv: muestra diferencias en el factor de calidad Q en JPEG\n")
print("              ===================================================")
print("               IMPACTO DE COMPRESIÓN JPEG EN EL DIÁMETRO ESTIMADO")
print("              ===================================================")
print(df_impacto_diametro.to_string(index=False, col_space=18, justify="left"))


# -----------------------------------------------------------------------------
# 5. Visualización de Degradación Estructural
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for idx, Q in enumerate([90, 50, 10, 5]):
    # Fila 1: ROI Grises
    axes[0, idx].imshow(imgs_jpeg_roi[Q], cmap="gray")
    axes[0, idx].set_title(
        f"ROI JPEG Q={Q}\nSSIM: {ssim(img_roi_gris, imgs_jpeg_roi[Q], data_range=255.0):.4f}",
        fontsize=10,
    )
    axes[0, idx].axis("off")

    # Fila 2: Máscara Binaria
    axes[1, idx].imshow(imgs_jpeg_mask[Q], cmap="gray")
    axes[1, idx].set_title(
        f"Máscara JPEG Q={Q}\nSSIM: {ssim(img_mascara_bin, imgs_jpeg_mask[Q], data_range=255.0):.4f}",
        fontsize=10,
    )
    axes[1, idx].axis("off")

plt.suptitle(
    "Comparación de Degradación Estructural: ROI Continua vs. Máscara Binaria",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()

path_fig_compresion = os.path.join(MOD_DIRS["4"], "01_degradacion_compresion_jpeg.png")
plt.savefig(path_fig_compresion, bbox_inches="tight")
plt.close()

print(f"\nFIGURA 01_degradacion_compresion_jpeg: se ve la compresión jpeg para distintos Q\n")

print("Llegó al final del procesamiento. Buen día! :)")













