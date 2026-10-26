# Pipeline de análisis de dinámica pupilar — Grupo 7

Script en Python que procesa un video del ojo para segmentar la pupila, extraer descriptores morfométricos y de textura, y obtener la curva de respuesta pupilar D(t) junto con parámetros fisiológicos (diámetro basal, diámetro mínimo, latencia, amplitud y velocidades de constricción y redilatación).

> El notebook `Guía práctica de procesamiento de Imágenes_Grupo 7.ipynb` resuelve la guía completa paso a paso. Ahí se explica la justificación de cada método, la elección de parámetros y la interpretación de cada figura y tabla que genera este script. Ante cualquier duda sobre por qué se hace algo, consultar primero la sección correspondiente del notebook (la numeración de módulos y secciones es la misma).

---

## Índice

1. [Requisitos e instalación](#1-requisitos-e-instalación)
2. [Estructura de carpetas](#2-estructura-de-carpetas)
3. [Manejo de rutas (path)](#3-manejo-de-rutas-path)
4. [Workflow del pipeline](#4-workflow-del-pipeline)
5. [Entradas por terminal (inputs)](#5-entradas-por-terminal-inputs)
6. [Resultados generados](#6-resultados-generados)
7. [Instante del estímulo (flash)](#7-instante-del-estímulo-flash).
---

## 1. Requisitos e instalación

- Python 3.9 o superior.
- Dependencias listadas en `requirements.txt`.

Se recomienda usar un entorno virtual:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

## 2. Estructura de carpetas

El video **debe estar dentro de la carpeta `videos/`**, ubicada en la misma carpeta que el script:

```
proyecto/
├── pipeline.py                                         # script principal
├── Guía práctica de procesamiento de Imágenes_Grupo 7.ipynb   # resolución de la guía
├── requirements.txt
├── README.md
├── Videos/
|   └── video_dilatacion_1.mp4
│   └── video_dilatacion_2.mp4                          # video de entrada
|   └── video_contraccion_1.mp4
|   └── video_contraccion_2.mp4
└── resultados_pipeline/                                # se crea automáticamente
    ├── Modulo_0/
    ├── Modulo_1/
    ├── Modulo_2/
    ├── Modulo_3/
    └── Modulo_4/
```

---

## 3. Manejo de rutas (path)

El script no usa rutas absolutas fijas. Al inicio se define:

```python
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "video_dilatacion_2.mp4")  # MODIFICAR CON EL NOMBRE DEL VIDEO
OUTPUT_DIR = os.path.join(BASE_DIR, "resultados_pipeline")               # CARPETA CON RESULTADOS
lado_pixel = 0.7  # mm, cambiar según corresponda
```

- **`BASE_DIR`**: carpeta donde está guardado el script, independientemente de desde dónde se lo ejecute.
- **`VIDEO_PATH`**: busca el video **dentro de la subcarpeta `videos/` de esa carpeta**, con el nombre indicado. Por ejemplo, si el script está en `C:\Users\usuario\TP\pipeline.py`, el video se busca en `C:\Users\usuario\TP\videos\video_dilatacion_2.mp4`.
- **`OUTPUT_DIR`**: carpeta donde se guardan todas las figuras, CSV y el video anotado. Se crea sola si no existe, junto con una subcarpeta por módulo.

Si el video no se encuentra, el script se detiene con un `FileNotFoundError` que muestra la ruta buscada. En ese caso, verificar que el archivo esté dentro de `videos/`, que esa carpeta esté junto al script, y que el nombre (incluida la extensión) coincida exactamente.


Si se prefiere tener el video en otra carpeta, se puede escribir la ruta completa:

```python
VIDEO_PATH = r"C:\Users\usuario\Videos\mi_video.mp4"
```

---

## 4. Workflow del pipeline

```bash
python pipeline.py
```

El script se ejecuta de principio a fin en un solo proceso, pero es **interactivo**: se detiene en cuatro momentos para pedir datos por la terminal. En cada pausa, el script ya guardó una figura de referencia; **hay que abrirla antes de responder**, porque las coordenadas se leen de sus ejes.

### Diagrama general

```
 video ──► MÓDULO 0 ──► MÓDULO 1 ──► MÓDULO 2 ──► MÓDULO 3 ──► resultados
            │                           │            │
            ├─ [Input 1] segundo        │            ├─ [Input 5] ¿validar GT? (si/no)
            └─ [Input 2] límites ROI    │            ├─ [Input 6] límites pupila × 6 keyframes
                                        |            └─ [Input 7] t_estimulo para param. fisiológicos
                                        ├─ [Input 3] ¿validar métodos de seg? (si/no)
                                        └─ [Input 4] límites pupila × 3 frames
```

### Módulo 0 — Inspección inicial y ROI

1. Abre el video y muestra por terminal FPS, resolución, cantidad de frames y duración.
2. **[Input 1]** Pide el segundo del frame representativo y lo guarda con ejes en píxeles (`frame_representativo.png`).
3. **[Input 2]** Pide los límites de la ROI del ojo, leídos sobre esa figura.
4. Recorta la ROI en color y en escala de grises (`roi_seleccionada.png`).

Todo lo que sigue trabaja sobre esta ROI, así que conviene que encierre el ojo completo con algo de margen.

### Módulo 1 — Fundamentos, espacios de color y mejoramiento

No requiere entradas. Sobre la ROI del frame representativo:

1. Submuestreo espacial (100 %, 50 %, 25 %, 5 %) y cuantización.
2. Separación en canales RGB y HSV.
3. Brillo, contraste e histogramas; escalado lineal, ecualización global y CLAHE.
4. Ruido gaussiano, sal y pimienta y speckle, con filtros de media, mediana y gaussiano.
5. Detección de bordes (Roberts, Prewitt, Sobel, Laplaciano, Canny).
6. Transformada de Fourier 2D: magnitud, fase y reconstrucciones cruzadas con el frame en t + 0.5 s.
7. Densidad espectral de potencia (2D y perfil central 1D).
8. Filtros pasa-bajos y pasa-altos en frecuencia, comparados con su equivalente espacial. Imprime PSNR y SSIM por terminal.

### Módulo 2 — Segmentación y morfología

1. Localiza automáticamente el centro aproximado de la pupila (mínimo de intensidad tras un suavizado gaussiano fuerte) y recorta una sub-ROI alrededor.
2. Segmenta con cuatro métodos: **Otsu** (global y jerárquico), **K-Means** (espacio L\*a\*b\*), **Snake** (contornos activos) y **Watershed** (con marcadores).
3. Compara los métodos y aplica morfología matemática (erosión, dilatación, apertura, cierre y la combinación propuesta) sobre la mejor máscara.
4. Esqueletización (eje medio y contorno pupilar).
5. **Validación multiframe:** extrae los frames en t + 0.5 s, t + 1.0 s y t + 1.5 s y guarda una vista ampliada con grilla (`09_inspeccion_anotacion_manual.png`).
6. **[Input 3]** Pregunta si se quiere hacer la validación métricas manuales.
7. **[Input 4]** Pide los límites de la pupila en esos 3 frames.
8. Construye una elipse de referencia con cada anotación y calcula Dice, Jaccard y error de área para cada método. Guarda el CSV y la superposición de contornos.

### Módulo 3 — Características, textura y dinámica temporal

**3.1 Frame segmentado (sin entradas)**

1. Posprocesamiento morfológico de la máscara de Watershed.
2. Descriptores morfométricos: área, perímetro, centroide, diámetro equivalente, circularidad, excentricidad, etc.
3. Textura de 1er orden (media, varianza, asimetría, curtosis, entropía por canal RGB).
4. Textura de 2do orden (GLCM / Haralick por canal RGB).

**3.2 Video completo**

1. Selecciona automáticamente **6 keyframes equiespaciados** entre 0.5 s y el final del video menos 0.5 s, y los guarda con grilla (`03_inspeccion_keyframes_video.png`).
2. **[Input 5]** Pregunta si se quiere hacer la validación con ground truth manual.
3. **[Input 6]** Si la respuesta es sí, pide los límites de la pupila en los 6 keyframes y construye la tabla de ground truth.
4. Segmenta la pupila **frame por frame** en todo el video, usando el centro del frame anterior para seguirla, y genera un video anotado.
5. Suaviza el diámetro con una mediana móvil de 5 muestras.
6. Si hay ground truth, compara automático vs. manual (MAE y RMSE de área y diámetro).
7. Grafica la curva D(t) y la trayectoria del centroide.
8. Extrae los parámetros fisiológicos característicos (ver Sección 7).
---

## 5. Entradas por terminal (inputs)

Todas las coordenadas se ingresan como **enteros en píxeles**. El eje Y crece **hacia abajo** (y = 0 es el borde superior de la imagen), igual que en las figuras.

### Input 1 — Segundo del frame representativo (Módulo 0)

```
Ingrese el segundo del video del cual quiere extraer el frame representativo:
```

| Qué ingresar | Formato | Ejemplo |
|---|---|---|
| Instante del video, en segundos | número decimal | `3.5` |

Elegir un instante con el ojo bien abierto, sin parpadeo y **antes del estímulo**, con la pupila bien visible. Debe ser menor que la duración del video, que se imprime justo antes. Este instante también define los frames de validación del Módulo 2 (t + 0.5, t + 1.0 y t + 1.5 s), así que **debe quedar al menos 1.5 s de video después**.

### Input 2 — Límites de la ROI del ojo (Módulo 0)

Figura de referencia: `Modulo_0/frame_representativo.png` (frame **completo**).

```
Ingrese X inferior (x_inicial):
Ingrese X superior (x_final):
Ingrese Y inferior (y_inicial [arriba]):
Ingrese Y superior (y_final):
```

| Variable | Significado |
|---|---|
| `x_inicial` | borde izquierdo de la ROI |
| `x_final` | borde derecho de la ROI |
| `y_inicial` | borde **superior** de la ROI |
| `y_final` | borde **inferior** de la ROI |

Debe cumplirse `x_inicial < x_final` e `y_inicial < y_final`. La ROI tiene que contener el ojo completo (pupila, iris y parte de la esclerótica) con un margen que permita que la pupila se mueva durante el video sin salirse.

### Input 3 — Límites de la pupila en 3 frames de validación (Módulo 2)

Figura de referencia: `Modulo_2/09_inspeccion_anotacion_manual.png` (vista ampliada alrededor de la pupila, con grilla cada 5 px y subgrilla cada 1 px).

Se repite para cada frame (t + 0.5 s, t + 1.0 s, t + 1.5 s):

```
Ingrese los límites del disco pupilar para el frame t + 0.5s:
  x_izq (límite izquierdo [px]):
  x_der (límite derecho [px]):
  y_sup (límite superior [px]):
  y_inf (límite inferior [px]):
```

| Variable | Significado |
|---|---|
| `x_izq` / `x_der` | bordes izquierdo y derecho del disco **pupilar** |
| `y_sup` / `y_inf` | bordes superior e inferior del disco pupilar |

- Son límites de la **pupila**, no del iris ni de la ROI.
- Las coordenadas son **relativas a la ROI** del Input 2. Se leen directamente de los ejes de la figura 09, que ya están en ese sistema.
- Con estos cuatro valores el script construye una elipse de referencia y la compara con cada método de segmentación.
- Si se ingresa algo que no es un entero, el script asigna valores por defecto (`98, 122, 57, 83`) que solo sirven para el video original. Con otro video, esto invalida la validación.

### Input 4 — ¿Validación con ground truth? (Módulo 3)

Figura de referencia: `Modulo_3/03_inspeccion_keyframes_video.png`.

```
¿Desea ingresar los límites para la validación Ground Truth? (si/no):
```

| Respuesta | Efecto |
|---|---|
| `no` o `n` | Se saltea el ground truth. El video se procesa igual, pero no se calculan MAE/RMSE, no se generan `04_keyframes_ground_truth.png` ni `ground_truth_keyframes.csv` y la curva D(t) no muestra los puntos manuales. |
| cualquier otra cosa (`si`) | Se piden los límites de la pupila en los 6 keyframes (Input 5). |

### Input 5 — Límites de la pupila en 6 keyframes (Módulo 3)

Figura de referencia: `Modulo_3/03_inspeccion_keyframes_video.png` (grilla mayor cada 25 px, menor cada 5 px).

Se repite para cada uno de los 6 keyframes; el número de frame y el tiempo se muestran en el título de cada panel:

```
Frame 14 (t = 0.47s):
  x1:
  x2:
  y1:
  y2:
```

| Variable | Significado |
|---|---|
| `x1` / `x2` | bordes izquierdo y derecho de la **pupila** |
| `y1` / `y2` | bordes superior e inferior de la **pupila** |

- Son límites de la **pupila**, no de la ROI.
- El sistema de coordenadas es el de los paneles de la figura 03: la ROI del Input 2 **ampliada 5 px por lado**. Por eso los valores no coinciden exactamente con los del Input 3, aunque sea la misma pupila. Siempre leerlos de los ejes de la figura 03.
- Una entrada inválida asigna un recuadro por defecto (30 %–70 % del panel), que no representa la pupila real.

---

## 6. Resultados generados

Todo queda en `resultados_pipeline/`, en la subcarpeta del módulo correspondiente.

### Módulo 0
| Archivo | Contenido |
|---|---|
| `frame_representativo.png` | Frame completo con ejes, para elegir la ROI |
| `roi_seleccionada.png` | ROI en color y en grises |

### Módulo 1
| Archivo | Contenido |
|---|---|
| `00_submuestreo_espacial.png` | ROI a distintas resoluciones |
| `01_cuantizacion.png` | Cuantización en color y en grises |
| `02_espacios_de_color.png` | Canales RGB y HSV |
| `03_histogramas_brillo_contraste.png` | Brillo, contraste e histogramas |
| `04_escalado_y_ecualizacion.png` | Escalado lineal, ecualización global y CLAHE |
| `05_ruido_y_filtrado_espacial.png` | Ruidos y filtros espaciales |
| `06_histogramas_ruido.png` | Histogramas de las imágenes ruidosas |
| `07_deteccion_de_bordes.png` | Detectores de bordes |
| `08_fourier_magnitud_fase.png` | Magnitud y fase de la TFD-2D |
| `09_fourier_reconstruccion.png` | Reconstrucciones con solo magnitud, solo fase y cruzadas |
| `10_psd_2d_comparacion.png` | PSD 2D original vs. filtrada |
| `11_psd_1d_perfil_central.png` | Perfil central de la PSD |
| `12_filtrado_frecuencial_pasabajos_pasaaltos.png` | Filtrado en frecuencia vs. espacial |

### Módulo 2
| Archivo | Contenido |
|---|---|
| `01_sub_roi_estimacion_centro.png` | Sub-ROI y centro estimado |
| `02_segmentacion_otsu.png` | Otsu global y jerárquico |
| `03_segmentacion_kmeans.png` | K-Means para distintos K |
| `04_segmentacion_snake.png` | Contorno activo |
| `05_segmentacion_watershed.png` | Watershed con marcadores |
| `06_comparativa_metodos_segmentacion.png` | Comparación de los cuatro métodos |
| `07_operaciones_morfologicas.png` | Erosión, dilatación, apertura, cierre y combinada |
| `08_esqueletizacion.png` | Eje medio y contorno pupilar |
| `09_inspeccion_anotacion_manual.png` | Frames de validación (para el Input 3) |
| CSV de validación | Dice, Jaccard y error de área por método y frame |
| Figura de superposición | Contornos de cada método sobre los frames de validación |

### Módulo 3
| Archivo | Contenido |
|---|---|
| `01_descriptores_morfometricos_frame.png` | Contorno, centroide y descriptores |
| `descriptores_textura_1er_orden.csv` | Textura de 1er orden por canal |
| `02_matrices_glcm_rgb.png` | Matrices GLCM |
| `descriptores_textura_2do_orden_glcm.csv` | Descriptores de Haralick |
| `03_inspeccion_keyframes_video.png` | Keyframes (para el Input 5) |
| `04_keyframes_ground_truth.png` | Anotaciones manuales *(solo si se validó)* |
| `ground_truth_keyframes.csv` | Área y diámetro manuales *(solo si se validó)* |
| `anotado_<nombre_del_video>.mp4` | Video con la pupila segmentada y el centroide |
| `seguimiento_temporal_pupilometria.csv` | Área, diámetro y centroide frame a frame |
| `05_curva_dinamica_pupilar.png` | Curva D(t) cruda, filtrada y ground truth |
| `06_trayectoria_centroide.png` | Movimiento del centroide |
| `parametros_fisiologicos.csv` | Parámetros fisiológicos de la respuesta |

### Por terminal
- Datos del video (FPS, resolución, duración).
- PSNR y SSIM del filtrado (Módulo 1).
- Métricas de validación de la segmentación (Módulo 2).
- Descriptores morfométricos y de textura (Módulo 3).
- MAE y RMSE automático vs. manual, si se validó.
- Tabla de parámetros fisiológicos.

Cada ejecución **sobrescribe** los resultados anteriores. Para conservarlos, renombrar o mover `resultados_pipeline/` antes de volver a correr el script.

## 7. Instante del estímulo (flash)

El instante en que se aplica el flash **no se pide por terminal**: está fijo en el código del Módulo 3 (sección 3.2, "Extracción y Exportación de Parámetros Fisiológicos"):

```python
t_flash = 9.0 if t_max_v >= 9.0 else (t_max_v * 0.5)
```

Es decir, el script asume que el estímulo ocurre en **t = 9 s**, o en la mitad del video si este dura menos de 9 s. Ese valor corresponde al video original.

### Para qué se usa

Todos los parámetros fisiológicos se calculan a partir de `t_flash`:

| Parámetro | Cómo depende de `t_flash` |
|---|---|
| Diámetro basal | Promedio del diámetro entre `0.1 · t_flash` y `t_flash` |
| Diámetro mínimo | Mínimo en la ventana `[t_flash, t_flash + 4 s]` |
| Latencia | Tiempo entre `t_flash` y el inicio de la constricción |
| Amplitud y velocidad de constricción | Se calculan a partir del basal y del mínimo |
| Velocidad de redilatación | Se calcula en el segundo posterior al mínimo |

Si `t_flash` no coincide con el estímulo real, la ventana basal incluye parte de la respuesta (o la ventana de respuesta empieza antes del flash), y **todos los parámetros fisiológicos quedan mal**, aunque la segmentación y la curva D(t) estén bien.

### Qué hay que cambiar

1. Identificar en qué segundo se produce el flash en el video nuevo. Se puede ver reproduciendo el video, o corriendo el script una vez y mirando en `05_curva_dinamica_pupilar.png` dónde empieza la caída del diámetro.
2. Reemplazar `9.0` por ese instante **en las dos apariciones** de la línea `t_flash = ...` (el bloque de parámetros fisiológicos está repetido en el script). Por ejemplo, para un flash en t = 6.5 s:

   ```python
   t_flash = 6.5
   ```

3. Volver a ejecutar el script.

