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

El script se ejecuta de principio a fin en un solo proceso, pero es **interactivo**: se detiene por momentos para pedir datos por la terminal. En cada pausa, el script ya guardó una figura de referencia; **hay que abrirla antes de responder**, porque las coordenadas se leen de sus ejes.


```
 video ──► MÓDULO 0 ──► MÓDULO 1 ──► MÓDULO 2 ──► MÓDULO 3 ──► MÓDULO 4 ──► resultados
            │                           │            │
            ├─ [Input 1] segundo        │            ├─ [Input 5] ¿validar GT? (si/no)
            └─ [Input 2] límites ROI    │            ├─ [Input 6] límites pupila × 6 keyframes
                                        |            └─ [Input 7] t_estimulo para param. fisiológicos
                                        ├─ [Input 3] ¿validar métodos de seg? (si/no)
                                        └─ [Input 4] límites pupila × 3 frames
```

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
