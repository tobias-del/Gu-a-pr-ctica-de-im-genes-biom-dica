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

> **Nota importante:** Prestar especial atención a los mensajes emitidos por la terminal durante la ejecución. En cada paso se explican en detalle los resultados procesados y se numera e identifica formalmente cada **IMAGEN**, **FIGURA** y **TABLA** exportada. Además, durante el ingreso de coordenadas de pupila en los Módulos 2 y 3, puede presionar la tecla **`v`** en cualquier momento para retroceder y corregir un valor ingresado por error.

El pipeline requiere un total de **7 entradas** organizadas en el siguiente flujo:

## Módulo 0 — Selección de escena y ROI
[Input 1] Segundo del frame representativo: Instante en segundos (ej: 3.5). Elegir un momento previo al estímulo con la pupila bien visible.

[Input 2] Límites de la ROI del ojo (x_inicial, x_final, y_inicial, y_final): Coordenadas enteras [px] obtenidas de Modulo_0/frame_representativo.png para encuadrar la región del ojo.

## Módulo 2 — Validación cuantitativa de métodos
[Input 3] Confirmación de validación (si/no): Activa o saltea la generación de la tabla de métricas (Dice, IoU, Error de área).

[Input 4] Límites de pupila en 3 frames (x_izq, x_der, y_sup, y_inf): (Si Input 3 = si) Coordenadas [px] leídas de Modulo_2/09_inspeccion_anotacion_manual.png para definir la referencia manual en $t+0.5\text{s}$, $t+1.0\text{s}$ y $t+1.5\text{s}$.

## Módulo 3 — Seguimiento temporal y parámetros fisiológicos
[Input 5] Confirmación Ground Truth (si/no): Habilita la evaluación de error cuantitativa (MAE/RMSE) del video automático frente a referencias manuales.

[Input 6] Límites de pupila en 6 keyframes (x1, x2, y1, y2): (Si Input 5 = si) Coordenadas [px] leídas de Modulo_3/03_inspeccion_keyframes_video.png para marcar la pupila en los cuadros seleccionados.

[Input 7] Instante del estímulo (t_flash): Segundo del evento luminoso/estímulo (ej: 9.0 o 10.0, presionar Enter para la mitad del tiempo total). Base temporal para calcular estado basal, latencia y velocidades fisiológicas de contracción o dilatación.