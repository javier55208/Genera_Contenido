# Avance 2 - Componente de vision (primera implementacion autorizada)

**Estado:** codigo preparado, NO se ha entrenado con las 900 portadas reales ni se ha medido accuracy de la CNN.
No modifica `chatbot.py`, `chatbot_hibrido.py`, ni el NLP.

## Datos de entrada (ya generados por el downloader)

Se necesita la carpeta completa `Dataset_IA`, por ejemplo:

```
C:\Users\javie\PycharmProjects\Images-Download\Dataset_IA\
    dataset_muestra.csv
    imagenes\MANGAS\...
    imagenes\DC_COMICS\...
    imagenes\MARVEL_COMICS\...
```

Si la moviste a otra ruta, sustituye la ruta en los comandos. No hay que volver a descargar las portadas.

## Paso 1: solo comprobar y dividir (sin instalar paquetes)

Ejecutar desde la RAIZ de `Genera Contenido` en PowerShell:

```powershell
py vision/preparar_dataset.py --dataset "C:\Users\javie\PycharmProjects\Images-Download\Dataset_IA"
```

Debe mostrar 900 imagenes, 300 por clase y los conteos de train, val, test.
Usa semilla 42, comprueba SHA256 de las 900 imagenes y evita que la **misma serie** `Contenido_Producto` de una categoria aparezca en dos particiones. Por esta medida, **70/15/15 es aproximado** (no siempre 630/135/135).

Genera `vision/resultados/particion_visual.csv` y `resumen_particion.json`. No altera ni copia fotos. El manifiesto contiene rutas de los archivos, por eso queda fuera de GitHub.

Comprobar las pruebas independientes sin instalar PyTorch:

```powershell
py -m unittest discover -s vision -p "test_preparar_dataset.py" -v
```

## Paso 2: entorno aislado y entrenamiento local (cuando el Paso 1 funcione)

En la computadora **personal**, no en la corporativa. Verificar la compatibilidad de los paquetes y del entorno Python 3.13; no cambiar el Python ni el NLP existente. Ejemplo de instalacion en entorno virtual independiente:

```powershell
py -m venv .venv-cnn
.\.venv-cnn\Scripts\python.exe -m pip install --upgrade pip
.\.venv-cnn\Scripts\python.exe -m pip install torch torchvision pillow scikit-learn
.\.venv-cnn\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

Para versiones/construcciones de CPU especificas, verificar las instrucciones oficiales https://pytorch.org/get-started/locally/ .

```powershell
.\.venv-cnn\Scripts\python.exe vision/entrenar_cnn.py --dataset "C:\Users\javie\PycharmProjects\Images-Download\Dataset_IA" --epocas 5 --batch 8 --autorizar-descarga-pesos
```

**Importante:** `--autorizar-descarga-pesos` permite la primera descarga (~20.5 MB) de los pesos oficiales ImageNet de EfficientNet-B0. Sin esa bandera el script no inicia el entrenamiento. Ejecutar solo en equipo personal y si se autoriza la descarga. Usa CPU, congela las capas de caracteristicas y entrena la capa final. Usa las particiones de entrenamiento y validacion; toca el test solo al terminar para reportar 5 evidencias: accuracy, precision, recall, F1-score y matriz de confusion.

Salidas en `vision/resultados/cnn/`:
- `metricas_cnn.json` (incluye metricas por clase)
- `matriz_confusion_cnn.csv`
- `historial_entrenamiento.csv`
- `mejor_modelo_cnn.pth` (pesos locales, no subir a GitHub)

**Limitaciones:** primer prototipo con CPU; no hemos probado tiempos ni calidad del modelo real. El conjunto de imagenes proviene de portadas de productos; cualquier publicacion de imagenes requiere revisar derechos y permisos. El control por `Contenido_Producto` reduce fuga entre portadas de la misma serie, pero no detecta variantes visuales casi identicas de series diferentes.

## GitHub - excluir datos y pesos

Fusionar el contenido de `gitignore_vision.txt` con el `.gitignore` actual. No reemplazar el archivo existente.

Se recomienda versionar: los dos scripts, sus pruebas, este README y el procedimiento; no publicar imágenes, manifests con rutas privadas ni checkpoints. Revisar los metadatos antes de publicar reportes.

## Bitacora
- Fase formal: Fase 0; excepcion autorizada para preparacion de dataset e implementacion CNN.
- Dataset: 900 imagenes, tres clases; generador original preservado.
- Arquitectura propuesta e implementada en codigo: EfficientNet-B0 + transferencia, clasificador de 3 salidas.
- Evidencia pendiente: ejecucion de particion con datos reales, entrenamiento, metrica visual e integracion posterior (NO autorizada en esta implementacion).
