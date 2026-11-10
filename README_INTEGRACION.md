# Integración multimodal incremental — Avance 2 (v0.1)

## Qué hace

Agrega al chatbot híbrido NLP una opción `imagen "ruta"` para inferencia local usando el
checkpoint EfficientNet-B0 **ya entrenado**. La etiqueta CNN se traduce de
`DC_COMICS -> DC`, `MARVEL_COMICS -> MARVEL`, `MANGAS -> MANGAS`. El usuario
**confirma o corrige** antes de registrar categoría; después proporciona el precio
y confirma la solicitud **por separado**. No genera publicidad, no consulta WooCommerce,
no modifica archivos del proyecto base y no envía imágenes a servicios externos.

## Integración segura

Copiar los archivos del ZIP dentro de la raíz del proyecto `Genera Contenido`,
**sin reemplazar** `chatbot.py`, `chatbot_hibrido.py`, `nlp/`,
`vision/entrenar_cnn.py` ni `vision/resultados/`.

Estructura nueva:

- `chatbot_multimodal.py` (nuevo)
- `vision/predecir_imagen.py` (nuevo)
- `test_multimodal.py` (nuevo)
- `README_INTEGRACION.md` (este archivo)

Requisitos: `.venv` con torch, torchvision, Pillow, scikit-learn, joblib **y nltk**;
si NLTK falta en el `.venv`, instalar en el equipo personal:

```powershell
.\.venv\Scripts\python.exe -m pip install nltk
```

Debe existir un modelo **generado por ti**, no descargado desde un tercero, en:
`vision/resultados/cnn/mejor_modelo_cnn.pth`.
Debe existir tu modelo NLP entrenado en `nlp/resultados/modelo_intenciones_svm.joblib`.
Estos archivos se cargan localmente: los checkpoints no confiables pueden ser riesgosos.

## Probar (sin ejecutar modelo CNN ni descargar pesos)

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_multimodal.py
```

## Ejecutar

```powershell
.\.venv\Scripts\python.exe chatbot_multimodal.py
```

Ejemplo de diálogo:

```text
Tu: quiero generar publicidad
Bot: ¿Cual es la categoria del producto? MANGAS, DC, MARVEL u OTROS.
Tu: imagen "C:\Users\javie\Pictures\portada.jpg"
Bot: La CNN sugiere la categoria DC (etiqueta visual: DC_COMICS). Es una estimacion ... ¿Es correcta? Responde si o no.
Tu: no
Bot: Indica la categoria correcta: MANGAS, DC, MARVEL u OTROS.
Tu: MARVEL
Bot: ¿Cual es el precio? Escribe por ejemplo: precio $199.
Tu: precio $199
Bot: Categoria: MARVEL. Precio: $199.00 MXN. ¿Confirmas los datos?
Tu: si
Bot: Solicitud confirmada: MARVEL, $199.00 MXN. En este avance no se genera la imagen publicitaria.
```

La ruta es solo un ejemplo; utiliza una imagen real del equipo personal.

## Limitaciones

- Prototipo académico por consola: entrada de imagen mediante ruta local, no adjunto gráfico.
- CNN accuracy prueba 60% sobre 135 imágenes según ejecución reportada por el estudiante;
  no fiarse de sus propuestas sin confirmación.
- No modifica ni corrige el validador previo de precios negativos.
- Pruebas automatizadas simulan CNN/NLP. Ejecutar además una conversación real con el checkpoint
  y conservar evidencia de pantalla para el informe. No se ha ejecutado el modelo real aquí.
- Evitar `git add .` sin revisar archivos. Excluir `/Dataset_IA/`, `/.venv/`, `*.pth`, `*.pt`,
  `*.joblib`, y los CSV de partición con rutas locales.
- No se deben publicar portadas, pesos, rutas privadas o datos corporativos.
