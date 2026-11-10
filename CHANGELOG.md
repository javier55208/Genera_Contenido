# Historial de cambios

Proyecto educativo: Asistente de publicidad (Maestría en IA).

## v2 — Avance 2

Versión con los componentes del Avance 2 sobre la base del Avance 1.

### Agregado
- **NLP** (`nlp/`): clasificador de intenciones con TF-IDF + SVM lineal, con tokenización, normalización, eliminación de stopwords y *stemming* (NLTK). Incluye entrenamiento, predicción y pruebas.
- **Visión** (`vision/`): preparación del dataset de portadas, entrenamiento de CNN, predicción de imágenes, partición de datos y pruebas. Estado y limitaciones en [vision/README.md](vision/README.md).
- **Chatbot híbrido** (`chatbot_hibrido.py`): integra el clasificador NLP al chatbot de reglas.
- **Chatbot multimodal** (`chatbot_multimodal.py`): opción `imagen "ruta"` para inferencia local, con correcciones de reinicio de conversación (v0.2) y de retención de categoría y precio (v0.3). Ver [LEEME_CORRECCION.md](LEEME_CORRECCION.md), [LEEME_CORRECCION_v03.md](LEEME_CORRECCION_v03.md) e [README_INTEGRACION.md](README_INTEGRACION.md).
- **Pruebas** controladas (`pruebas_controladas_avance2.py`, `test_*.py`) y sus evidencias en `evidencias_pruebas/`.

### Cambiado
- `README.md` actualizado con el estado de las funcionalidades del Avance 2.
- `requirements.txt` ampliado con las dependencias de NLP.
- `.gitignore` ampliado para excluir entornos virtuales, dataset local, modelos entrenados, cachés y copias de archivos.

### Sin cambios
- El chatbot de reglas (`chatbot.py`) sigue funcionando de forma independiente.

## v1 — Avance 1

Primera versión del proyecto.

### Incluye
- `chatbot.py`: asistente de publicidad basado en reglas.
- `requirements.txt` con las dependencias iniciales.
- `README.md` con la descripción del proyecto.
