# Avance 2 — Clasificador NLP de intenciones (TF-IDF + SVM)

## Alcance de esta entrega

Módulo **independiente** que preserva el chatbot de reglas del Avance 1. Implementa las cuatro técnicas exigidas (tokenización, normalización, eliminación de stopwords, *stemming*) usando NLTK, y un clasificador de intenciones con TF-IDF y SVM lineal. No genera anuncios, no descarga portadas, no entrena CNN y **todavía no se integra al chatbot**.

El conjunto inicial contiene **300 mensajes de ejemplo redactados para fines académicos**, 30 por cada una de las diez intenciones del Avance 1. **No son conversaciones de clientes ni evidencias de uso real**. El estudiante debe revisar su pertinencia antes de presentar el modelo como solución operativa. Los ejemplos están agrupados en 100 familias de paráfrasis (tres variaciones por familia).

## Archivos

- `datos/ejemplos_intenciones.csv`: dataset textual (`id`, `texto`, `intencion`, `grupo`, `origen`, `estado_contextual`).
- `preprocesamiento.py`: cuatro pasos NLP auditable y explicables.
- `entrenar_nlp.py`: divide por familia, entrena TF-IDF + SVM y evalúa.
- `predecir_nlp.py`: prueba el modelo sobre un mensaje sin ejecutar acciones.
- `test_nlp.py`: pruebas reproducibles.
- `resultados/`: modelo entrenado, manifiesto de partición, predicciones, métricas e informes.

## Uso local (Python 3.10+)

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_nlp.py
python entrenar_nlp.py --semilla 42
python predecir_nlp.py "Quiero preparar un anuncio de Batman"
```

Para utilizar otro CSV con las mismas columnas:

```bash
python entrenar_nlp.py --csv datos/ejemplos_intenciones.csv --salida resultados --semilla 42
```

No requiere descargar recursos NLTK: `wordpunct_tokenize` y `SnowballStemmer('spanish')` funcionan localmente.

## Separación para evaluación

Se reservan **dos de las diez familias** de paráfrasis de cada intención para prueba: 240 ejemplos de entrenamiento y 60 ejemplos de prueba, con semilla reproducible. Las tres frases de la misma familia nunca se reparten entre entrenamiento y prueba. Esto mitiga la fuga entre paráfrasis próximas, aunque un conjunto sintético y pequeño sigue sin representar conversaciones reales.

Los resultados de `metricas_nlp.json` son **métricas reales de la ejecución sobre ejemplos sintéticos**, no resultados de clientes, ni garantías de desempeño futuro. La matriz de confusión está en `matriz_confusion.csv` y las fallas concretas en `predicciones_prueba.csv`.

## Límites y seguridad del diseño

1. Las intenciones `confirmar_categoria` / `corregir_categoria` dependen del estado conversacional. Un simple «sí» fuera del estado de confirmación **no debe confirmar nada**. La integración deberá conservar ese control por reglas.
2. La predicción no equivale a autorización para ejecutar operaciones. Precios y categorías deben seguir validándose por reglas.
3. Los mensajes fuera de dominio no están modelados como clase independiente; introducir rechazo/abstención y pruebas adversarias en etapas futuras.
4. `joblib` utiliza pickle: cargar **únicamente** el modelo generado localmente y nunca artefactos arbitrarios externos.
5. La eliminación de stopwords utiliza una **lista conservadora** que preserva `no` y `si`, porque son señales útiles para confirmaciones y correcciones.
6. El código fuente de Avance 1, el generador WooCommerce y las imágenes no se modificaron.
7. Los 10 tipos de intenciones se tomaron del prototipo de reglas del Avance 1; no se incorporaron automatismos para el precio o imágenes.

## Evidencias para documento final (todavía pendiente)

- Captura de los cuatro pasos para una frase (`resultados/ejemplo_preprocesamiento.json`).
- Tabla de frecuencias por intención y partición (`resultados/metricas_nlp.json`).
- Resultados reales de SVM (`resultados/reporte_clasificacion.csv`, `resultados/predicciones_prueba.csv`).
- Matriz de confusión (`resultados/matriz_confusion.csv`).
- Análisis académico de errores y limitaciones del dataset sintético.

**Control de fase:** implementación puntual autorizada del componente NLP del Avance 2; no implica autorización de CNN, despliegue ni modificación del chatbot original.
