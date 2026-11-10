# Resultado tecnico preliminar - NLP del Avance 2

**Ejecucion real sobre el dataset de ejemplos sinteticos** (no conversaciones reales).

- **Muestras:** 300 mensajes, 10 intenciones, 30 por clase.
- **Entrenamiento:** 240. **Prueba:** 60.
- **Separacion:** familias completas de parafrasis y semilla 42.
- **Modelo:** TF-IDF (unigramas + bigramas) y SVM lineal.
- **Accuracy:** 0.6000 (36/60 casos correctos).
- **F1 macro:** 0.5784.
- **Errores observados:** 24 de 60 ejemplos de prueba.

## Limitaciones observadas

Los mayores problemas de recall afectan las intenciones `indicar_categoria`, `modificar_precio` y `cancelar` sobre las familias de ejemplos de prueba. La separacion por familias evita compartir formulaciones excesivamente parecidas, pero aumenta la dificultad del ejercicio con un corpus pequeno.

No usar este resultado para ejecutar acciones automaticamente. `confirmar_categoria` solo tiene sentido cuando el dialogo se encuentra en estado de confirmacion. Valores numericos y categorias continuan requiriendo validacion determinista.

## Siguientes decisiones, no implementadas

1. Validar manualmente el dataset sintetico y ampliar ejemplos de lenguaje natural con supervision del estudiante.
2. Definir como integrar SVM con el estado y los validadores del chatbot original, sin sustituirlos.
3. Conservar una prueba independiente al mejorar el modelo; no optimizar iterativamente contra el conjunto de prueba de esta version.
4. Preparar la CNN y la integracion multimodal solo despues de autorizacion especifica.

**Bitacora:** desarrollo del modulo NLP autorizado el 5 de noviembre de 2026; fuentes y chatbot base preservados.
