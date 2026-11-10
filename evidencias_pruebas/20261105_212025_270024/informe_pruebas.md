# Evidencias de pruebas controladas — Avance 2

Fecha: 2026-11-05T21:20:26
Python: 3.12.0
Modos: reglas, hibrido

## Resultado de pruebas funcionales

Escenarios funcionales que cumplen: 10/12
Escenarios diagnósticos u observacionales: 6

| Modo | Escenario | Tipo | Pasos correctos | Cumple |
|---|---|---|---|---|
| reglas | E01 — Flujo completo DC | funcional | 5/5 | Sí |
| reglas | E02 — Flujo MARVEL con confirmacion | funcional | 4/4 | Sí |
| reglas | E03 — Correccion de categoria | funcional | 6/6 | Sí |
| reglas | E04 — Ayuda y despedida | observacion | 1/2 | No |
| reglas | E05 — Cancelacion y reinicio | funcional | 4/4 | Sí |
| reglas | E06 — Precio cero rechazado | funcional | 3/3 | Sí |
| reglas | E07 — Precio negativo rechazado | funcional | 2/3 | No |
| reglas | E08 — Editorial fuera de categorias | diagnostico | 2/2 | Sí |
| reglas | E09 — Saludo y solicitud con errata | diagnostico | 1/2 | No |
| hibrido | E01 — Flujo completo DC | funcional | 5/5 | Sí |
| hibrido | E02 — Flujo MARVEL con confirmacion | funcional | 4/4 | Sí |
| hibrido | E03 — Correccion de categoria | funcional | 6/6 | Sí |
| hibrido | E04 — Ayuda y despedida | observacion | 2/2 | Sí |
| hibrido | E05 — Cancelacion y reinicio | funcional | 4/4 | Sí |
| hibrido | E06 — Precio cero rechazado | funcional | 3/3 | Sí |
| hibrido | E07 — Precio negativo rechazado | funcional | 2/3 | No |
| hibrido | E08 — Editorial fuera de categorias | diagnostico | 1/2 | No |
| hibrido | E09 — Saludo y solicitud con errata | diagnostico | 2/2 | Sí |

## Sondeos exploratorios del modelo SVM

Coincidencias observadas: 6/8. NO interpretar como accuracy independiente.

## Interpretación y límites

- La ejecución usa mensajes prefijados y verifica respuesta y estado del diálogo.
- Se crea un chatbot nuevo en cada escenario para evitar contaminación del estado.
- Las pruebas diagnósticas no bloquean por sí solas el prototipo; se documentan como mejoras.
- Las reglas tienen prioridad; la columna `prediccion_svm_fallback` muestra dónde se consultó el modelo.
- No se entrena el modelo durante estas pruebas, ni se comprueba la CNN o la integración texto-imagen.
- Las métricas estadísticas de SVM se deben tomar del conjunto test usado en `entrenar_nlp.py`.
- El contenido depende del modelo instalado y de la versión de sus bibliotecas.

## Archivos adjuntos

- `conversaciones_terminal.txt`: diálogos completos tal como se muestran en la terminal.
- `conversaciones_controladas.csv`: detalle técnico por paso.
- `resumen_escenarios.csv`: resultados por escenario.
- `sondeos_svm.csv`: comparación exploratoria de intenciones.
- `resumen_ejecucion.json`: configuración y recuentos.
