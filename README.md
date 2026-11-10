# Asistente de publicidad — Proyecto incremental de Maestria en IA

Este repositorio **continua el Avance 1** y suma los experimentos del **Avance 2** sobre la misma raiz. El chatbot de reglas sigue funcionando de forma autonoma; el SVM es un complemento experimental opcional, no lo reemplaza.

## Estado de funcionalidades

| Componente | Estado | Archivo |
|---|---|---|
| Chatbot de reglas, categorias, precios, confirmaciones | Implementado desde Avance 1 | `chatbot.py` |
| Pruebas del chatbot | Implementadas desde Avance 1 | `test_chatbot.py` |
| Dataset NLP sintetico (300 frases, 10 intenciones) | Prototipo de Avance 2 | `nlp/datos/ejemplos_intenciones.csv` |
| Tokenizacion, normalizacion, stopwords, stemming | Implementado en NLP | `nlp/preprocesamiento.py` |
| Vectorizacion TF-IDF y clasificador SVM | Entrenado y evaluado inicialmente | `nlp/entrenar_nlp.py`, `nlp/resultados/` |
| Acoplamiento NLP-reglas (respaldo restringido) | Experimental y opcional | `chatbot_hibrido.py` |
| CNN, imagenes, multimodalidad visual | Pendiente; no incluido | — |

## Estructura

```text
proyecto_maestria_incremental_v02/
  chatbot.py                # Chatbot Avance 1: ejecutable y retrocompatible
  test_chatbot.py           # Regresion del Avance 1
  chatbot_hibrido.py        # Ejecucion opcional de NLP con reglas
  test_hibrido.py           # Pruebas del acoplamiento controlado
  requirements.txt          # Dependencias compartidas
  README.md                 # Guia maestra para todo el proyecto
  nlp/
    datos/ejemplos_intenciones.csv
    preprocesamiento.py
    entrenar_nlp.py
    predecir_nlp.py
    test_nlp.py
    resultados/             # Metricas y modelo inicial
    README.md               # Detalles metodologicos del modulo
```

## Instalacion desde la raiz

Se requiere Python 3.10 o superior. Desde una terminal situada en esta carpeta:

```bash
python -m pip install -r requirements.txt
```

## Avance 1 intacto en comportamiento

```bash
python chatbot.py
python -m unittest -v test_chatbot.py
```

El unico cambio en `chatbot.py` es un punto de extension `detectar_intencion` que por defecto ejecuta la MISMA funcion de reglas. Se conserva la interfaz original de `Chatbot.responder` y todas las validaciones existentes. Este cambio permite complementarlo mediante una subclase sin duplicar el flujo conversacional.

## Avance 2 — experimentacion NLP

```bash
python nlp/entrenar_nlp.py --semilla 42
python nlp/predecir_nlp.py "Quiero preparar un anuncio de Batman"
python -m unittest discover -s nlp -p test_nlp.py -v
```

Los mensajes son **ejemplos sinteticos academicos**, no mensajes de clientes. La primera prueba reporta accuracy 60% y F1 macro 57.8%; NO representa efectividad comercial real. Ver `nlp/README.md` para metodologia, limites y resultados.

## Acoplamiento opcional: modo hibrido experimental

```bash
python -m unittest -v test_hibrido.py
python chatbot_hibrido.py
```

`chatbot_hibrido.py` ejecuta la misma conversacion del Avance 1, pero cuando las reglas NO identifican la intencion, consulta el modelo SVM. Por seguridad solo acepta del SVM estas etiquetas: `generar_publicidad`, `solicitar_ayuda`, `saludo` y `despedida`. La categoria, precio, correcciones, cancelaciones y confirmaciones permanecen controladas por reglas. **No es la integracion multimodal exigida por la rubrica**: falta entrenar la CNN e integrar la clasificacion visual.

Un clasificador SVM siempre puede proponer alguna de las 10 etiquetas, aun para mensajes fuera de dominio; esta prueba no contiene umbral de abstencion fiable. No debe usarse en produccion.

## Proceso previo de datos visuales

Se preparo, mediante herramienta auxiliar separada desarrollada con apoyo de Gemini, un CSV con 900 registros de imagen (300 MANGAS, 300 DC_COMICS, 300 MARVEL_COMICS), a partir del catalogo WooCommerce. La herramienta de adquisicion y muestreo **no se incorpora al chatbot**; permanece separada. Falta dividir imagenes, entrenar CNN y obtener metricas visuales. No se incluyen portadas en este ZIP.

## Evidencias y continuidad

- Se preservan `chatbot.py` y las pruebas del Avance 1; este ZIP sustituye al paquete aislado NLP para trabajar incrementalmente.
- Los resultados en `nlp/resultados/` provienen del experimento inicial; no equivalen a metricas obtenidas tras integracion o con usuarios reales.
- No sobrescribas tu proyecto original sin antes crear respaldo y comparar versiones, pues tu repositorio local pudiera contener cambios posteriores.
- El proyecto final podra agregar vision, gestion de modelos, interfaz y API como modulos complementarios, manteniendo esta raiz.

## Bitacora / decisiones

1. **Confirmado:** preservar chatbot de reglas y la estructura unica del proyecto.
2. **Confirmado:** preparar NLP con TF-IDF + SVM y 10 intenciones como experimento del Avance 2.
3. **Implementado como opcion experimental:** adaptador de reglas + SVM sin automatizar acciones sensibles.
4. **Pendiente:** mejorar calidad del NLP, evaluar sus errores, entrenar CNN e integrarla, recopilar evidencias y redactar PDF.
5. **Control:** no se modifica el repositorio GitHub del usuario ni los archivos de WooCommerce. Esta entrega es una copia local para revision.
