# Integracion multimodal v0.3 - correcciones de conversacion

Alcance autorizado: correcciones del chatbot multimodal del Avance 2. No cambia la CNN, el SVM, el chatbot original, los modelos, los datasets ni WooCommerce.

## Cambios comprobados
- En la etapa de captura de precio acepta `500`, `900`, `$500`, `900 pesos` y rechaza importes cero o negativos.
- Responde a preguntas como `Donde te paso la imagen?` y `Puedo mandarte una nueva imagen?` con el formato `imagen "C:\ruta\portada.jpg"`.
- Si ya hay categoria, no la reemplaza silenciosamente: informa que `nueva estimacion` permite iniciar una nueva captura.
- Reconoce `nueva imagen publicitaria` como inicio de otra solicitud.
- Ante `no` al resumen final, pregunta si se desea corregir la categoria o el precio. Es posible responder `categoria`, `precio` o directamente con la categoria/cifra nueva.
- Mantiene prioridad de reglas y confirmaciones explicitas sobre las sugerencias SVM/CNN.

## Instalacion en computadora personal
1. Respaldar `chatbot_multimodal.py` y `test_multimodal.py` existentes.
2. Reemplazar **solo** esos dos archivos en la raiz de `Genera Contenido`.
3. Ejecutar desde PowerShell, en la raiz del repositorio:

```
.\.venv\Scripts\python.exe -m unittest -v test_multimodal.py test_hibrido.py test_chatbot.py
```

4. Probar:

```
.\.venv\Scripts\python.exe chatbot_multimodal.py
```

Ejemplo: `imagen "C:\ruta\portada.jpg"` -> `si` -> `900` -> `no` -> `precio` -> `500` -> `si` -> `nueva imagen publicitaria`.

## Evidencia y limites
37 pruebas automatizadas aprobaron en el entorno de desarrollo (29 originales + 8 nuevas). Se usan dobles CNN/SVM para el dialogo; es necesario comprobar en Windows con los modelos reales. No se realizaron reentrenamientos, descargas de pesos ni cambios en archivos externos.

No subir datasets de portadas, rutas privadas, archivos .pth, ni .venv a GitHub.
