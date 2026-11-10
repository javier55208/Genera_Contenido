# Correccion de reinicio multimodal v0.2

Corrige exclusivamente la retencion de categoria/precio tras una solicitud confirmada.

## Instalacion en el proyecto existente
1. Haz copia de seguridad de `chatbot_multimodal.py` y `test_multimodal.py`.
2. Reemplaza **solo esos dos archivos** en la raiz de `Genera Contenido`.
3. Conserva `chatbot.py`, `chatbot_hibrido.py`, `nlp/`, `vision/` y todos los modelos.

## Pruebas en PowerShell

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_multimodal.py test_hibrido.py test_chatbot.py
```

## Prueba interactiva

```powershell
.\.venv\Scripts\python.exe chatbot_multimodal.py
```

Secuencia: `imagen "C:\ruta\portada1.jpg"` -> `si` -> `$199` -> `si` -> `quiero otra estimacion` -> `imagen "C:\ruta\portada2.jpg"` -> `si` -> `$249` -> `si`.

Alternativamente, despues de la primera confirmacion: `quiero generar publicidad`, `quiero generar otra publicidad`, o directamente una nueva ruta `imagen "..."`.

No se modifican modelos, pesos, datasets ni se hace publicacion a GitHub.
