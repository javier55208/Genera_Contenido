"""Pruebas controladas y evidencia academica: chatbot original + hibrido + SVM.

Ejecutar desde la raiz del proyecto:
    py pruebas_controladas_avance2.py
    py pruebas_controladas_avance2.py --solo-reglas

No entrena modelos, no modifica codigo ni sobrescribe resultados de entrenamiento.
Los informes se guardan en evidencias_pruebas/<marca_de_tiempo>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# id, descripcion, pasos (mensaje, texto esperado, estado esperado), clase de prueba
ESCENARIOS = [
    ("E01", "Flujo completo DC", [
        ("hola", "Hola", "inicio"),
        ("quiero generar publicidad", "categoria del producto", "captura"),
        ("DC Comics", "precio", "captura"),
        ("precio $199", "Confirmas", "confirmar"),
        ("si", "Solicitud confirmada: DC, $199.00", "finalizado"),
    ], "funcional"),
    ("E02", "Flujo MARVEL con confirmacion", [
        ("Quiero generar publicidad", "categoria del producto", "captura"),
        ("MARVEL", "precio", "captura"),
        ("$250", "Confirmas", "confirmar"),
        ("SI", "Solicitud confirmada: MARVEL, $250.00", "finalizado"),
    ], "funcional"),
    ("E03", "Correccion de categoria", [
        ("quiero generar publicidad", "categoria del producto", "captura"),
        ("OTROS", "precio", "captura"),
        ("$190", "Confirmas", "confirmar"),
        ("no", "categoria correcta", "captura"),
        ("MANGAS", "Confirmas", "confirmar"),
        ("si", "Solicitud confirmada: MANGAS, $190.00", "finalizado"),
    ], "funcional"),
    ("E04", "Ayuda y despedida", [
        ("¿Me explicas cómo funciona?", "Puedo registrar", "inicio"),
        ("hasta luego", "Hasta luego", "inicio"),
    ], "observacion"),
    ("E05", "Cancelacion y reinicio", [
        ("quiero generar publicidad", "categoria del producto", "captura"),
        ("DC", "precio", "captura"),
        ("cancelar", "Solicitud cancelada", "inicio"),
        ("quiero generar publicidad", "categoria del producto", "captura"),
    ], "funcional"),
    ("E06", "Precio cero rechazado", [
        ("quiero generar publicidad", "categoria del producto", "captura"),
        ("MARVEL", "precio", "captura"),
        ("precio $0", "precio", "captura"),
    ], "funcional"),
    ("E07", "Precio negativo rechazado", [
        ("quiero generar publicidad", "categoria del producto", "captura"),
        ("DC", "precio", "captura"),
        ("precio -50 pesos", "precio", "captura"),
    ], "funcional"),
    ("E08", "Editorial fuera de categorias", [
        ("quiero generar publicidad", "categoria del producto", "captura"),
        ("Planeta", "categoria del producto", "captura"),
    ], "diagnostico"),
    ("E09", "Saludo y solicitud con errata", [
        ("hola", "Hola", "inicio"),
        ("quiero generar una ublicidad, ¿Se puede?", "categoria del producto", "captura"),
    ], "diagnostico"),
]

# Mensajes fuera del entrenamiento usados como casos exploratorios, NO conjunto de prueba independiente.
SONDEOS_SVM = [
    ("S01", "Quiero preparar un anuncio de Batman", "generar_publicidad"),
    ("S02", "Necesito ayuda para utilizar el asistente", "solicitar_ayuda"),
    ("S03", "Hasta luego", "despedida"),
    ("S04", "Hola, buenas tardes", "saludo"),
    ("S05", "Quisiera anular la solicitud", "cancelar"),
    ("S06", "El precio es 125 pesos", "indicar_precio"),
    ("S07", "La categoría es Marvel", "indicar_categoria"),
    ("S08", "Quiero corregir la categoría", "corregir_categoria"),
]


def escribir_csv(ruta: Path, columnas: list[str], filas: list[dict]):
    with ruta.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)



def construir_transcripcion(pasos: list[dict], resumen: list[dict], info: dict) -> str:
    """Genera diálogos legibles con respuestas REALES de la ejecución."""
    lineas = [
        "PRUEBAS CONTROLADAS — AVANCE 2 (CHATBOT ORIGINAL E HÍBRIDO)",
        "=" * 72,
        f"Fecha de ejecución: {info['fecha_generacion']}",
        f"Python: {info['version_python']}",
        "",
        "Cada respuesta fue producida por el chatbot durante esta ejecución.",
        "[OK] significa que la respuesta Y el estado coincidieron con lo esperado.",
        "[FALLO] registra diferencias para analizarlas, sin ocultarlas.",
        "",
    ]
    for caso in resumen:
        lineas += [
            "=" * 72,
            f"{caso['modo'].upper()} | {caso['escenario']} - {caso['descripcion']}",
            f"Tipo: {caso['tipo']} | Resultado: {caso['pasos_correctos']}/{caso['pasos']} pasos correctos",
            "-" * 72,
        ]
        coincidentes = [p for p in pasos if p['modo'] == caso['modo'] and p['escenario'] == caso['escenario']]
        for p in coincidentes:
            lineas.append(f"Usuario: {p['mensaje']}")
            lineas.append(f"Chatbot: {p['respuesta_obtenida'] if p['respuesta_obtenida'] else '(sin respuesta)'}")
            lineas.append(f"Evaluación: {'[OK]' if p['cumple_paso'] else '[FALLO]'} | Estado: {p['estado_obtenido'] or 'no disponible'}")
            if p['prediccion_svm_fallback']:
                lineas.append(f"SVM consultado: {p['prediccion_svm_fallback']}")
            if not p['cumple_paso']:
                lineas.append(f"Esperado (fragmento): {p['texto_esperado']}; estado: {p['estado_esperado']}")
            if p['error']:
                lineas.append(f"ERROR: {p['error']}")
            lineas.append("")
    lineas += [
        "=" * 72,
        "RESUMEN",
        f"Escenarios funcionales aprobados: {info['escenarios_funcionales_aprobados']}/{info['escenarios_funcionales_totales']}",
        "Nota: son pruebas guiadas, no una medida de accuracy de SVM.",
        "Para métricas estadísticas consultar nlp/resultados/metricas_nlp.json.",
    ]
    if info['advertencias']:
        lineas.extend(["", "ADVERTENCIAS:"] + info['advertencias'])
    return "\n".join(lineas) + "\n"


def evaluar_dialogos(bot_clase, nombre_modo: str):
    pasos, resumen = [], []
    for ident, titulo, dialogo, tipo in ESCENARIOS:
        bot = bot_clase()  # Evitar heredar estado entre escenarios.
        buenos = 0
        for numero, (mensaje, esperado, estado_esperado) in enumerate(dialogo, start=1):
            try:
                respuesta = bot.responder(mensaje)
                estado = str(getattr(bot, "estado", ""))
                coincide = esperado.casefold() in respuesta.casefold()
                estado_ok = estado == estado_esperado
                ok = coincide and estado_ok
                pred_svm = getattr(bot, "ultima_prediccion_svm", None)
                error = ""
            except Exception as exc:
                respuesta, estado, pred_svm = "", "", None
                coincide, estado_ok, ok, error = False, False, False, f"{type(exc).__name__}: {exc}"
            buenos += int(ok)
            pasos.append({
                "modo": nombre_modo, "escenario": ident, "descripcion": titulo,
                "tipo": tipo, "paso": numero, "mensaje": mensaje,
                "respuesta_obtenida": respuesta, "texto_esperado": esperado,
                "estado_obtenido": estado, "estado_esperado": estado_esperado,
                "cumple_texto": int(coincide), "cumple_estado": int(estado_ok),
                "cumple_paso": int(ok), "prediccion_svm_fallback": pred_svm or "",
                "error": error,
            })
        resumen.append({
            "modo": nombre_modo, "escenario": ident, "descripcion": titulo,
            "tipo": tipo, "pasos": len(dialogo), "pasos_correctos": buenos,
            "cumple_escenario": int(buenos == len(dialogo)),
        })
    return pasos, resumen


def evaluar_svm(modelo):
    filas = []
    for ident, mensaje, esperado in SONDEOS_SVM:
        try:
            predicha = str(modelo.predict([mensaje])[0])
            error = ""
        except Exception as exc:
            predicha, error = "", f"{type(exc).__name__}: {exc}"
        filas.append({
            "caso": ident, "mensaje": mensaje, "intencion_esperada": esperado,
            "intencion_predicha": predicha, "coincide": int(predicha == esperado), "error": error,
        })
    return filas


def main():
    p = argparse.ArgumentParser(description="Pruebas y evidencias de NLP y chatbot híbrido")
    p.add_argument("--solo-reglas", action="store_true", help="No cargar modelo SVM")
    p.add_argument("--salida", default="evidencias_pruebas", help="Carpeta para evidencias nuevas")
    args = p.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destino = (ROOT / args.salida / stamp).resolve()
    # No permitir rutas que sustituyan archivos; se crea subdirectorio único.
    destino.mkdir(parents=True, exist_ok=False)

    from chatbot import Chatbot
    pasos, resumen = evaluar_dialogos(Chatbot, "reglas")
    advertencias = []
    sondeos = []
    modelos_ejecutados = ["reglas"]

    if not args.solo_reglas:
        try:
            from chatbot_hibrido import ChatbotHibrido
            import joblib
            model_path = ROOT / "nlp" / "resultados" / "modelo_intenciones_svm.joblib"
            # SOLO cargar modelo generado localmente y confiable; joblib deserializa pickle.
            if str(ROOT / "nlp") not in sys.path:
                sys.path.insert(0, str(ROOT / "nlp"))
            import preprocesamiento  # Importación del tokenizador para modelos serializados.
            modelo = joblib.load(model_path)
            class HibridoLocal(ChatbotHibrido):
                def __init__(self):
                    super().__init__(modelo=modelo)
            p2, r2 = evaluar_dialogos(HibridoLocal, "hibrido")
            pasos.extend(p2)
            resumen.extend(r2)
            sondeos = evaluar_svm(modelo)
            modelos_ejecutados.append("hibrido")
        except Exception as exc:
            advertencias.append(f"No se pudo ejecutar híbrido/SVM: {type(exc).__name__}: {exc}")

    escribir_csv(destino / "conversaciones_controladas.csv", [
        "modo", "escenario", "descripcion", "tipo", "paso", "mensaje", "respuesta_obtenida",
        "texto_esperado", "estado_obtenido", "estado_esperado", "cumple_texto", "cumple_estado",
        "cumple_paso", "prediccion_svm_fallback", "error",
    ], pasos)
    escribir_csv(destino / "resumen_escenarios.csv", [
        "modo", "escenario", "descripcion", "tipo", "pasos", "pasos_correctos", "cumple_escenario",
    ], resumen)
    escribir_csv(destino / "sondeos_svm.csv", [
        "caso", "mensaje", "intencion_esperada", "intencion_predicha", "coincide", "error",
    ], sondeos)

    info = {
        "fecha_generacion": datetime.now().isoformat(timespec="seconds"),
        "version_python": platform.python_version(),
        "modos_ejecutados": modelos_ejecutados,
        "escenarios": len(resumen),
        "escenarios_funcionales_aprobados": sum(int(r["cumple_escenario"]) for r in resumen if r["tipo"] == "funcional"),
        "escenarios_funcionales_totales": sum(r["tipo"] == "funcional" for r in resumen),
        "escenarios_diagnosticos_y_observacion": sum(r["tipo"] != "funcional" for r in resumen),
        "sondeos_svm": len(sondeos),
        "sondeos_svm_coincidentes": sum(x["coincide"] for x in sondeos),
        "advertencias": advertencias,
        "nota_metodologica": (
            "Los escenarios guiados comprueban funciones especificas; los sondeos SVM son casos "
            "exploratorios y NO equivalen a accuracy de un conjunto test independiente. "
            "Para metricas del clasificador, usar nlp/resultados/metricas_nlp.json generado por entrenamiento."
        ),
    }
    (destino / "resumen_ejecucion.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# Evidencias de pruebas controladas — Avance 2", "",
          f"Fecha: {info['fecha_generacion']}",
          f"Python: {info['version_python']}",
          f"Modos: {', '.join(modelos_ejecutados)}", "",
          "## Resultado de pruebas funcionales", "",
          f"Escenarios funcionales que cumplen: {info['escenarios_funcionales_aprobados']}/{info['escenarios_funcionales_totales']}",
          f"Escenarios diagnósticos u observacionales: {info['escenarios_diagnosticos_y_observacion']}", "",
          "| Modo | Escenario | Tipo | Pasos correctos | Cumple |",
          "|---|---|---|---|---|",]
    for r in resumen:
        md.append(f"| {r['modo']} | {r['escenario']} — {r['descripcion']} | {r['tipo']} | "
                  f"{r['pasos_correctos']}/{r['pasos']} | {'Sí' if r['cumple_escenario'] else 'No'} |")
    md += ["", "## Sondeos exploratorios del modelo SVM", "",
           f"Coincidencias observadas: {info['sondeos_svm_coincidentes']}/{info['sondeos_svm']}. "
           "NO interpretar como accuracy independiente.", "",
           "## Interpretación y límites", "",
           "- La ejecución usa mensajes prefijados y verifica respuesta y estado del diálogo.",
           "- Se crea un chatbot nuevo en cada escenario para evitar contaminación del estado.",
           "- Las pruebas diagnósticas no bloquean por sí solas el prototipo; se documentan como mejoras.",
           "- Las reglas tienen prioridad; la columna `prediccion_svm_fallback` muestra dónde se consultó el modelo.",
           "- No se entrena el modelo durante estas pruebas, ni se comprueba la CNN o la integración texto-imagen.",
           "- Las métricas estadísticas de SVM se deben tomar del conjunto test usado en `entrenar_nlp.py`.",
           "- El contenido depende del modelo instalado y de la versión de sus bibliotecas."]
    if advertencias:
        md += ["", "## Advertencias", *["- " + a for a in advertencias]]
    md += ["", "## Archivos adjuntos", "",
           "- `conversaciones_terminal.txt`: diálogos completos tal como se muestran en la terminal.",
           "- `conversaciones_controladas.csv`: detalle técnico por paso.",
           "- `resumen_escenarios.csv`: resultados por escenario.",
           "- `sondeos_svm.csv`: comparación exploratoria de intenciones.",
           "- `resumen_ejecucion.json`: configuración y recuentos.", ""]
    (destino / "informe_pruebas.md").write_text("\n".join(md), encoding="utf-8")

    transcripcion = construir_transcripcion(pasos, resumen, info)
    (destino / "conversaciones_terminal.txt").write_text(transcripcion, encoding="utf-8-sig")
    print(transcripcion, end="")
    print(f"Carpeta de evidencias: {destino}")
    print("Conversaciones en texto: conversaciones_terminal.txt")
    print(f"Escenarios funcionales aprobados: {info['escenarios_funcionales_aprobados']}/{info['escenarios_funcionales_totales']}")
    print(f"Sondeos SVM coincidentes: {info['sondeos_svm_coincidentes']}/{info['sondeos_svm']} (diagnóstico, NO accuracy)")
    for a in advertencias:
        print("ADVERTENCIA:", a)
    print("Abrir conversaciones_terminal.txt para leer los diálogos completos.")
    return 0 if not advertencias else 1


if __name__ == "__main__":
    raise SystemExit(main())
