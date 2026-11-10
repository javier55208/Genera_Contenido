"""Prototipo opcional: chatbot de reglas con apoyo NLP SVM.

El SVM NO controla precios, confirmaciones, correcciones ni cancelaciones.
Se utiliza solo como respaldo para algunas intenciones no reconocidas.
Modelo academicamente exploratorio (accuracy inicial 60% en dataset sintetico).
"""
from pathlib import Path
import sys

from chatbot import Chatbot, intencion, normalizar

ROOT = Path(__file__).resolve().parent
NLP_DIR = ROOT / "nlp"
MODEL_PATH = NLP_DIR / "resultados" / "modelo_intenciones_svm.joblib"


class ChatbotHibrido(Chatbot):
    """Preserva el flujo del Avance 1; utiliza SVM solo si las reglas no reconocen."""

    def __init__(self, modelo=None):
        super().__init__()
        if modelo is None:
            import joblib
            # El modelo existente serializa el tokenizador desde preprocesamiento.py.
            if str(NLP_DIR) not in sys.path:
                sys.path.insert(0, str(NLP_DIR))
            import preprocesamiento  # noqa: F401
            modelo = joblib.load(MODEL_PATH)  # Cargar solo el artefacto local confiable.
        self.modelo_nlp = modelo
        self.ultima_prediccion_svm = None

    def detectar_intencion(self, mensaje):
        regla = intencion(mensaje, self.estado)
        self.ultima_prediccion_svm = None
        if regla != "no_reconocida":
            return regla
        propuesta = str(self.modelo_nlp.predict([mensaje])[0])
        self.ultima_prediccion_svm = propuesta
        # Lista deliberadamente restrictiva; el SVM no autoriza acciones sensibles.
        if propuesta in {"generar_publicidad", "solicitar_ayuda", "saludo", "despedida"}:
            return propuesta
        return "no_reconocida"


def main():
    bot = ChatbotHibrido()
    print("Chatbot hibrido experimental (reglas + NLP SVM). Escribe salir para terminar.")
    while True:
        try:
            mensaje = input("Tu: ")
        except (EOFError, KeyboardInterrupt):
            break
        if normalizar(mensaje) == "salir":
            break
        print("Bot:", bot.responder(mensaje))


if __name__ == "__main__":
    main()
