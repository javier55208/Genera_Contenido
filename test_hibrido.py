"""Pruebas de regresion del acoplamiento opcional, sin cargar el modelo real."""
import unittest
from chatbot_hibrido import ChatbotHibrido


class ModeloFalso:
    def __init__(self, prediccion):
        self.prediccion = prediccion

    def predict(self, textos):
        return [self.prediccion]


class Integracion(unittest.TestCase):
    def test_reglas_tienen_prioridad(self):
        bot = ChatbotHibrido(modelo=ModeloFalso("cancelar"))
        self.assertEqual(bot.detectar_intencion("hola"), "saludo")
        self.assertIsNone(bot.ultima_prediccion_svm)

    def test_respaldo_svm_para_iniciar(self):
        bot = ChatbotHibrido(modelo=ModeloFalso("generar_publicidad"))
        self.assertEqual(bot.detectar_intencion("Me gustaria preparar un cartel"), "generar_publicidad")

    def test_svm_no_puede_cancelar(self):
        bot = ChatbotHibrido(modelo=ModeloFalso("cancelar"))
        self.assertEqual(bot.detectar_intencion("xyzxyz"), "no_reconocida")

    def test_confirmacion_conserva_contexto(self):
        bot = ChatbotHibrido(modelo=ModeloFalso("saludo"))
        bot.responder("quiero generar publicidad")
        bot.responder("Marvel")
        bot.responder("precio $199")
        self.assertIn("confirmada", bot.responder("si"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
