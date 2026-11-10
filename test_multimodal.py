"""Pruebas multimodales SIN descargar pesos ni acceder a imagenes reales."""
import unittest
from chatbot_multimodal import ChatbotMultimodal


class SVMFalso:
    def predict(self, textos):
        return ['cancelar']  # debe quedar bloqueado por las reglas


class CNNFalsa:
    def __init__(self, categoria='DC_COMICS'):
        self.categoria = categoria
        self.llamadas = 0
    def predecir(self, ruta):
        self.llamadas += 1
        return self.categoria


class PruebasIntegracion(unittest.TestCase):
    def bot(self, clase='DC_COMICS'):
        cnn = CNNFalsa(clase)
        return ChatbotMultimodal(modelo=SVMFalso(), clasificador=cnn), cnn

    def test_sugerencia_no_asigna_categoria_automaticamente(self):
        bot, cnn = self.bot()
        r = bot.enviar_imagen('portada.jpg')
        self.assertIn('DC', r)
        self.assertIsNone(bot.categoria)
        self.assertEqual(cnn.llamadas, 1)

    def test_confirmacion_visual_no_confirma_solicitud_final(self):
        bot, _ = self.bot()
        bot.enviar_imagen('portada.jpg')
        respuesta = bot.responder('si')
        self.assertEqual(bot.categoria, 'DC')
        self.assertIsNone(bot.precio)
        self.assertIn('precio', respuesta.lower())
        self.assertEqual(bot.estado, 'captura')
        bot.responder('precio $199')
        self.assertEqual(bot.estado, 'confirmar')
        self.assertIn('confirmada', bot.responder('si'))

    def test_correccion_por_usuario_tiene_prioridad(self):
        bot, _ = self.bot('MARVEL_COMICS')
        bot.enviar_imagen('portada.jpg')
        self.assertIn('categoria correcta', bot.responder('no'))
        self.assertIn('precio', bot.responder('MANGAS'))
        self.assertEqual(bot.categoria, 'MANGAS')

    def test_cancelacion_limpia_estado_visual(self):
        bot, _ = self.bot()
        bot.enviar_imagen('portada.jpg')
        self.assertIn('cancelada', bot.responder('cancelar'))
        self.assertIsNone(bot.categoria_visual_pendiente)
        self.assertFalse(bot.activo)

    def test_svm_no_puede_cancelar(self):
        bot, _ = self.bot()
        bot.enviar_imagen('portada.jpg')
        self.assertIn('si o no', bot.responder('texto ambiguo'))
        self.assertTrue(bot.activo)

    def test_evitar_sobrescribir_categoria(self):
        bot, cnn = self.bot()
        bot.enviar_imagen('primera.png')
        bot.responder('si')
        r = bot.enviar_imagen('segunda.png')
        self.assertIn('Ya hay una categoria', r)
        self.assertEqual(cnn.llamadas, 1)

    def test_etiqueta_visual_no_reconocida(self):
        bot, _ = self.bot('OTRA_CLASE')
        self.assertIn('no reconocida', bot.enviar_imagen('imagen.png'))
        self.assertFalse(bot.activo)

    def _completar_solicitud(self, bot):
        bot.enviar_imagen('primera.png')
        bot.responder('si')
        bot.responder('$199')
        self.assertIn('confirmada', bot.responder('si'))
        self.assertEqual(bot.estado, 'finalizado')

    def test_nueva_estimacion_no_recicla_precio_y_categoria(self):
        bot, cnn = self.bot()
        self._completar_solicitud(bot)
        r = bot.responder('quiero otra estimacion')
        self.assertIn('Nueva solicitud', r)
        self.assertIsNone(bot.categoria)
        self.assertIsNone(bot.precio)
        self.assertTrue(bot.activo)
        self.assertEqual(bot.estado, 'captura')
        self.assertIn('sugiere', bot.enviar_imagen('segunda.png'))
        self.assertEqual(cnn.llamadas, 2)
        self.assertIn('precio', bot.responder('si').lower())
        self.assertIsNone(bot.precio)

    def test_generar_publicidad_tras_confirmar_limpia_datos(self):
        bot, _ = self.bot()
        self._completar_solicitud(bot)
        respuesta = bot.responder('quiero generar publicidad')
        self.assertIn('categoria', respuesta.lower())
        self.assertIsNone(bot.categoria)
        self.assertIsNone(bot.precio)
        self.assertEqual(bot.estado, 'captura')
        self.assertIn('precio', bot.responder('MARVEL').lower())

    def test_generar_otra_publicidad_tras_confirmar_limpia_datos(self):
        bot, _ = self.bot()
        self._completar_solicitud(bot)
        respuesta = bot.responder('quiero generar otra publicidad')
        self.assertIn('Nueva solicitud', respuesta)
        self.assertIsNone(bot.categoria)
        self.assertIsNone(bot.precio)

    def test_segunda_imagen_directamente_despues_confirmacion(self):
        bot, cnn = self.bot('MANGAS')
        self._completar_solicitud(bot)
        self.assertIn('sugiere', bot.enviar_imagen('segunda.jpg'))
        self.assertEqual(cnn.llamadas, 2)
        self.assertIsNone(bot.precio)
        self.assertIsNone(bot.categoria)
        self.assertIn('precio', bot.responder('si').lower())

    def test_mensaje_irrelevante_despues_confirmacion_no_recicla_datos(self):
        bot, _ = self.bot()
        self._completar_solicitud(bot)
        self.assertIn('solicitud anterior', bot.responder('si').lower())
        self.assertEqual(bot.estado, 'finalizado')
        self.assertEqual(bot.categoria, 'DC')

    def test_solicitud_iniciada_por_texto_puede_usar_imagen(self):
        bot, _ = self.bot('MANGAS')
        bot.responder('quiero generar publicidad')
        self.assertIn('CNN sugiere', bot.enviar_imagen('portada.jpg'))
        self.assertIn('precio', bot.responder('si').lower())


if __name__ == '__main__':
    unittest.main(verbosity=2)

class PruebasErroresUsuario(unittest.TestCase):
    def bot(self, categoria='DC_COMICS'):
        return ChatbotMultimodal(modelo=SVMFalso(), clasificador=CNNFalsa(categoria))

    def test_precio_numero_sin_simbolo(self):
        bot = self.bot()
        bot.enviar_imagen('portada.png')
        bot.responder('si')
        self.assertIn('$900.00', bot.responder('900'))
        self.assertIn('confirmada', bot.responder('si'))

    def test_precio_cero_y_negativo_bloqueados(self):
        bot = self.bot()
        bot.enviar_imagen('portada.png')
        bot.responder('si')
        self.assertIn('mayor a cero', bot.responder('0'))
        self.assertIn('mayor a cero', bot.responder('-50'))
        self.assertIn('mayor a cero', bot.responder('precio -50 pesos'))
        self.assertIsNone(bot.precio)
        self.assertIn('$500.00', bot.responder('500'))

    def test_como_enviar_imagen_durante_captura(self):
        bot = self.bot()
        bot.responder('quiero generar publicidad')
        bot.responder('DC')
        self.assertIn('imagen "', bot.responder('Donde te paso la imagen para que estime el segmento?'))
        self.assertIn('nueva estimacion', bot.responder('si, pero puedo mandarte unanueva imagen?'))
        self.assertEqual(bot.categoria, 'DC')
        self.assertIsNone(bot.precio)

    def test_nueva_imagen_publicitaria_reinicia(self):
        bot = self.bot()
        bot.enviar_imagen('primera.png')
        bot.responder('si')
        bot.responder('$199')
        bot.responder('si')
        self.assertIn('Nueva solicitud', bot.responder('nueva imagen publicitaria'))
        self.assertIsNone(bot.categoria)
        self.assertIsNone(bot.precio)
        self.assertIn('sugiere', bot.enviar_imagen('segunda.png'))

    def test_no_confirmacion_permite_corregir_precio(self):
        bot = self.bot()
        bot.enviar_imagen('portada.png')
        bot.responder('si')
        bot.responder('$199')
        respuesta = bot.responder('no')
        self.assertIn('categoria o precio', respuesta)
        self.assertIn('nuevo precio', bot.responder('precio'))
        self.assertIn('$500.00', bot.responder('500'))
        self.assertIn('confirmada', bot.responder('si'))
        self.assertEqual(bot.categoria, 'DC')

    def test_no_confirmacion_permite_corregir_categoria(self):
        bot = self.bot()
        bot.enviar_imagen('portada.png')
        bot.responder('si')
        bot.responder('$199')
        bot.responder('no')
        self.assertIn('categoria correcta', bot.responder('categoria'))
        self.assertIn('MARVEL', bot.responder('MARVEL'))
        self.assertIn('confirmada', bot.responder('si'))
        self.assertEqual(bot.categoria, 'MARVEL')

    def test_pregunta_informativa_no_sobrescribe_propuesta_visual(self):
        bot = self.bot()
        bot.enviar_imagen('portada.png')
        self.assertIn('imagen "', bot.responder('Como mando una imagen?'))
        self.assertEqual(bot.categoria_visual_pendiente, 'DC')
        self.assertIn('precio', bot.responder('si'))

    def test_no_continua_sin_elegir_tipo_correccion(self):
        bot = self.bot()
        bot.enviar_imagen('portada.png')
        bot.responder('si')
        bot.responder('$199')
        bot.responder('no')
        self.assertIn('categoria o precio', bot.responder('si'))
        self.assertEqual(bot.estado, 'confirmar')
        self.assertTrue(bot.esperando_tipo_correccion)

