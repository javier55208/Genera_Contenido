"""Diez escenarios independientes, sin inventar resultados de usuarios reales."""
import unittest
from chatbot import Chatbot, intencion, precio_en

class PruebasConversacion(unittest.TestCase):
    def test_01_saludo(self):
        self.assertIn('Hola', Chatbot().responder('Hola'))
    def test_02_inicio(self):
        self.assertIn('categoria', Chatbot().responder('Quiero generar publicidad').lower())
    def test_03_categoria_mangas(self):
        b=Chatbot(); b.responder('quiero publicidad'); self.assertIn('precio', b.responder('es un manga').lower())
    def test_04_categoria_dc(self):
        b=Chatbot(); b.responder('quiero publicidad'); b.responder('DC'); self.assertEqual(b.categoria,'DC')
    def test_05_categoria_marvel(self):
        b=Chatbot(); b.responder('quiero publicidad'); b.responder('Marvel'); self.assertEqual(b.categoria,'MARVEL')
    def test_06_precio_y_confirmacion(self):
        b=Chatbot(); b.responder('quiero publicidad'); b.responder('Mangas'); self.assertIn('Confirmas',b.responder('precio $199'))
        self.assertIn('confirmada', b.responder('si'))
    def test_07_corregir_categoria(self):
        b=Chatbot(); b.responder('quiero publicidad'); b.responder('Marvel'); b.responder('precio $199'); b.responder('cambia a DC'); self.assertEqual(b.categoria,'DC')
    def test_08_modificar_precio(self):
        b=Chatbot(); b.responder('quiero publicidad'); b.responder('DC'); b.responder('precio $199'); b.responder('cambia el precio a $249'); self.assertEqual(str(b.precio),'249')
    def test_09_cancelar(self):
        b=Chatbot(); b.responder('quiero publicidad'); b.responder('Marvel'); self.assertIn('cancelada', b.responder('cancela')); self.assertIsNone(b.categoria)
    def test_10_ayuda_y_entrada_desconocida(self):
        b=Chatbot(); self.assertIn('MANGAS',b.responder('ayuda')); b.responder('quiero publicidad'); self.assertIn('categoria',b.responder('xyzxyz').lower())
    def test_11_precio_invalido(self):
        b=Chatbot(); b.responder('quiero publicidad'); b.responder('DC'); self.assertIn('precio',b.responder('precio $0').lower())
    def test_12_intenciones(self):
        self.assertEqual(intencion('modifica el precio a $249','confirmar'),'modificar_precio')
        self.assertEqual(precio_en('precio $249'),249)

if __name__ == '__main__': unittest.main(verbosity=2)
