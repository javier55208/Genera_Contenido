"""Pruebas automáticas de separación de datos y preprocesamiento."""
import unittest
from pathlib import Path
from collections import Counter
from entrenar_nlp import INTENCIONES, dividir_por_familias, leer_dataset, crear_modelo
from preprocesamiento import (
    tokenizar, normalizar_tokens, eliminar_stopwords, aplicar_stemming,
    preprocesar_texto, explicar_preprocesamiento,
)

DATA = Path(__file__).resolve().parent / 'datos' / 'ejemplos_intenciones.csv'


class TestNLP(unittest.TestCase):
    def test_pasos_preprocesamiento(self):
        p = explicar_preprocesamiento('Sí, cambia el precio a $249.')
        self.assertIn('Sí', p['tokenizacion'])
        self.assertIn('si', p['normalizacion'])
        self.assertIn('numero', p['normalizacion'])
        self.assertNotIn('el', p['sin_stopwords'])
        self.assertTrue(len(p['stemming']) > 0)

    def test_negacion_no_se_elimina(self):
        tokens = preprocesar_texto('No, esa categoría está incorrecta')
        self.assertIn('no', tokens)
        self.assertTrue(tokens)

    def test_corpus_bien_etiquetado(self):
        datos = leer_dataset(DATA)
        self.assertEqual(len(datos), 300)
        self.assertEqual(set(Counter(x['intencion'] for x in datos)), set(INTENCIONES))
        self.assertTrue(all(n == 30 for n in Counter(x['intencion'] for x in datos).values()))

    def test_particion_reproducible_y_sin_fuga_grupos(self):
        datos = leer_dataset(DATA)
        tr1, te1 = dividir_por_familias(datos, semilla=42)
        tr2, te2 = dividir_por_familias(datos, semilla=42)
        self.assertEqual([x['id'] for x in tr1], [x['id'] for x in tr2])
        self.assertEqual([x['id'] for x in te1], [x['id'] for x in te2])
        self.assertEqual(len(tr1), 240)
        self.assertEqual(len(te1), 60)
        self.assertFalse({x['grupo'] for x in tr1} & {x['grupo'] for x in te1})
        self.assertTrue(all(n == 6 for n in Counter(x['intencion'] for x in te1).values()))

    def test_modelo_entrena_y_predice(self):
        data = leer_dataset(DATA)
        tr, te = dividir_por_familias(data, 42)
        model = crear_modelo()
        model.fit([x['texto'] for x in tr], [x['intencion'] for x in tr])
        preds = model.predict([x['texto'] for x in te])
        self.assertEqual(len(preds), 60)
        self.assertTrue(set(preds).issubset(INTENCIONES))


if __name__ == '__main__':
    unittest.main(verbosity=2)
