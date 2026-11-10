"""Pruebas del particionador, sin PyTorch ni imágenes externas."""
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

try:
    from vision.preparar_dataset import preparar
except ModuleNotFoundError:
    from preparar_dataset import preparar


class PruebasDataset(unittest.TestCase):
    def generar_dataset(self, raiz, repetido=False):
        base = raiz / 'Dataset_IA'
        base.mkdir()
        campos = ['Categoria_Dataset', 'Ruta_Dataset_Relativa', 'Hash_Imagen', 'Contenido_Producto']
        filas = []
        for clase in ('MANGAS', 'DC_COMICS', 'MARVEL_COMICS'):
            (base / 'imagenes' / clase).mkdir(parents=True)
            for i in range(300):
                ruta = f'imagenes/{clase}/{i}.jpg'
                datos = f'{clase}-{i}'.encode('utf-8')
                if repetido and clase == 'DC_COMICS' and i == 1:
                    datos = b'MANGAS-0'
                (base / ruta).write_bytes(datos)
                filas.append({'Categoria_Dataset': clase, 'Ruta_Dataset_Relativa': ruta,
                              'Hash_Imagen': hashlib.sha256(datos).hexdigest(),
                              'Contenido_Producto': f'Serie {i//3}'})
        with (base / 'dataset_muestra.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=campos); w.writeheader(); w.writerows(filas)
        return base

    def test_particion_grupos_sin_fuga(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dataset = self.generar_dataset(root)
            salida = root / 'salida'
            resumen = preparar(dataset, salida)
            self.assertEqual(resumen['total'], 900)
            with (salida / 'particion_visual.csv').open(newline='', encoding='utf-8') as f:
                filas = list(csv.DictReader(f))
            self.assertEqual(len(filas), 900)
            grupos = {}
            for f in filas:
                llave = (f['clase'], f['grupo'])
                grupos.setdefault(llave, f['split'])
                self.assertEqual(grupos[llave], f['split'])
            for clase in ('MANGAS', 'DC_COMICS', 'MARVEL_COMICS'):
                partes = {s: sum(1 for f in filas if f['clase'] == clase and f['split'] == s)
                          for s in ('train','val','test')}
                self.assertEqual(partes, {'train': 210, 'val': 45, 'test': 45})

    def test_detecta_archivo_duplicado(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dataset = self.generar_dataset(root, repetido=True)
            with self.assertRaisesRegex(ValueError, 'repetida'):
                preparar(dataset, root / 'salida')


if __name__ == '__main__':
    unittest.main()
