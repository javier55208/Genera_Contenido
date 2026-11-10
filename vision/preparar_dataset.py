"""Avance 2: auditar y dividir el dataset visual sin mover las imagenes.

Uso: py vision/preparar_dataset.py --dataset "C:\\...\\Dataset_IA"
La salida se guarda dentro de vision/resultados (privada, no subir a GitHub).
"""
import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

CLASES = ('MANGAS', 'DC_COMICS', 'MARVEL_COMICS')
PORCENTAJES = {'train': 0.70, 'val': 0.15, 'test': 0.15}


def clave_grupo(texto, respaldo):
    texto = unicodedata.normalize('NFKD', (texto or '').strip().lower())
    texto = ''.join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r'[^a-z0-9]+', ' ', texto).strip()
    return texto or respaldo


def sha256(archivo):
    h = hashlib.sha256()
    with archivo.open('rb') as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloque)
    return h.hexdigest()


def separar_grupos(grupos, semilla, clase):
    """Asignar grupos de la misma serie a un solo subconjunto.

    Si existen series de muchos volumenes, los porcentajes son aproximados.
    """
    rng = random.Random(f'{semilla}-{clase}')
    grupos = list(grupos.items())
    rng.shuffle(grupos)
    grupos.sort(key=lambda par: len(par[1]), reverse=True)
    total = sum(len(items) for _, items in grupos)
    objetivos = {k: total * p for k, p in PORCENTAJES.items()}
    asignados = {k: [] for k in PORCENTAJES}
    for _, items in grupos:
        candidatos = sorted(PORCENTAJES, key=lambda k: (len(asignados[k]) - objetivos[k], len(asignados[k])))
        asignados[candidatos[0]].extend(items)
    return asignados


def preparar(dataset: Path, salida: Path, semilla: int = 42):
    dataset = dataset.resolve()
    csv_origen = dataset / 'dataset_muestra.csv'
    if not csv_origen.is_file():
        raise FileNotFoundError(f'No existe {csv_origen}. Indica la carpeta Dataset_IA completa.')
    registros = []
    hashes = {}
    with csv_origen.open(encoding='utf-8-sig', newline='') as f:
        lector = csv.DictReader(f)
        requeridas = {'Categoria_Dataset', 'Ruta_Dataset_Relativa', 'Hash_Imagen'}
        if not requeridas.issubset(lector.fieldnames or []):
            raise ValueError(f'Faltan columnas del CSV: {sorted(requeridas-set(lector.fieldnames or []))}')
        for i, fila in enumerate(lector, start=2):
            clase = fila['Categoria_Dataset'].strip()
            if clase not in CLASES:
                raise ValueError(f'Fila {i}: categoria inesperada {clase!r}')
            relativo = Path(fila['Ruta_Dataset_Relativa'].strip().replace('\\', '/'))
            ruta = (dataset / relativo).resolve()
            if not ruta.is_relative_to(dataset) or not ruta.is_file():
                raise FileNotFoundError(f'Fila {i}: imagen fuera de carpeta o inexistente: {relativo}')
            hash_real = sha256(ruta)
            hash_esperado = fila['Hash_Imagen'].strip().lower()
            if hash_esperado and hash_real != hash_esperado:
                raise ValueError(f'Fila {i}: SHA256 distinto del CSV: {relativo}')
            if hash_real in hashes:
                raise ValueError(f'Imagen repetida en {relativo} y {hashes[hash_real]}')
            hashes[hash_real] = str(relativo)
            grupo = clave_grupo(fila.get('Contenido_Producto'), f'__imagen_{hash_real}')
            registros.append({'clase': clase, 'ruta_relativa': relativo.as_posix(),
                              'grupo': grupo, 'sha256': hash_real})
    conteo = Counter(r['clase'] for r in registros)
    if set(conteo) != set(CLASES) or any(conteo[c] != 300 for c in CLASES):
        raise ValueError(f'Esperamos 300 imagenes por clase (900 total). Encontrado: {dict(conteo)}')
    asignacion = []
    detalle = {}
    for clase in CLASES:
        grupos = defaultdict(list)
        for registro in registros:
            if registro['clase'] == clase:
                grupos[registro['grupo']].append(registro)
        partes = separar_grupos(grupos, semilla, clase)
        detalle[clase] = {parte: len(items) for parte, items in partes.items()}
        for parte, items in partes.items():
            asignacion.extend({**item, 'split': parte} for item in items)
    salida.mkdir(parents=True, exist_ok=True)
    rng = random.Random(semilla)
    rng.shuffle(asignacion)
    with (salida / 'particion_visual.csv').open('w', newline='', encoding='utf-8') as f:
        escritor = csv.DictWriter(f, fieldnames=['ruta_relativa', 'clase', 'split', 'grupo', 'sha256'])
        escritor.writeheader()
        escritor.writerows(asignacion)
    resumen = {'dataset': str(dataset), 'semilla': semilla, 'total': len(asignacion),
               'agrupacion': 'Contenido_Producto dentro de cada clase; evita fuga de una misma serie',
               'objetivo': PORCENTAJES, 'conteos': detalle,
               'nota': 'Los tamaños pueden variar respecto a 70/15/15 para mantener series juntas.'}
    (salida / 'resumen_particion.json').write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding='utf-8')
    print('AUDITORIA Y PARTICION VISUAL COMPLETADAS')
    print(f'Imagenes: {len(asignacion)} | Hashes unicos: {len(hashes)}')
    for clase, partes in detalle.items():
        print(f'{clase}: entrenamiento={partes["train"]}, validacion={partes["val"]}, prueba={partes["test"]}')
    print(f'Manifiesto: {salida / "particion_visual.csv"}')
    print('No se copiaron ni publicaron imagenes.')
    return resumen


def main():
    p = argparse.ArgumentParser(description='Validar y separar las 900 portadas')
    p.add_argument('--dataset', type=Path, required=True, help='Directorio Dataset_IA')
    p.add_argument('--salida', type=Path, default=Path(__file__).resolve().parent / 'resultados')
    p.add_argument('--semilla', type=int, default=42)
    args = p.parse_args()
    preparar(args.dataset, args.salida, args.semilla)


if __name__ == '__main__':
    main()
