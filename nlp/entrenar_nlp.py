"""Entrena y evalúa TF-IDF + SVM sobre ejemplos etiquetados del chatbot.

Uso: python entrenar_nlp.py [--semilla 42] [--csv datos/ejemplos_intenciones.csv]
No modifica el código del Avance 1 ni realiza entrenamiento visual.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocesamiento import explicar_preprocesamiento, preprocesar_texto

ROOT = Path(__file__).resolve().parent
INTENCIONES = (
    'saludo', 'generar_publicidad', 'indicar_categoria', 'indicar_precio',
    'confirmar_categoria', 'corregir_categoria', 'modificar_precio',
    'solicitar_ayuda', 'cancelar', 'despedida',
)


def leer_dataset(ruta: Path) -> list[dict[str, str]]:
    with ruta.open('r', encoding='utf-8-sig', newline='') as fp:
        datos = list(csv.DictReader(fp))
    if not datos:
        raise ValueError('El dataset está vacío.')
    required = {'id', 'texto', 'intencion', 'grupo', 'origen'}
    missing = required - set(datos[0])
    if missing:
        raise ValueError(f'Faltan columnas: {sorted(missing)}')
    for d in datos:
        if not d['texto'].strip() or d['intencion'] not in INTENCIONES or not d['grupo']:
            raise ValueError(f'Ejemplo inválido: {d!r}')
    id_set = [d['id'] for d in datos]
    if len(id_set) != len(set(id_set)):
        raise ValueError('Existen IDs repetidos en el dataset.')
    # No basta con separar filas aleatorias si hay paráfrasis de la misma frase.
    return datos


def dividir_por_familias(datos: list[dict[str, str]], semilla: int, proporcion: float = 0.20):
    """Reservar familias completas de frases por clase para prueba sin fuga por paráfrasis."""
    rng = random.Random(semilla)
    grupos_test = set()
    for categoria in INTENCIONES:
        grupos = sorted({d['grupo'] for d in datos if d['intencion'] == categoria})
        if len(grupos) < 5:
            raise ValueError(f'Se necesitan al menos 5 familias para {categoria}; hay {len(grupos)}')
        # Un grupo no puede contener etiquetas distintas.
        for grp in grupos:
            etiquetas = {d['intencion'] for d in datos if d['grupo'] == grp}
            if etiquetas != {categoria}:
                raise ValueError(f'El grupo {grp} mezcla etiquetas')
        cuantos = max(1, round(len(grupos) * proporcion))
        grupos_test.update(rng.sample(grupos, cuantos))
    train = [d for d in datos if d['grupo'] not in grupos_test]
    test = [d for d in datos if d['grupo'] in grupos_test]
    assert not ({d['grupo'] for d in train} & {d['grupo'] for d in test})
    return train, test


def crear_modelo() -> Pipeline:
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=preprocesar_texto,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )),
        ('svm', LinearSVC(class_weight='balanced', random_state=42)),
    ])


def guardar_csv(ruta: Path, nombres: list[str], filas: list[dict]):
    with ruta.open('w', encoding='utf-8-sig', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=nombres)
        w.writeheader()
        w.writerows(filas)


def entrenar(csv_path: Path, salida: Path, semilla: int = 42):
    datos = leer_dataset(csv_path)
    train, test = dividir_por_familias(datos, semilla)
    modelo = crear_modelo()
    modelo.fit([d['texto'] for d in train], [d['intencion'] for d in train])
    reales = [d['intencion'] for d in test]
    predicciones = list(modelo.predict([d['texto'] for d in test]))
    resultado = classification_report(
        reales, predicciones, labels=list(INTENCIONES),
        output_dict=True, zero_division=0,
    )
    matriz = confusion_matrix(reales, predicciones, labels=list(INTENCIONES))

    salida.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, salida / 'modelo_intenciones_svm.joblib')
    guardar_csv(salida / 'particion_textual.csv', ['id', 'intencion', 'grupo', 'conjunto'],
                [{'id': d['id'], 'intencion': d['intencion'], 'grupo': d['grupo'],
                  'conjunto': 'train' if d in train else 'test'} for d in datos])
    guardar_csv(salida / 'predicciones_prueba.csv',
                ['id', 'texto', 'intencion_real', 'intencion_predicha', 'correcto', 'grupo'],
                [{'id': d['id'], 'texto': d['texto'], 'intencion_real': d['intencion'],
                  'intencion_predicha': p, 'correcto': int(d['intencion'] == p), 'grupo': d['grupo']}
                 for d, p in zip(test, predicciones)])
    guardar_csv(salida / 'matriz_confusion.csv',
                ['real\\predicha', *INTENCIONES],
                [{'real\\predicha': cat, **{c: int(v) for c, v in zip(INTENCIONES, fila)}}
                 for cat, fila in zip(INTENCIONES, matriz)])
    filas_reporte = []
    for etiqueta in [*INTENCIONES, 'macro avg', 'weighted avg']:
        fila = resultado[etiqueta]
        filas_reporte.append({'intencion': etiqueta, 'precision': fila['precision'],
                              'recall': fila['recall'], 'f1_score': fila['f1-score'],
                              'support': int(fila['support'])})
    guardar_csv(salida / 'reporte_clasificacion.csv',
                ['intencion', 'precision', 'recall', 'f1_score', 'support'], filas_reporte)
    metrics = {
        'algoritmo': 'LinearSVC (SVM lineal)',
        'representacion': 'TF-IDF unigramas y bigramas',
        'preprocesamiento': ['tokenizacion', 'normalizacion', 'eliminacion_stopwords', 'stemming_es'],
        'procedencia_dataset': 'ejemplos sinteticos redactados para prototipo; NO son conversaciones reales',
        'semilla': semilla,
        'separacion': '80/20 por familias de parafrasis, estratificada por intencion',
        'filas_total': len(datos), 'filas_entrenamiento': len(train), 'filas_prueba': len(test),
        'clases': list(INTENCIONES),
        'distribucion_entrenamiento': dict(Counter(d['intencion'] for d in train)),
        'distribucion_prueba': dict(Counter(d['intencion'] for d in test)),
        'accuracy_prueba': float(accuracy_score(reales, predicciones)),
        'f1_macro_prueba': float(f1_score(reales, predicciones, average='macro', zero_division=0)),
        'informe_por_clase': resultado,
        'advertencias': [
            'Las metricas solo representan el conjunto sintetico de prueba, no usuarios reales.',
            'Las familias de parafrasis se mantienen separadas entre entrenamiento y prueba.',
            'Mensajes breves como si/no requieren evaluar estado conversacional en la futura integracion.',
            'El clasificador NLP no valida precios ni categorias ni reemplaza las reglas de seguridad.',
            'No se incluyo una clase abierta para mensajes no reconocidos.',
        ],
    }
    (salida / 'metricas_nlp.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    (salida / 'ejemplo_preprocesamiento.json').write_text(
        json.dumps({'texto': 'Cambia el precio a $249, por favor',
                    'etapas': explicar_preprocesamiento('Cambia el precio a $249, por favor')},
                   ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print('Dataset:',len(datos),'ejemplos | Entrenamiento:',len(train),'| Prueba:',len(test))
    print('Accuracy:',round(metrics['accuracy_prueba'],4),'| F1 macro:',round(metrics['f1_macro_prueba'],4))
    print('Resultados guardados en:', salida)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar modelo NLP SVM del Avance 2')
    parser.add_argument('--csv', default=str(ROOT / 'datos' / 'ejemplos_intenciones.csv'))
    parser.add_argument('--salida', default=str(ROOT / 'resultados'))
    parser.add_argument('--semilla', type=int, default=42)
    args = parser.parse_args()
    entrenar(Path(args.csv), Path(args.salida), args.semilla)
