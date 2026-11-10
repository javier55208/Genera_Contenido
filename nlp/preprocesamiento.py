"""Preprocesamiento de texto en español para el Avance 2.

Implementa, de forma auditable y sin descargar corpus externos:
1) tokenización (NLTK), 2) normalización, 3) eliminación de stopwords,
4) stemming en español (NLTK SnowballStemmer).

Mantener 'no' y 'si' es deliberado: son señales de confirmación y corrección.
"""
from __future__ import annotations

import re
import unicodedata
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize

# Lista conservadora, explícita y revisable de palabras funcionales.
# No se utiliza la lista general de NLTK porque podría borrar negaciones.
STOPWORDS_ES = frozenset({
    'a', 'al', 'ante', 'bajo', 'con', 'contra', 'de', 'del', 'desde', 'durante',
    'el', 'ella', 'ellos', 'en', 'entre', 'era', 'es', 'esa', 'ese', 'eso',
    'esta', 'este', 'estos', 'hay', 'la', 'las', 'le', 'les', 'lo', 'los',
    'mi', 'mis', 'para', 'pero', 'por', 'que', 'se', 'su', 'sus', 'tu', 'tus',
    'una', 'unas', 'uno', 'unos', 'un', 'y', 'o', 'como', 'muy', 'me', 'te',
})
_STEMMER = SnowballStemmer('spanish')
_PATRON_NUMERO = re.compile(r'^\d+(?:[.,]\d+)?$')
_PATRON_PALABRA = re.compile(r'[a-z0-9]+')


def tokenizar(texto: str) -> list[str]:
    """Tokenización inicial con NLTK sin requerir nltk.download()."""
    if not isinstance(texto, str):
        raise TypeError('El texto debe ser una cadena (str)')
    return wordpunct_tokenize(texto)


def normalizar_tokens(tokens: list[str]) -> list[str]:
    """Minúsculas, eliminación de acentos y símbolos; precios a token 'numero'."""
    resultado = []
    for token in tokens:
        token = ''.join(
            c for c in unicodedata.normalize('NFD', token.lower())
            if unicodedata.category(c) != 'Mn'
        )
        if token == '$':
            resultado.append('pesos')
        elif _PATRON_NUMERO.fullmatch(token):
            resultado.append('numero')
        else:
            resultado.extend(_PATRON_PALABRA.findall(token))
    return resultado


def eliminar_stopwords(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in STOPWORDS_ES]


def aplicar_stemming(tokens: list[str]) -> list[str]:
    return [_STEMMER.stem(token) for token in tokens]


def preprocesar_texto(texto: str) -> list[str]:
    """Función usada por el vectorizador TF-IDF; definida a nivel de módulo para joblib."""
    return aplicar_stemming(eliminar_stopwords(normalizar_tokens(tokenizar(texto))))


def explicar_preprocesamiento(texto: str) -> dict[str, list[str]]:
    """Devuelve los cuatro pasos explícitos como evidencia académica."""
    t = tokenizar(texto)
    n = normalizar_tokens(t)
    s = eliminar_stopwords(n)
    return {
        'tokenizacion': t,
        'normalizacion': n,
        'sin_stopwords': s,
        'stemming': aplicar_stemming(s),
    }
