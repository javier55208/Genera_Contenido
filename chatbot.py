"""Prototipo académico: chatbot de reglas para solicitudes publicitarias.
No clasifica imágenes ni genera publicidad. Python 3.10+, NLTK.
"""
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from nltk.tokenize import wordpunct_tokenize

CATEGORIAS = ('MANGAS', 'DC', 'MARVEL', 'OTROS')


def normalizar(texto):
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto.lower()) if unicodedata.category(c) != 'Mn')
    return ' '.join(wordpunct_tokenize(texto)).replace(' $ ', ' $')


def categoria_en(texto):
    t = normalizar(texto)
    if re.search(r'\b(manga|mangas|manhwa|manhwas|novela ligera|novelas ligeras)\b', t):
        return 'MANGAS'
    if re.search(r'\b(marvel|spider man|spiderman)\b', t):
        return 'MARVEL'
    if re.search(r'\b(dc|batman|superman)\b', t):
        return 'DC'
    if re.search(r'\b(otros|otro|generica|generico)\b', t):
        return 'OTROS'
    return None



def precio_en(texto):
    # Detectar precios explicitos y conservar el signo negativo.
    t = normalizar(texto)

    patron = (
        r'(?:\$\s*|precio\s*(?:de|es|a|:)?\s*|'
        r'cuesta\s*|vale\s*|a\s+)'
        r'(-?\s*\d{1,7}(?:[.,]\d{1,2})?)'
        r'\s*(?:pesos|mxn)?\b'
    )

    m = re.search(patron, t)

    if not m:
        m = re.search(
            r'(?<!\w)(-?\s*\d{1,7}(?:[.,]\d{1,2})?)'
            r'\s*(?:pesos|mxn)\b',
            t
        )

    if not m:
        return None

    try:
        numero = m.group(1).replace(' ', '').replace(',', '.')
        valor = Decimal(numero)

        if not valor.is_finite() or valor <= 0:
            return None

        return valor

    except InvalidOperation:
        return None

    # Solo precios explícitos: $199, 199 pesos, precio 199, cuesta 199.
    t = normalizar(texto)
    m = re.search(r'(?:\$\s*|precio\s*(?:de|es|a|:)?\s*|cuesta\s*|vale\s*|a\s+)(\d{1,7}(?:[.,]\d{1,2})?)\s*(?:pesos|mxn)?\b', t)
    if not m:
        m = re.search(r'\b(\d{1,7}(?:[.,]\d{1,2})?)\s*(?:pesos|mxn)\b', t)
    if not m:
        return None
    try:
        valor = Decimal(m.group(1).replace(',', '.'))
        return valor if valor > 0 else None
    except InvalidOperation:
        return None


def intencion(texto, estado):
    t = normalizar(texto)
    if re.search(r'\b(cancelar|cancela|detener|anular)\b', t): return 'cancelar'
    if re.search(r'\b(adios|hasta luego|nos vemos)\b', t): return 'despedida'
    if re.search(r'\b(ayuda|opciones|que puedes hacer)\b', t): return 'solicitar_ayuda'
    if re.search(r'\b(cambia|cambiar|corrige|corregir|modifica|modificar)\b', t):
        return 'modificar_precio' if re.search(r'\b(precio|cuesta|vale)\b|\$', t) else 'corregir_categoria'
    if re.search(r'\b(no|incorrecto|equivocado)\b', t) and estado == 'confirmar': return 'corregir_categoria'
    if re.search(r'\b(si|correcto|confirmo|confirmar|adelante|esta bien)\b', t) and estado == 'confirmar': return 'confirmar_categoria'
    if re.search(r'\b(hola|buenos dias|buenas tardes|buenas noches)\b', t): return 'saludo'
    if re.search(r'\b(publicidad|anuncio|promocion|publicar)\b', t): return 'generar_publicidad'
    if categoria_en(texto): return 'indicar_categoria'
    if precio_en(texto) is not None: return 'indicar_precio'
    return 'no_reconocida'


@dataclass
class Chatbot:
    categoria: str | None = None
    precio: Decimal | None = None
    activo: bool = False
    estado: str = 'inicio'

    def resumen(self):
        return f'Categoria: {self.categoria}. Precio: ${self.precio:.2f} MXN. ¿Confirmas los datos?'

    def responder(self, mensaje):
        if not isinstance(mensaje, str) or not mensaje.strip():
            return 'Escribe una solicitud. Puedes pedir ayuda para conocer las opciones.'
        intent = intencion(mensaje, self.estado)
        if intent == 'solicitar_ayuda':
            return 'Puedo registrar una solicitud de publicidad: categoria MANGAS, DC, MARVEL u OTROS; precio; confirmacion, correccion o cancelacion. Aun no genero imagenes.'
        if intent == 'cancelar':
            self.categoria, self.precio, self.activo, self.estado = None, None, False, 'inicio'
            return 'Solicitud cancelada. No se genero publicidad.'
        if intent == 'despedida': return 'Hasta luego. Gracias por utilizar el asistente.'
        if intent == 'saludo' and not self.activo:
            return 'Hola. Puedo ayudarte a registrar una solicitud publicitaria. Escribe: quiero generar publicidad.'
        if intent == 'generar_publicidad':
            # Cada nueva solicitud comienza sin datos anteriores.
            self.categoria = None
            self.precio = None
            self.activo = True
            self.estado = 'captura'
        if not self.activo:
            return 'Para iniciar, escribe: quiero generar publicidad. O escribe ayuda.'
        if intent == 'confirmar_categoria' and self.estado == 'confirmar':
            self.estado, self.activo = 'finalizado', False
            return f'Solicitud confirmada: {self.categoria}, ${self.precio:.2f} MXN. Fin de chatbot de pruebaayuda.'
        if intent in ('corregir_categoria', 'modificar_precio'):
            if intent == 'corregir_categoria':
                cat = categoria_en(mensaje)
                if cat: self.categoria = cat
                else:
                    self.categoria, self.estado = None, 'captura'
                    return 'Indica la categoria correcta: MANGAS, DC, MARVEL u OTROS.'
            else:
                val = precio_en(mensaje)
                if val is None:
                    self.precio, self.estado = None, 'captura'
                    return 'Indica un precio valido mayor a cero, por ejemplo: precio $199.'
                self.precio = val
        else:
            cat = categoria_en(mensaje)
            val = precio_en(mensaje)
            if cat: self.categoria = cat
            if val is not None: self.precio = val
        if self.categoria is None:
            self.estado = 'captura'
            return '¿Cual es la categoria del producto? MANGAS, DC, MARVEL u OTROS.'
        if self.precio is None:
            self.estado = 'captura'
            return '¿Cual es el precio? Escribe por ejemplo: precio $199.'
        self.estado = 'confirmar'
        return self.resumen()


def main():
    bot = Chatbot()
    print('Asistente de publicidad (prototipo académico). Escribe salir para terminar.')
    while True:
        try: mensaje = input('Tu: ')
        except (EOFError, KeyboardInterrupt): break
        if normalizar(mensaje) == 'salir': break
        print('Bot:', bot.responder(mensaje))


if __name__ == '__main__':
    main()
