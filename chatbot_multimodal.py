"""Integracion incremental: reglas + NLP SVM + clasificador de portada CNN.

La CNN sugiere categoria, el usuario decide. No genera anuncios ni cambia WooCommerce.
Uso (PowerShell): .\\.venv\\Scripts\\python.exe chatbot_multimodal.py
En el chat: imagen "C:\\ruta\\portada.jpg"
"""
from pathlib import Path
import re
from decimal import Decimal
from chatbot import categoria_en, normalizar
from chatbot_hibrido import ChatbotHibrido
from vision.predecir_imagen import ClasificadorPortadas

CLASE_VISUAL_A_CHAT = {'MANGAS': 'MANGAS', 'DC_COMICS': 'DC', 'MARVEL_COMICS': 'MARVEL'}


class ChatbotMultimodal(ChatbotHibrido):
    def __init__(self, modelo=None, clasificador=None):
        # Permite un doble de prueba, pero usa SVM real en ejecucion normal.
        super().__init__(modelo=modelo)
        self.clasificador = clasificador if clasificador is not None else ClasificadorPortadas()
        self.categoria_visual_pendiente = None
        self.esperando_correccion_visual = False
        self.esperando_tipo_correccion = False

    def _reiniciar_solicitud(self):
        """Limpia solo los datos de la solicitud; conserva modelos NLP y CNN cargados."""
        self.categoria = None
        self.precio = None
        self.activo = False
        self.estado = 'inicio'
        self.categoria_visual_pendiente = None
        self.esperando_correccion_visual = False
        self.esperando_tipo_correccion = False

    @staticmethod
    def _pide_nueva_solicitud(texto):
        # Frases explicitas para reiniciar sin interpretar un 'si' como nueva orden.
        return bool(re.fullmatch(
            r'(?:quiero\s+)?(?:(?:generar|hacer|iniciar|preparar)\s+)?'
            r'(?:(?:una|un)\s+)?(?:otra|nuevo|nueva)\s+'
            r'(?:estimacion|clasificacion|publicidad|solicitud|anuncio)'
            r'(?:\s+de\s+imagen)?', texto) or re.fullmatch(
            r'(?:quiero\s+)?(?:una\s+)?nueva\s+imagen\s+publicitaria', texto))

    @staticmethod
    def _consulta_sobre_imagen(texto):
        """Dudas de uso: nunca confundir con una nueva solicitud comercial."""
        return (bool(re.search(r'\b(imagen|foto|portada|archivo)\b', texto))
                and bool(re.search(r'\b(como|donde|puedo|mandar|enviar|pasar|adjuntar|subir|compartir)\b', texto)))

    @staticmethod
    def _precio_numero_aislado(mensaje):
        """Solo una cifra completa sin moneda; no extraer numeros de texto arbitrario."""
        entrada = mensaje.strip()
        if not re.fullmatch(r'\d{1,7}(?:[.,]\d{1,2})?', entrada):
            return None
        valor = Decimal(entrada.replace(',', '.'))
        return valor if valor > 0 else None

    @staticmethod
    def _precio_negativo_explicito(mensaje):
        """Bloquea importes negativos antes de delegar al parser legado."""
        texto = normalizar(mensaje)
        return bool(re.search(r'(?:^|\bprecio\s*|\bcuesta\s*|\bvale\s*|\$\s*)-\s*\d', mensaje.lower())) or bool(
            re.fullmatch(r'-\s*\d+(?:[.,]\d+)?(?:\s*(?:pesos|mxn))?', texto))

    def enviar_imagen(self, ruta):
        """Permite una sola propuesta a confirmar, sin sobrescribir datos ya capturados."""
        if self.estado == 'finalizado':
            self._reiniciar_solicitud()
        if self.categoria_visual_pendiente is not None or self.esperando_correccion_visual:
            return 'Primero confirma o corrige la categoria visual pendiente.'
        if self.categoria is not None:
            return ('Ya hay una categoria registrada. Para evitar reemplazarla sin permiso, '
                    'escribe cancelar y comienza una nueva solicitud antes de analizar otra imagen.')
        if not str(ruta).strip():
            return 'Indica una ruta: imagen "C:\\ruta\\portada.jpg".'
        try:
            clase_cnn = self.clasificador.predecir(Path(str(ruta).strip().strip('"')))
            categoria = CLASE_VISUAL_A_CHAT[clase_cnn]
        except KeyError:
            return 'La CNN devolvio una categoria no reconocida. No se modifico la solicitud.'
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
            return f'No se pudo clasificar la imagen: {exc}. La solicitud no cambio.'
        self.categoria_visual_pendiente = categoria
        self.activo = True
        self.estado = 'captura'
        return (f'La CNN sugiere la categoria {categoria} (etiqueta visual: {clase_cnn}). '
                'Es una estimacion, no una confirmacion. ¿Es correcta? Responde si o no.')

    def responder(self, mensaje):
        if not isinstance(mensaje, str) or not mensaje.strip():
            return 'Escribe un mensaje o usa imagen "ruta_de_portada".'
        texto = normalizar(mensaje)
        # Mantener cancelacion bajo control de la regla original, incluso durante confirmacion visual.
        if texto in {'cancelar', 'cancela', 'anular', 'detener'}:
            self.categoria_visual_pendiente = None
            self.esperando_correccion_visual = False
            self.esperando_tipo_correccion = False
            return super().responder('cancelar')
        if self._pide_nueva_solicitud(texto):
            self._reiniciar_solicitud()
            self.activo = True
            self.estado = 'captura'
            return ('Nueva solicitud iniciada. Puedes escribir imagen "C:\\ruta\\portada.jpg" '
                    'o indicar MANGAS, DC, MARVEL u OTROS.')
        if self._consulta_sobre_imagen(texto):
            if self.categoria is not None and self.estado != 'finalizado':
                return ('Si, puedes analizar una nueva portada. Ya hay una categoria registrada; '
                        'para evitar reemplazarla por accidente escribe nueva estimacion y luego '
                        'imagen "C:\\ruta\\portada.jpg". Se limpiaran categoria y precio anteriores.')
            return ('Si. Para analizarla escribe imagen "C:\\ruta\\portada.jpg". '
                    'Tambien puedes escribir nueva estimacion para iniciar otra solicitud.')
        if self._precio_negativo_explicito(mensaje) and self.activo:
            return 'El precio debe ser mayor a cero. Escribe por ejemplo $199 o 199 pesos.'
        if self.esperando_tipo_correccion:
            if texto in {'precio', 'cambiar precio', 'corregir precio', 'modificar precio'}:
                self.precio = None
                self.esperando_tipo_correccion = False
                self.estado = 'captura'
                return 'Indica el nuevo precio, por ejemplo $199 o 199.'
            if texto in {'categoria', 'cambiar categoria', 'corregir categoria'}:
                self.categoria = None
                self.esperando_tipo_correccion = False
                self.estado = 'captura'
                return 'Indica la categoria correcta: MANGAS, DC, MARVEL u OTROS.'
            if categoria_en(mensaje):
                self.categoria = categoria_en(mensaje)
                self.esperando_tipo_correccion = False
                self.estado = 'captura'
                return super().responder('quiero generar publicidad')
            precio = self._precio_numero_aislado(mensaje)
            if precio is not None or re.fullmatch(r'\s*\$\s*\d{1,7}(?:[.,]\d{1,2})?\s*', mensaje):
                self.esperando_tipo_correccion = False
                self.estado = 'captura'
                self.precio = None
                return super().responder('precio ' + mensaje.strip())
            return '¿Que deseas corregir: categoria o precio? Tambien puedes escribir directamente el nuevo valor.'
        if self.estado == 'confirmar' and texto in {'no', 'incorrecto', 'equivocado'}:
            self.esperando_tipo_correccion = True
            return '¿Que deseas corregir: categoria o precio? Tambien puedes escribir directamente el nuevo valor.'
        if (self.activo and self.estado == 'captura' and self.categoria is not None
                and self.precio is None):
            valor = self._precio_numero_aislado(mensaje)
            if valor is not None:
                return super().responder('precio ' + mensaje.strip())
            if re.fullmatch(r'0+(?:[.,]0+)?', mensaje.strip()):
                return 'El precio debe ser mayor a cero. Escribe por ejemplo $199 o 199 pesos.'
        if self.estado == 'finalizado':
            # No reciclar categoria y precio de una solicitud ya confirmada.
            if texto in {'quiero generar publicidad', 'generar publicidad',
                         'quiero hacer publicidad', 'hacer publicidad'}:
                self._reiniciar_solicitud()
                return super().responder('quiero generar publicidad')
            return ('La solicitud anterior ya esta confirmada. '
                    'Escribe nueva estimacion, quiero generar publicidad '
                    'o imagen "C:\\ruta\\portada.jpg" para empezar otra.')
        if self.categoria_visual_pendiente is not None:
            if texto in {'si', 'correcto', 'confirmo', 'esta bien'}:
                self.categoria = self.categoria_visual_pendiente
                self.categoria_visual_pendiente = None
                self.estado = 'captura'  # no confundir con confirmacion FINAL de precio
                return super().responder('quiero generar publicidad')
            if texto in {'no', 'incorrecto', 'equivocado'}:
                self.categoria_visual_pendiente = None
                self.esperando_correccion_visual = True
                return 'Indica la categoria correcta: MANGAS, DC, MARVEL u OTROS.'
            return 'Primero confirma la categoria visual con si o no, o escribe cancelar.'
        if self.esperando_correccion_visual:
            categoria = categoria_en(mensaje)
            if not categoria:
                return 'No reconoci la categoria. Escribe MANGAS, DC, MARVEL u OTROS; o cancelar.'
            self.categoria = categoria
            self.esperando_correccion_visual = False
            self.estado = 'captura'
            return super().responder('quiero generar publicidad')
        return super().responder(mensaje)


def main():
    bot = ChatbotMultimodal()
    print('Asistente multimodal experimental: texto + CNN local. No genera publicidad.')
    print('Escribe imagen "C:\\ruta\\portada.jpg" para clasificar; salir para terminar.')
    while True:
        try:
            mensaje = input('Tu: ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if normalizar(mensaje) == 'salir':
            break
        if mensaje.lower().startswith(('imagen ', '/imagen ')):
            ruta = mensaje.split(' ', 1)[1].strip()
            respuesta = bot.enviar_imagen(ruta)
        elif mensaje.lower() in ('imagen', '/imagen'):
            respuesta = 'Usa imagen "C:\\ruta\\portada.jpg".'
        else:
            respuesta = bot.responder(mensaje)
        print('Bot:', respuesta)


if __name__ == '__main__':
    main()
