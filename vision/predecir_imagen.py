"""Inferencia local CNN EfficientNet-B0 del Avance 2.

Solo lee imagenes locales y el checkpoint creado por vision/entrenar_cnn.py.
No descarga modelos ni envia datos a servicios externos.
"""
from pathlib import Path

CLASES_ESPERADAS = ('MANGAS', 'DC_COMICS', 'MARVEL_COMICS')
EXTENSIONES_PERMITIDAS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


class ClasificadorPortadas:
    def __init__(self, modelo_path=None):
        self.modelo_path = Path(modelo_path) if modelo_path else Path(__file__).resolve().parent / 'resultados' / 'cnn' / 'mejor_modelo_cnn.pth'
        self.modelo = None
        self.transformacion = None
        self.clases = None

    def cargar(self):
        """Carga diferida: ningun paquete pesado se importa hasta la primera imagen."""
        if self.modelo is not None:
            return
        if not self.modelo_path.is_file():
            raise FileNotFoundError(f'No se encuentra el modelo CNN: {self.modelo_path}')
        import torch
        from torch import nn
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        # Solo checkpoint propio de confianza. weights_only reduce superficie de deserializacion.
        ck = torch.load(self.modelo_path, map_location='cpu', weights_only=True)
        if ck.get('arquitectura') != 'efficientnet_b0':
            raise ValueError('Checkpoint con arquitectura desconocida.')
        clases = tuple(ck.get('clases', []))
        if clases != CLASES_ESPERADAS:
            raise ValueError('Las clases o su orden no coinciden con el entrenamiento.')
        modelo = efficientnet_b0(weights=None)  # Nunca se descarga aqui.
        modelo.classifier[1] = nn.Linear(modelo.classifier[1].in_features, len(clases))
        modelo.load_state_dict(ck['state_dict'], strict=True)
        modelo.eval()
        self.modelo = modelo
        self.transformacion = EfficientNet_B0_Weights.DEFAULT.transforms()
        self.clases = clases

    def predecir(self, ruta_imagen):
        ruta = Path(ruta_imagen).expanduser()
        if not ruta.is_file():
            raise FileNotFoundError(f'No se encuentra la imagen: {ruta}')
        if ruta.suffix.lower() not in EXTENSIONES_PERMITIDAS:
            raise ValueError('Formato no admitido: usa JPG, JPEG, PNG, WEBP o BMP.')
        self.cargar()
        import torch
        from PIL import Image, ImageOps, UnidentifiedImageError
        try:
            with Image.open(ruta) as im:
                imagen = ImageOps.exif_transpose(im).convert('RGB')
                tensor = self.transformacion(imagen).unsqueeze(0)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError('No se pudo leer la imagen; verifica que no este danada.') from exc
        with torch.inference_mode():
            logits = self.modelo(tensor)
            indice = int(logits.argmax(dim=1).item())
        return self.clases[indice]
