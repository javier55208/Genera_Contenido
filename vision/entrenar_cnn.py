"""Entrenamiento local EfficientNet-B0: clasificador 3 clases, CPU, transferencia.

Descarga de pesos solamente si se usa --autorizar-descarga-pesos.
El conjunto de prueba se evalua una sola vez tras seleccionar por validacion.
"""
import argparse
import copy
import csv
import json
import random
from pathlib import Path

CLASES = ('MANGAS', 'DC_COMICS', 'MARVEL_COMICS')


def cargar_datos(manifiesto, raiz):
    registros = {'train': [], 'val': [], 'test': []}
    with manifiesto.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['split'] not in registros or r['clase'] not in CLASES:
                raise ValueError('Split/clase desconocida en manifiesto')
            ruta = (raiz / r['ruta_relativa']).resolve()
            if not ruta.is_relative_to(raiz.resolve()) or not ruta.is_file():
                raise FileNotFoundError(ruta)
            registros[r['split']].append((ruta, CLASES.index(r['clase'])))
    for split, items in registros.items():
        if not items:
            raise ValueError(f'Split vacio: {split}')
    return registros


def entrenar(args):
    # Importaciones diferidas: preparar dataset no necesita paquetes de deep learning.
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    from PIL import Image, ImageOps
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    if args.epocas < 1 or args.batch < 1:
        raise ValueError('epocas y batch deben ser positivos')
    random.seed(args.semilla)
    torch.manual_seed(args.semilla)
    raiz = args.dataset.resolve()
    registros = cargar_datos(args.manifiesto, raiz)
    pesos = EfficientNet_B0_Weights.DEFAULT
    # Politica explicita: sin este indicador, NO se descargan pesos.
    if not args.autorizar_descarga_pesos:
        raise SystemExit('Se requiere --autorizar-descarga-pesos para usar pesos ImageNet. '
                         'Obtén tu aprobacion para esta descarga antes de ejecutar.')
    print('Carga de pesos oficiales EfficientNet-B0; pueden descargarse una sola vez de download.pytorch.org.')
    modelo = efficientnet_b0(weights=pesos)
    for p in modelo.features.parameters():
        p.requires_grad = False
    entrada = modelo.classifier[1].in_features
    modelo.classifier[1] = nn.Linear(entrada, len(CLASES))
    # CPU intencionadamente; no se requiere software para aceleracion AMD.
    dispositivo = torch.device('cpu')
    modelo.to(dispositivo)
    normalizar = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    evaluar_transform = pesos.transforms()
    entrenar_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(), normalizar,
    ])

    class Portadas(Dataset):
        def __init__(self, items, transform):
            self.items, self.transform = items, transform
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            archivo, clase = self.items[idx]
            with Image.open(archivo) as imagen:
                imagen = ImageOps.exif_transpose(imagen).convert('RGB')
                tensor = self.transform(imagen)
            return tensor, clase

    loaders = {
        k: DataLoader(Portadas(v, entrenar_transform if k == 'train' else evaluar_transform),
                      batch_size=args.batch, shuffle=(k == 'train'), num_workers=0)
        for k, v in registros.items()
    }
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.classifier.parameters(), lr=args.lr)
    mejor = -1.0
    mejor_estado = None
    historial = []
    for epoca in range(1, args.epocas + 1):
        fila = {'epoca': epoca}
        for split in ('train', 'val'):
            modelo.train(split == 'train')
            # La base congelada conserva BatchNorm preentrenado en modo eval.
            modelo.features.eval()
            aciertos, n, perdida = 0, 0, 0.0
            contexto = torch.enable_grad() if split == 'train' else torch.no_grad()
            with contexto:
                for imagenes, etiquetas in loaders[split]:
                    imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
                    if split == 'train':
                        optimizador.zero_grad()
                    logits = modelo(imagenes)
                    loss = criterio(logits, etiquetas)
                    if split == 'train':
                        loss.backward()
                        optimizador.step()
                    n += len(etiquetas)
                    aciertos += (logits.argmax(1) == etiquetas).sum().item()
                    perdida += loss.item() * len(etiquetas)
            fila[f'{split}_accuracy'] = round(aciertos/n, 5)
            fila[f'{split}_loss'] = round(perdida/n, 5)
        historial.append(fila)
        print(f'Epoca {epoca}/{args.epocas}: train={fila["train_accuracy"]:.3f}, val={fila["val_accuracy"]:.3f}')
        if fila['val_accuracy'] > mejor:
            mejor = fila['val_accuracy']
            mejor_estado = copy.deepcopy(modelo.state_dict())
    modelo.load_state_dict(mejor_estado)
    modelo.eval()
    reales, predichos = [], []
    with torch.no_grad():
        for imagenes, etiquetas in loaders['test']:
            logits = modelo(imagenes.to(dispositivo))
            reales.extend(etiquetas.tolist())
            predichos.extend(logits.argmax(1).cpu().tolist())
    reporte = classification_report(reales, predichos, labels=list(range(3)), target_names=CLASES,
                                    output_dict=True, zero_division=0)
    matriz = confusion_matrix(reales, predichos, labels=list(range(3)))
    salida = args.salida.resolve()
    salida.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': modelo.state_dict(), 'clases': list(CLASES),
                'arquitectura': 'efficientnet_b0', 'semilla': args.semilla}, salida / 'mejor_modelo_cnn.pth')
    with (salida / 'historial_entrenamiento.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(historial[0])); w.writeheader(); w.writerows(historial)
    with (salida / 'matriz_confusion_cnn.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['real/predicha', *CLASES]);
        for clase, fila in zip(CLASES, matriz): w.writerow([clase, *fila.tolist()])
    metricas = {'accuracy': accuracy_score(reales, predichos),
                'precision_macro': reporte['macro avg']['precision'],
                'recall_macro': reporte['macro avg']['recall'],
                'f1_macro': reporte['macro avg']['f1-score'],
                'mejor_accuracy_validacion': mejor,
                'n_train': len(registros['train']), 'n_val': len(registros['val']), 'n_test': len(registros['test']),
                'reporte_por_clase': reporte}
    (salida / 'metricas_cnn.json').write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Evaluacion independiente del conjunto de prueba:')
    print(json.dumps({k: v for k,v in metricas.items() if k != 'reporte_por_clase'}, indent=2))
    print('Resultados guardados en:', salida)
    print('ATENCION: no publiques pesos, particiones con rutas privadas o imagenes en GitHub.')


def main():
    p = argparse.ArgumentParser(description='Entrenar EfficientNet-B0 con CPU')
    p.add_argument('--dataset', type=Path, required=True, help='Carpeta Dataset_IA (con imagenes/)')
    p.add_argument('--manifiesto', type=Path, default=Path(__file__).resolve().parent / 'resultados/particion_visual.csv')
    p.add_argument('--salida', type=Path, default=Path(__file__).resolve().parent / 'resultados/cnn')
    p.add_argument('--epocas', type=int, default=5)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--semilla', type=int, default=42)
    p.add_argument('--autorizar-descarga-pesos', action='store_true', help='Permite obtener pesos ImageNet si no estan en cache')
    entrenar(p.parse_args())


if __name__ == '__main__':
    main()
