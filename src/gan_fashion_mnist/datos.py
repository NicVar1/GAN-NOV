"""
datos.py
========
Carga Fashion-MNIST y se queda SOLO con las imágenes de la categoría elegida.
Esas imágenes son los ejemplos "reales" que verá el Discriminador.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .configuracion import Configuracion


def crear_cargador(cfg: Configuracion) -> tuple[DataLoader, int]:
    """Devuelve (DataLoader con imágenes reales de la categoría, cantidad de imágenes)."""

    # Transformaciones aplicadas a cada imagen:
    #  1) ToTensor   -> convierte la imagen a tensor con valores en [0, 1]
    #  2) Normalize  -> (x - 0.5) / 0.5 lleva los valores a [-1, 1]
    # Se usa [-1, 1] porque la última capa del Generador es Tanh, que produce
    # justamente valores en [-1, 1]. Reales y falsas quedan en la misma escala.
    transformacion = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    # Descarga (solo la primera vez) el conjunto de entrenamiento: 60 000 imágenes.
    dataset = datasets.FashionMNIST(
        root=cfg.carpeta_datos, train=True, download=True, transform=transformacion
    )

    # Filtra por la categoría elegida (~6 000 imágenes por clase).
    indices = torch.nonzero(dataset.targets == cfg.categoria).flatten().tolist()
    subconjunto = Subset(dataset, indices)

    cargador = DataLoader(
        subconjunto,
        batch_size=cfg.tam_lote,
        shuffle=True,      # Mezcla el orden en cada época
        drop_last=True,    # Descarta el último lote incompleto (BatchNorm prefiere lotes completos)
        num_workers=0,     # 0 = sin procesos extra (lo más seguro en Windows)
    )
    return cargador, len(subconjunto)
