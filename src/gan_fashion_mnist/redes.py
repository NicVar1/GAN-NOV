"""
redes.py
========
Aquí se DEFINEN y se CREAN las dos redes neuronales de la GAN:

    1) Red Neuronal GENERADORA     (clase Generador     -> función crear_generador)
    2) Red Neuronal DISCRIMINADORA (clase Discriminador -> función crear_discriminador)

Idea general de una GAN (Generative Adversarial Network):
    - El GENERADOR G recibe ruido aleatorio z y produce una imagen falsa G(z).
    - El DISCRIMINADOR D recibe una imagen y devuelve la probabilidad de que sea REAL.
    - Ambos compiten: G intenta engañar a D y D intenta no dejarse engañar.
      Al final G aprende a producir imágenes que parecen de la categoría real.

La arquitectura usada es una DCGAN (Deep Convolutional GAN) pequeña,
adaptada a imágenes de 28x28.
"""

import copy

import torch
from torch import nn

from .configuracion import Configuracion


# ============================================================================
# 1) RED NEURONAL GENERADORA (G)
# ============================================================================
#   Entrada : vector de ruido z de tamaño [lote, 100]
#   Salida  : imagen falsa de tamaño       [lote, 1, 28, 28] con valores en [-1, 1]
#
#   z(100) -> Linear -> 128 x 7 x 7 -> ConvTranspose -> 64 x 14 x 14
#          -> ConvTranspose -> 1 x 28 x 28 -> Tanh
#
#   La ConvTranspose2d (convolución transpuesta) hace lo contrario de una
#   convolución normal: AUMENTA el tamaño espacial (7 -> 14 -> 28). Así el
#   Generador "pinta" la imagen de lo general a lo detallado.
# ============================================================================
class Generador(nn.Module):
    def __init__(self, dim_latente: int = 100, canales_base: int = 64, canales_imagen: int = 1):
        super().__init__()
        self.dim_latente = dim_latente
        c = canales_base

        self.red = nn.Sequential(
            # --- Capa 1: proyección del ruido -------------------------------
            # Lleva el vector de 100 números a 128*7*7 = 6272 números.
            # bias=False porque la BatchNorm siguiente ya tiene su propio desplazamiento.
            nn.Linear(dim_latente, c * 2 * 7 * 7, bias=False),
            # BatchNorm: normaliza las activaciones del lote (media 0, varianza 1)
            # y estabiliza el entrenamiento. Aprende 2 parámetros por neurona (gamma, beta).
            nn.BatchNorm1d(c * 2 * 7 * 7),
            # ReLU: deja pasar solo valores positivos (no linealidad).
            nn.ReLU(inplace=True),
            # Convierte el vector plano de 6272 en un "mini mapa" de 128 canales de 7x7.
            nn.Unflatten(1, (c * 2, 7, 7)),

            # --- Capa 2: 128x7x7 -> 64x14x14 -------------------------------
            # kernel 4, stride 2, padding 1  =>  duplica alto y ancho.
            nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),

            # --- Capa 3 (salida): 64x14x14 -> 1x28x28 ----------------------
            nn.ConvTranspose2d(c, canales_imagen, kernel_size=4, stride=2, padding=1),
            # Tanh deja cada píxel en [-1, 1], igual que las imágenes reales normalizadas.
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Pasada hacia adelante: ruido -> imagen falsa
        return self.red(z)


# ============================================================================
# 2) RED NEURONAL DISCRIMINADORA (D)
# ============================================================================
#   Entrada : imagen (real o falsa) [lote, 1, 28, 28]
#   Salida  : probabilidad de ser REAL [lote, 1] con valores en (0, 1)
#
#   1x28x28 -> Conv -> 64x14x14 -> Conv -> 128x7x7 -> Flatten -> Linear -> Sigmoid
#
#   Es un clasificador binario convolucional: las convoluciones con stride 2
#   REDUCEN el tamaño (28 -> 14 -> 7) mientras extraen bordes, texturas y formas.
# ============================================================================
class Discriminador(nn.Module):
    def __init__(self, canales_base: int = 64, canales_imagen: int = 1,
                 pendiente_leaky: float = 0.2, dropout: float = 0.3):
        super().__init__()
        c = canales_base

        self.red = nn.Sequential(
            # --- Capa 1: 1x28x28 -> 64x14x14 -------------------------------
            # 64 filtros de 4x4 que recorren la imagen saltando de 2 en 2.
            # (En la primera capa de D no se usa BatchNorm, según DCGAN.)
            nn.Conv2d(canales_imagen, c, kernel_size=4, stride=2, padding=1),
            # LeakyReLU: como ReLU pero deja pasar un poco de los negativos (0.2*x).
            # Evita "neuronas muertas" y da gradientes más útiles al Generador.
            nn.LeakyReLU(pendiente_leaky, inplace=True),
            # Dropout: apaga neuronas al azar durante el entrenamiento para que
            # D no memorice y no se vuelva demasiado fuerte frente a G.
            nn.Dropout(dropout),

            # --- Capa 2: 64x14x14 -> 128x7x7 -------------------------------
            nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(pendiente_leaky, inplace=True),
            nn.Dropout(dropout),

            # --- Capa 3 (salida): 128x7x7 -> 1 número ----------------------
            nn.Flatten(),                       # 128*7*7 = 6272 valores en fila
            nn.Linear(c * 2 * 7 * 7, 1),        # Un solo "puntaje"
            # Sigmoid convierte el puntaje en probabilidad: ~1 = real, ~0 = falsa.
            nn.Sigmoid(),
        )

    def forward(self, imagen: torch.Tensor) -> torch.Tensor:
        # Pasada hacia adelante: imagen -> probabilidad de ser real
        return self.red(imagen)


# ============================================================================
# Inicialización de pesos (recomendación del paper DCGAN, Radford et al. 2015)
#   - Pesos de capas Conv / ConvTranspose / Linear ~ Normal(media=0, desv=0.02)
#   - BatchNorm: gamma ~ Normal(1, 0.02), beta = 0
# ============================================================================
def inicializar_pesos(modulo: nn.Module) -> None:
    if isinstance(modulo, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
        if modulo.bias is not None:
            nn.init.zeros_(modulo.bias)
    elif isinstance(modulo, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(modulo.weight, mean=1.0, std=0.02)
        nn.init.zeros_(modulo.bias)


# ############################################################################
# >>>>>>>>>>>>>>  AQUÍ SE CREA LA RED NEURONAL GENERADORA  <<<<<<<<<<<<<<<<<<<
# ############################################################################
def crear_generador(cfg: Configuracion) -> Generador:
    generador = Generador(
        dim_latente=cfg.dim_latente,
        canales_base=cfg.canales_base_g,
        canales_imagen=cfg.canales_imagen,
    )
    generador.apply(inicializar_pesos)        # Pesos iniciales aleatorios pequeños
    return generador.to(cfg.dispositivo)      # Se envía a CPU o GPU


# ############################################################################
# >>>>>>>>>>>>  AQUÍ SE CREA LA RED NEURONAL DISCRIMINADORA  <<<<<<<<<<<<<<<<<
# ############################################################################
def crear_discriminador(cfg: Configuracion) -> Discriminador:
    discriminador = Discriminador(
        canales_base=cfg.canales_base_d,
        canales_imagen=cfg.canales_imagen,
        pendiente_leaky=cfg.pendiente_leaky,
        dropout=cfg.dropout_d,
    )
    discriminador.apply(inicializar_pesos)
    return discriminador.to(cfg.dispositivo)


def discriminador_para_evaluar(discriminador: Discriminador) -> Discriminador:
    """
    Copia del Discriminador lista para PUNTUAR imágenes después del entrenamiento.

    Durante el entrenamiento D siempre vio lotes "solo reales" o "solo falsos" y su
    BatchNorm normalizó con las estadísticas DE ESE LOTE. Si usamos .eval(), BatchNorm
    cambia a promedios acumulados (mezcla de reales y falsas) y los puntajes se
    distorsionan. Por eso se deja BatchNorm igual que en entrenamiento y solo se
    apaga el Dropout. Se trabaja sobre una copia para no alterar el modelo guardado.
    """
    copia = copy.deepcopy(discriminador).train()
    for modulo in copia.modules():
        if isinstance(modulo, nn.Dropout):
            modulo.eval()
    return copia
