"""
perdidas.py
===========
FUNCIONES DE PÉRDIDA de la GAN.

Ambas redes usan la Entropía Cruzada Binaria (BCE, Binary Cross-Entropy):

    BCE(p, y) = -[ y * log(p) + (1 - y) * log(1 - p) ]

    p = salida del Discriminador (probabilidad de "real")
    y = etiqueta objetivo (1 = real, 0 = falsa)

    - Si y = 1  ->  BCE = -log(p)      : es pequeña cuando p se acerca a 1
    - Si y = 0  ->  BCE = -log(1 - p)  : es pequeña cuando p se acerca a 0

Juego minimax original (Goodfellow et al., 2014):

    min_G max_D  V(D, G) = E_x[ log D(x) ] + E_z[ log(1 - D(G(z))) ]

Valor de referencia: cuando D no distingue nada, D(x) = D(G(z)) = 0.5 y
    pérdida_D = -log(0.5) - log(0.5) = 2 * 0.693 = 1.386
    pérdida_G = -log(0.5)            =     0.693
Una GAN "equilibrada" oscila alrededor de esos valores.
"""

import torch
from torch import nn

# Criterio BCE compartido. Recibe probabilidades (por eso D termina en Sigmoid).
criterio_bce = nn.BCELoss()


# ============================================================================
# PÉRDIDA DEL DISCRIMINADOR
# ============================================================================
#   pérdida_D = BCE(D(x), etiqueta_real) + BCE(D(G(z)), etiqueta_falsa)
#             = -log D(x)              -   log(1 - D(G(z)))
#
#   D quiere: D(x) -> 1 (reconocer reales) y D(G(z)) -> 0 (detectar falsas).
#   Se devuelven las dos partes por separado para poder graficarlas.
# ============================================================================
def perdida_discriminador(salida_real: torch.Tensor, salida_falsa: torch.Tensor,
                          etiqueta_real: float, etiqueta_falsa: float):
    objetivo_real = torch.full_like(salida_real, etiqueta_real)     # p.ej. 0.9 (label smoothing)
    objetivo_falso = torch.full_like(salida_falsa, etiqueta_falsa)  # 0.0

    perdida_real = criterio_bce(salida_real, objetivo_real)     # ¿Qué tan bien reconoce reales?
    perdida_falsa = criterio_bce(salida_falsa, objetivo_falso)  # ¿Qué tan bien detecta falsas?
    perdida_total = perdida_real + perdida_falsa
    return perdida_total, perdida_real, perdida_falsa


# ============================================================================
# PÉRDIDA DEL GENERADOR (versión "no saturante")
# ============================================================================
#   pérdida_G = BCE(D(G(z)), 1) = -log D(G(z))
#
#   G quiere que D crea que sus imágenes son reales: D(G(z)) -> 1.
#   En vez de minimizar log(1 - D(G(z))) (fórmula original, que casi no da
#   gradiente al inicio cuando D gana fácilmente) se maximiza log D(G(z)).
#   Por eso el objetivo es la etiqueta 1 aunque las imágenes sean falsas.
# ============================================================================
def perdida_generador(salida_falsa_para_g: torch.Tensor) -> torch.Tensor:
    objetivo = torch.ones_like(salida_falsa_para_g)   # "Haz que D diga REAL"
    return criterio_bce(salida_falsa_para_g, objetivo)
