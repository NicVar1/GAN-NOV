"""
generar.py
==========
Usa la red Generadora YA ENTRENADA para crear imágenes nuevas de la categoría.

    uv run gan-generar                    (categoría 7, 64 imágenes)
    uv run gan-generar --categoria 1 --cantidad 32
"""

import argparse
import sys
from pathlib import Path

import torch

from .configuracion import CATEGORIAS, Configuracion
from .redes import crear_discriminador, crear_generador, discriminador_para_evaluar
from . import visualizacion as vis


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    analizador = argparse.ArgumentParser(description="Genera imágenes con un Generador entrenado")
    analizador.add_argument("--categoria", type=int, default=7, choices=range(10))
    analizador.add_argument("--cantidad", type=int, default=64)
    analizador.add_argument("--semilla", type=int, default=None, help="Sin semilla = imágenes distintas cada vez")
    args = analizador.parse_args()

    cfg = Configuracion(categoria=args.categoria)
    carpeta = Path(cfg.carpeta_salidas) / f"categoria_{cfg.categoria}"
    ruta_g = carpeta / "generador.pth"
    if not ruta_g.exists():
        sys.exit(f"No existe {ruta_g}. Primero ejecute: uv run gan-entrenar --categoria {cfg.categoria}")

    if args.semilla is not None:
        torch.manual_seed(args.semilla)

    # Se vuelven a crear las redes con la misma arquitectura y se cargan los pesos aprendidos
    punto_control = torch.load(ruta_g, map_location=cfg.dispositivo)
    cfg.epocas = punto_control["config"]["epocas"]
    generador = crear_generador(cfg)
    generador.load_state_dict(punto_control["estado"])
    generador.eval()

    discriminador = crear_discriminador(cfg)
    discriminador.load_state_dict(torch.load(carpeta / "discriminador.pth", map_location=cfg.dispositivo)["estado"])
    juez = discriminador_para_evaluar(discriminador)   # D sin Dropout (ver redes.py)

    with torch.no_grad():
        z = torch.randn(args.cantidad, cfg.dim_latente, device=cfg.dispositivo)  # Ruido nuevo
        imagenes = generador(z).cpu()
        puntajes = juez(imagenes.to(cfg.dispositivo)).cpu().flatten()

    ruta = carpeta / "nuevas_generadas.png"
    vis.guardar_resultado(imagenes, CATEGORIAS[cfg.categoria], cfg.epocas, ruta)
    vis.guardar_con_puntajes(imagenes, puntajes, CATEGORIAS[cfg.categoria], carpeta / "nuevas_con_puntajes.png")
    print(f"Categoría: {CATEGORIAS[cfg.categoria]}")
    print(f"D(G(z)) promedio de las nuevas imágenes: {puntajes.mean():.4f}")
    print(f"Guardado en: {ruta.resolve()}")
