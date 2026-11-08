"""
visualizacion.py
================
Gráficas y figuras que se guardan en la carpeta de salidas:

  01_muestras_reales.png        -> ejemplos reales de la categoría usada
  02_funciones_perdida.png      -> pérdida de D y de G durante el entrenamiento
  03_salidas_discriminador.png  -> D(x) y D(G(z)) a lo largo del entrenamiento
  04_histogramas_pesos_G.png    -> distribución de pesos del Generador (antes / después)
  05_histogramas_pesos_D.png    -> distribución de pesos del Discriminador (antes / después)
  06_evolucion_generador.png    -> lo que dibuja G en distintas épocas (mismo ruido)
  07_resultado_gan.png          -> RESULTADO FINAL de la GAN (64 imágenes generadas)
  08_real_vs_generado.png       -> comparación lado a lado
  09_puntajes_discriminador.png -> imágenes generadas con el valor D(G(z)) de cada una
  evolucion.gif                 -> animación del aprendizaje
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Backend sin ventana: solo guarda archivos
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid


def _a_grilla(imagenes: torch.Tensor, filas: int = 8) -> np.ndarray:
    """Convierte un lote [N,1,28,28] en [-1,1] a una imagen de grilla en [0,1]."""
    grilla = make_grid(imagenes, nrow=filas, padding=2, normalize=True, value_range=(-1, 1))
    return grilla.permute(1, 2, 0).numpy()


def _suavizar(valores, ventana: int = 25) -> np.ndarray:
    """Media móvil para que las curvas por lote se lean mejor."""
    v = np.asarray(valores)
    if len(v) < ventana:
        return v
    return np.convolve(v, np.ones(ventana) / ventana, mode="valid")


def guardar_muestras_reales(imagenes: torch.Tensor, categoria: str, ruta: Path) -> None:
    plt.figure(figsize=(7, 7))
    plt.imshow(_a_grilla(imagenes))
    plt.title(f"Imágenes REALES - Categoría: {categoria}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(ruta, dpi=120)
    plt.close()


def graficar_perdidas(hist_iter: dict, hist_epoca: list[dict], categoria: str, ruta: Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5))

    # Izquierda: por lote (crudo + suavizado)
    ax = ejes[0]
    ax.plot(hist_iter["perdida_d"], color="tab:red", alpha=0.2)
    ax.plot(hist_iter["perdida_g"], color="tab:blue", alpha=0.2)
    ax.plot(_suavizar(hist_iter["perdida_d"]), color="tab:red", label="Pérdida Discriminador")
    ax.plot(_suavizar(hist_iter["perdida_g"]), color="tab:blue", label="Pérdida Generador")
    ax.axhline(np.log(4), color="tab:red", ls="--", lw=0.8, label="Equilibrio D = 2·ln2 ≈ 1.386")
    ax.axhline(np.log(2), color="tab:blue", ls="--", lw=0.8, label="Equilibrio G = ln2 ≈ 0.693")
    ax.set_title("Funciones de pérdida (por lote)")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("BCE")
    ax.legend()
    ax.grid(alpha=0.3)

    # Derecha: promedios por época, con D separada en real y falsa
    ax = ejes[1]
    ep = [h["epoca"] for h in hist_epoca]
    ax.plot(ep, [h["perdida_d"] for h in hist_epoca], "o-", color="tab:red", label="Pérdida D total")
    ax.plot(ep, [h["perdida_d_real"] for h in hist_epoca], "--", color="salmon", label="Pérdida D (reales)")
    ax.plot(ep, [h["perdida_d_falsa"] for h in hist_epoca], ":", color="darkred", label="Pérdida D (falsas)")
    ax.plot(ep, [h["perdida_g"] for h in hist_epoca], "o-", color="tab:blue", label="Pérdida G")
    ax.set_title("Funciones de pérdida (promedio por época)")
    ax.set_xlabel("Época")
    ax.set_ylabel("BCE")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"GAN - Categoría: {categoria}")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def graficar_salidas_discriminador(hist_iter: dict, hist_epoca: list[dict], categoria: str, ruta: Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5))

    ax = ejes[0]
    ax.plot(_suavizar(hist_iter["d_x"]), color="tab:green", label="D(x)  - imágenes reales")
    ax.plot(_suavizar(hist_iter["d_g_z"]), color="tab:orange", label="D(G(z)) - imágenes generadas")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="0.5 = D no distingue")
    ax.set_ylim(0, 1)
    ax.set_title("Salida del Discriminador (por lote, suavizado)")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Probabilidad de 'real'")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = ejes[1]
    ep = [h["epoca"] for h in hist_epoca]
    ax.plot(ep, [h["d_x"] for h in hist_epoca], "o-", color="tab:green", label="D(x)")
    ax.plot(ep, [h["d_g_z_antes"] for h in hist_epoca], "o-", color="tab:orange", label="D(G(z)) antes de actualizar D")
    ax.plot(ep, [h["d_g_z_despues"] for h in hist_epoca], "s--", color="tab:purple", label="D(G(z)) después de actualizar D")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_title("Salida del Discriminador (promedio por época)")
    ax.set_xlabel("Época")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"Valores del Discriminador - Categoría: {categoria}")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def graficar_histogramas_pesos(antes: dict, despues: dict, nombre_red: str, ruta: Path) -> None:
    """Un histograma por capa con pesos: antes (gris) vs después (color) del entrenamiento."""
    n = len(antes)
    fig, ejes = plt.subplots(1, n, figsize=(5 * n, 4))
    ejes = np.atleast_1d(ejes)
    for ax, (nombre, w_antes) in zip(ejes, antes.items()):
        w_despues = despues[nombre]
        ax.hist(w_antes.numpy(), bins=80, alpha=0.5, color="gray", label="Antes (N(0, 0.02))", density=True)
        ax.hist(w_despues.numpy(), bins=80, alpha=0.6, color="tab:blue", label="Después", density=True)
        ax.set_title(f"{nombre}\n{w_antes.numel():,} pesos")
        ax.set_xlabel("Valor del peso")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Distribución de los pesos - {nombre_red}")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def graficar_evolucion(evolucion: list[torch.Tensor], categoria: str, ruta: Path, columnas: int = 5) -> None:
    """Muestra 16 imágenes del ruido fijo en algunas épocas representativas."""
    total = len(evolucion)
    elegidas = sorted(set(np.linspace(0, total - 1, min(10, total)).round().astype(int).tolist()))
    filas = int(np.ceil(len(elegidas) / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(3.2 * columnas, 3.4 * filas))
    ejes = np.atleast_1d(ejes).flatten()
    for ax in ejes:
        ax.axis("off")
    for ax, idx in zip(ejes, elegidas):
        ax.imshow(_a_grilla(evolucion[idx][:16], filas=4))
        ax.set_title(f"Época {idx + 1}")
    fig.suptitle(f"Evolución del Generador (mismo ruido z) - Categoría: {categoria}")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def guardar_resultado(imagenes: torch.Tensor, categoria: str, epocas: int, ruta: Path) -> None:
    plt.figure(figsize=(8, 8.4))
    plt.imshow(_a_grilla(imagenes))
    plt.title(f"RESULTADO DE LA GAN - Categoría usada: {categoria}\n"
              f"64 imágenes generadas por la red Generadora tras {epocas} épocas", fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(ruta, dpi=130)
    plt.close()


def guardar_real_vs_generado(reales: torch.Tensor, falsas: torch.Tensor, d_real: float, d_falsa: float,
                             categoria: str, ruta: Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(12, 6.4))
    ejes[0].imshow(_a_grilla(reales[:36], filas=6))
    ejes[0].set_title(f"REALES (Fashion-MNIST)\nD(x) promedio = {d_real:.3f}")
    ejes[1].imshow(_a_grilla(falsas[:36], filas=6))
    ejes[1].set_title(f"GENERADAS por G\nD(G(z)) promedio = {d_falsa:.3f}")
    for ax in ejes:
        ax.axis("off")
    fig.suptitle(f"Real vs Generado - Categoría: {categoria}")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def guardar_con_puntajes(imagenes: torch.Tensor, puntajes: torch.Tensor, categoria: str, ruta: Path) -> None:
    """Cada imagen generada con la probabilidad de 'real' que le asigna el Discriminador."""
    n = min(24, imagenes.size(0))
    fig, ejes = plt.subplots(3, 8, figsize=(14, 7.4))
    for i, ax in enumerate(ejes.flatten()):
        ax.axis("off")
        if i < n:
            img = (imagenes[i, 0].numpy() + 1) / 2
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            p = puntajes[i].item()
            ax.set_title(f"D = {p:.2f}", fontsize=10, color="tab:green" if p >= 0.5 else "tab:red")
    fig.suptitle(f"Imágenes generadas y su puntaje en el Discriminador (verde ≥ 0.5 = 'parece real')\n"
                 f"Categoría: {categoria}")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def guardar_gif(evolucion: list[torch.Tensor], ruta: Path) -> None:
    cuadros = []
    for lote in evolucion:
        arr = (_a_grilla(lote[:64]) * 255).astype(np.uint8)
        cuadros.append(Image.fromarray(arr).resize((arr.shape[1] * 2, arr.shape[0] * 2), Image.NEAREST))
    cuadros[0].save(ruta, save_all=True, append_images=cuadros[1:], duration=250, loop=0)
