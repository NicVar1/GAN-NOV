"""
entrenamiento.py
================
Bucle de entrenamiento ADVERSARIAL de la GAN.

En cada lote se hacen dos pasos:

  PASO 1 - Entrenar el DISCRIMINADOR (G queda congelado)
      a) D ve imágenes REALES          -> debe decir "real"   (D(x) alto)
      b) D ve imágenes FALSAS de G     -> debe decir "falsa"  (D(G(z)) bajo)
      c) pérdida_D = BCE(D(x), 0.9) + BCE(D(G(z)), 0) ; se actualizan SOLO los pesos de D

  PASO 2 - Entrenar el GENERADOR (D queda fijo)
      a) G genera imágenes y D las evalúa
      b) pérdida_G = BCE(D(G(z)), 1)  -> G quiere que D diga "real"
      c) se actualizan SOLO los pesos de G
"""

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .configuracion import Configuracion
from .perdidas import perdida_discriminador, perdida_generador


def entrenar(generador: nn.Module, discriminador: nn.Module, cargador: DataLoader,
             optim_g: torch.optim.Optimizer, optim_d: torch.optim.Optimizer,
             cfg: Configuracion, ruido_fijo: torch.Tensor):
    """
    Entrena la GAN y devuelve:
      historial_iter  : métricas de cada lote (para curvas detalladas)
      historial_epoca : promedios por época (para tablas / CSV)
      evolucion       : imágenes generadas con el MISMO ruido al final de cada época
    """
    disp = cfg.dispositivo
    historial_iter = {"perdida_d": [], "perdida_g": [], "d_x": [], "d_g_z": []}
    historial_epoca = []
    evolucion = []

    print("\n" + "=" * 100)
    print(f" ENTRENAMIENTO  |  épocas={cfg.epocas}  lotes/época={len(cargador)}  dispositivo={disp}")
    print("=" * 100)
    print(f" {'Época':>6} {'Pérd. D':>9} {'D real':>8} {'D falsa':>8} {'Pérd. G':>9}"
          f" {'D(x)':>7} {'D(G(z))1':>9} {'D(G(z))2':>9} {'Tiempo':>8}")
    print("-" * 100)

    for epoca in range(1, cfg.epocas + 1):
        inicio = time.time()
        acum = {"perdida_d": 0.0, "perdida_d_real": 0.0, "perdida_d_falsa": 0.0,
                "perdida_g": 0.0, "d_x": 0.0, "d_g_z_antes": 0.0, "d_g_z_despues": 0.0}

        generador.train()
        discriminador.train()

        for imagenes_reales, _ in cargador:
            imagenes_reales = imagenes_reales.to(disp)
            lote = imagenes_reales.size(0)

            # ==============================================================
            # PASO 1: ACTUALIZAR EL DISCRIMINADOR
            # ==============================================================
            optim_d.zero_grad()

            # (a) Imágenes reales -> D(x)
            salida_real = discriminador(imagenes_reales)

            # (b) Imágenes falsas: ruido z ~ N(0,1) -> G(z) -> D(G(z))
            z = torch.randn(lote, cfg.dim_latente, device=disp)
            imagenes_falsas = generador(z)
            # .detach() corta el grafo: al entrenar D NO se calculan gradientes para G
            salida_falsa = discriminador(imagenes_falsas.detach())

            # (c) Pérdida de D y retropropagación
            perdida_d, perdida_d_real, perdida_d_falsa = perdida_discriminador(
                salida_real, salida_falsa, cfg.etiqueta_real, cfg.etiqueta_falsa)
            perdida_d.backward()
            optim_d.step()

            # ==============================================================
            # PASO 2: ACTUALIZAR EL GENERADOR
            # ==============================================================
            optim_g.zero_grad()

            # D (ya actualizado) vuelve a evaluar las imágenes falsas, esta vez SIN
            # detach, para que el gradiente fluya desde D hasta los pesos de G.
            salida_para_g = discriminador(imagenes_falsas)
            perdida_g = perdida_generador(salida_para_g)
            perdida_g.backward()
            optim_g.step()   # Solo cambia G (optim_g solo contiene parámetros de G)

            # ---------------- Registro de valores -------------------------
            valores = {
                "perdida_d": perdida_d.item(),
                "perdida_d_real": perdida_d_real.item(),
                "perdida_d_falsa": perdida_d_falsa.item(),
                "perdida_g": perdida_g.item(),
                "d_x": salida_real.mean().item(),            # Prob. media que D da a las reales
                "d_g_z_antes": salida_falsa.mean().item(),   # Prob. media a las falsas (antes de actualizar D)
                "d_g_z_despues": salida_para_g.mean().item(),  # Prob. media a las falsas (después de actualizar D)
            }
            for k, v in valores.items():
                acum[k] += v
            historial_iter["perdida_d"].append(valores["perdida_d"])
            historial_iter["perdida_g"].append(valores["perdida_g"])
            historial_iter["d_x"].append(valores["d_x"])
            historial_iter["d_g_z"].append(valores["d_g_z_antes"])

        # ---------------- Promedios de la época ---------------------------
        n = len(cargador)
        promedio = {k: v / n for k, v in acum.items()}
        promedio["epoca"] = epoca
        promedio["segundos"] = time.time() - inicio
        historial_epoca.append(promedio)

        # Imágenes con el ruido fijo para ver cómo mejora G época a época
        generador.eval()
        with torch.no_grad():
            evolucion.append(generador(ruido_fijo).cpu())

        print(f" {epoca:>6} {promedio['perdida_d']:>9.4f} {promedio['perdida_d_real']:>8.4f}"
              f" {promedio['perdida_d_falsa']:>8.4f} {promedio['perdida_g']:>9.4f}"
              f" {promedio['d_x']:>7.3f} {promedio['d_g_z_antes']:>9.3f}"
              f" {promedio['d_g_z_despues']:>9.3f} {promedio['segundos']:>7.1f}s")

    print("-" * 100)
    print(" D(G(z))1 = prob. que D asigna a las falsas ANTES de actualizar D")
    print(" D(G(z))2 = prob. que D asigna a las falsas DESPUÉS de actualizar D (la que usa G para aprender)")
    return historial_iter, historial_epoca, evolucion
