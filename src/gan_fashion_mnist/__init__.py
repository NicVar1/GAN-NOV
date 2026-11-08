"""
GAN explicada paso a paso sobre Fashion-MNIST
=============================================
Ejecutar:   uv run gan-entrenar                 (categoría 7 = Zapatilla deportiva)
            uv run gan-entrenar --categoria 1   (otra categoría, p.ej. Pantalón)
            uv run gan-entrenar --epocas 20

Flujo del programa (función main):
    1. Configuración e hiperparámetros
    2. Carga de datos reales (solo la categoría elegida)
    3. CREACIÓN de la red Generadora y de la red Discriminadora
    4. Parámetros de cada red (arquitectura + valores de los pesos)
    5. Funciones de pérdida y optimizadores
    6. Entrenamiento adversarial
    7. Valores finales del Generador y del Discriminador
    8. Resultado de la GAN (imágenes generadas) indicando la categoría usada
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from .configuracion import CATEGORIAS, Configuracion
from .datos import crear_cargador
from .entrenamiento import entrenar
from .inspeccion import (contar_parametros, estadisticas_pesos, imprimir_comparacion,
                         imprimir_estadisticas, pesos_aplanados, resumen_arquitectura)
from .perdidas import criterio_bce
from .redes import crear_discriminador, crear_generador, discriminador_para_evaluar
from . import visualizacion as vis


def _titulo(texto: str) -> None:
    print("\n" + "#" * 100 + f"\n#  {texto}\n" + "#" * 100)


def _leer_argumentos() -> Configuracion:
    analizador = argparse.ArgumentParser(description="Entrena una GAN (DCGAN) sobre una categoría de Fashion-MNIST")
    analizador.add_argument("--categoria", type=int, default=7, choices=range(10),
                            help="Clase de Fashion-MNIST: " + ", ".join(f"{k}={v}" for k, v in CATEGORIAS.items()))
    analizador.add_argument("--epocas", type=int, default=40)
    analizador.add_argument("--lote", type=int, default=64)
    analizador.add_argument("--semilla", type=int, default=42)
    args = analizador.parse_args()
    return Configuracion(categoria=args.categoria, epocas=args.epocas, tam_lote=args.lote, semilla=args.semilla)


def main() -> None:
    # Permite imprimir tildes y ñ en la consola de Windows
    sys.stdout.reconfigure(encoding="utf-8")

    # ======================================================================
    # 1. CONFIGURACIÓN
    # ======================================================================
    cfg = _leer_argumentos()
    salidas = Path(cfg.carpeta_salidas) / f"categoria_{cfg.categoria}"
    salidas.mkdir(parents=True, exist_ok=True)

    # Semillas: el mismo número produce siempre el mismo resultado
    random.seed(cfg.semilla)
    np.random.seed(cfg.semilla)
    torch.manual_seed(cfg.semilla)

    _titulo(f"GAN - CATEGORÍA USADA: {cfg.categoria} = {cfg.nombre_categoria}")
    print(" Hiperparámetros:")
    for clave, valor in vars(cfg).items():
        print(f"   {clave:<22} = {valor}")

    # ======================================================================
    # 2. DATOS REALES
    # ======================================================================
    cargador, n_imagenes = crear_cargador(cfg)
    reales_ejemplo, _ = next(iter(cargador))
    print(f"\n Imágenes reales de '{cfg.nombre_categoria}': {n_imagenes}  ->  {len(cargador)} lotes de {cfg.tam_lote}")
    vis.guardar_muestras_reales(reales_ejemplo, cfg.nombre_categoria, salidas / "01_muestras_reales.png")

    # ======================================================================
    # 3. CREACIÓN DE LAS DOS REDES NEURONALES
    # ======================================================================
    _titulo("3. CREACIÓN DE LAS REDES NEURONALES")

    # ----------------------------------------------------------------------
    # >>> AQUÍ SE CREA LA RED NEURONAL GENERADORA (ver redes.py: Generador) <<<
    # ----------------------------------------------------------------------
    generador = crear_generador(cfg)

    # ----------------------------------------------------------------------
    # >>> AQUÍ SE CREA LA RED NEURONAL DISCRIMINADORA (ver redes.py: Discriminador) <<<
    # ----------------------------------------------------------------------
    discriminador = crear_discriminador(cfg)

    print(" Generador creado      ->", generador.__class__.__name__)
    print(" Discriminador creado  ->", discriminador.__class__.__name__)

    # ======================================================================
    # 4. PARÁMETROS DE CADA RED
    # ======================================================================
    _titulo("4. PARÁMETROS DE LA RED GENERADORA Y DE LA RED DISCRIMINADORA")
    z_ejemplo = torch.randn(2, cfg.dim_latente, device=cfg.dispositivo)
    img_ejemplo = torch.randn(2, cfg.canales_imagen, cfg.tam_imagen, cfg.tam_imagen, device=cfg.dispositivo)

    arq_g = resumen_arquitectura(generador, z_ejemplo, "RED NEURONAL GENERADORA (G)  z -> imagen")
    arq_d = resumen_arquitectura(discriminador, img_ejemplo, "RED NEURONAL DISCRIMINADORA (D)  imagen -> P(real)")

    stats_g_antes = estadisticas_pesos(generador)
    stats_d_antes = estadisticas_pesos(discriminador)
    pesos_g_antes = pesos_aplanados(generador)
    pesos_d_antes = pesos_aplanados(discriminador)
    # Copia de los pesos iniciales (la usa el front-end para comparar antes / después)
    torch.save(generador.state_dict(), salidas / "generador_inicial.pth")
    torch.save(discriminador.state_dict(), salidas / "discriminador_inicial.pth")
    imprimir_estadisticas(stats_g_antes, "GENERADOR - ANTES de entrenar (inicialización DCGAN)")
    imprimir_estadisticas(stats_d_antes, "DISCRIMINADOR - ANTES de entrenar (inicialización DCGAN)")

    # ======================================================================
    # 5. FUNCIONES DE PÉRDIDA Y OPTIMIZADORES
    # ======================================================================
    _titulo("5. FUNCIONES DE PÉRDIDA Y OPTIMIZADORES")
    print(f" Criterio: {criterio_bce}  (Entropía Cruzada Binaria)")
    print("   BCE(p, y) = -[ y·log(p) + (1-y)·log(1-p) ]")
    print(f"   Pérdida Discriminador = BCE(D(x), {cfg.etiqueta_real}) + BCE(D(G(z)), {cfg.etiqueta_falsa})"
          "  = -log D(x) - log(1 - D(G(z)))")
    print("   Pérdida Generador     = BCE(D(G(z)), 1)  = -log D(G(z))   (versión no saturante)")
    print("   Equilibrio teórico: D(x)=D(G(z))=0.5 -> Pérdida D ≈ 1.386, Pérdida G ≈ 0.693")

    # Cada optimizador SOLO contiene los parámetros de su red:
    # así al entrenar D no se modifican los pesos de G y viceversa.
    optim_g = torch.optim.Adam(generador.parameters(), lr=cfg.tasa_aprendizaje_g, betas=(cfg.beta1, cfg.beta2))
    optim_d = torch.optim.Adam(discriminador.parameters(), lr=cfg.tasa_aprendizaje_d, betas=(cfg.beta1, cfg.beta2))
    print(f"\n Optimizador G: Adam(lr={cfg.tasa_aprendizaje_g}, betas=({cfg.beta1}, {cfg.beta2}))")
    print(f" Optimizador D: Adam(lr={cfg.tasa_aprendizaje_d}, betas=({cfg.beta1}, {cfg.beta2}))")

    # ======================================================================
    # 6. ENTRENAMIENTO
    # ======================================================================
    _titulo("6. ENTRENAMIENTO ADVERSARIAL")
    # Ruido fijo: siempre el mismo z, para comparar qué dibuja G en cada época
    ruido_fijo = torch.randn(cfg.num_muestras_fijas, cfg.dim_latente, device=cfg.dispositivo)
    hist_iter, hist_epoca, evolucion = entrenar(generador, discriminador, cargador,
                                                optim_g, optim_d, cfg, ruido_fijo)

    # ======================================================================
    # 7. VALORES DEL GENERADOR Y DEL DISCRIMINADOR DESPUÉS DE ENTRENAR
    # ======================================================================
    _titulo("7. VALORES DEL GENERADOR Y DEL DISCRIMINADOR (DESPUÉS DE ENTRENAR)")
    stats_g_despues = estadisticas_pesos(generador)
    stats_d_despues = estadisticas_pesos(discriminador)
    imprimir_estadisticas(stats_g_despues, "GENERADOR - DESPUÉS de entrenar")
    imprimir_estadisticas(stats_d_despues, "DISCRIMINADOR - DESPUÉS de entrenar")
    imprimir_comparacion(stats_g_antes, stats_g_despues, "GENERADOR")
    imprimir_comparacion(stats_d_antes, stats_d_despues, "DISCRIMINADOR")
    vis.graficar_histogramas_pesos(pesos_g_antes, pesos_aplanados(generador), "Red Generadora",
                                   salidas / "04_histogramas_pesos_G.png")
    vis.graficar_histogramas_pesos(pesos_d_antes, pesos_aplanados(discriminador), "Red Discriminadora",
                                   salidas / "05_histogramas_pesos_D.png")

    # ======================================================================
    # 8. RESULTADO DE LA GAN
    # ======================================================================
    _titulo(f"8. RESULTADO DE LA GAN - CATEGORÍA: {cfg.nombre_categoria}")
    generador.eval()
    juez = discriminador_para_evaluar(discriminador)   # D sin Dropout (ver redes.py)
    with torch.no_grad():
        falsas = generador(ruido_fijo).cpu()                                   # Resultado final de G
        puntajes_falsas = juez(falsas.to(cfg.dispositivo)).cpu().flatten()
        puntajes_reales = juez(reales_ejemplo.to(cfg.dispositivo)).cpu().flatten()

    d_real, d_falsa = puntajes_reales.mean().item(), puntajes_falsas.mean().item()
    engañadas = (puntajes_falsas >= 0.5).float().mean().item() * 100
    print(f" D(x) promedio sobre imágenes reales     : {d_real:.4f}")
    print(f" D(G(z)) promedio sobre imágenes generadas: {d_falsa:.4f}")
    print(f" Imágenes generadas que D cree reales     : {engañadas:.1f}%")

    vis.graficar_perdidas(hist_iter, hist_epoca, cfg.nombre_categoria, salidas / "02_funciones_perdida.png")
    vis.graficar_salidas_discriminador(hist_iter, hist_epoca, cfg.nombre_categoria,
                                       salidas / "03_salidas_discriminador.png")
    vis.graficar_evolucion(evolucion, cfg.nombre_categoria, salidas / "06_evolucion_generador.png")
    vis.guardar_resultado(falsas, cfg.nombre_categoria, cfg.epocas, salidas / "07_resultado_gan.png")
    vis.guardar_real_vs_generado(reales_ejemplo, falsas, d_real, d_falsa, cfg.nombre_categoria,
                                 salidas / "08_real_vs_generado.png")
    vis.guardar_con_puntajes(falsas, puntajes_falsas, cfg.nombre_categoria, salidas / "09_puntajes_discriminador.png")
    vis.guardar_gif(evolucion, salidas / "evolucion.gif")

    # ---------------- Guardar modelos y reportes --------------------------
    torch.save({"estado": generador.state_dict(), "config": vars(cfg)}, salidas / "generador.pth")
    torch.save({"estado": discriminador.state_dict(), "config": vars(cfg)}, salidas / "discriminador.pth")

    # Datos crudos para el front-end: imágenes de cada época, reales y pérdidas por lote
    torch.save({
        "evolucion": torch.stack(evolucion),   # [épocas, 64, 1, 28, 28]
        "reales": reales_ejemplo.cpu(),        # [64, 1, 28, 28]
        "historial_iter": hist_iter,
    }, salidas / "datos_entrenamiento.pt")

    with open(salidas / "historial_epocas.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(hist_epoca[0].keys()))
        escritor.writeheader()
        escritor.writerows(hist_epoca)

    total_g, _ = contar_parametros(generador)
    total_d, _ = contar_parametros(discriminador)
    resumen = {
        "categoria": {"indice": cfg.categoria, "nombre": cfg.nombre_categoria, "imagenes_reales": n_imagenes},
        "hiperparametros": vars(cfg),
        "generador": {"parametros_totales": total_g, "arquitectura": arq_g,
                      "pesos_antes": stats_g_antes, "pesos_despues": stats_g_despues},
        "discriminador": {"parametros_totales": total_d, "arquitectura": arq_d,
                          "pesos_antes": stats_d_antes, "pesos_despues": stats_d_despues},
        "resultado_final": {"D_x_promedio": d_real, "D_G_z_promedio": d_falsa,
                            "porcentaje_generadas_que_engañan_a_D": engañadas,
                            "ultima_epoca": hist_epoca[-1]},
    }
    with open(salidas / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"\n Archivos guardados en: {salidas.resolve()}")
    for archivo in sorted(salidas.iterdir()):
        print(f"   - {archivo.name}")
