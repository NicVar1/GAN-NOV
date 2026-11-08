"""
web.py
======
Exporta todo lo que produjo el entrenamiento a frontend/datos.js y abre el
front-end en el navegador.

    uv run gan-web                  (categoría 7)
    uv run gan-web --categoria 1
    uv run gan-web --no-abrir       (solo exporta)

Qué se exporta:
  - Pesos ENTRENADOS del Generador y del Discriminador (float32 en base64).
    Con ellos el navegador ejecuta las dos redes en vivo (frontend/gan.js).
  - Arquitecturas, hiperparámetros, estadísticas e histogramas de los pesos.
  - Historial de pérdidas y de salidas de D (por lote y por época).
  - Imágenes generadas en cada época, imágenes reales y el resultado final.
  - Un caso de prueba calculado con PyTorch para verificar que el JavaScript
    produce exactamente lo mismo.
"""

import argparse
import base64
import csv
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np
import torch

from .configuracion import CATEGORIAS, Configuracion
from .redes import crear_discriminador, crear_generador, discriminador_para_evaluar

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
CARPETA_FRONTEND = RAIZ_PROYECTO / "frontend"


def _b64_float32(tensor: torch.Tensor) -> str:
    """Tensor -> bytes float32 little-endian -> texto base64."""
    return base64.b64encode(tensor.detach().cpu().float().numpy().astype("<f4").tobytes()).decode("ascii")


def _b64_imagenes(imagenes: torch.Tensor) -> str:
    """Imágenes en [-1, 1] -> enteros 0..255 (uint8) -> base64. Ocupa 4 veces menos que float."""
    pixeles = ((imagenes.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return base64.b64encode(pixeles.numpy().tobytes()).decode("ascii")


def _exportar_pesos(estado: dict) -> dict:
    """Solo tensores de punto flotante (se omite el contador entero de BatchNorm)."""
    return {
        nombre: {"forma": list(t.shape), "datos": _b64_float32(t)}
        for nombre, t in estado.items() if t.is_floating_point()
    }


def _histogramas(inicial: dict, final: dict, bins: int = 60) -> list[dict]:
    """Histogramas de los pesos 'weight' de capas Conv/Linear, antes y después."""
    salida = []
    for nombre, t_final in final.items():
        if not nombre.endswith("weight") or t_final.dim() < 2:
            continue
        a = inicial[nombre].flatten().numpy()
        d = t_final.flatten().numpy()
        minimo, maximo = float(min(a.min(), d.min())), float(max(a.max(), d.max()))
        cuentas_a, bordes = np.histogram(a, bins=bins, range=(minimo, maximo))
        cuentas_d, _ = np.histogram(d, bins=bins, range=(minimo, maximo))
        salida.append({
            "nombre": nombre, "forma": list(t_final.shape), "cantidad": int(d.size),
            "min": minimo, "max": maximo,
            "antes": cuentas_a.tolist(), "despues": cuentas_d.tolist(),
        })
    return salida


def exportar(categoria: int) -> Path:
    cfg = Configuracion(categoria=categoria, dispositivo="cpu")
    carpeta = Path(cfg.carpeta_salidas) / f"categoria_{categoria}"
    requeridos = ["generador.pth", "discriminador.pth", "generador_inicial.pth",
                  "discriminador_inicial.pth", "datos_entrenamiento.pt", "resumen.json", "historial_epocas.csv"]
    faltan = [r for r in requeridos if not (carpeta / r).exists()]
    if faltan:
        sys.exit(f"Faltan archivos en {carpeta}: {faltan}\nPrimero ejecute: uv run gan-entrenar --categoria {categoria}")

    # ---------------- Cargar modelos entrenados --------------------------
    pc_g = torch.load(carpeta / "generador.pth", map_location="cpu")
    pc_d = torch.load(carpeta / "discriminador.pth", map_location="cpu")
    cfg.epocas = pc_g["config"]["epocas"]
    generador = crear_generador(cfg)
    generador.load_state_dict(pc_g["estado"])
    generador.eval()
    discriminador = crear_discriminador(cfg)
    discriminador.load_state_dict(pc_d["estado"])
    juez = discriminador_para_evaluar(discriminador)

    inicial_g = torch.load(carpeta / "generador_inicial.pth", map_location="cpu")
    inicial_d = torch.load(carpeta / "discriminador_inicial.pth", map_location="cpu")
    entrenamiento = torch.load(carpeta / "datos_entrenamiento.pt", map_location="cpu")
    resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
    with open(carpeta / "historial_epocas.csv", encoding="utf-8") as f:
        historial_epocas = [{k: float(v) for k, v in fila.items()} for fila in csv.DictReader(f)]

    evolucion = entrenamiento["evolucion"]          # [E, 64, 1, 28, 28]
    reales = entrenamiento["reales"]                # [64, 1, 28, 28]
    finales = evolucion[-1]                         # Resultado final de G (ruido fijo)

    with torch.no_grad():
        puntajes_finales = juez(finales).flatten()
        puntajes_reales = juez(reales).flatten()

        # Caso de prueba para validar la implementación JavaScript
        generador_prueba = torch.Generator().manual_seed(123)
        z_prueba = torch.randn(16, cfg.dim_latente, generator=generador_prueba)
        g_prueba = generador(z_prueba)
        d_prueba = juez(g_prueba).flatten()

    hist_iter = {k: [round(v, 4) for v in vals] for k, vals in entrenamiento["historial_iter"].items()}

    datos = {
        "categoria": {"indice": categoria, "nombre": CATEGORIAS[categoria],
                      "imagenes_reales": resumen["categoria"]["imagenes_reales"]},
        "categorias": CATEGORIAS,
        "hiperparametros": resumen["hiperparametros"],
        "arquitectura": {
            "generador": resumen["generador"]["arquitectura"],
            "discriminador": resumen["discriminador"]["arquitectura"],
        },
        "total_parametros": {
            "generador": resumen["generador"]["parametros_totales"],
            "discriminador": resumen["discriminador"]["parametros_totales"],
        },
        "estadisticas_pesos": {
            "generador": {"antes": resumen["generador"]["pesos_antes"], "despues": resumen["generador"]["pesos_despues"]},
            "discriminador": {"antes": resumen["discriminador"]["pesos_antes"],
                              "despues": resumen["discriminador"]["pesos_despues"]},
        },
        "histogramas": {
            "generador": _histogramas(inicial_g, pc_g["estado"]),
            "discriminador": _histogramas(inicial_d, pc_d["estado"]),
        },
        "historial_epocas": historial_epocas,
        "historial_iter": hist_iter,
        "lotes_por_epoca": len(hist_iter["perdida_d"]) // len(historial_epocas),
        "imagenes": {
            "lado": 28,
            "evolucion": _b64_imagenes(evolucion[:, :16]),   # 16 por época
            "evolucion_forma": [evolucion.shape[0], 16],
            "reales": _b64_imagenes(reales),
            "finales": _b64_imagenes(finales),
            "puntajes_finales": [round(v, 4) for v in puntajes_finales.tolist()],
            "puntajes_reales": [round(v, 4) for v in puntajes_reales.tolist()],
        },
        "pesos": {
            "generador": _exportar_pesos(pc_g["estado"]),
            "discriminador": _exportar_pesos(pc_d["estado"]),
        },
        "prueba": {
            "z": _b64_float32(z_prueba),
            "imagen0": _b64_float32(g_prueba[0]),
            "puntajes": d_prueba.tolist(),
        },
    }

    CARPETA_FRONTEND.mkdir(exist_ok=True)
    destino = CARPETA_FRONTEND / "datos.js"
    destino.write_text(
        "// Archivo generado automáticamente por `uv run gan-web`. No editar a mano.\n"
        "window.GAN_DATOS = " + json.dumps(datos, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    return destino


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    analizador = argparse.ArgumentParser(description="Exporta los resultados y abre el front-end de la GAN")
    analizador.add_argument("--categoria", type=int, default=7, choices=range(10))
    analizador.add_argument("--no-abrir", action="store_true", help="Solo exporta, no abre el navegador")
    args = analizador.parse_args()

    destino = exportar(args.categoria)
    print(f"Datos exportados: {destino}  ({destino.stat().st_size / 1e6:.1f} MB)")
    indice = CARPETA_FRONTEND / "index.html"
    print(f"Front-end: {indice}")
    if not args.no_abrir:
        webbrowser.open(indice.as_uri())
