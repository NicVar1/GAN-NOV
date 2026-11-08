"""
inspeccion.py
=============
Herramientas para VER los parámetros y valores internos de cada red:

  - resumen_arquitectura : tabla capa por capa (tipo, forma de salida, nº de parámetros)
  - estadisticas_pesos   : media, desviación, mínimo y máximo de cada tensor de pesos
  - imprimir_comparacion : cómo cambiaron los pesos antes vs después de entrenar
"""

import torch
from torch import nn


def contar_parametros(modelo: nn.Module) -> tuple[int, int]:
    """(total de parámetros, parámetros entrenables)."""
    total = sum(p.numel() for p in modelo.parameters())
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return total, entrenables


def _describir_capa(capa: nn.Module) -> str:
    """Texto corto con la configuración de la capa (filtros, kernel, stride...)."""
    if isinstance(capa, nn.Linear):
        return f"in={capa.in_features}, out={capa.out_features}, bias={capa.bias is not None}"
    if isinstance(capa, (nn.Conv2d, nn.ConvTranspose2d)):
        return (f"in={capa.in_channels}, out={capa.out_channels}, k={capa.kernel_size[0]}, "
                f"s={capa.stride[0]}, p={capa.padding[0]}")
    if isinstance(capa, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return f"features={capa.num_features}"
    if isinstance(capa, nn.LeakyReLU):
        return f"pendiente={capa.negative_slope}"
    if isinstance(capa, nn.Dropout):
        return f"p={capa.p}"
    if isinstance(capa, nn.Unflatten):
        return f"-> {tuple(capa.unflattened_size)}"
    return ""


def resumen_arquitectura(modelo: nn.Module, entrada_ejemplo: torch.Tensor, nombre: str) -> list[dict]:
    """
    Pasa una entrada de ejemplo por la red y registra, con "hooks", la forma
    de salida de cada capa. Imprime una tabla y la devuelve como lista de dicts.
    """
    filas, ganchos = [], []

    def registrar(indice, capa):
        def gancho(_modulo, _entrada, salida):
            filas.append({
                "indice": indice,
                "capa": capa.__class__.__name__,
                "configuracion": _describir_capa(capa),
                "forma_salida": list(salida.shape[1:]),   # sin la dimensión del lote
                "parametros": sum(p.numel() for p in capa.parameters()),
            })
        return gancho

    for i, capa in enumerate(modelo.red):
        ganchos.append(capa.register_forward_hook(registrar(i, capa)))

    estaba_entrenando = modelo.training
    modelo.eval()
    with torch.no_grad():
        modelo(entrada_ejemplo)
    modelo.train(estaba_entrenando)
    for g in ganchos:
        g.remove()

    total, entrenables = contar_parametros(modelo)
    linea = "-" * 100
    print(f"\n{linea}\n ARQUITECTURA: {nombre}\n{linea}")
    print(f" Entrada: {list(entrada_ejemplo.shape[1:])}")
    print(f" {'#':>2}  {'Capa':<16}{'Configuración':<42}{'Forma de salida':<20}{'Parámetros':>12}")
    print(linea)
    for f in filas:
        print(f" {f['indice']:>2}  {f['capa']:<16}{f['configuracion']:<42}"
              f"{str(f['forma_salida']):<20}{f['parametros']:>12,}")
    print(linea)
    print(f" Parámetros totales: {total:,}   |   Entrenables: {entrenables:,}")
    print(linea)
    return filas


def estadisticas_pesos(modelo: nn.Module) -> list[dict]:
    """Media, desviación estándar, mínimo, máximo y primeros valores de cada tensor."""
    stats = []
    for nombre, p in modelo.named_parameters():
        datos = p.detach().float().cpu()
        stats.append({
            "nombre": nombre,
            "forma": list(datos.shape),
            "cantidad": datos.numel(),
            "media": datos.mean().item(),
            "desv": datos.std().item() if datos.numel() > 1 else 0.0,
            "min": datos.min().item(),
            "max": datos.max().item(),
            "primeros_valores": [round(v, 5) for v in datos.flatten()[:5].tolist()],
        })
    return stats


def imprimir_estadisticas(stats: list[dict], titulo: str) -> None:
    linea = "-" * 112
    print(f"\n{linea}\n VALORES DE LOS PARÁMETROS: {titulo}\n{linea}")
    print(f" {'Tensor':<16}{'Forma':<20}{'Media':>10}{'Desv.':>10}{'Mín':>10}{'Máx':>10}   Primeros valores")
    print(linea)
    for s in stats:
        print(f" {s['nombre']:<16}{str(s['forma']):<20}{s['media']:>10.4f}{s['desv']:>10.4f}"
              f"{s['min']:>10.4f}{s['max']:>10.4f}   {s['primeros_valores']}")
    print(linea)


def imprimir_comparacion(antes: list[dict], despues: list[dict], titulo: str) -> None:
    """Muestra cuánto cambió la desviación estándar de cada tensor con el entrenamiento."""
    linea = "-" * 84
    print(f"\n{linea}\n CAMBIO DE LOS PESOS CON EL ENTRENAMIENTO: {titulo}\n{linea}")
    print(f" {'Tensor':<16}{'Media antes':>14}{'Media después':>16}{'Desv. antes':>14}{'Desv. después':>16}")
    print(linea)
    for a, d in zip(antes, despues):
        print(f" {a['nombre']:<16}{a['media']:>14.4f}{d['media']:>16.4f}{a['desv']:>14.4f}{d['desv']:>16.4f}")
    print(linea)


def pesos_aplanados(modelo: nn.Module) -> dict[str, torch.Tensor]:
    """Copia de los pesos ('weight') de capas Conv/Linear, para los histogramas."""
    return {
        n: p.detach().cpu().flatten().clone()
        for n, p in modelo.named_parameters()
        if n.endswith("weight") and p.dim() > 1
    }
