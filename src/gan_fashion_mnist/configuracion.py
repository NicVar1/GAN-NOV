"""
configuracion.py
================
Todos los HIPERPARÁMETROS de la GAN en un solo lugar.

Un hiperparámetro es un valor que NO aprende la red: lo elegimos nosotros
antes de entrenar (tamaño del ruido, tasa de aprendizaje, épocas, etc.).
Los PARÁMETROS (pesos y sesgos) sí los aprende la red; esos se ven en
inspeccion.py.
"""

from dataclasses import dataclass, field

import torch

# ---------------------------------------------------------------------------
# Categorías del dataset Fashion-MNIST (10 clases de ropa en escala de grises,
# imágenes de 28x28 píxeles). La GAN se entrena con UNA sola categoría para
# que aprenda a dibujar exactamente ese tipo de prenda.
# ---------------------------------------------------------------------------
CATEGORIAS = {
    0: "Camiseta / Top (T-shirt/top)",
    1: "Pantalón (Trouser)",
    2: "Suéter (Pullover)",
    3: "Vestido (Dress)",
    4: "Abrigo (Coat)",
    5: "Sandalia (Sandal)",
    6: "Camisa (Shirt)",
    7: "Zapatilla deportiva (Sneaker)",
    8: "Bolso (Bag)",
    9: "Botín (Ankle boot)",
}


def _dispositivo_por_defecto() -> str:
    """Usa la GPU si PyTorch la detecta; si no, la CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Configuracion:
    # ----------------------------- Datos ----------------------------------
    categoria: int = 7              # Clase de Fashion-MNIST usada (7 = Zapatilla deportiva)
    tam_lote: int = 64              # Imágenes por lote (batch size)
    tam_imagen: int = 28            # Las imágenes son de 28x28 píxeles
    canales_imagen: int = 1         # 1 canal = escala de grises

    # ------------------------ Red Generadora (G) --------------------------
    dim_latente: int = 100          # Tamaño del vector de ruido z ~ N(0, 1) que recibe G
    canales_base_g: int = 64        # Número base de mapas de características en G

    # ---------------------- Red Discriminadora (D) ------------------------
    canales_base_d: int = 64        # Número base de filtros convolucionales en D
    pendiente_leaky: float = 0.2    # Pendiente negativa de LeakyReLU en D
    dropout_d: float = 0.3          # Probabilidad de apagar neuronas en D (evita que D domine)

    # ------------------------- Entrenamiento ------------------------------
    epocas: int = 40                # Pasadas completas por el dataset
    tasa_aprendizaje_g: float = 2e-4  # Learning rate del optimizador Adam de G
    tasa_aprendizaje_d: float = 2e-4  # Learning rate del optimizador Adam de D
    beta1: float = 0.5              # Momento de Adam (0.5 es lo recomendado en DCGAN)
    beta2: float = 0.999            # Segundo momento de Adam

    # Etiquetas usadas en la función de pérdida BCE.
    # "One-sided label smoothing": en vez de decirle a D que las reales valen 1.0
    # le decimos 0.9; así D no se vuelve sobreconfiado y G recibe mejores gradientes.
    etiqueta_real: float = 0.9
    etiqueta_falsa: float = 0.0

    # ----------------------------- Otros ----------------------------------
    semilla: int = 42               # Semilla para que los resultados sean reproducibles
    carpeta_datos: str = "datos"    # Donde se descarga Fashion-MNIST
    carpeta_salidas: str = "salidas"  # Donde se guardan gráficas, modelos y reportes
    num_muestras_fijas: int = 64    # Imágenes generadas con ruido fijo para ver la evolución
    dispositivo: str = field(default_factory=_dispositivo_por_defecto)

    @property
    def nombre_categoria(self) -> str:
        return CATEGORIAS[self.categoria]
