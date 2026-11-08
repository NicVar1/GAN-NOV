/*
 * gan.js
 * ======
 * Implementación en JavaScript puro de las DOS redes neuronales de la GAN,
 * usando los pesos ya entrenados en PyTorch (exportados a datos.js).
 *
 * Es exactamente la misma arquitectura que src/gan_fashion_mnist/redes.py:
 *
 *   GENERADOR      z(100) -> Linear -> BatchNorm1d -> ReLU -> (128,7,7)
 *                         -> ConvTranspose2d -> BatchNorm2d -> ReLU -> (64,14,14)
 *                         -> ConvTranspose2d -> Tanh -> (1,28,28)
 *
 *   DISCRIMINADOR  (1,28,28) -> Conv2d -> LeakyReLU -> (64,14,14)
 *                            -> Conv2d -> BatchNorm2d -> LeakyReLU -> (128,7,7)
 *                            -> Flatten -> Linear -> Sigmoid -> P(real)
 *
 * (El Dropout solo actúa durante el entrenamiento, por eso aquí no aparece.)
 */
(function (global) {
  "use strict";

  const EPS = 1e-5; // epsilon de BatchNorm en PyTorch

  // ---------- Decodificación base64 -> arreglos tipados -------------------
  function base64ABytes(texto) {
    if (typeof atob === "function") {
      const bin = atob(texto);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return bytes;
    }
    return new Uint8Array(Buffer.from(texto, "base64")); // Node.js (pruebas)
  }
  const base64AFloat32 = (t) => new Float32Array(base64ABytes(t).buffer);

  function cargarPesos(pesosExportados) {
    const salida = {};
    for (const [nombre, info] of Object.entries(pesosExportados)) {
      salida[nombre] = { forma: info.forma, datos: base64AFloat32(info.datos) };
    }
    return salida;
  }

  // ---------- Operaciones básicas ------------------------------------------

  // Convolución transpuesta (kernel 4, stride 2, padding 1): duplica alto y ancho.
  // Pesos en formato PyTorch: [canales_entrada, canales_salida, 4, 4]
  function convTranspuesta(entrada, cIn, alto, ancho, peso, sesgo, cOut) {
    const k = 4, s = 2, p = 1;
    const aOut = (alto - 1) * s - 2 * p + k, anOut = (ancho - 1) * s - 2 * p + k;
    const plano = aOut * anOut;
    const salida = new Float32Array(cOut * plano);
    if (sesgo) for (let co = 0; co < cOut; co++) salida.fill(sesgo[co], co * plano, (co + 1) * plano);
    for (let ci = 0; ci < cIn; ci++) {
      for (let iy = 0; iy < alto; iy++) {
        for (let ix = 0; ix < ancho; ix++) {
          const v = entrada[ci * alto * ancho + iy * ancho + ix];
          if (v === 0) continue; // tras ReLU muchos valores son 0: se omiten
          for (let co = 0; co < cOut; co++) {
            const base = (ci * cOut + co) * 16;
            const baseSalida = co * plano;
            for (let ky = 0; ky < k; ky++) {
              const oy = iy * s - p + ky;
              if (oy < 0 || oy >= aOut) continue;
              for (let kx = 0; kx < k; kx++) {
                const ox = ix * s - p + kx;
                if (ox < 0 || ox >= anOut) continue;
                salida[baseSalida + oy * anOut + ox] += v * peso[base + ky * k + kx];
              }
            }
          }
        }
      }
    }
    return salida;
  }

  // Convolución normal (kernel 4, stride 2, padding 1): reduce a la mitad.
  // Pesos en formato PyTorch: [canales_salida, canales_entrada, 4, 4]
  function convolucion(entrada, cIn, alto, ancho, peso, sesgo, cOut) {
    const k = 4, s = 2, p = 1;
    const aOut = Math.floor((alto + 2 * p - k) / s) + 1, anOut = Math.floor((ancho + 2 * p - k) / s) + 1;
    const salida = new Float32Array(cOut * aOut * anOut);
    for (let co = 0; co < cOut; co++) {
      for (let oy = 0; oy < aOut; oy++) {
        for (let ox = 0; ox < anOut; ox++) {
          let suma = sesgo ? sesgo[co] : 0;
          for (let ci = 0; ci < cIn; ci++) {
            const baseP = (co * cIn + ci) * 16, baseE = ci * alto * ancho;
            for (let ky = 0; ky < k; ky++) {
              const iy = oy * s - p + ky;
              if (iy < 0 || iy >= alto) continue;
              for (let kx = 0; kx < k; kx++) {
                const ix = ox * s - p + kx;
                if (ix < 0 || ix >= ancho) continue;
                suma += entrada[baseE + iy * ancho + ix] * peso[baseP + ky * k + kx];
              }
            }
          }
          salida[co * aOut * anOut + oy * anOut + ox] = suma;
        }
      }
    }
    return salida;
  }

  // BatchNorm en modo evaluación: usa la media y varianza acumuladas en el entrenamiento.
  function batchNormEval(x, canales, porCanal, g, b, media, varianza) {
    for (let c = 0; c < canales; c++) {
      const escala = g[c] / Math.sqrt(varianza[c] + EPS);
      for (let i = c * porCanal; i < (c + 1) * porCanal; i++) x[i] = (x[i] - media[c]) * escala + b[c];
    }
  }

  // BatchNorm con estadísticas DEL LOTE (como en el entrenamiento del Discriminador).
  function batchNormLote(lote, canales, porCanal, g, b) {
    for (let c = 0; c < canales; c++) {
      let suma = 0, suma2 = 0, n = 0;
      for (const x of lote) for (let i = c * porCanal; i < (c + 1) * porCanal; i++) { suma += x[i]; suma2 += x[i] * x[i]; n++; }
      const media = suma / n, varianza = suma2 / n - media * media;
      const escala = g[c] / Math.sqrt(varianza + EPS);
      for (const x of lote) for (let i = c * porCanal; i < (c + 1) * porCanal; i++) x[i] = (x[i] - media) * escala + b[c];
    }
  }

  const relu = (x) => { for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0; };
  const leaky = (x, m) => { for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] *= m; };
  const sigmoide = (v) => 1 / (1 + Math.exp(-v));

  // =========================================================================
  // RED GENERADORA: z -> imagen (devuelve también las activaciones internas)
  // =========================================================================
  function generador(P, z) {
    const W0 = P["red.0.weight"].datos; // [6272, 100]
    const nOcultas = P["red.0.weight"].forma[0], nZ = P["red.0.weight"].forma[1];

    // Capa 0: Linear 100 -> 6272
    const h = new Float32Array(nOcultas);
    for (let j = 0; j < nOcultas; j++) {
      let s = 0;
      const base = j * nZ;
      for (let i = 0; i < nZ; i++) s += W0[base + i] * z[i];
      h[j] = s;
    }
    // Capas 1-2: BatchNorm1d + ReLU
    batchNormEval(h, nOcultas, 1, P["red.1.weight"].datos, P["red.1.bias"].datos,
      P["red.1.running_mean"].datos, P["red.1.running_var"].datos);
    relu(h);
    // Capa 3: Unflatten -> 128 x 7 x 7 (la memoria ya está en ese orden)
    const c1 = P["red.4.weight"].forma[0], c2 = P["red.4.weight"].forma[1];

    // Capas 4-6: ConvTranspose2d 128->64 (7->14) + BatchNorm2d + ReLU
    const m14 = convTranspuesta(h, c1, 7, 7, P["red.4.weight"].datos, null, c2);
    batchNormEval(m14, c2, 14 * 14, P["red.5.weight"].datos, P["red.5.bias"].datos,
      P["red.5.running_mean"].datos, P["red.5.running_var"].datos);
    relu(m14);

    // Capas 7-8: ConvTranspose2d 64->1 (14->28) + Tanh
    const img = convTranspuesta(m14, c2, 14, 14, P["red.7.weight"].datos, P["red.7.bias"].datos, 1);
    for (let i = 0; i < img.length; i++) img[i] = Math.tanh(img[i]);

    return { mapa7: h, mapa14: m14, imagen: img };
  }

  // =========================================================================
  // RED DISCRIMINADORA: lote de imágenes -> probabilidad de ser REAL
  // =========================================================================
  function discriminador(P, imagenes, pendiente = 0.2) {
    const c1 = P["red.0.weight"].forma[0], c2 = P["red.3.weight"].forma[0];

    // Capas 0-1: Conv2d 1->64 (28->14) + LeakyReLU
    const m14 = imagenes.map((img) => {
      const x = convolucion(img, 1, 28, 28, P["red.0.weight"].datos, P["red.0.bias"].datos, c1);
      leaky(x, pendiente);
      return x;
    });
    // Capas 3-5: Conv2d 64->128 (14->7) + BatchNorm2d (estadísticas del lote) + LeakyReLU
    const m7 = m14.map((x) => convolucion(x, c1, 14, 14, P["red.3.weight"].datos, null, c2));
    batchNormLote(m7, c2, 49, P["red.4.weight"].datos, P["red.4.bias"].datos);
    m7.forEach((x) => leaky(x, pendiente));

    // Capas 7-9: Flatten + Linear 6272 -> 1 + Sigmoid
    const W = P["red.8.weight"].datos, b = P["red.8.bias"].datos[0];
    return m14.map((_, n) => {
      let s = b;
      for (let i = 0; i < W.length; i++) s += W[i] * m7[n][i];
      return { mapa14: m14[n], mapa7: m7[n], logit: s, prob: sigmoide(s) };
    });
  }

  // Ruido gaussiano N(0,1) con Box-Muller (z del espacio latente)
  function ruido(n, aleatorio = Math.random) {
    const z = new Float32Array(n);
    for (let i = 0; i < n; i += 2) {
      const u = 1 - aleatorio(), v = aleatorio();
      const r = Math.sqrt(-2 * Math.log(u));
      z[i] = r * Math.cos(2 * Math.PI * v);
      if (i + 1 < n) z[i + 1] = r * Math.sin(2 * Math.PI * v);
    }
    return z;
  }

  const API = { base64ABytes, base64AFloat32, cargarPesos, generador, discriminador, ruido };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else global.GAN = API;
})(typeof window !== "undefined" ? window : globalThis);
