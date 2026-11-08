/*
 * app.js
 * ======
 * Lógica de la interfaz. Lee window.GAN_DATOS (generado por `uv run gan-web`)
 * y usa gan.js para ejecutar el Generador y el Discriminador en vivo.
 */
(function () {
  "use strict";

  const D = window.GAN_DATOS;
  if (!D || !window.GAN) {
    document.getElementById("aviso-error").hidden = false;
    return;
  }

  // =========================================================================
  // Utilidades
  // =========================================================================
  const $ = (id) => document.getElementById(id);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmt = (n, d = 3) => Number(n).toFixed(d);
  const miles = (n) => Number(n).toLocaleString("es-CO");
  const NS = "http://www.w3.org/2000/svg";
  const LADO = 28, PIX = LADO * LADO;

  function el(etiqueta, atributos = {}, padre = null, texto = null) {
    const nodo = etiqueta.startsWith("svg:") ? document.createElementNS(NS, etiqueta.slice(4)) : document.createElement(etiqueta);
    for (const [k, v] of Object.entries(atributos)) nodo.setAttribute(k, v);
    if (texto !== null) nodo.textContent = texto;
    if (padre) padre.appendChild(nodo);
    return nodo;
  }

  // Imágenes uint8 (0..255) exportadas desde Python
  const bytesEvolucion = GAN.base64ABytes(D.imagenes.evolucion);
  const bytesReales = GAN.base64ABytes(D.imagenes.reales);
  const bytesFinales = GAN.base64ABytes(D.imagenes.finales);
  const imagenU8 = (bytes, i) => bytes.subarray(i * PIX, (i + 1) * PIX);
  const u8AFloat = (u8) => Float32Array.from(u8, (v) => v / 127.5 - 1);

  // Dibuja una imagen 28x28. Acepta uint8 (0..255) o float (-1..1).
  function pintarImagen(canvas, valores) {
    canvas.width = LADO; canvas.height = LADO;
    const ctx = canvas.getContext("2d");
    const datos = ctx.createImageData(LADO, LADO);
    const esU8 = valores instanceof Uint8Array;
    for (let i = 0; i < PIX; i++) {
      const g = esU8 ? valores[i] : Math.max(0, Math.min(255, (valores[i] + 1) * 127.5));
      datos.data.set([g, g, g, 255], i * 4);
    }
    ctx.putImageData(datos, 0, 0);
  }

  // Dibuja 16 mapas de activación en una grilla 4x4 (cada mapa normalizado por separado)
  function pintarMapas(canvas, mapas, lado, color, cantidad = 16) {
    const columnas = 4, hueco = 1;
    const ancho = columnas * lado + (columnas - 1) * hueco;
    canvas.width = ancho; canvas.height = ancho;
    const ctx = canvas.getContext("2d");
    const datos = ctx.createImageData(ancho, ancho);
    for (let i = 0; i < datos.data.length; i += 4) datos.data.set([40, 40, 38, 255], i);
    const plano = lado * lado;
    for (let m = 0; m < cantidad; m++) {
      const off = m * plano;
      let min = Infinity, max = -Infinity;
      for (let i = 0; i < plano; i++) { const v = mapas[off + i]; if (v < min) min = v; if (v > max) max = v; }
      const rango = max - min || 1;
      const ox = (m % columnas) * (lado + hueco), oy = Math.floor(m / columnas) * (lado + hueco);
      for (let y = 0; y < lado; y++) for (let x = 0; x < lado; x++) {
        const t = (mapas[off + y * lado + x] - min) / rango;
        const p = ((oy + y) * ancho + ox + x) * 4;
        datos.data.set([color[0] * t, color[1] * t, color[2] * t, 255], p);
      }
    }
    ctx.putImageData(datos, 0, 0);
  }

  // Ruido z como grilla 10x10: azul = positivo, naranja = negativo
  function pintarZ(canvas, z) {
    canvas.width = 10; canvas.height = 10;
    const ctx = canvas.getContext("2d");
    const datos = ctx.createImageData(10, 10);
    for (let i = 0; i < 100; i++) {
      const t = Math.min(1, Math.abs(z[i]) / 2.5);
      const c = z[i] >= 0 ? [90, 140, 255] : [240, 120, 70];
      datos.data.set([c[0] * t + 20 * (1 - t), c[1] * t + 20 * (1 - t), c[2] * t + 20 * (1 - t), 255], i * 4);
    }
    ctx.putImageData(datos, 0, 0);
  }

  function hexARgb(hex) {
    const h = hex.replace("#", "");
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }

  // Celda de galería con su puntaje D
  function celdaGaleria(padre, valores, puntaje, alClic) {
    const celda = el("div", { class: "celda" }, padre);
    pintarImagen(el("canvas", {}, celda), valores);
    if (puntaje !== undefined && puntaje !== null) {
      el("span", { class: "puntaje " + (puntaje >= 0.5 ? "alto" : "bajo") }, celda, "D=" + fmt(puntaje, 2));
    }
    if (alClic) celda.addEventListener("click", alClic);
    return celda;
  }

  function cifra(padre, rotulo, numero, sub = "", clase = "") {
    const c = el("div", { class: "cifra " + clase }, padre);
    el("div", { class: "rotulo" }, c, rotulo);
    el("div", { class: "numero" }, c, numero);
    if (sub) el("div", { class: "sub" }, c, sub);
  }

  // Media móvil
  function suavizar(v, ventana = 25) {
    const salida = [];
    let suma = 0;
    for (let i = 0; i < v.length; i++) {
      suma += v[i];
      if (i >= ventana) suma -= v[i - ventana];
      salida.push(suma / Math.min(i + 1, ventana));
    }
    return salida;
  }

  // =========================================================================
  // Gráfica de líneas en SVG (sin librerías)
  // =========================================================================
  function graficaLineas(svg, opciones) {
    const { series, xMax, yMin, yMax, lineasH = [], ancho, alto, etiquetaX = "", ticksY = 5, formatoX = (v) => v } = opciones;
    svg.innerHTML = "";
    const m = { izq: 38, der: 10, arr: 10, aba: 28 };
    const W = ancho - m.izq - m.der, H = alto - m.arr - m.aba;
    const sx = (x) => m.izq + (x / xMax) * W;
    const sy = (y) => m.arr + H - ((y - yMin) / (yMax - yMin)) * H;

    for (let i = 0; i <= ticksY; i++) {
      const v = yMin + ((yMax - yMin) * i) / ticksY;
      el("svg:line", { x1: m.izq, x2: m.izq + W, y1: sy(v), y2: sy(v), class: "rejilla" }, svg);
      el("svg:text", { x: m.izq - 6, y: sy(v) + 4, "text-anchor": "end" }, svg, fmt(v, v % 1 === 0 ? 0 : 1));
    }
    for (let i = 0; i <= 5; i++) {
      const v = (xMax * i) / 5;
      el("svg:text", { x: sx(v), y: alto - 8, "text-anchor": "middle" }, svg, formatoX(v));
    }
    if (etiquetaX) el("svg:text", { x: m.izq + W, y: alto - 8, "text-anchor": "end" }, svg, "");
    el("svg:line", { x1: m.izq, x2: m.izq + W, y1: m.arr + H, y2: m.arr + H, class: "eje" }, svg);

    for (const l of lineasH) {
      el("svg:line", { x1: m.izq, x2: m.izq + W, y1: sy(l.y), y2: sy(l.y), stroke: l.color, "stroke-dasharray": "5 4", "stroke-width": 1 }, svg);
      if (l.texto) el("svg:text", { x: m.izq + W - 4, y: sy(l.y) - 4, "text-anchor": "end", style: `fill:${l.color}` }, svg, l.texto);
    }
    for (const s of series) {
      const paso = Math.max(1, Math.floor(s.datos.length / 900));
      let d = "";
      for (let i = 0; i < s.datos.length; i += paso) {
        const y = Math.max(yMin, Math.min(yMax, s.datos[i]));
        d += (d ? "L" : "M") + sx(s.x ? s.x[i] : i).toFixed(1) + "," + sy(y).toFixed(1);
      }
      el("svg:path", { d, fill: "none", stroke: s.color, "stroke-width": s.grosor || 2, opacity: s.opacidad || 1,
        "stroke-linejoin": "round", "stroke-dasharray": s.discontinua ? "6 4" : "none" }, svg);
    }
    // Sombra del "futuro" y cursor
    const futuro = el("svg:rect", { x: sx(xMax), y: m.arr, width: 0, height: H, fill: css("--fondo"), opacity: 0.7 }, svg);
    const cursor = el("svg:line", { x1: 0, x2: 0, y1: m.arr, y2: m.arr + H, class: "cursor" }, svg);
    const puntos = [];
    return {
      sx, sy,
      ponerCursor(x) {
        const px = sx(x);
        cursor.setAttribute("x1", px); cursor.setAttribute("x2", px);
        futuro.setAttribute("x", px); futuro.setAttribute("width", Math.max(0, m.izq + W - px));
      },
      ocultarCursor() { cursor.setAttribute("opacity", 0); futuro.setAttribute("width", 0); },
      punto(x, y, color, radio = 5) {
        const c = el("svg:circle", { cx: sx(x), cy: sy(y), r: radio, fill: color, stroke: css("--superficie"), "stroke-width": 2 }, svg);
        puntos.push(c);
        return c;
      },
    };
  }

  function leyenda(contenedor, items) {
    contenedor.innerHTML = "";
    for (const it of items) {
      const s = el("span", {}, contenedor);
      el("i", { class: it.discontinua ? "discontinua" : "", style: it.discontinua ? `border-color:${it.color}` : `background:${it.color}` }, s);
      s.appendChild(document.createTextNode(it.texto));
    }
  }

  // =========================================================================
  // Pesos cargados en memoria (para ejecutar las redes en vivo)
  // =========================================================================
  const PG = GAN.cargarPesos(D.pesos.generador);
  const PD = GAN.cargarPesos(D.pesos.discriminador);
  const realesFloat = Array.from({ length: 64 }, (_, i) => u8AFloat(imagenU8(bytesReales, i)));
  const nombreCat = D.categoria.nombre;
  const epocas = D.historial_epocas.length;
  const hist = D.historial_epocas;
  const ultima = hist[epocas - 1];
  const promedio = (a) => a.reduce((s, v) => s + v, 0) / a.length;

  // =========================================================================
  // 0. Encabezado y diagrama
  // =========================================================================
  $("nombre-categoria").textContent = `${D.categoria.indice} · ${nombreCat}`;
  $("detalle-categoria").textContent = `${miles(D.categoria.imagenes_reales)} imágenes reales de 28×28`;

  const cif = $("cifras");
  cifra(cif, "Parámetros del Generador", miles(D.total_parametros.generador), "pesos y sesgos aprendidos", "g");
  cifra(cif, "Parámetros del Discriminador", miles(D.total_parametros.discriminador), "pesos y sesgos aprendidos", "d");
  cifra(cif, "Épocas de entrenamiento", epocas, `${D.lotes_por_epoca} lotes de ${D.hiperparametros.tam_lote} por época`);
  cifra(cif, "Iteraciones totales", miles(D.historial_iter.perdida_d.length), `≈ ${Math.round(hist.reduce((s, h) => s + h.segundos, 0) / 60)} min en ${D.hiperparametros.dispositivo.toUpperCase()}`);

  function urlImagen(valores) {
    const c = document.createElement("canvas");
    pintarImagen(c, valores);
    return c.toDataURL();
  }

  // Dibuja el diagrama de flujo de la GAN; devuelve los elementos por clave
  function dibujarDiagrama(svg, conAnimacion) {
    svg.innerHTML = "";
    const cG = css("--g"), cD = css("--d"), cR = css("--real"), cT = css("--texto"), cT2 = css("--texto-2");
    const sup = css("--superficie"), gS = css("--g-suave"), dS = css("--d-suave"), rS = css("--real-suave"), s2 = css("--superficie-2");
    const defs = el("svg:defs", {}, svg);
    const partes = {};
    for (const [id, color] of [["fg", cG], ["fd", cD], ["fr", cR], ["ft", cT2]]) {
      const mk = el("svg:marker", { id: svg.id + id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
      el("svg:path", { d: "M0,0 L10,5 L0,10 z", fill: color }, mk);
    }
    const grupo = (clave) => (partes[clave] = el("svg:g", {}, svg));
    const caja = (clave, x, y, w, h, relleno, borde, titulo, sub) => {
      const g = grupo(clave);
      el("svg:rect", { x, y, width: w, height: h, rx: 12, fill: relleno, stroke: borde, class: "caja" }, g);
      el("svg:text", { x: x + w / 2, y: y + h / 2 - (sub ? 4 : -4), "text-anchor": "middle", style: `fill:${cT};font-size:15px;font-weight:600` }, g, titulo);
      if (sub) el("svg:text", { x: x + w / 2, y: y + h / 2 + 15, "text-anchor": "middle", style: "font-size:11.5px" }, g, sub);
      return g;
    };
    const flecha = (clave, d, color, marcador, discontinua, texto, tx, ty) => {
      const g = grupo(clave);
      el("svg:path", { d, fill: "none", stroke: color, "stroke-width": 2, "marker-end": `url(#${svg.id + marcador})`,
        "stroke-dasharray": discontinua ? "6 5" : conAnimacion ? "8 4" : "none", class: conAnimacion && !discontinua ? "punto-flujo" : "" }, g);
      if (texto) el("svg:text", { x: tx, y: ty, "text-anchor": "middle", style: `fill:${color};font-size:11.5px` }, g, texto);
      return g;
    };
    const imagen = (clave, x, y, url, borde, titulo) => {
      const g = grupo(clave);
      el("svg:rect", { x: x - 3, y: y - 3, width: 76, height: 76, rx: 8, fill: "#000", stroke: borde, "stroke-width": 2 }, g);
      el("svg:image", { x, y, width: 70, height: 70, href: url, style: "image-rendering:pixelated" }, g);
      el("svg:text", { x: x + 35, y: y + 92, "text-anchor": "middle", style: `fill:${borde};font-size:12px;font-weight:600` }, g, titulo);
      return g;
    };

    caja("z", 14, 172, 112, 66, s2, cT2, "Ruido z", "100 números al azar");
    caja("g", 170, 166, 150, 78, gS, cG, "Generador G", miles(D.total_parametros.generador) + " parámetros");
    imagen("falsa", 372, 170, urlImagen(imagenU8(bytesFinales, 0)), cG, "G(z): falsa");
    imagen("real", 372, 20, urlImagen(imagenU8(bytesReales, 0)), cR, "x: real");
    caja("d", 520, 98, 160, 84, dS, cD, "Discriminador D", miles(D.total_parametros.discriminador) + " parámetros");
    caja("salida", 728, 105, 104, 70, sup, cD, "P(real)", "0 falsa · 1 real");
    caja("perdida", 716, 214, 230, 62, sup, cT2, "Pérdidas BCE", "L_D (para D) · L_G (para G)");

    flecha("fZG", "M126,205 L166,205", cT2, "ft");
    flecha("fGF", "M320,205 L366,205", cG, "fg");
    flecha("fFD", "M448,205 C485,205 490,160 516,152", cG, "fg");
    flecha("fRD", "M448,55 C485,55 490,115 516,124", cR, "fr");
    flecha("fDS", "M680,140 L724,140", cD, "fd");
    flecha("fSP", "M780,175 L780,210", cT2, "ft");
    flecha("retroD", "M716,232 C650,232 620,215 605,186", cD, "fd", true, "actualiza D", 640, 248);
    flecha("retroG", "M716,262 C560,292 300,292 252,248", cG, "fg", true, "actualiza G (el gradiente pasa por D)", 470, 296);
    return partes;
  }
  dibujarDiagrama($("diagrama"), true);

  // =========================================================================
  // 1. Laboratorio en vivo
  // =========================================================================
  const lab = { zs: [], salidasG: [], resD: [], sel: 0, fuente: "falsa" };
  const colorG = hexARgb("#7aa2ff"), colorD = hexARgb("#f08a5d");

  function evaluarD() {
    const lote = lab.fuente === "falsa" ? lab.salidasG.map((s) => s.imagen) : realesFloat.slice(0, 16);
    lab.resD = GAN.discriminador(PD, lote, D.hiperparametros.pendiente_leaky);
  }

  function nuevoLote() {
    lab.zs = Array.from({ length: 16 }, () => GAN.ruido(D.hiperparametros.dim_latente));
    lab.salidasG = lab.zs.map((z) => GAN.generador(PG, z));
    evaluarD();
    pintarLab();
  }

  function moverZ() {
    const eps = GAN.ruido(D.hiperparametros.dim_latente);
    const z = lab.zs[lab.sel];
    for (let i = 0; i < z.length; i++) z[i] = z[i] * 0.94 + 0.35 * eps[i];
    lab.salidasG[lab.sel] = GAN.generador(PG, z);
    evaluarD();
    pintarLab();
  }

  function pintarLab() {
    const i = lab.sel, falsa = lab.fuente === "falsa";
    const sg = lab.salidasG[i], rd = lab.resD[i];
    pintarZ($("cv-z"), lab.zs[i]);
    pintarMapas($("cv-g7"), sg.mapa7, 7, colorG);
    pintarMapas($("cv-g14"), sg.mapa14, 14, colorG);
    for (const id of ["cv-z", "cv-g7", "cv-g14"]) $(id).style.opacity = falsa ? 1 : 0.25;
    pintarImagen($("cv-img"), falsa ? sg.imagen : realesFloat[i]);
    $("titulo-imagen").textContent = falsa ? "Imagen G(z)" : "Imagen real x";
    $("cv-img").parentElement.style.background = falsa ? css("--g-suave") : css("--real-suave");
    pintarMapas($("cv-d14"), rd.mapa14, 14, colorD);
    pintarMapas($("cv-d7"), rd.mapa7, 7, colorD);
    $("medidor").style.height = rd.prob * 100 + "%";
    $("valor-d").textContent = fmt(rd.prob, 3);
    $("veredicto").textContent = rd.prob >= 0.5 ? "D cree que es REAL" : "D cree que es FALSA";

    const gal = $("galeria-lote");
    gal.innerHTML = "";
    lab.resD.forEach((r, n) => {
      const valores = falsa ? lab.salidasG[n].imagen : realesFloat[n];
      const c = celdaGaleria(gal, valores, r.prob, () => { lab.sel = n; pintarLab(); });
      if (n === i) c.classList.add("sel");
    });
    const prom = promedio(lab.resD.map((r) => r.prob));
    const engaña = lab.resD.filter((r) => r.prob >= 0.5).length;
    $("promedio-lote").textContent = falsa
      ? `D(G(z)) promedio del lote = ${fmt(prom)} · ${engaña} de 16 engañan a D (≥ 0.5). Una GAN perfecta llevaría este valor a 0.5.`
      : `D(x) promedio sobre 16 imágenes reales = ${fmt(prom)}. Compárelo con el puntaje de las generadas.`;
  }

  function interpolar() {
    const zA = GAN.ruido(D.hiperparametros.dim_latente), zB = GAN.ruido(D.hiperparametros.dim_latente);
    const tira = $("tira-interpolacion");
    tira.innerHTML = "";
    for (let k = 0; k < 10; k++) {
      const t = k / 9;
      const z = zA.map((a, i) => a * (1 - t) + zB[i] * t);
      const envoltorio = el("div", { class: k === 0 || k === 9 ? "extremo" : "" }, tira);
      pintarImagen(el("canvas", {}, envoltorio), GAN.generador(PG, z).imagen);
      if (k === 0) el("span", {}, envoltorio, "A");
      if (k === 9) el("span", {}, envoltorio, "B");
    }
  }

  $("btn-ruido").addEventListener("click", nuevoLote);
  $("btn-mover").addEventListener("click", moverZ);
  $("btn-interpolar").addEventListener("click", interpolar);
  $("selector-fuente").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    lab.fuente = b.dataset.fuente;
    $("selector-fuente").querySelectorAll("button").forEach((x) => x.classList.toggle("activo", x === b));
    evaluarD();
    pintarLab();
  });

  // =========================================================================
  // 2. Arquitectura
  // =========================================================================
  const EXPLICACION = {
    generador: [
      ["Proyecta el ruido de 100 números a 6 272 = 128·7·7 neuronas. Cada neurona combina los 100 valores de z con sus propios pesos. Es la capa con más parámetros de toda la GAN.",
        "100 entradas × 6 272 salidas = 627 200 pesos (sin bias)", "nn.Linear(dim_latente, c*2*7*7, bias=False)"],
      ["Normaliza cada neurona con la media y la varianza del lote, y luego aplica una escala γ y un desplazamiento β aprendidos. Estabiliza el entrenamiento.",
        "2 × 6 272 = 12 544 (γ y β por neurona)", "nn.BatchNorm1d(c*2*7*7)"],
      ["ReLU(x) = max(0, x). Deja pasar solo lo positivo. Sin esta no linealidad, toda la red sería una sola multiplicación lineal.", "0, no tiene parámetros", "nn.ReLU(inplace=True)"],
      ["Reacomoda el vector de 6 272 valores en 128 mapas de 7×7. Es el primer «boceto» espacial de la prenda.", "0, no tiene parámetros", "nn.Unflatten(1, (c*2, 7, 7))"],
      ["Convolución transpuesta: cada píxel de entrada «pinta» un parche 4×4 en la salida, avanzando de 2 en 2, y así duplica el tamaño (7 → 14). Pasa de 128 a 64 mapas.",
        "128 entradas × 64 salidas × 4 × 4 = 131 072 (sin bias)", "nn.ConvTranspose2d(c*2, c, kernel_size=4, stride=2, padding=1, bias=False)"],
      ["Normaliza cada uno de los 64 mapas.", "2 × 64 = 128 (γ y β)", "nn.BatchNorm2d(c)"],
      ["No linealidad ReLU.", "0", "nn.ReLU(inplace=True)"],
      ["Última convolución transpuesta: combina los 64 mapas en 1 imagen de 28×28.", "64 × 1 × 4 × 4 + 1 sesgo = 1 025", "nn.ConvTranspose2d(c, 1, kernel_size=4, stride=2, padding=1)"],
      ["Tanh comprime cada píxel a [−1, 1], la misma escala que las imágenes reales normalizadas con (x − 0.5) / 0.5.", "0", "nn.Tanh()"],
    ],
    discriminador: [
      ["64 filtros de 4×4 recorren la imagen saltando de 2 en 2 (28 → 14). Detectan bordes y trazos simples. Según DCGAN, aquí no se usa BatchNorm.",
        "64 filtros × 1 canal × 4 × 4 + 64 sesgos = 1 088", "nn.Conv2d(1, c, kernel_size=4, stride=2, padding=1)"],
      ["LeakyReLU(x) = x si x > 0; 0.2·x si no. Mantiene un gradiente pequeño en los negativos, y así G sigue recibiendo señal.", "0", "nn.LeakyReLU(0.2, inplace=True)"],
      ["Durante el entrenamiento apaga al azar el 30 % de las activaciones. Así D no se vuelve demasiado fuerte ni memoriza. Al evaluar se desactiva.", "0", "nn.Dropout(0.3)"],
      ["128 filtros que combinan los 64 mapas anteriores (14 → 7). Detectan partes de la prenda, como la suela, la puntera o los cordones.",
        "128 filtros × 64 canales × 4 × 4 = 131 072 (sin bias)", "nn.Conv2d(c, c*2, kernel_size=4, stride=2, padding=1, bias=False)"],
      ["Normaliza cada uno de los 128 mapas con las estadísticas del lote.", "2 × 128 = 256 (γ y β)", "nn.BatchNorm2d(c*2)"],
      ["LeakyReLU con pendiente 0.2.", "0", "nn.LeakyReLU(0.2, inplace=True)"],
      ["Dropout del 30 %.", "0", "nn.Dropout(0.3)"],
      ["Aplana los 128×7×7 en un vector de 6 272 características.", "0", "nn.Flatten()"],
      ["Combina las 6 272 características en un solo puntaje (logit).", "6 272 pesos × 1 salida + 1 sesgo = 6 273", "nn.Linear(c*2*7*7, 1)"],
      ["σ(s) = 1 / (1 + e^(−s)) convierte el puntaje en la probabilidad de que la imagen sea real.", "0", "nn.Sigmoid()"],
    ],
  };

  let capaSel = null;
  function listaCapas(red, contenedor, color) {
    const filas = D.arquitectura[red];
    const max = Math.max(...filas.map((f) => f.parametros));
    contenedor.innerHTML = "";
    filas.forEach((f) => {
      const c = el("div", { class: "capa" }, contenedor);
      el("span", { class: "idx" }, c, f.indice);
      const n = el("span", { class: "nombre" }, c);
      el("b", {}, n, f.capa);
      el("small", {}, n, f.configuracion || "—");
      const p = el("span", { class: "n" }, c);
      el("strong", {}, p, miles(f.parametros));
      p.appendChild(document.createTextNode("[" + f.forma_salida.join("×") + "]"));
      el("span", { class: "barra-param", style: `width:${(f.parametros / max) * 100}%;background:${color}` }, c);
      c.addEventListener("click", () => mostrarCapa(red, f, c));
      if (!capaSel && red === "generador" && f.indice === 0) mostrarCapa(red, f, c);
    });
  }
  function mostrarCapa(red, f, nodo) {
    document.querySelectorAll(".capa.sel").forEach((x) => x.classList.remove("sel"));
    nodo.classList.add("sel");
    capaSel = nodo;
    const [texto, formula, codigo] = EXPLICACION[red][f.indice];
    const caja = $("detalle-capa");
    caja.innerHTML = "";
    const titulo = el("h3", { class: red === "generador" ? "g" : "d" }, caja,
      `${red === "generador" ? "Generador" : "Discriminador"} · capa ${f.indice}: ${f.capa}`);
    titulo.style.marginBottom = "4px";
    el("p", {}, caja, texto);
    el("p", { class: "dim" }, caja, `Forma de salida: [${f.forma_salida.join(" × ")}]  ·  Parámetros: ${miles(f.parametros)}`);
    el("div", { class: "formula-param" }, caja, "Parámetros = " + formula);
    const cod = el("p", { class: "dim" }, caja, "En redes.py: ");
    el("code", {}, cod, codigo);
  }
  listaCapas("generador", $("capas-g"), css("--g"));
  listaCapas("discriminador", $("capas-d"), css("--d"));
  $("total-g").textContent = miles(D.total_parametros.generador) + " parámetros";
  $("total-d").textContent = miles(D.total_parametros.discriminador) + " parámetros";

  const HIPER = {
    categoria: "Clase de Fashion-MNIST usada para entrenar",
    tam_lote: "Imágenes por lote (batch)",
    dim_latente: "Tamaño del ruido z que recibe G",
    canales_base_g: "Mapas base en G (128 → 64 → 1)",
    canales_base_d: "Filtros base en D (64 → 128)",
    pendiente_leaky: "Pendiente negativa de LeakyReLU en D",
    dropout_d: "Probabilidad de Dropout en D",
    epocas: "Pasadas completas por el dataset",
    tasa_aprendizaje_g: "Learning rate de Adam para G",
    tasa_aprendizaje_d: "Learning rate de Adam para D",
    beta1: "Momento β1 de Adam (0.5 según DCGAN)",
    beta2: "Momento β2 de Adam",
    etiqueta_real: "Objetivo para imágenes reales (label smoothing)",
    etiqueta_falsa: "Objetivo para imágenes falsas",
    semilla: "Semilla aleatoria (reproducibilidad)",
  };
  const th = $("tabla-hiper");
  const cab = el("tr", {}, el("thead", {}, th));
  ["Hiperparámetro", "Valor", "Significado"].forEach((t) => el("th", {}, cab, t));
  const cuerpo = el("tbody", {}, th);
  for (const [k, txt] of Object.entries(HIPER)) {
    const tr = el("tr", {}, cuerpo);
    el("code", {}, el("td", {}, tr), k);
    el("td", {}, tr, String(D.hiperparametros[k]));
    el("td", {}, tr, txt);
  }

  // =========================================================================
  // 3. Un paso de entrenamiento
  // =========================================================================
  const PASOS = [
    { clase: "d", titulo: "1 · D mira imágenes reales", texto: "Se toma un lote de 64 imágenes reales y D calcula D(x).",
      claves: ["real", "fRD", "d", "fDS", "salida"],
      codigo: "# entrenamiento.py: PASO 1\noptim_d.zero_grad()\nsalida_real = discriminador(imagenes_reales)   # D(x)" },
    { clase: "d", titulo: "2 · G crea falsas y D las mira", texto: "Se genera ruido z, G crea 64 imágenes falsas y D calcula D(G(z)). Con .detach(), G no recibe gradiente en este paso.",
      claves: ["z", "fZG", "g", "fGF", "falsa", "fFD", "d", "fDS", "salida"],
      codigo: "z = torch.randn(lote, 100)\nimagenes_falsas = generador(z)                       # G(z)\nsalida_falsa = discriminador(imagenes_falsas.detach())  # D(G(z))" },
    { clase: "d", titulo: "3 · Se actualiza SOLO D", texto: "L_D = BCE(D(x), 0.9) + BCE(D(G(z)), 0). El gradiente solo cambia los pesos de D.",
      claves: ["salida", "fSP", "perdida", "retroD", "d"],
      codigo: "perdida_d, real, falsa = perdida_discriminador(\n    salida_real, salida_falsa, 0.9, 0.0)\nperdida_d.backward()\noptim_d.step()          # solo cambian los pesos de D" },
    { clase: "g", titulo: "4 · D (ya actualizado) juzga de nuevo", texto: "D vuelve a evaluar las imágenes falsas, esta vez sin detach, para que el gradiente pueda llegar hasta G.",
      claves: ["falsa", "fFD", "d", "fDS", "salida", "fSP", "perdida"],
      codigo: "# PASO 2\noptim_g.zero_grad()\nsalida_para_g = discriminador(imagenes_falsas)\nperdida_g = perdida_generador(salida_para_g)  # -log D(G(z))" },
    { clase: "g", titulo: "5 · Se actualiza SOLO G", texto: "El gradiente de L_G atraviesa D sin modificarlo y llega a G. G cambia para que D diga «real».",
      claves: ["perdida", "retroG", "g", "d", "z", "fZG"],
      codigo: "perdida_g.backward()\noptim_g.step()          # solo cambian los pesos de G\n\n# ... y se repite con el siguiente lote" },
  ];
  let pasoActual = 0, partesPaso = null;
  function dibujarPasos() {
    partesPaso = dibujarDiagrama($("diagrama-paso"), false);
    const lista = $("lista-pasos");
    lista.innerHTML = "";
    PASOS.forEach((p, i) => {
      const li = el("li", { class: p.clase }, lista);
      el("b", {}, li, p.titulo);
      el("span", {}, li, p.texto);
      li.addEventListener("click", () => irAPaso(i));
    });
    irAPaso(pasoActual);
  }
  function irAPaso(i) {
    pasoActual = (i + PASOS.length) % PASOS.length;
    const p = PASOS[pasoActual];
    for (const [clave, nodo] of Object.entries(partesPaso)) nodo.setAttribute("class", p.claves.includes(clave) ? "encendido" : "apagado");
    $("lista-pasos").querySelectorAll("li").forEach((li, k) => li.classList.toggle("activo", k === pasoActual));
    $("codigo-paso").textContent = p.codigo;
  }
  $("btn-paso-ant").addEventListener("click", () => irAPaso(pasoActual - 1));
  $("btn-paso-sig").addEventListener("click", () => irAPaso(pasoActual + 1));

  // =========================================================================
  // 4. Funciones de pérdida (explorador BCE)
  // =========================================================================
  const ER = D.hiperparametros.etiqueta_real;
  function actualizarBCE() {
    const dx = +$("sl-dx").value, dgz = +$("sl-dgz").value;
    $("o-dx").textContent = fmt(dx, 2);
    $("o-dgz").textContent = fmt(dgz, 2);
    const ldReal = -(ER * Math.log(dx) + (1 - ER) * Math.log(1 - dx));
    const ldFalsa = -Math.log(1 - dgz);
    const lg = -Math.log(dgz);
    const c = $("cifras-bce");
    c.innerHTML = "";
    cifra(c, "L_D en reales", fmt(ldReal), `BCE(D(x), ${ER})`, "d");
    cifra(c, "L_D en falsas", fmt(ldFalsa), "−log(1 − D(G(z)))", "d");
    cifra(c, "L_D total", fmt(ldReal + ldFalsa), "equilibrio ≈ 1.386", "d");
    cifra(c, "L_G", fmt(lg), "−log D(G(z)) · eq. ≈ 0.693", "g");
    let nota;
    if (dgz >= 0.5 && dx <= 0.6) nota = "G está engañando a D: L_G es baja y L_D es alta. D tendrá que mejorar.";
    else if (Math.abs(dx - 0.5) < 0.12 && Math.abs(dgz - 0.5) < 0.12) nota = "Zona de equilibrio: D no logra distinguir las imágenes reales de las falsas. Es la meta teórica de una GAN.";
    else if (dx > 0.7 && dgz < 0.3) nota = "D está ganando: distingue bien las reales de las falsas, así que L_D es baja. L_G es alta y G recibe una señal fuerte para mejorar. Así se ve la época 1 del entrenamiento.";
    else nota = "Mueve los deslizadores. D quiere D(x) alto y D(G(z)) bajo; G quiere D(G(z)) alto.";
    $("nota-bce").textContent = nota;

    const xs = Array.from({ length: 99 }, (_, i) => (i + 1) / 100);
    const g = graficaLineas($("grafica-bce"), {
      ancho: 460, alto: 300, xMax: 1, yMin: 0, yMax: 5, formatoX: (v) => fmt(v, 1),
      series: [
        { datos: xs.map((p) => -Math.log(p)), x: xs, color: css("--g") },
        { datos: xs.map((p) => -Math.log(1 - p)), x: xs, color: css("--d") },
      ],
    });
    g.ocultarCursor();
    g.punto(dx, -Math.log(dx), css("--real"));
    g.punto(dgz, -Math.log(1 - dgz), css("--d"));
    g.punto(dgz, -Math.log(dgz), css("--g"));
  }
  leyenda($("leyenda-bce"), [
    { color: css("--g"), texto: "−log(p): costo si el objetivo es «real» (lo usa G)" },
    { color: css("--d"), texto: "−log(1−p): costo si el objetivo es «falsa» (lo usa D)" },
    { color: css("--real"), texto: "● D(x)" },
  ]);
  $("sl-dx").addEventListener("input", actualizarBCE);
  $("sl-dgz").addEventListener("input", actualizarBCE);

  // =========================================================================
  // 5. Entrenamiento época por época
  // =========================================================================
  const HI = D.historial_iter;
  const totalIter = HI.perdida_d.length;
  let grafPerdidas, grafSalidas, epocaActual = 1, reproduciendo = null;

  function dibujarGraficasEntrenamiento() {
    const yMaxP = Math.min(7, Math.ceil(Math.max(...suavizar(HI.perdida_g)) + 0.5));
    grafPerdidas = graficaLineas($("grafica-perdidas"), {
      ancho: 620, alto: 260, xMax: totalIter, yMin: 0, yMax: yMaxP, formatoX: (v) => "época " + Math.round(v / D.lotes_por_epoca),
      lineasH: [{ y: Math.log(4), color: css("--d"), texto: "equilibrio D 1.386" }, { y: Math.log(2), color: css("--g"), texto: "equilibrio G 0.693" }],
      series: [
        { datos: HI.perdida_g, color: css("--g"), grosor: 1, opacidad: 0.18 },
        { datos: HI.perdida_d, color: css("--d"), grosor: 1, opacidad: 0.18 },
        { datos: suavizar(HI.perdida_g), color: css("--g") },
        { datos: suavizar(HI.perdida_d), color: css("--d") },
      ],
    });
    grafSalidas = graficaLineas($("grafica-salidas"), {
      ancho: 620, alto: 220, xMax: totalIter, yMin: 0, yMax: 1, ticksY: 4, formatoX: (v) => "época " + Math.round(v / D.lotes_por_epoca),
      lineasH: [{ y: 0.5, color: css("--texto-2"), texto: "0.5 = D no distingue" }],
      series: [
        { datos: suavizar(HI.d_x), color: css("--real") },
        { datos: suavizar(HI.d_g_z), color: css("--g") },
      ],
    });
    leyenda($("leyenda-perdidas"), [
      { color: css("--d"), texto: "Pérdida del Discriminador L_D" }, { color: css("--g"), texto: "Pérdida del Generador L_G" },
      { color: css("--texto-2"), texto: "equilibrio teórico", discontinua: true },
    ]);
    leyenda($("leyenda-salidas"), [
      { color: css("--real"), texto: "D(x): puntaje a imágenes reales" }, { color: css("--g"), texto: "D(G(z)): puntaje a imágenes generadas" },
    ]);
  }

  function pintarEvolucion(epoca) {
    const canvas = $("cv-evolucion");
    const n = D.imagenes.evolucion_forma[1], col = 4, hueco = 1, lado = col * LADO + (col - 1) * hueco;
    canvas.width = lado; canvas.height = lado;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#222"; ctx.fillRect(0, 0, lado, lado);
    const datos = ctx.createImageData(LADO, LADO);
    for (let k = 0; k < n; k++) {
      const img = imagenU8(bytesEvolucion, (epoca - 1) * n + k);
      for (let i = 0; i < PIX; i++) datos.data.set([img[i], img[i], img[i], 255], i * 4);
      ctx.putImageData(datos, (k % col) * (LADO + hueco), Math.floor(k / col) * (LADO + hueco));
    }
  }

  function fase(h) {
    if (h.d_g_z_antes < 0.15) return ["Fase 1 · El Discriminador domina", "G apenas produce manchas y D las detecta sin esfuerzo (D(G(z)) ≈ 0.1). L_G es muy alta y G recibe gradientes fuertes."];
    if (h.perdida_g > 1.35) return ["Fase 2 · El Generador aprende la forma", "Las manchas toman la silueta de la prenda. D(G(z)) sube y L_G baja rápido. D empieza a dudar."];
    return ["Fase 3 · Competencia equilibrada", "Las dos pérdidas oscilan cerca del equilibrio (L_D ≈ 1.386, L_G ≈ 0.693) y G pule los detalles. Ninguna red aplasta a la otra."];
  }

  function irAEpoca(e) {
    epocaActual = e;
    const h = hist[e - 1];
    $("sl-epoca").value = e;
    $("txt-epoca").textContent = e;
    pintarEvolucion(e);
    grafPerdidas.ponerCursor(e * D.lotes_por_epoca);
    grafSalidas.ponerCursor(e * D.lotes_por_epoca);
    const [t, txt] = fase(h);
    $("fase").innerHTML = "";
    el("b", {}, $("fase"), t);
    $("fase").appendChild(document.createTextNode(txt));
    const c = $("cifras-epoca");
    c.innerHTML = "";
    cifra(c, "Pérdida D (total)", fmt(h.perdida_d), "promedio de la época", "d");
    cifra(c, "Pérdida D · reales", fmt(h.perdida_d_real), "BCE(D(x), 0.9)", "d");
    cifra(c, "Pérdida D · falsas", fmt(h.perdida_d_falsa), "BCE(D(G(z)), 0)", "d");
    cifra(c, "Pérdida G", fmt(h.perdida_g), "−log D(G(z))", "g");
    cifra(c, "D(x)", fmt(h.d_x), "puntaje medio a reales", "real");
    cifra(c, "D(G(z))", fmt(h.d_g_z_antes), "puntaje medio a falsas", "g");
  }

  $("sl-epoca").max = epocas;
  $("txt-total-epocas").textContent = epocas;
  $("sl-epoca").addEventListener("input", (e) => irAEpoca(+e.target.value));
  $("btn-play").addEventListener("click", () => {
    if (reproduciendo) {
      clearInterval(reproduciendo); reproduciendo = null;
      $("btn-play").textContent = "▶ Reproducir";
      return;
    }
    if (epocaActual >= epocas) irAEpoca(1);
    $("btn-play").textContent = "❚❚ Pausar";
    reproduciendo = setInterval(() => {
      if (epocaActual >= epocas) { $("btn-play").click(); return; }
      irAEpoca(epocaActual + 1);
    }, 320);
  });

  // =========================================================================
  // 6. Pesos
  // =========================================================================
  let redPesos = "generador";
  function llenarSelectorTensores() {
    const sel = $("selector-tensor");
    sel.innerHTML = "";
    D.histogramas[redPesos].forEach((h, i) =>
      el("option", { value: i }, sel, `${h.nombre}  [${h.forma.join("×")}] · ${miles(h.cantidad)} pesos`));
    dibujarHistograma();
    tablaPesos();
  }
  function dibujarHistograma() {
    const h = D.histogramas[redPesos][+$("selector-tensor").value || 0];
    const svg = $("grafica-hist");
    svg.innerHTML = "";
    const ancho = 460, alto = 260, m = { izq: 12, der: 12, arr: 10, aba: 28 };
    const W = ancho - m.izq - m.der, H = alto - m.arr - m.aba, n = h.antes.length;
    const max = Math.max(...h.antes, ...h.despues);
    const color = redPesos === "generador" ? css("--g") : css("--d");
    const area = (cuentas) => {
      let d = `M${m.izq},${m.arr + H}`;
      cuentas.forEach((c, i) => {
        const y = m.arr + H - (c / max) * H;
        d += `L${m.izq + (i / n) * W},${y}L${m.izq + ((i + 1) / n) * W},${y}`;
      });
      return d + `L${m.izq + W},${m.arr + H}Z`;
    };
    el("svg:path", { d: area(h.antes), fill: css("--neutro"), opacity: 0.45 }, svg);
    el("svg:path", { d: area(h.despues), fill: color, opacity: 0.55 }, svg);
    el("svg:line", { x1: m.izq, x2: m.izq + W, y1: m.arr + H, y2: m.arr + H, class: "eje" }, svg);
    const x0 = m.izq + ((0 - h.min) / (h.max - h.min)) * W;
    el("svg:line", { x1: x0, x2: x0, y1: m.arr, y2: m.arr + H, class: "rejilla" }, svg);
    el("svg:text", { x: m.izq, y: alto - 8 }, svg, fmt(h.min, 3));
    el("svg:text", { x: x0, y: alto - 8, "text-anchor": "middle" }, svg, "0");
    el("svg:text", { x: m.izq + W, y: alto - 8, "text-anchor": "end" }, svg, fmt(h.max, 3));
    $("titulo-hist").textContent = `${redPesos === "generador" ? "Generador" : "Discriminador"} · ${h.nombre}`;
    leyenda($("leyenda-hist"), [{ color: css("--neutro"), texto: "Antes de entrenar: N(0, 0.02)" }, { color, texto: "Después de entrenar" }]);
    $("tabla-pesos").querySelectorAll("tbody tr").forEach((tr) => tr.classList.toggle("sel", tr.dataset.nombre === h.nombre));
  }
  function tablaPesos() {
    const t = $("tabla-pesos");
    t.innerHTML = "";
    const tr0 = el("tr", {}, el("thead", {}, t));
    ["Tensor", "Forma", "Media antes → después", "Desv. antes → después", "Mín / Máx (después)", "Primeros 5 valores (después)"].forEach((x, i) =>
      el("th", { class: i >= 2 && i <= 4 ? "num" : "" }, tr0, x));
    const tb = el("tbody", {}, t);
    const { antes, despues } = D.estadisticas_pesos[redPesos];
    despues.forEach((s, i) => {
      const a = antes[i];
      const tr = el("tr", { "data-nombre": s.nombre }, tb);
      el("code", {}, el("td", {}, tr), s.nombre);
      el("td", {}, tr, "[" + s.forma.join("×") + "]");
      el("td", { class: "num" }, tr, `${fmt(a.media, 4)} → ${fmt(s.media, 4)}`);
      el("td", { class: "num" }, tr, `${fmt(a.desv, 4)} → ${fmt(s.desv, 4)}`);
      el("td", { class: "num" }, tr, `${fmt(s.min, 3)} / ${fmt(s.max, 3)}`);
      el("td", {}, tr, s.primeros_valores.map((v) => fmt(v, 3)).join(", "));
    });
    const h = D.histogramas[redPesos][+$("selector-tensor").value || 0];
    tb.querySelectorAll("tr").forEach((tr) => tr.classList.toggle("sel", tr.dataset.nombre === h.nombre));
  }
  $("selector-red").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    redPesos = b.dataset.red;
    $("selector-red").querySelectorAll("button").forEach((x) => x.classList.toggle("activo", x === b));
    llenarSelectorTensores();
  });
  $("selector-tensor").addEventListener("change", dibujarHistograma);

  // =========================================================================
  // 7. Resultado
  // =========================================================================
  function pintarResultado() {
    $("titulo-resultado").textContent = `Resultado de la GAN · Categoría usada: ${nombreCat}`;
    const gr = $("galeria-reales"), gf = $("galeria-finales");
    gr.innerHTML = ""; gf.innerHTML = "";
    D.imagenes.puntajes_reales.forEach((p, i) => celdaGaleria(gr, imagenU8(bytesReales, i), p));
    D.imagenes.puntajes_finales.forEach((p, i) => celdaGaleria(gf, imagenU8(bytesFinales, i), p));
    const pr = promedio(D.imagenes.puntajes_reales), pf = promedio(D.imagenes.puntajes_finales);
    $("prom-reales").textContent = `· D(x) promedio ${fmt(pr)}`;
    $("prom-falsas").textContent = `· D(G(z)) promedio ${fmt(pf)}`;
    const h1 = hist[0];
    $("conclusion").textContent =
      `Después de ${epocas} épocas, el Generador aprendió a dibujar «${nombreCat}» a partir de ruido puro. ` +
      `En la época 1, D le daba a las imágenes falsas un puntaje de ${fmt(h1.d_g_z_antes, 2)}; en la última época, ${fmt(ultima.d_g_z_antes, 2)}. ` +
      `En ese mismo tiempo, D(x) bajó de ${fmt(h1.d_x, 2)} a ${fmt(ultima.d_x, 2)}. Las dos se acercaron a 0.5: al Discriminador le cuesta cada vez más ` +
      `distinguir las imágenes reales de las generadas. Las pérdidas terminaron en L_D = ${fmt(ultima.perdida_d)} y L_G = ${fmt(ultima.perdida_g)}, ` +
      `cerca de los valores de equilibrio teórico (1.386 y 0.693), y no hubo colapso de modo: las imágenes generadas son variadas.`;
  }

  // =========================================================================
  // Navegación activa y arranque
  // =========================================================================
  const enlaces = [...document.querySelectorAll("nav a")];
  const observador = new IntersectionObserver((entradas) => {
    entradas.forEach((en) => {
      if (en.isIntersecting) enlaces.forEach((a) => a.classList.toggle("activo", a.getAttribute("href") === "#" + en.target.id));
    });
  }, { rootMargin: "-45% 0px -50% 0px" });
  document.querySelectorAll(".seccion").forEach((s) => observador.observe(s));

  function dibujarTodo() {
    dibujarDiagrama($("diagrama"), true);
    dibujarPasos();
    actualizarBCE();
    dibujarGraficasEntrenamiento();
    irAEpoca(epocaActual);
    llenarSelectorTensores();
  }

  nuevoLote();
  interpolar();
  pintarResultado();
  dibujarTodo();
  // Si el sistema cambia entre tema claro y oscuro, se redibujan los SVG con los nuevos colores
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", dibujarTodo);
})();
