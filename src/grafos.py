"""Dibujos del grafo de Markov (Anderl et al., 2016) a partir del modelo ajustado.

El grafo ya existe en `MarkovK.matriz_transicion()`: es la matriz W con los estados
INICIO, los canales, CONVERSIÓN y NULO. Este módulo solo lo convierte en tablas
legibles y lo dibuja. Hay cuatro figuras:

1. `dibujar_grafo_orden1`    el grafo de orden 1 (nodo = canal) y su matriz W.
2. `dibujar_detalle_orden2`  cómo un nodo de orden 1 se parte en parejas en el orden 2.
3. `dibujar_remocion`        el efecto de remoción visto en el grafo (antes y después).
4. `dibujar_grafo_escuela`   grafo ilustrativo del recorrido en la escuela (sin datos).
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch

from . import params as P
from .markov import MarkovK, BASE

GRIS, ROJO, TINTA = "#8C8C8B", "#E30613", "#2B2B2A"
GRIS_CLARO = "#D9D9D8"
ROJOS = LinearSegmentedColormap.from_list("rojos", ["#FBE3E5", "#F4A6AE", "#EE6B79", ROJO, "#8E0410"])

CORTO = {"SEA": "Búsqueda\npagada", "SEO": "Búsqueda\norgánica", "PriceComp": "Comparador",
         "Display": "Display", "Newsletter": "Newsletter", "Retargeting": "Retargeting",
         "Affiliate": "Afiliado", "Other": "Otro"}
INICIO, CONV, NULO = "INICIO", "CONVERSIÓN", "NULO"


# -----------------------------------------------------------------------------
# Tablas
# -----------------------------------------------------------------------------
def matriz_orden1(clics, recorridos):
    """Matriz de transición W de orden 1 como DataFrame (filas = desde, columnas = hacia).

    Filas y columnas: INICIO, los 8 canales, CONVERSIÓN y NULO. Cada fila suma 1
    (salvo CONVERSIÓN y NULO, que son absorbentes y se dejan en cero)."""
    m = MarkovK(1).fit(clics, recorridos)
    W, estados, _ = m.matriz_transicion()
    etiquetas = [INICIO] + [P.CANALES[e - 1] for e in estados] + [CONV, NULO]
    return pd.DataFrame(W.toarray(), index=etiquetas, columns=etiquetas)


def prob_conversion(W):
    """P(llegar de INICIO a CONVERSIÓN) en un grafo W (DataFrame).
    Resuelve (I − Q) x = r, con Q = transiciones entre estados transitorios
    y r = probabilidad de pasar a CONVERSIÓN desde cada uno."""
    trans = [e for e in W.index if e not in (CONV, NULO)]
    Q = W.loc[trans, trans].to_numpy()
    r = W.loc[trans, CONV].to_numpy()
    x = np.linalg.solve(np.eye(len(trans)) - Q, r)
    return float(x[trans.index(INICIO)])


def quitar_canal(W, canal):
    """Grafo sin `canal`: todo lo que entraba a él se manda a NULO (definición del paper)."""
    W2 = W.copy()
    W2[NULO] += W2[canal]
    W2[canal] = 0.0
    W2.loc[canal] = 0.0
    return W2


def remocion_por_canal(W):
    """Efecto de remoción de cada canal en el grafo W (orden 1): caída absoluta de
    P(INICIO → CONVERSIÓN) al quitarlo y su porcentaje sobre la suma de efectos."""
    base = prob_conversion(W)
    filas = [(c, base, prob_conversion(quitar_canal(W, c))) for c in P.CANALES]
    t = pd.DataFrame(filas, columns=["canal", "p_con_canal", "p_sin_canal"])
    t["efecto_remocion"] = t.p_con_canal - t.p_sin_canal
    t["efecto_%"] = 100 * t.efecto_remocion / t.efecto_remocion.sum()
    return t


def tasas_orden2(clics, recorridos, canal="SEA", min_visitas=300):
    """Para los estados de orden 2 que terminan en `canal` (X > canal): visitas y
    probabilidad de pasar a CONVERSIÓN, junto a la del mismo canal en orden 1.
    Son tasas sin suavizar (lo que el grafo ve directamente)."""
    m = MarkovK(2, suavizado=0).fit(clics, recorridos)
    c_idx = P.CANALES.index(canal) + 1
    g1 = m.tablas_[1]
    p1 = g1.loc[c_idx, "conv"] / g1.loc[c_idx, "salidas"]
    g2 = m.tablas_[2].reset_index()
    actual, previo = g2["s"] % BASE, g2["s"] // BASE
    g2 = g2[(actual == c_idx) & (previo > 0)].copy()
    g2["desde"] = [P.CANALES[d - 1] for d in g2["s"] // BASE]
    g2["estado"] = g2["desde"] + " > " + canal
    g2["p_conversion"] = g2["conv"] / g2["salidas"]
    g2 = g2[g2["salidas"] >= min_visitas]
    tabla = (g2[["estado", "desde", "salidas", "conv", "p_conversion"]]
             .rename(columns={"salidas": "visitas", "conv": "compras"})
             .sort_values("p_conversion", ascending=False).reset_index(drop=True))
    return tabla, float(p1)


# -----------------------------------------------------------------------------
# Utilidades de dibujo
# -----------------------------------------------------------------------------
def _posiciones_orden1():
    """Display (59 % de los clics) va al centro: así todas las flechas que llegan a él
    son radios y no se cruzan. Los demás canales rodean a Display; INICIO a la
    izquierda y los dos estados absorbentes a la derecha."""
    pos = {INICIO: (-3.35, 0.0), CONV: (3.45, 1.0), NULO: (3.45, -1.0), "Display": (0.0, 0.0)}
    alrededor = {"SEA": 90, "SEO": 40, "Affiliate": -10, "Newsletter": -60,
                 "Retargeting": -110, "Other": -150, "PriceComp": 145}
    for c, a in alrededor.items():
        pos[c] = (1.75 * np.cos(np.radians(a)), 1.6 * np.sin(np.radians(a)))
    return pos


def _radio_canal(c):
    """Radio del nodo (unidades de datos) según la participación en clics (Tabla 3)."""
    return 0.22 + 0.36 * np.sqrt(P.PARTICIPACION_CLICS[c])


RADIO_FIJO = {INICIO: 0.36, CONV: 0.42, NULO: 0.42}


def _flecha(ax, p0, p1, ancho, color, rad=0.0, alpha=1.0, estilo="-", r0=0.0, r1=0.0, z=1):
    """Flecha de p0 a p1 recortada para no tapar los nodos (r0, r1 en unidades de datos)."""
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    if np.linalg.norm(d) == 0:
        return
    u = d / np.linalg.norm(d)
    a = FancyArrowPatch(p0 + u * r0, p1 - u * (r1 + 0.03), arrowstyle="-|>",
                        mutation_scale=7 + 1.5 * min(ancho, 6),
                        lw=ancho, color=color, alpha=alpha, linestyle=estilo,
                        connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def _nodo(ax, xy, r, fc, ec, texto, color_texto, fs=7.2, forma="circulo", estilo="-", negrita=False):
    from matplotlib.patches import Circle, FancyBboxPatch
    if forma == "circulo":
        p = Circle(xy, r, fc=fc, ec=ec, lw=1.4, ls=estilo, zorder=3)
    else:
        p = FancyBboxPatch((xy[0] - r, xy[1] - r * 0.75), 2 * r, 1.5 * r,
                           boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2, zorder=3)
    ax.add_patch(p)
    ax.text(*xy, texto, ha="center", va="center", fontsize=fs, color=color_texto, zorder=4,
            fontweight="bold" if negrita else "normal", linespacing=1.15)


# -----------------------------------------------------------------------------
# Figura 1 — grafo de orden 1 y matriz W
# -----------------------------------------------------------------------------
def dibujar_grafo_orden1(W, ax, umbral=0.05, quitado=None, titulo=None, mostrar_barra=True):
    """Dibuja el grafo de orden 1.

    - Tamaño del nodo: participación del canal en los clics (Tabla 3).
    - Color del nodo y número rojo: probabilidad de pasar a CONVERSIÓN desde ese canal.
    - ↻: probabilidad de repetir el mismo canal en el clic siguiente (carryover).
    - Grosor de la flecha: probabilidad de transición w_ij.
    - Entre canales solo se dibujan las transiciones ≥ `umbral`; las demás existen
      (ver la matriz W), pero ensucian el dibujo.
    - `quitado`: canal removido; se dibuja vacío y punteado (figura 3).
    """
    pos = _posiciones_orden1()
    canales = P.CANALES
    p_conv = W.loc[canales, CONV]
    norm = Normalize(vmin=0, vmax=max(0.02, p_conv.max()))
    R = {c: _radio_canal(c) for c in canales}
    R.update(RADIO_FIJO)

    # Aristas
    for c in canales:                                   # INICIO → canal
        w = W.loc[INICIO, c]
        if w > 0:
            _flecha(ax, pos[INICIO], pos[c], 0.5 + 7 * w, GRIS, rad=0.0, alpha=0.5, r0=R[INICIO], r1=R[c])
    for a in canales:                                   # canal → canal
        for b in canales:
            w = W.loc[a, b]
            if a != b and w >= umbral:
                _flecha(ax, pos[a], pos[b], 0.4 + 7 * w, TINTA, rad=0.12, alpha=0.28, r0=R[a], r1=R[b])
    for c in canales:                                   # canal → absorbentes
        wc, wn = W.loc[c, CONV], W.loc[c, NULO]
        if wc > 0:
            _flecha(ax, pos[c], pos[CONV], 0.6 + 110 * wc, ROJO, rad=-0.15, alpha=0.7,
                    r0=R[c], r1=R[CONV], z=2)
        if wn > 0:
            _flecha(ax, pos[c], pos[NULO], 0.4 + 2 * wn, GRIS, rad=0.15, alpha=0.25, r0=R[c], r1=R[NULO])

    # Nodos. Display (grande) lleva el texto adentro; los demás, afuera hacia el borde.
    for c in canales:
        x, y = pos[c]
        if c == quitado:
            _nodo(ax, pos[c], R[c], "white", GRIS, "", GRIS, estilo="--")
        else:
            _nodo(ax, pos[c], R[c], ROJOS(norm(p_conv[c])), "white", "", TINTA)
        nombre = CORTO[c].replace("\n", " ")
        texto = nombre if c == quitado else f"{nombre}\n↻ {W.loc[c, c]:.0%} · conv. {p_conv[c]:.1%}"
        if c == "Display":
            ax.text(x, y, texto.replace(" · ", "\n"), ha="center", va="center", fontsize=7.4,
                    color=TINTA, zorder=4, linespacing=1.2)
            continue
        ang = np.arctan2(y, x)
        d = R[c] + 0.08
        tx, ty = x + d * np.cos(ang), y + d * np.sin(ang)
        ha = "left" if np.cos(ang) > 0.3 else ("right" if np.cos(ang) < -0.3 else "center")
        va = "bottom" if np.sin(ang) > 0.3 else ("top" if np.sin(ang) < -0.3 else "center")
        ax.text(tx, ty, texto, ha=ha, va=va, fontsize=7, color=GRIS if c == quitado else TINTA,
                zorder=5, linespacing=1.2,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    _nodo(ax, pos[INICIO], R[INICIO], "white", TINTA, INICIO, TINTA, negrita=True)
    _nodo(ax, pos[CONV], R[CONV], ROJO, ROJO, "CONVER-\nSIÓN", "white", forma="caja", negrita=True)
    _nodo(ax, pos[NULO], R[NULO], GRIS_CLARO, GRIS, NULO, TINTA, forma="caja", negrita=True)

    ax.set_xlim(-3.85, 3.95); ax.set_ylim(-2.3, 2.75); ax.set_aspect("equal"); ax.axis("off")
    if titulo:
        ax.set_title(titulo, loc="left", fontsize=10.5)
    if mostrar_barra:
        sm = plt.cm.ScalarMappable(cmap=ROJOS, norm=norm)
        cb = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, shrink=0.55)
        cb.set_label("P(pasar a CONVERSIÓN desde el canal)", fontsize=7.5)
        cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
        cb.ax.tick_params(labelsize=7)
    return ax


def dibujar_matriz(W, ax):
    """Mapa de calor de W: cada celda = P(siguiente = columna | actual = fila)."""
    filas = [INICIO, *P.CANALES]
    cols = [*P.CANALES, CONV, NULO]
    M = W.loc[filas, cols].to_numpy()
    ax.imshow(M, cmap=ROJOS, vmin=0, vmax=0.45, aspect="auto")
    nombres = lambda l: [CORTO.get(e, e).replace("\n", " ") for e in l]
    ax.set_xticks(range(len(cols)), nombres(cols), rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(filas)), nombres(filas), fontsize=7.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = f"{v:.1%}" if 0 < v < 0.1 else f"{v:.0%}"
            ax.text(j, i, txt if v > 0 else "·", ha="center", va="center", fontsize=6.3,
                    color="white" if v > 0.28 else TINTA)
    ax.set_xlabel("hacia"); ax.set_ylabel("desde")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Matriz de transición W (cada fila suma 100 %)", loc="left", fontsize=10.5)
    return ax


# -----------------------------------------------------------------------------
# Figura 2 — del orden 1 al orden 2
# -----------------------------------------------------------------------------
def dibujar_detalle_orden2(tabla, p_orden1, canal="SEA", ax=None):
    """Izquierda: en orden 1 hay un solo nodo `canal` con una sola probabilidad de
    conversión. Derecha: en orden 2 ese nodo se parte según el canal anterior, y cada
    pareja tiene su propia probabilidad. El grosor de la flecha a CONVERSIÓN es esa
    probabilidad."""
    ax = ax or plt.gca()
    n = len(tabla)
    ys = np.linspace(1, -1, n) * min(1.6, 0.32 * n)
    vmax = max(tabla.p_conversion.max(), p_orden1)
    norm = Normalize(vmin=0, vmax=vmax)
    nombre = CORTO[canal].replace("\n", " ")

    # Orden 1
    x1, xc1 = -3.6, -1.9
    ax.scatter(x1, 0, s=2600, c=[ROJOS(norm(p_orden1))], edgecolors="white", zorder=3)
    ax.text(x1, 0, CORTO[canal], ha="center", va="center", fontsize=7.5, zorder=4,
            color="white" if norm(p_orden1) > 0.55 else TINTA)
    ax.scatter(xc1, 0, s=1500, c=[ROJO], marker="s", zorder=3)
    ax.text(xc1, 0, "CONV.", ha="center", va="center", fontsize=7, color="white", zorder=4, fontweight="bold")
    _flecha(ax, (x1, 0), (xc1, 0), 0.8 + 120 * p_orden1, ROJO, alpha=0.8, r0=0.42, r1=0.3)
    ax.text((x1 + xc1) / 2, 0.22, f"{p_orden1:.2%}", ha="center", fontsize=8, color=ROJO, fontweight="bold")
    ax.text(x1 - 0.1, 1.95, "Orden 1: un solo nodo", fontsize=9, color=TINTA, fontweight="bold", ha="left")

    # Separador
    ax.plot([-1.2, -1.2], [-1.9, 1.9], color=GRIS_CLARO, lw=1)

    # Orden 2
    xp, xs, xc2 = -0.3, 1.5, 3.4
    ax.scatter(xc2, 0, s=1500, c=[ROJO], marker="s", zorder=3)
    ax.text(xc2, 0, "CONV.", ha="center", va="center", fontsize=7, color="white", zorder=4, fontweight="bold")
    for y, (_, f) in zip(ys, tabla.iterrows()):
        prev = CORTO[f.desde].replace("\n", " ")
        ax.text(xp, y, prev, ha="right", va="center", fontsize=7.5, color=GRIS)
        _flecha(ax, (xp + 0.05, y), (xs - 0.28, y), 0.8, GRIS_CLARO, alpha=0.9)
        ax.scatter(xs, y, s=900, c=[ROJOS(norm(f.p_conversion))], edgecolors="white", zorder=3)
        ax.text(xs, y, f"{prev[:4]}.›\n{nombre[:4]}.", ha="center", va="center", fontsize=5.8, zorder=4,
                color="white" if norm(f.p_conversion) > 0.55 else TINTA)
        _flecha(ax, (xs, y), (xc2, 0), 0.6 + 120 * f.p_conversion, ROJO, alpha=0.75, r0=0.26, r1=0.3)
        ax.text(xs - 0.34, y + 0.05, f"{f.p_conversion:.2%}   n = {f.visitas:,.0f}",
                fontsize=6.8, color=ROJO, va="bottom", ha="right", fontweight="bold")
    ax.text(xp - 1.0, 1.95, f"Orden 2: el nodo se parte según el canal anterior", fontsize=9,
            color=TINTA, fontweight="bold", ha="left")
    ax.text(xp - 1.0, -1.98, "canal anterior", fontsize=7, color=GRIS, ha="left", va="top")
    ax.text(xs, -1.98, "estado (pareja)", fontsize=7, color=GRIS, ha="center", va="top")
    ax.set_xlim(-4.3, 4.0); ax.set_ylim(-2.15, 2.15); ax.axis("off")
    return ax


# -----------------------------------------------------------------------------
# Figura 3 — efecto de remoción
# -----------------------------------------------------------------------------
def dibujar_remocion(W, canal, axes):
    """Dos paneles: el grafo completo y el grafo sin `canal`. Devuelve
    (p_base, p_sin, efecto) con efecto = p_base − p_sin."""
    p_base = prob_conversion(W)
    W2 = quitar_canal(W, canal)
    p_sin = prob_conversion(W2)
    dibujar_grafo_orden1(W, axes[0], mostrar_barra=False,
                         titulo=f"Grafo completo · P(INICIO → CONVERSIÓN) = {p_base:.2%}")
    dibujar_grafo_orden1(W2, axes[1], quitado=canal, mostrar_barra=False,
                         titulo=f"Sin {CORTO[canal].replace(chr(10), ' ').lower()} · "
                                f"P(INICIO → CONVERSIÓN) = {p_sin:.2%}")
    # Nota sobre el canal quitado
    pos = _posiciones_orden1()
    x, y = pos[canal]
    axes[1].text(x - _radio_canal(canal) - 0.1, y - 0.1,
                 "quitado: lo que entraba\naquí se manda a NULO", fontsize=7, color=ROJO,
                 ha="right", va="center", zorder=6)
    return p_base, p_sin, p_base - p_sin


# -----------------------------------------------------------------------------
# Figura 4 — grafo ilustrativo de la escuela (sin datos)
# -----------------------------------------------------------------------------
def dibujar_grafo_escuela(ax):
    """Traducción del grafo de Anderl al recorrido de un prospecto en la escuela.
    Es un ESQUEMA: no hay probabilidades porque aún no hay datos. Los touchpoints
    digitales y las experiencias gratuitas son estados transitorios; el inicio de un
    programa pagado y el NULO son absorbentes."""
    MAT, NUL = "MATRÍCULA\n(inicio de\nprograma pagado)", "NULO\n(no se matricula\nen el horizonte)"
    nodos = {
        INICIO: ((-3.0, 0.0), "inicio"),
        "Pauta\ndigital": ((-1.6, 1.1), "digital"), "Orgánico /\nweb": ((-1.6, 0.0), "digital"),
        "Correo\n(mail mkt)": ((-1.6, -1.1), "digital"),
        "Formulario\n(lead)": ((0.0, 0.55), "digital"), "Contacto\ncomercial": ((0.0, -0.75), "experiencia"),
        "Evento /\nmasterclass": ((1.6, 1.1), "experiencia"), "LS / IL": ((1.6, -0.2), "experiencia"),
        MAT: ((3.3, 0.7), "matricula"), NUL: ((3.3, -1.0), "nulo"),
    }
    estilos = {"inicio": dict(fc="white", ec=TINTA), "digital": dict(fc="#F2F2F1", ec=GRIS),
               "experiencia": dict(fc="#FBE3E5", ec=ROJO), "matricula": dict(fc=ROJO, ec=ROJO),
               "nulo": dict(fc=GRIS_CLARO, ec=GRIS)}
    caja = {}
    for n, ((x, y), tipo) in nodos.items():
        caja[n] = ax.text(x, y, n, ha="center", va="center", fontsize=7.6, zorder=4,
                          color="white" if tipo == "matricula" else TINTA,
                          fontweight="bold" if tipo in ("inicio", "matricula", "nulo") else "normal",
                          bbox=dict(boxstyle="round,pad=0.45", lw=1.2, **estilos[tipo]))
    aristas = [
        (INICIO, "Pauta\ndigital"), (INICIO, "Orgánico /\nweb"), (INICIO, "Correo\n(mail mkt)"),
        ("Pauta\ndigital", "Formulario\n(lead)"), ("Orgánico /\nweb", "Formulario\n(lead)"),
        ("Correo\n(mail mkt)", "Contacto\ncomercial"),
        ("Formulario\n(lead)", "Contacto\ncomercial"), ("Formulario\n(lead)", "Evento /\nmasterclass"),
        ("Contacto\ncomercial", "LS / IL"), ("Evento /\nmasterclass", "LS / IL"),
        ("Evento /\nmasterclass", MAT), ("LS / IL", MAT), ("Contacto\ncomercial", MAT),
        ("LS / IL", NUL), ("Contacto\ncomercial", NUL),
    ]
    ax.figure.canvas.draw()  # necesario para que las cajas tengan tamaño
    for a, b in aristas:
        rojo = b == MAT
        rad = -0.3 if (a, b) == ("Contacto\ncomercial", MAT) else 0.06
        ax.annotate("", xy=nodos[b][0], xytext=nodos[a][0], zorder=2,
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, color=ROJO if rojo else GRIS,
                                    alpha=0.85 if rojo else 0.65, shrinkA=3, shrinkB=3,
                                    patchA=caja[a].get_bbox_patch(), patchB=caja[b].get_bbox_patch(),
                                    connectionstyle=f"arc3,rad={rad}"))
    for x, texto, tipo in [(-3.1, "digital (transitorio)", "digital"),
                           (-0.9, "experiencia o contacto (transitorio)", "experiencia"),
                           (2.2, "absorbente", "matricula"), (3.1, "absorbente", "nulo")]:
        ax.text(x, -1.95, texto, fontsize=7.5, ha="left", color="white" if tipo == "matricula" else TINTA,
                bbox=dict(boxstyle="round,pad=0.3", **estilos[tipo]))
    ax.set_xlim(-3.5, 4.0); ax.set_ylim(-2.15, 1.65); ax.axis("off")
    return ax


# -----------------------------------------------------------------------------
# Grafos completos construidos con los datos (órdenes 1 a 4)
# -----------------------------------------------------------------------------
def grafo_datos(clics, recorridos, k):
    """Grafo dirigido de orden k construido con los datos (networkx).

    Nodos: estados de orden k (tuplas de los últimos k canales; al inicio del
    recorrido, tuplas más cortas), más INICIO, CONVERSIÓN y NULO.
    Atributos de cada estado: `visitas`, `compras` (veces que fue el último estado
    de un recorrido con compra) y `p_conv` = compras / visitas.
    Aristas: `n` = veces observada la transición y `w` = n / salidas del origen
    (la probabilidad de transición w_ij del paper)."""
    from .markov import codificar_estados, decodificar
    s = codificar_estados(clics, k)
    jid = clics["journey_id"].to_numpy()
    pos = clics["pos"].to_numpy()
    ultimo = np.r_[jid[1:] != jid[:-1], True]
    y = recorridos.set_index("journey_id")["convierte"].reindex(jid[ultimo]).to_numpy().astype(bool)
    origen = np.r_[np.where(pos == 0, -1, np.r_[-9, s[:-1]])]  # -1 = INICIO
    df = pd.DataFrame({"o": origen, "d": s})
    pares = df.groupby(["o", "d"]).size()
    G = nx.DiGraph()
    visitas = pd.Series(s).value_counts()
    compras = pd.Series(s[ultimo][y]).value_counts()
    nombre = lambda c: decodificar(int(c), k, P.CANALES)
    for c, v in visitas.items():
        cp = int(compras.get(c, 0))
        G.add_node(nombre(c), visitas=int(v), compras=cp, p_conv=cp / v, ultimo=P.CANALES[int(c) % BASE - 1])
    G.add_node(INICIO); G.add_node(CONV); G.add_node(NULO)
    salidas = df.groupby("o").size()
    for (o, d), n in pares.items():
        a = INICIO if o == -1 else nombre(o)
        G.add_edge(a, nombre(d), n=int(n))
    fin = pd.DataFrame({"s": s[ultimo], "y": y}).groupby(["s", "y"]).size()
    for (c, conv), n in fin.items():
        G.add_edge(nombre(c), CONV if conv else NULO, n=int(n))
    for a in G.nodes:
        tot = sum(G.edges[a, b]["n"] for b in G.successors(a))
        for b in G.successors(a):
            G.edges[a, b]["w"] = G.edges[a, b]["n"] / tot
    return G


def resumen_grafos(clics, recorridos, ordenes=(1, 2, 3, 4)):
    """Tamaño del grafo según el orden: estados, aristas y qué tan vistos están."""
    filas = []
    for k in ordenes:
        G = grafo_datos(clics, recorridos, k)
        v = np.array([d["visitas"] for n, d in G.nodes(data=True) if "visitas" in d])
        aristas = sum(1 for a, b in G.edges if b not in (CONV, NULO) and a != INICIO)
        filas.append((k, len(v), aristas, int(np.median(v)), 100 * (v < 50).mean()))
    return pd.DataFrame(filas, columns=["orden", "estados", "aristas_entre_estados",
                                        "visitas_mediana", "% estados con < 50 visitas"])


def dibujar_grafo_orden2(G2, ax, top_aristas=45):
    """Grafo de orden 2 completo, con los datos. Cada nodo es una pareja
    (anterior › actual) y se ubica en una rejilla: columna = canal actual,
    fila = canal anterior (fila de arriba = primer clic del recorrido, sin anterior).
    Tamaño = visitas; color = probabilidad de pasar a CONVERSIÓN desde la pareja.
    Se dibujan las `top_aristas` transiciones con más recorridos (las demás existen)."""
    canales = P.CANALES
    filas = ["(inicio)"] + canales
    pos = {}
    for n, d in G2.nodes(data=True):
        if "visitas" not in d:
            continue
        partes = n.split(" > ")
        prev = "(inicio)" if len(partes) == 1 else partes[0]
        pos[n] = (canales.index(partes[-1]) * 1.0, -filas.index(prev) * 0.82)
    vmax = max(d["visitas"] for n, d in G2.nodes(data=True) if n in pos)
    pconv = {n: G2.nodes[n]["p_conv"] for n in pos}
    norm = Normalize(vmin=0, vmax=np.quantile(list(pconv.values()), 0.97))
    radio = {n: 0.05 + 0.3 * np.sqrt(G2.nodes[n]["visitas"] / vmax) for n in pos}
    aristas = sorted(((a, b, d["n"]) for a, b, d in G2.edges(data=True) if a in pos and b in pos),
                     key=lambda t: -t[2])[:top_aristas]
    nmax = aristas[0][2]
    for a, b, n in aristas:
        _flecha(ax, pos[a], pos[b], 0.3 + 4 * n / nmax, TINTA, rad=0.25, alpha=0.22,
                r0=radio[a], r1=radio[b])
    from matplotlib.patches import Circle
    for n in pos:
        ax.add_patch(Circle(pos[n], radio[n], fc=ROJOS(norm(pconv[n])), ec="white", lw=0.8, zorder=3))
    for j, c in enumerate(canales):
        ax.text(j, 0.62, CORTO[c], ha="center", va="bottom", fontsize=7.2, color=TINTA)
    for i, f in enumerate(filas):
        ax.text(-0.62, -i * 0.82, CORTO.get(f, f).replace("\n", " "), ha="right", va="center",
                fontsize=7.2, color=TINTA if i else GRIS)
    ax.text(3.5, 1.15, "canal actual →", ha="center", fontsize=8, color=GRIS)
    ax.text(-2.3, -3.3, "canal anterior ↓", ha="center", fontsize=8, color=GRIS, rotation=90)
    ax.set_xlim(-2.6, 7.5); ax.set_ylim(-7.0, 1.4); ax.set_aspect("equal"); ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=ROJOS, norm=norm)
    cb = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, shrink=0.5)
    cb.set_label("P(pasar a CONVERSIÓN desde la pareja)", fontsize=7.5)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    cb.ax.tick_params(labelsize=7)
    return ax


def dibujar_crecimiento(grafos, axes, semilla=1):
    """Un panel por orden: el grafo completo con los datos (solo estados; sin INICIO,
    CONVERSIÓN ni NULO). Muestra cómo se multiplican los nodos al dar memoria y que
    la mayoría de los nodos de orden 3 y 4 se ven pocas veces (puntos claros)."""
    for ax, (k, G) in zip(axes, grafos.items()):
        H = G.subgraph([n for n, d in G.nodes(data=True) if "visitas" in d]).copy()
        H.remove_edges_from(nx.selfloop_edges(H))
        pesos = {e: H.edges[e]["n"] for e in H.edges}
        nx.set_edge_attributes(H, {e: np.log1p(v) for e, v in pesos.items()}, "peso")
        pos = nx.spring_layout(H.to_undirected(), weight="peso", seed=semilla,
                               k=1.6 / np.sqrt(len(H)), iterations=60)
        v = np.array([H.nodes[n]["visitas"] for n in H.nodes])
        tam = 4 + 260 * np.sqrt(v / v.max()) if k > 1 else 120 + 900 * np.sqrt(v / v.max())
        poco = v < 50
        color = [GRIS_CLARO if p else ROJO for p in poco]
        aristas = sorted(H.edges, key=lambda e: -pesos[e])[: min(len(H.edges), 25 * len(H))]
        nx.draw_networkx_edges(H, pos, edgelist=aristas, ax=ax, alpha=0.06 if k > 2 else 0.25,
                               width=0.4, arrows=False, edge_color=TINTA)
        nx.draw_networkx_nodes(H, pos, ax=ax, node_size=tam, node_color=color, linewidths=0)
        if k == 1:
            nx.draw_networkx_labels(H, pos, {n: CORTO[n] for n in H.nodes}, ax=ax, font_size=6.5)
        ax.set_title(f"Orden {k}: {len(H):,} nodos".replace(",", ".") +
                     f" · {poco.mean():.0%} en gris", loc="left", fontsize=9.5)
        ax.axis("off")
    return axes


def dibujar_recorridos(W, clics, recorridos, ids, ax, titulo=None):
    """Recorridos concretos de los datos dibujados sobre el grafo de orden 1:
    cada recorrido es un camino INICIO → canales → CONVERSIÓN o NULO."""
    dibujar_grafo_orden1(W, ax, mostrar_barra=False, titulo=titulo, umbral=1.1)
    for p in list(ax.patches):
        if isinstance(p, FancyArrowPatch):
            p.set_alpha(0.07)
    pos = _posiciones_orden1()
    colores = [ROJO, "#1F5FAD", "#2E8B57", "#B8860B"]
    leyenda = []
    for j, jid in enumerate(ids):
        ruta = [P.CANALES[c] for c in clics.loc[clics.journey_id == jid, "canal"]]
        compra = bool(recorridos.set_index("journey_id").loc[jid, "convierte"])
        camino = [INICIO] + ruta + [CONV if compra else NULO]
        col = colores[j % len(colores)]
        for i, (a, b) in enumerate(zip(camino[:-1], camino[1:])):
            if a == b:   # repetir canal: un pequeño lazo
                x, y = pos[a]
                r = _radio_canal(a)
                ax.add_patch(FancyArrowPatch((x - 0.14, y + r - 0.02), (x + 0.14, y + r - 0.02), arrowstyle="-|>",
                             connectionstyle="arc3,rad=-1.6", mutation_scale=9, lw=1.6, color=col, zorder=6))
                continue
            ra = RADIO_FIJO.get(a, _radio_canal(a) if a in P.CANALES else 0.3)
            rb = RADIO_FIJO.get(b, _radio_canal(b) if b in P.CANALES else 0.3)
            _flecha(ax, pos[a], pos[b], 2.0, col, rad=0.12 + 0.07 * j, alpha=0.95, r0=ra, r1=rb, z=6)
        leyenda.append((col, " → ".join(CORTO[c].replace("\n", " ") for c in ruta) +
                        ("  ⇒ compra" if compra else "  ⇒ sin compra")))
    for i, (col, txt) in enumerate(leyenda):
        ax.text(-3.8, -2.45 - 0.28 * i, "■ ", color=col, fontsize=9, va="center")
        ax.text(-3.55, -2.45 - 0.28 * i, txt, color=TINTA, fontsize=7.4, va="center")
    ax.set_ylim(-2.45 - 0.28 * len(leyenda), 2.75)
    return ax
