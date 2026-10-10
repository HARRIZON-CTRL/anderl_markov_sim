"""Simulador de recorridos de compra calibrado con Anderl et al. (2016), Data Set 1.

Qué produce
-----------
Dos tablas:

    clics       una fila por clic:        journey_id, pos (0 = primer clic), canal (entero)
    recorridos  una fila por recorrido:   journey_id, largo, convierte (0/1), p_verdadera

Cómo lo hace (cuatro pasos, uno por función)
--------------------------------------------
1. _muestrear_largos        cuántos clics tiene cada recorrido           (Tabla 1 + S2)
2. _muestrear_canales       qué canal tiene cada clic                    (Tabla 3 + S1)
3. _indice_conversion       qué tan "comprador" es cada recorrido         (Tabla 8 + S3)
4. _calibrar_intercepto     ajusta el nivel para que compre el 1,64 %     (Tabla 1)

Los canales se codifican como enteros 0..7 en el orden de params.CANALES.
"""
import numpy as np
import pandas as pd

from . import params as P


# -----------------------------------------------------------------------------
# Paso 1. Largo de cada recorrido
# -----------------------------------------------------------------------------
def _muestrear_largos(n, rng, ds=P.DS1, s=P.SUPUESTOS):
    """Sortea el número de clics de cada recorrido.

    La Tabla 1 da tres proporciones exactas:
        P(L = 1)       = 1 - 206.519 / 600.978 = 65,6 %
        P(2 <= L <= 4) = (206.519 - 48.344) / 600.978 = 26,3 %
        P(L >= 5)      = 48.344 / 600.978 = 8,0 %
    Lo que no da es la forma dentro de cada tramo (supuesto S2):
        - L = 2, 3, 4 se reparten 55 / 28 / 17 %.
        - L >= 5 es 5 + una binomial negativa de cola larga, porque la DE (8,86) es
          mucho mayor que la media (2,46): hay pocos recorridos muy largos.
    """
    p_ge2 = ds["n_largo_ge2"] / ds["n_recorridos"]
    p_ge5 = ds["n_largo_ge5"] / ds["n_recorridos"]
    tramo = rng.choice(3, size=n, p=[1 - p_ge2, p_ge2 - p_ge5, p_ge5])

    largos = np.ones(n, dtype=np.int64)                       # tramo 0: un clic
    t1 = tramo == 1
    largos[t1] = rng.choice([2, 3, 4], size=t1.sum(), p=s["p_largo_2_4"])
    t2 = tramo == 2
    r, mu = s["cola_dispersion"], s["cola_media_extra"]
    extra = rng.negative_binomial(r, r / (r + mu), size=t2.sum())  # media mu, varianza grande
    largos[t2] = np.minimum(5 + extra, s["largo_max"])
    return largos


# -----------------------------------------------------------------------------
# Paso 2. Canal de cada clic
# -----------------------------------------------------------------------------
def _muestrear_canales(largos, rng, s=P.SUPUESTOS):
    """Sortea el canal de cada clic (supuesto S1).

    Regla: el primer clic se sortea según la participación de la Tabla 3. Cada clic
    siguiente repite el canal anterior con probabilidad `arrastre` (carryover); si no
    lo repite, se vuelve a sortear según la Tabla 3.

    Equivale a una cadena de Markov con matriz W = a·I + (1 - a)·π, cuya distribución
    estacionaria es π (la Tabla 3). Por eso la participación simulada coincide con la
    del paper.

    Implementación vectorizada: se sortea un canal "nuevo" para todos los clics y se
    marca cuáles repiten; los que repiten heredan el último canal nuevo (forward fill).
    """
    pi = np.array([P.PARTICIPACION_CLICS[c] for c in P.CANALES])
    pi = pi / pi.sum()
    total = int(largos.sum())
    journey_id = np.repeat(np.arange(len(largos)), largos)
    inicio = np.r_[0, np.cumsum(largos)[:-1]]
    pos = np.arange(total) - np.repeat(inicio, largos)        # 0, 1, 2... dentro de cada recorrido

    nuevo = rng.choice(len(pi), size=total, p=pi)
    repite = (rng.random(total) < s["arrastre"]) & (pos > 0)  # el primer clic nunca repite
    canal = pd.Series(np.where(repite, np.nan, nuevo)).ffill().astype(int).to_numpy()
    return journey_id, pos, canal


# -----------------------------------------------------------------------------
# Paso 3. Índice de conversión de cada recorrido
# -----------------------------------------------------------------------------
def _indice_conversion(clics, largos, escala_conteo, peso_ultimo, escala_orden, efecto_retorno=0.0, s=P.SUPUESTOS):
    """Calcula, para cada recorrido, la parte variable del logit de comprar (supuesto S3).

        índice = escala_conteo · Σ_canales β_c · (clics en c)      (a) Tabla 8, escala calibrada
               + peso_ultimo · β_(último canal)                    (b) recencia, calibrado
               + escala_orden · [efecto de la pareja final          (c) secuencia, calibrado
                                 + efecto del trío final]
               + efecto_retorno · log(número de clics)              (d) volver, calibrado

    Lectura de cada parte:
      (a) Lo que el paper estimó con su Logit 1: cada clic en afiliado sube la propensión
          (β = 1,15) y cada clic en display la baja mucho (β = -3,43). Se usan los signos y
          las proporciones de la Tabla 8, pero su escala se calibra: el Logit 1 del paper no
          tenía términos de recencia ni de orden, así que parte de lo que atribuyó a los
          conteos aquí lo explican (b) y (c).
      (b) El último contacto pesa más que los anteriores. Sin este término, la regla
          del último clic no tendría ventaja sobre la del primero, y en el paper sí la tiene.
      (c) Ciertas secuencias al final del recorrido (p. ej. display -> búsqueda) suben la
          propensión más allá de los conteos. Esto es lo que permite que un modelo que
          recuerda el orden (Markov de orden 2-4, Logit 2) gane a uno que no (Logit 1).
      (d) Volver al sitio indica interés sostenido: quien regresa varias veces compra más
          que quien entra una sola vez. Con log, el segundo clic suma mucho y el vigésimo
          casi nada. Este término es el que hace que distinguir "primer contacto" de
          "contacto repetido" (lo que ya hace un Markov de orden 2) mejore tanto la
          predicción, como se ve en el salto de 0,74 a 0,82 de la Tabla 5.
    """
    idx = {c: i for i, c in enumerate(P.CANALES)}
    beta = np.array([P.TABLA8_DS1_BETA[c] for c in P.CANALES])
    jid = clics["journey_id"].to_numpy()
    can = clics["canal"].to_numpy()

    # (a) suma de β por clic, agregada por recorrido
    z = escala_conteo * np.bincount(jid, weights=beta[can], minlength=len(largos))

    # (b) último, penúltimo y antepenúltimo canal de cada recorrido (-1 si no existe)
    fin = np.cumsum(largos) - 1
    ult = can[fin]
    pen = np.where(largos >= 2, can[np.maximum(fin - 1, 0)], -1)
    ant = np.where(largos >= 3, can[np.maximum(fin - 2, 0)], -1)
    z = z + peso_ultimo * beta[ult]

    # (c) efectos de secuencia al final del recorrido
    orden = np.zeros(len(largos))
    for (a, b), v in s["efectos_pares"].items():
        orden += v * ((pen == idx[a]) & (ult == idx[b]))
    for (a, b, c), v in s["efectos_trios"].items():
        orden += v * ((ant == idx[a]) & (pen == idx[b]) & (ult == idx[c]))
    # (d) volver: un recorrido con más de un clic indica interés sostenido
    z = z + efecto_retorno * np.log(largos)
    return z + escala_orden * orden


def _sigmoide(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


# -----------------------------------------------------------------------------
# Paso 4. Intercepto que fija la tasa de conversión
# -----------------------------------------------------------------------------
def _calibrar_intercepto(z, objetivo):
    """Busca (bisección) el intercepto b0 tal que el promedio de sigmoide(b0 + z) sea
    la tasa objetivo. Así la simulación compra exactamente lo que dice la Tabla 1."""
    lo, hi = -20.0, 10.0
    for _ in range(70):
        mid = (lo + hi) / 2
        tasa = _sigmoide(mid + z).mean()
        lo, hi = (mid, hi) if tasa < objetivo else (lo, mid)
    return (lo + hi) / 2


# -----------------------------------------------------------------------------
# Función principal
# -----------------------------------------------------------------------------
def simular(n_recorridos=P.DS1["n_recorridos"], tasa_conversion=P.DS1["tasa_conversion"],
            semilla=P.SEMILLA, escala_conteo=None, peso_ultimo=None, escala_orden=None, efecto_retorno=None,
            largo_minimo=1):
    """Genera un conjunto de datos simulado con las características de DS1.

    Parámetros
    ----------
    n_recorridos     número de recorridos (por defecto, los 600.978 de la Tabla 1)
    tasa_conversion  proporción que compra (por defecto, 1,64 % de la Tabla 1)
    semilla          semilla aleatoria (reproducibilidad)
    escala_conteo, peso_ultimo, escala_orden
                     parámetros de S3; si no se pasan, se usan los calibrados (params.CALIBRADOS)
    largo_minimo     mínimo de clics por recorrido (1 = como el paper). Con 3, todo recorrido
                     de 1 o 2 clics pasa a tener 3: personas con más acciones (análisis 7.2).

    Qué es cada fila
    ----------------
    Un RECORRIDO es una persona (en el paper, una cookie): la secuencia de sus clics hasta que
    compra o pasan 30 días sin contacto. Un CLIC es una acción dentro del recorrido. No hay
    identificador de persona por encima del recorrido: si alguien vuelve, cuenta como otro
    recorrido (no se modela la recurrencia). Tampoco hay fechas: solo el orden (supuesto S4).
    """
    escala_conteo = P.CALIBRADOS["escala_conteo"] if escala_conteo is None else escala_conteo
    efecto_retorno = P.CALIBRADOS["efecto_retorno"] if efecto_retorno is None else efecto_retorno
    peso_ultimo = P.CALIBRADOS["peso_ultimo"] if peso_ultimo is None else peso_ultimo
    escala_orden = P.CALIBRADOS["escala_orden"] if escala_orden is None else escala_orden
    rng = np.random.default_rng(semilla)

    largos = _muestrear_largos(n_recorridos, rng)                         # paso 1
    largos = np.maximum(largos, largo_minimo)
    jid, pos, canal = _muestrear_canales(largos, rng)                      # paso 2
    clics = pd.DataFrame({"journey_id": jid, "pos": pos, "canal": canal})
    z = _indice_conversion(clics, largos, escala_conteo, peso_ultimo, escala_orden, efecto_retorno)  # paso 3
    b0 = _calibrar_intercepto(z, tasa_conversion)                          # paso 4
    p = _sigmoide(b0 + z)
    y = (rng.random(n_recorridos) < p).astype(int)                         # compra sí/no

    recorridos = pd.DataFrame({"journey_id": np.arange(n_recorridos), "largo": largos,
                               "convierte": y, "p_verdadera": p})
    return clics, recorridos


def mostrar_recorridos(clics, recorridos, n=8, solo_largos=True, semilla=1):
    """Muestra algunos recorridos simulados como texto legible (para el notebook)."""
    r = recorridos[recorridos.largo.between(2, 6)] if solo_largos else recorridos
    ids = r.sample(n, random_state=semilla).journey_id
    sub = clics[clics.journey_id.isin(ids)]
    filas = []
    for j, g in sub.groupby("journey_id"):
        ruta = " → ".join(P.CANALES[c] for c in g.sort_values("pos").canal)
        rec = recorridos.loc[j]
        filas.append((j, ruta, "compra" if rec.convierte else "sin compra", round(rec.p_verdadera, 4)))
    return pd.DataFrame(filas, columns=["recorrido", "ruta", "resultado", "p_verdadera"])


def resumen_vs_paper(clics, recorridos, ds=P.DS1):
    """Compara lo simulado con las Tablas 1, 3 y 7 (último y primer clic) del paper."""
    t1 = pd.DataFrame([
        ("Recorridos", ds["n_recorridos"], len(recorridos)),
        ("Clics", ds["n_clics"], len(clics)),
        ("% largo ≥ 2", 100 * ds["n_largo_ge2"] / ds["n_recorridos"], 100 * (recorridos.largo >= 2).mean()),
        ("% largo ≥ 5", 100 * ds["n_largo_ge5"] / ds["n_recorridos"], 100 * (recorridos.largo >= 5).mean()),
        ("Largo medio", ds["largo_media"], recorridos.largo.mean()),
        ("Largo DE", ds["largo_de"], recorridos.largo.std()),
        ("Conversiones", ds["n_conversiones"], int(recorridos.convierte.sum())),
        ("Tasa de conversión (%)", 100 * ds["tasa_conversion"], 100 * recorridos.convierte.mean()),
    ], columns=["medida", "paper", "simulado"])

    part = clics["canal"].value_counts(normalize=True)
    g = clics.groupby("journey_id")["canal"]
    conv = recorridos.set_index("journey_id")["convierte"].astype(bool)
    ult = g.last()[conv]
    pri = g.first()[conv]
    t3 = pd.DataFrame({
        "canal": P.CANALES,
        "clics_paper_%": [100 * P.PARTICIPACION_CLICS[c] for c in P.CANALES],
        "clics_sim_%": [100 * part.get(i, 0.0) for i in range(len(P.CANALES))],
        "ultimo_clic_paper_%": [P.TABLA7_DS1[c][1] for c in P.CANALES],
        "ultimo_clic_sim_%": [100 * (ult == i).mean() for i in range(len(P.CANALES))],
        "primer_clic_paper_%": [P.TABLA7_DS1[c][2] for c in P.CANALES],
        "primer_clic_sim_%": [100 * (pri == i).mean() for i in range(len(P.CANALES))],
    })
    return t1, t3


def explicar_simulacion(recorridos_a_mostrar=None, n=200_000, semilla=P.SEMILLA):
    """Para el notebook: corre los 4 pasos del simulador y devuelve, para unos pocos
    recorridos, lo que produce cada paso. Se simulan `n` recorridos para que el
    intercepto del paso 4 sea el real (fija la compra en 1,64 %) y se muestran algunos.

    Devuelve un diccionario con:
      paso1  largo de cada recorrido
      paso2  tabla de clics (una fila por clic) con el canal en texto
      paso3  desglose del índice de compra: (a) conteos, (b) recencia, (c) secuencia, (d) volver
      paso4  intercepto, probabilidad de compra y si compró
    """
    c = P.CALIBRADOS
    rng = np.random.default_rng(semilla)
    largos = _muestrear_largos(n, rng)
    jid, pos, canal = _muestrear_canales(largos, rng)
    clics = pd.DataFrame({"journey_id": jid, "pos": pos, "canal": canal})
    # índice por partes (cada una con las demás en cero)
    a = _indice_conversion(clics, largos, c["escala_conteo"], 0, 0, 0)
    b = _indice_conversion(clics, largos, 0, c["peso_ultimo"], 0, 0)
    cc = _indice_conversion(clics, largos, 0, 0, c["escala_orden"], 0)
    d = _indice_conversion(clics, largos, 0, 0, 0, c["efecto_retorno"])
    z = a + b + cc + d
    b0 = _calibrar_intercepto(z, P.DS1["tasa_conversion"])
    p = _sigmoide(b0 + z)
    y = (rng.random(n) < p).astype(int)
    if recorridos_a_mostrar is None:   # 2 que compran y 2 que no, de 1 a 5 clics
        compran = np.nonzero((y == 1) & (largos >= 2) & (largos <= 5))[0][:2]
        no = np.nonzero((y == 0) & (largos <= 4))[0][:2]
        recorridos_a_mostrar = sorted(np.r_[compran, no].tolist())
    ids = list(recorridos_a_mostrar)
    sel = clics[clics.journey_id.isin(ids)].copy()
    sel["canal_texto"] = [P.CANALES[k] for k in sel.canal]
    ruta = sel.groupby("journey_id")["canal_texto"].apply(lambda s: " → ".join(s))
    return {
        "paso1": pd.DataFrame({"recorrido": ids, "largo (clics)": largos[ids]}),
        "paso2": sel[["journey_id", "pos", "canal", "canal_texto"]].rename(
            columns={"journey_id": "recorrido", "pos": "posición", "canal": "canal (código)", "canal_texto": "canal"}),
        "paso3": pd.DataFrame({"recorrido": ids, "ruta": ruta.reindex(ids).to_numpy(),
                               "(a) conteos": a[ids], "(b) recencia": b[ids], "(c) secuencia": cc[ids],
                               "(d) volver": d[ids], "índice z": z[ids]}),
        "paso4": pd.DataFrame({"recorrido": ids, "intercepto b0": b0, "índice z": z[ids],
                               "p = sigmoide(b0 + z)": p[ids], "compra (sorteo)": y[ids]}),
        "tasa_simulada": y.mean(),
    }
