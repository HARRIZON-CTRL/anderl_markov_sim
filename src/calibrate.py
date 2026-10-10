"""Calibración del supuesto S3 (mecanismo de conversión).

Problema
--------
El paper no publica cómo se relaciona la secuencia de canales con la compra; solo
publica sus consecuencias: el AUC que obtiene cada modelo (Tabla 5). El simulador
necesita cuatro números que el paper no da:

    escala_conteo   cuánto pesan los conteos por canal (Tabla 8) en la compra
    peso_ultimo     cuánto pesa, además, el último canal (recencia)
    escala_orden    cuánto pesan las secuencias al final del recorrido
    efecto_retorno  cuánto sube la propensión al volver (log del número de clics)

Regla de calibración (método de momentos simulados, versión simple)
-------------------------------------------------------------------
1. Se define una rejilla de valores candidatos para los cuatro parámetros, en dos
   etapas: una gruesa (rango amplio) y una fina (alrededor de lo mejor de la gruesa).
2. Para cada combinación se simula una muestra de calibración (150.000 recorridos,
   semilla SEMILLA_CALIBRACION, distinta de la del análisis final), se separa 70/30
   y se mide el AUC fuera de muestra de cinco modelos ancla.
3. Se elige la combinación que minimiza la suma de errores al cuadrado frente a la
   Tabla 5 (columna "fuera de muestra"):

       pérdida = Σ_ancla (AUC_simulado − AUC_paper)²

Modelos ancla: último clic, primer clic, Logit 1, Markov orden 2 y Markov orden 3.
Quedan FUERA de la calibración, y sirven como verificación independiente:
Markov orden 1 y 4, Logit 2, todos los lift, la Tabla 7 y la Tabla 8.

Honestidad metodológica: como los anclas se usaron para calibrar, su parecido con el
paper no es evidencia de nada; la evidencia está en los modelos que no se usaron.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import params as P
from .baselines import Logit, ReglaClic
from .markov import MarkovK
from .simulate import simular

ANCLAS = {
    "Último clic": lambda: ReglaClic("ultimo"),
    "Primer clic": lambda: ReglaClic("primero"),
    "Logit 1": lambda: Logit(1, len(P.CANALES)),
    "Markov orden 2": lambda: MarkovK(2),
    "Markov orden 3": lambda: MarkovK(3),
}

# Etapa 1 (gruesa): cubre un rango amplio.
REJILLA = {
    "escala_conteo": [0.15, 0.25, 0.35],
    "peso_ultimo": [0.0, 0.25, 0.5],
    "escala_orden": [0.5, 1.0, 1.5],
    "efecto_retorno": [1.0, 1.5, 2.0, 2.5],
}

# Etapa 2 (fina): alrededor de las mejores combinaciones de la etapa 1, ampliando
# escala_conteo, que quedó en el borde superior de la rejilla gruesa.
REJILLA_FINA = {
    "escala_conteo": [0.35, 0.45, 0.55],
    "peso_ultimo": [0.0, 0.25],
    "escala_orden": [0.5, 1.0, 1.5, 2.0],
    "efecto_retorno": [1.5, 2.0, 2.5],
}


def auc_anclas(escala_conteo, peso_ultimo, escala_orden, efecto_retorno,
               n=150_000, semilla=P.SEMILLA_CALIBRACION):
    """AUC fuera de muestra (partición 70/30) de los modelos ancla para una combinación."""
    clics, rec = simular(n_recorridos=n, semilla=semilla, escala_conteo=escala_conteo,
                         peso_ultimo=peso_ultimo, escala_orden=escala_orden,
                         efecto_retorno=efecto_retorno)
    prueba = np.random.default_rng(0).random(n) < 0.3
    ids_tr = rec.journey_id.to_numpy()[~prueba]
    c_tr = clics[clics.journey_id.isin(ids_tr)]
    c_te = clics[~clics.journey_id.isin(ids_tr)]
    y = rec.set_index("journey_id")["convierte"]
    out = {}
    for nombre, crear in ANCLAS.items():
        m = crear().fit(c_tr, rec[~prueba])
        s = m.predecir(c_te)
        out[nombre] = roc_auc_score(y.reindex(s.index), s)
    return out


def calibrar(rejilla=REJILLA, verbose=True):
    """Recorre la rejilla y devuelve una tabla ordenada por pérdida (la mejor primero)."""
    objetivo = {k: P.RESULTADOS_PAPER_DS1[k][1] for k in ANCLAS}
    filas = []
    combos = list(itertools.product(*rejilla.values()))
    for i, (ec, pu, eo, er) in enumerate(combos, 1):
        auc = auc_anclas(ec, pu, eo, er)
        perdida = sum((auc[k] - objetivo[k]) ** 2 for k in ANCLAS)
        filas.append({"escala_conteo": ec, "peso_ultimo": pu, "escala_orden": eo, "efecto_retorno": er,
                      **{f"AUC {k}": v for k, v in auc.items()}, "perdida": perdida})
        if verbose and i % 12 == 0:
            print(f"{i}/{len(combos)} combinaciones evaluadas")
    tabla = pd.DataFrame(filas).sort_values("perdida").reset_index(drop=True)
    return tabla, objetivo
