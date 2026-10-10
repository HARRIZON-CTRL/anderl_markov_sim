"""Evaluación como en Anderl et al. (2016): validación cruzada de 10 particiones,
AUC y lift del decil superior, dentro y fuera de muestra."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import params as P
from .baselines import Logit, ReglaClic
from .markov import MarkovK


def lift_decil_superior(y, score):
    n = max(1, int(round(0.10 * len(y))))
    top = np.argsort(-np.asarray(score), kind="stable")[:n]
    return np.asarray(y)[top].mean() / np.mean(y)


def modelos_paper(n_canales=len(P.CANALES)):
    return {
        "Markov orden 1": lambda: MarkovK(1),
        "Markov orden 2": lambda: MarkovK(2),
        "Markov orden 3": lambda: MarkovK(3),
        "Markov orden 4": lambda: MarkovK(4),
        "Último clic": lambda: ReglaClic("ultimo"),
        "Primer clic": lambda: ReglaClic("primero"),
        "Logit 1": lambda: Logit(1, n_canales),
        "Logit 2": lambda: Logit(2, n_canales),
    }


def _evaluar_particion(f, ids_tr, ids_te, clics, recorridos, modelos):
    """Entrena los modelos con una partición y los evalúa dentro y fuera de muestra.
    Es la unidad de trabajo que se reparte entre los workers."""
    y_all = recorridos.set_index("journey_id")["convierte"]
    jid_clic = clics["journey_id"].to_numpy()
    c_tr = clics[np.isin(jid_clic, ids_tr)]
    c_te = clics[np.isin(jid_clic, ids_te)]
    r_tr = recorridos[recorridos["journey_id"].isin(ids_tr)]
    filas = []
    for nombre, crear in modelos.items():
        m = crear().fit(c_tr, r_tr)
        for muestra, c in (("dentro", c_tr), ("fuera", c_te)):
            s = m.predecir(c)
            y = y_all.reindex(s.index).to_numpy()
            filas.append((nombre, f, muestra, roc_auc_score(y, s.to_numpy()),
                          lift_decil_superior(y, s.to_numpy())))
    return filas


def validacion_cruzada(clics, recorridos, modelos=None, k_folds=10, semilla=P.SEMILLA,
                       verbose=True, n_jobs=-1):
    """Validación cruzada estratificada de k particiones (protocolo del paper).

    Las particiones son independientes, así que se reparten entre workers con joblib
    (n_jobs = -1 usa todos los núcleos del equipo; n_jobs = 1 corre en serie).
    El resultado es idéntico en serie o en paralelo: las particiones las fija la semilla."""
    from joblib import Parallel, delayed
    modelos = modelos or modelos_paper()
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=semilla)
    ids = recorridos["journey_id"].to_numpy()
    tareas = (delayed(_evaluar_particion)(f, ids[tr], ids[te], clics, recorridos, modelos)
              for f, (tr, te) in enumerate(skf.split(recorridos, recorridos["convierte"])))
    resultados = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(tareas)
    filas = [fila for r in resultados for fila in r]
    return pd.DataFrame(filas, columns=["modelo", "particion", "muestra", "auc", "lift"])


def tabla_comparacion(res):
    """Promedio y DE por modelo, junto a lo reportado en el paper (Tabla 5, DS1)."""
    t = res.groupby(["modelo", "muestra"])[["auc", "lift"]].agg(["mean", "std"]).unstack("muestra")
    t.columns = [f"{m}_{a}_{b}" for m, a, b in t.columns]
    paper = pd.DataFrame(P.RESULTADOS_PAPER_DS1,
                         index=["paper_auc_dentro", "paper_auc_fuera",
                                "paper_lift_dentro", "paper_lift_fuera"]).T
    orden = list(P.RESULTADOS_PAPER_DS1)
    out = paper.join(t).reindex(orden)
    return out[["paper_auc_fuera", "auc_mean_fuera", "auc_std_fuera",
                "paper_lift_fuera", "lift_mean_fuera", "lift_std_fuera",
                "paper_auc_dentro", "auc_mean_dentro"]]


# -----------------------------------------------------------------------------
# Comparaciones adicionales con el paper (Tablas 7 y 8) y extensión
# -----------------------------------------------------------------------------
def comparar_tabla7(clics, recorridos, orden=3):
    """Efecto de remoción por canal (en %) del Markov de orden `orden` frente a la Tabla 7.

    Simplificación declarada: para el canal c se quitan a la vez todos los estados cuyo
    último canal es c; el paper promedia los efectos de esos estados uno por uno.
    """
    m = MarkovK(orden).fit(clics, recorridos)
    ef, p_base = m.efecto_remocion(P.CANALES)
    ef["paper_markov3_%"] = [P.TABLA7_DS1[c][0] for c in ef.canal]
    ef = ef.rename(columns={"efecto_%": "sim_markov%d_%%" % orden})
    return ef[["canal", "paper_markov3_%", "sim_markov%d_%%" % orden, "efecto_remocion"]], p_base


def comparar_tabla8(clics, recorridos):
    """Ajusta el Logit 1 sin regularización sobre los datos simulados y lo pone al lado
    de los coeficientes de la Tabla 8. Sirve para ver cuánto del efecto de los conteos
    se recupera cuando el verdadero mecanismo también tiene recencia, orden y retorno."""
    from sklearn.linear_model import LogisticRegression
    from .baselines import matriz_conteos
    X = matriz_conteos(clics, len(P.CANALES))
    y = recorridos.set_index("journey_id")["convierte"].reindex(X.index)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(penalty=None, max_iter=2000).fit(X.to_numpy(), y.to_numpy())
    t = pd.DataFrame({"canal": P.CANALES,
                      "paper_beta": [P.TABLA8_DS1_BETA[c] for c in P.CANALES],
                      "sim_beta": lr.coef_[0]})
    t.loc[len(t)] = ["Intercepto", P.TABLA8_DS1_INTERCEPTO, lr.intercept_[0]]
    return t


def extension_tamano(tamanos, k_folds=10, semilla=P.SEMILLA, verbose=True, n_jobs=-1, largo_minimo=1):
    """Paso 7 de la guía: repite la validación cruzada cambiando SOLO el número de
    recorridos (personas), con la misma tasa de conversión y el mismo mecanismo.
    Devuelve el AUC y el lift fuera de muestra (promedio y DE entre particiones).
    `largo_minimo` > 1 sirve para el análisis adicional de recorridos más largos."""
    from .simulate import simular
    filas = []
    for n in tamanos:
        c, r = simular(n_recorridos=n, semilla=semilla, largo_minimo=largo_minimo)
        res = validacion_cruzada(c, r, k_folds=k_folds, verbose=False, n_jobs=n_jobs)
        res = (res[res.muestra == "fuera"].groupby("modelo")
               .agg(auc=("auc", "mean"), auc_de=("auc", "std"), lift=("lift", "mean")).reset_index())
        res["n_recorridos"] = n
        res["clics"] = len(c)
        res["conversiones"] = int(r.convierte.sum())
        filas.append(res)
        if verbose:
            print(f"n = {n:,} listo ({len(c):,} clics, {int(r.convierte.sum()):,} compras)")
    return pd.concat(filas, ignore_index=True)
