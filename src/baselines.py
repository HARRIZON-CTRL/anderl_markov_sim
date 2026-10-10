"""Modelos de comparación de la Tabla 5 de Anderl et al. (2016)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _ultimo_y_primero(clics):
    g = clics.groupby("journey_id")["canal"]
    return g.last(), g.first()


class ReglaClic:
    """Último o primer clic: tasa de conversión observada según ese canal."""

    def __init__(self, cual="ultimo"):
        self.cual = cual

    def fit(self, clics, recorridos):
        ult, pri = _ultimo_y_primero(clics)
        canal = ult if self.cual == "ultimo" else pri
        y = recorridos.set_index("journey_id")["convierte"].reindex(canal.index)
        self.tasa_ = y.groupby(canal.to_numpy()).mean()
        self.global_ = y.mean()
        return self

    def predecir(self, clics):
        ult, pri = _ultimo_y_primero(clics)
        canal = ult if self.cual == "ultimo" else pri
        return pd.Series(self.tasa_.reindex(canal.to_numpy()).fillna(self.global_).to_numpy(),
                         index=canal.index)


def _indice_recorridos(clics):
    """Posición 0..n-1 de cada clic según su recorrido (en orden de journey_id)."""
    ids, inv = np.unique(clics["journey_id"].to_numpy(), return_inverse=True)
    return ids, inv


def matriz_conteos(clics, n_canales):
    """Logit 1: número de clics por canal (sin orden).
    Se arma con numpy (bincount) en vez de pd.crosstab: mismo resultado, ~20 veces más rápido."""
    ids, inv = _indice_recorridos(clics)
    canal = clics["canal"].to_numpy()
    m = np.bincount(inv * n_canales + canal, minlength=len(ids) * n_canales)
    return pd.DataFrame(m.reshape(len(ids), n_canales), index=pd.Index(ids, name="journey_id"))


def matriz_posiciones(clics, n_canales, posiciones=4):
    """Logit 2: dummy canal i en la posición t contada desde el final (t = 1..4)."""
    ids, inv = _indice_recorridos(clics)
    pos = clics["pos"].to_numpy()
    largo = np.zeros(len(ids), dtype=np.int64)
    np.maximum.at(largo, inv, pos + 1)
    desde_final = largo[inv] - pos
    ok = desde_final <= posiciones
    col = (desde_final[ok] - 1) * n_canales + clics["canal"].to_numpy()[ok]
    ncol = posiciones * n_canales
    m = np.zeros(len(ids) * ncol, dtype=np.int64)
    m[inv[ok] * ncol + col] = 1
    return pd.DataFrame(m.reshape(len(ids), ncol), index=pd.Index(ids, name="journey_id"))


class Logit:
    def __init__(self, tipo, n_canales):
        self.tipo, self.n = tipo, n_canales

    def _X(self, clics):
        return matriz_conteos(clics, self.n) if self.tipo == 1 else matriz_posiciones(clics, self.n)

    def fit(self, clics, recorridos):
        X = self._X(clics)
        y = recorridos.set_index("journey_id")["convierte"].reindex(X.index)
        self.m_ = LogisticRegression(max_iter=1000).fit(X.to_numpy(), y.to_numpy())
        return self

    def predecir(self, clics):
        X = self._X(clics)
        return pd.Series(self.m_.predict_proba(X.to_numpy())[:, 1], index=X.index)
