"""Grafos de Markov de orden k para recorridos de clientes (Anderl et al., 2016; Archak et al., 2010).

Estados: tuplas con los últimos k canales (al inicio del recorrido, tuplas más cortas).
Estados especiales: START, CONV (conversión) y NULL (sin conversión).

Predicción de un recorrido: probabilidad de pasar a CONV desde su último estado.
(Con k = 1 coincide con la regla de último clic, como en la Tabla 5 del paper.)
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

BASE = 16  # codificación de tuplas: dígito = canal + 1 (0 = vacío)


def codificar_estados(clics, k):
    """Agrega a cada clic el código del estado de orden k (últimos k canales)."""
    can = clics["canal"].to_numpy() + 1
    pos = clics["pos"].to_numpy()
    codigo = np.zeros(len(can), dtype=np.int64)
    for j in range(k):  # j = 0 canal actual, j = 1 anterior, ...
        prev = np.roll(can, j)
        valido = pos >= j
        codigo += np.where(valido, prev, 0) * (BASE ** j)
    return codigo


def decodificar(codigo, k, canales):
    partes = []
    for j in reversed(range(k)):
        d = (codigo // BASE ** j) % BASE
        if d > 0:
            partes.append(canales[d - 1])
    return " > ".join(partes)


class MarkovK:
    """Modelo de Markov de orden k.

    suavizado (m): decisión de implementación, no del paper. Un estado de orden alto
    que se vio pocas veces da una tasa muy ruidosa (p. ej. 1 compra en 3 visitas = 33 %).
    Por eso la tasa de cada estado se "encoge" hacia la de su estado de orden inferior:

        p_k(s) = (compras_k(s) + m · p_{k-1}(s')) / (visitas_k(s) + m)

    donde s' es el mismo estado sin el canal más antiguo. Con muchas visitas manda el
    dato; con pocas, manda el orden inferior. m = 0 reproduce la estimación sin suavizar.
    """

    def __init__(self, k, suavizado=20):
        self.k = k
        self.m = suavizado

    def fit(self, clics, recorridos):
        self.tasa_global_ = recorridos["convierte"].mean()
        self.tablas_ = {}
        y = recorridos.set_index("journey_id")["convierte"]
        es_ultimo = np.r_[clics["journey_id"].to_numpy()[1:] != clics["journey_id"].to_numpy()[:-1], True]
        for k in range(1, self.k + 1):  # se guardan órdenes menores para respaldo (backoff)
            s = codificar_estados(clics, k)
            df = pd.DataFrame({"s": s, "ult": es_ultimo,
                               "conv": es_ultimo & y.reindex(clics["journey_id"]).to_numpy().astype(bool)})
            g = df.groupby("s").agg(salidas=("s", "size"), conv=("conv", "sum"))
            self.tablas_[k] = g
        self._clics_fit = clics
        self._y_fit = y
        return self

    def predecir(self, clics):
        """Probabilidad de pasar a CONV desde el último estado de cada recorrido,
        construida de orden 1 a orden k con el suavizado descrito arriba."""
        jid = clics["journey_id"].to_numpy()
        es_ultimo = np.r_[jid[1:] != jid[:-1], True]
        pred = np.full(es_ultimo.sum(), self.tasa_global_)   # orden 0: tasa global
        for k in range(1, self.k + 1):
            s = codificar_estados(clics, k)[es_ultimo]
            g = self.tablas_[k]
            n = g["salidas"].reindex(s).fillna(0).to_numpy()
            c = g["conv"].reindex(s).fillna(0).to_numpy()
            if self.m > 0:
                pred = (c + self.m * pred) / (n + self.m)
            else:
                pred = np.where(n > 0, c / np.maximum(n, 1), pred)
        return pd.Series(pred, index=jid[es_ultimo])

    # ------------------------------------------------------------------
    # Grafo y efecto de remoción
    # ------------------------------------------------------------------
    def matriz_transicion(self):
        """Matriz W sobre estados {START, estados..., CONV, NULL}."""
        clics, y = self._clics_fit, self._y_fit
        s = codificar_estados(clics, self.k)
        jid = clics["journey_id"].to_numpy()
        pos = clics["pos"].to_numpy()
        es_ultimo = np.r_[jid[1:] != jid[:-1], True]
        estados = np.unique(s)
        idx = {e: i + 1 for i, e in enumerate(estados)}
        n = len(estados) + 3
        START, CONV, NULL = 0, n - 2, n - 1
        si = np.vectorize(idx.get)(s)
        origen, destino = [], []
        origen.append(np.full((pos == 0).sum(), START)); destino.append(si[pos == 0])
        sig = ~es_ultimo
        origen.append(si[sig]); destino.append(si[np.r_[False, sig[:-1]]])
        conv = y.reindex(jid[es_ultimo]).to_numpy().astype(bool)
        origen.append(si[es_ultimo]); destino.append(np.where(conv, CONV, NULL))
        o, d = np.concatenate(origen), np.concatenate(destino)
        C = sparse.coo_matrix((np.ones(len(o)), (o, d)), shape=(n, n)).tocsr()
        filas = np.asarray(C.sum(axis=1)).ravel()
        filas[filas == 0] = 1
        W = sparse.diags(1 / filas) @ C
        return W.tocsr(), estados, (START, CONV, NULL)

    @staticmethod
    def _prob_conversion(W, START, CONV, NULL, quitar=None):
        n = W.shape[0]
        transitorios = np.array([i for i in range(n) if i not in (CONV, NULL)])
        W = W.tolil(copy=True)
        if quitar is not None and len(quitar):
            for q in quitar:  # aristas entrantes a q se redirigen a NULL
                col = W[:, q].toarray().ravel()
                for i in np.nonzero(col)[0]:
                    W[i, NULL] = W[i, NULL] + col[i]
                    W[i, q] = 0
        W = W.tocsr()
        Q = W[transitorios][:, transitorios]
        r = W[transitorios][:, [CONV]].toarray().ravel()
        A = sparse.identity(len(transitorios)) - Q
        x = spsolve(A.tocsc(), r)
        return x[np.where(transitorios == START)[0][0]]

    def efecto_remocion(self, canales):
        """Efecto de remoción por canal: cuánto cae P(START -> CONV) al quitar
        todos los estados cuyo último canal es c. Se reporta también en % del total."""
        W, estados, (START, CONV, NULL) = self.matriz_transicion()
        base = self._prob_conversion(W, START, CONV, NULL)
        ultimo = estados % BASE  # dígito del canal actual
        filas = []
        for c_idx, c in enumerate(canales):
            quitar = np.nonzero(ultimo == c_idx + 1)[0] + 1
            p = self._prob_conversion(W, START, CONV, NULL, quitar)
            filas.append((c, base - p))
        df = pd.DataFrame(filas, columns=["canal", "efecto_remocion"])
        df["efecto_%"] = 100 * df["efecto_remocion"] / df["efecto_remocion"].sum()
        return df, base
